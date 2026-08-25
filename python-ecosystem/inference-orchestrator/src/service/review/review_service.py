import os
import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from dotenv import load_dotenv
from mcp_use import MCPClient

from model.dtos import ReviewRequestDto
from utils.mcp_config import MCPConfigBuilder
from llm.llm_factory import LLMFactory
from utils.response_parser import ResponseParser
from service.rag.rag_client import RagClient
from service.review.issue_processor import post_process_analysis_result
from service.review.plugin_context import apply_plugin_file_policy
from service.review.quality_capture import (
    ReviewQualityCaptureSession,
    create_quality_capture_session,
    review_response_indicates_failure,
    wrap_quality_capture_llm,
)
from service.review.evidence_scopes import (
    process_review_evidence_scopes,
    select_review_evidence_diff,
)
from utils.hunk_coverage import validate_acquired_diff_manifest
from utils.error_sanitizer import create_user_friendly_error
from service.review.orchestrator import MultiStageReviewOrchestrator
from service.review.snapshot_identity import validate_review_snapshot_identity

logger = logging.getLogger(__name__)

class ReviewService:
    """Service class for handling code review requests with streaming support."""
    
    # Maximum retries for LLM-based response fixing
    MAX_FIX_RETRIES = 2

    # Maximum concurrent reviews (each spawns a JVM subprocess + LLM calls)
    MAX_CONCURRENT_REVIEWS = int(os.environ.get("MAX_CONCURRENT_REVIEWS", "20"))

    # Hard timeout ceiling per review (seconds). Configurable via .env
    REVIEW_TIMEOUT_SECONDS = int(os.environ.get("REVIEW_TIMEOUT_SECONDS", "1500"))
    MCP_SESSION_INITIALIZATION_TIMEOUT_SECONDS = float(os.environ.get(
        "MCP_SESSION_INITIALIZATION_TIMEOUT_SECONDS",
        "30",
    ))
    def __init__(self):
        load_dotenv(interpolate=False)
        self.default_jar_path = os.environ.get(
            "MCP_SERVER_JAR",
            #"/var/www/html/codecrow/codecrow-public/java-ecosystem/mcp-servers/vcs-mcp/target/codecrow-vcs-mcp-1.0.jar",
            "/app/codecrow-vcs-mcp-1.0.jar"
        )
        self.rag_client = RagClient()
        self._review_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REVIEWS)

    async def process_review_request(
            self,
            request: ReviewRequestDto,
            event_callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a review request with optional event streaming.

        Args:
            request: The review request data
            event_callback: Optional callback to receive progress events
                          Expected signature: callback(event: Dict) -> None
                          Events have structure: {"type": "status|progress|error|final", ...}

        Returns:
            Dict with "result" key containing the analysis result or error
        """
        # Validate before provider construction, MCP
        # startup, or dry-run dispatch. Every review mode must describe the
        # same exact immutable repository snapshot.
        validate_review_snapshot_identity(request)
        async with self._review_semaphore:
            if request.promptDryRun:
                return await self._process_prompt_dry_run(request, event_callback)
            quality_capture = create_quality_capture_session(request)
            review_event_callback = (
                quality_capture.wrap_event_callback(event_callback)
                if quality_capture is not None
                else event_callback
            )
            try:
                response = await self._process_review(
                    request=request,
                    repo_path=None,
                    event_callback=review_event_callback,
                    quality_capture=quality_capture,
                )
            except BaseException as exception:
                if quality_capture is not None:
                    await quality_capture.complete(None, exception)
                raise
            if quality_capture is not None:
                await quality_capture.complete(
                    response,
                    failed=review_response_indicates_failure(response),
                )
                self._emit_event(review_event_callback, {
                    "type": "status",
                    "state": "review_quality_capture_completed",
                    "message": "Review quality capture completed",
                    "qualityCapture": quality_capture.receipt(),
                })
            return response

    async def _process_prompt_dry_run(
            self,
            request: ReviewRequestDto,
            event_callback: Optional[Callable[[Dict], None]],
    ) -> Dict[str, Any]:
        """Run real context assembly with a capturing model and store its prompts."""
        enabled = os.environ.get(
            "ANALYSIS_PROMPT_DRY_RUN_ENABLED", "false"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            raise ValueError(
                "promptDryRun was requested while ANALYSIS_PROMPT_DRY_RUN_ENABLED is false"
            )

        try:
            simulated_findings = int(os.environ.get(
                "ANALYSIS_PROMPT_DRY_RUN_SYNTHETIC_FINDINGS_PER_FILE", "6"
            ))
        except ValueError as exception:
            raise ValueError(
                "ANALYSIS_PROMPT_DRY_RUN_SYNTHETIC_FINDINGS_PER_FILE must be an integer"
            ) from exception
        try:
            simulated_findings_max_total = int(os.environ.get(
                "ANALYSIS_PROMPT_DRY_RUN_SYNTHETIC_FINDINGS_MAX_TOTAL", "24"
            ))
        except ValueError as exception:
            raise ValueError(
                "ANALYSIS_PROMPT_DRY_RUN_SYNTHETIC_FINDINGS_MAX_TOTAL must be an integer"
            ) from exception

        self._emit_event(event_callback, {
            "type": "status",
            "state": "prompt_dry_run_started",
            "message": (
                "Prompt dry run started: real project context will be assembled "
                "without calling the review LLM"
            ),
        })
        logger.info(
            "prompt_dry_run_started project=%s pr=%s job=%s",
            request.projectId,
            request.pullRequestId,
            request.promptDryRunId,
        )

        from service.review.prompt_dry_run import capture_and_store_review_prompts

        summary = await capture_and_store_review_prompts(
            request,
            self._rag_client_for_request(request),
            simulated_findings_per_file=simulated_findings,
            simulated_findings_max_total=simulated_findings_max_total,
            event_callback=event_callback,
        )
        self._emit_event(event_callback, {
            "type": "status",
            "state": "prompt_dry_run_completed",
            "message": (
                "Prompt dry run completed; artifact: "
                f"{summary['promptArtifact']['containerPath']}"
            ),
            "promptArtifact": summary["promptArtifact"],
        })
        logger.info(
            "prompt_dry_run_completed project=%s pr=%s job=%s artifact=%s "
            "provider_calls=0",
            request.projectId,
            request.pullRequestId,
            request.promptDryRunId,
            summary["promptArtifact"]["containerPath"],
        )
        return {"result": summary}

    async def _process_review(
            self,
            request: ReviewRequestDto,
            repo_path: Optional[str] = None,
            event_callback: Optional[Callable[[Dict], None]] = None,
            quality_capture: Optional[ReviewQualityCaptureSession] = None,
    ) -> Dict[str, Any]:
        """
        Internal method that handles both regular and local repo reviews.
        
        When rawDiff is provided:
        - Diff is embedded directly in prompt (no need to call getPullRequestDiff)
        - MCP agent still has access to all other tools (getFile, getComments, etc.)
        
        When rawDiff is not provided:
        - MCP agent fetches diff via getPullRequestDiff tool

        Emits events via event_callback:
        - {"type": "status", "state": "started", "message": "..."}
        - {"type": "status", "state": "mcp_initialized", "message": "..."}
        - {"type": "progress", "step": N, "max_steps": M, "message": "..."}
        - {"type": "mcp_output", "content": "...", "step": N}
        - {"type": "final", "result": {...}}
        - {"type": "error", "message": "..."}
        """
        jar_path = self.default_jar_path

        # An incremental execution owns the delta manifest. The full PR diff is
        # still carried as snapshot context, but must not be validated or
        # planned as though every historical PR path belonged to this run.
        review_evidence_diff = select_review_evidence_diff(request)
        has_raw_diff = bool(review_evidence_diff)

        # ── MCP-free branch reconciliation fast path ──
        # When Java provides pre-fetched file contents AND there are previous
        # issues to reconcile, skip MCP entirely: no JVM subprocess, no tool
        # calls — just a direct LLM call.
        # This check is done BEFORE the jar existence check since MCP-free
        # reconciliation doesn't need the jar at all.
        # NOTE: When there are no previous issues (e.g. direct push with no
        # prior review history), we fall through to the standard path which
        # runs a full multi-stage review of the diff.
        is_branch_reconciliation = request.analysisType == "BRANCH_ANALYSIS"
        has_file_contents = bool(request.reconciliationFileContents)
        has_previous_issues = bool(request.previousCodeAnalysisIssues)
        needs_multistage_review = not (
            is_branch_reconciliation and has_previous_issues
        )

        # Parse and prove the acquired diff before MCP startup, provider
        # construction, or any repository-context query. Reconciliation
        # requests intentionally carry an issue-scoped diff rather than the
        # complete changed-file manifest, so their separate direct path is not
        # subject to this full-review equality check.
        processed_diff = None
        full_pr_processed_diff = None
        if has_raw_diff and needs_multistage_review:
            evidence_scopes = process_review_evidence_scopes(request)
            processed_diff = evidence_scopes.review
            full_pr_processed_diff = evidence_scopes.full_pr
            validate_acquired_diff_manifest(
                request.changedFiles or (),
                request.deletedFiles or (),
                processed_diff,
            )

            logger.info(
                f"Diff pre-processed: {processed_diff.total_files} files, "
                f"+{processed_diff.total_additions}/-{processed_diff.total_deletions}, "
                f"skipped: {processed_diff.skipped_files}"
            )

            # Incremental review and PR-wide reasoning use deliberately separate
            # evidence scopes. Stage 0/1, hunk coverage, RAG overlay indexing,
            # and publication anchors continue to use only ``processed_diff``
            # (the delta). Stage 2 receives this bounded base-to-head parse so it
            # cannot mistake a one-file delta for the complete PR state.
            if (
                request.analysisMode == "INCREMENTAL"
                and request.deltaDiff
            ):
                if full_pr_processed_diff is not None:
                    logger.info(
                        "Full PR evidence scope prepared separately: %d files; "
                        "review/publication scope remains %d delta files",
                        len(full_pr_processed_diff.files),
                        len(processed_diff.files),
                    )
                else:
                    logger.warning(
                        "Full PR evidence scope unavailable; continuing the "
                        "delta review with PR-wide omission claims disabled"
                    )
            else:
                full_pr_processed_diff = processed_diff

            if processed_diff.truncated:
                self._emit_event(event_callback, {
                    "type": "warning",
                    "message": processed_diff.truncation_reason
                })

        if is_branch_reconciliation and has_file_contents and has_previous_issues:
            try:
                async with asyncio.timeout(self.REVIEW_TIMEOUT_SECONDS):
                    logger.info(
                        "Branch reconciliation with %d pre-fetched files — skipping MCP",
                        len(request.reconciliationFileContents),
                    )
                    self._emit_event(event_callback, {
                        "type": "status",
                        "state": "direct_reconciliation",
                        "message": f"Direct reconciliation mode ({len(request.reconciliationFileContents)} files pre-fetched)"
                    })

                    llm = self._create_llm(request, quality_capture)
                    pr_metadata = self._build_pr_metadata(request)
                    num_issues = len(pr_metadata.get("previousCodeAnalysisIssues", []))
                    logger.info(f"Branch reconciliation: {num_issues} previous issues to process (MCP-free)")

                    orchestrator = MultiStageReviewOrchestrator(
                        llm=llm,
                        mcp_client=None,  # No MCP needed
                        rag_client=None,
                        event_callback=event_callback,
                    )

                    result = await orchestrator.execute_batched_branch_analysis(
                        request, pr_metadata
                    )

                    # Post-process
                    if result and 'issues' in result:
                        result = post_process_analysis_result(result)

                    self._emit_event(event_callback, {
                        "type": "status",
                        "state": "completed",
                        "message": "Branch reconciliation completed (MCP-free)"
                    })
                    return {"result": result}

            except TimeoutError:
                timeout_msg = f"Review timed out after {self.REVIEW_TIMEOUT_SECONDS} seconds"
                logger.error(timeout_msg)
                self._emit_event(event_callback, {"type": "error", "message": timeout_msg})
                error_response = ResponseParser.create_error_response(
                    "Review timed out", timeout_msg
                )
                return {"result": error_response}

            except Exception as e:
                logger.error(f"Direct reconciliation failed: {str(e)}", exc_info=True)
                sanitized_message = create_user_friendly_error(e)
                error_response = ResponseParser.create_error_response(
                    "Direct reconciliation failed", sanitized_message
                )
                self._emit_event(event_callback, {
                    "type": "error",
                    "message": sanitized_message
                })
                return {"result": error_response}

        use_mcp_tools = bool(request.useMcpTools)
        mcp_available = use_mcp_tools and os.path.exists(jar_path)
        if use_mcp_tools and not mcp_available:
            logger.warning(
                "Agentic repository tools requested but the VCS MCP server is "
                "unavailable at %s; continuing with direct file-review prompts",
                jar_path,
            )
            self._emit_event(event_callback, {
                "type": "status",
                "state": "mcp_degraded",
                "message": (
                    "Repository tools are unavailable; analysis will continue "
                    "with the diff and prepared context"
                ),
            })
        
        client = None
        try:
            async with asyncio.timeout(self.REVIEW_TIMEOUT_SECONDS):
                context = "with pre-fetched diff" if has_raw_diff else "fetching diff via MCP"
                self._emit_event(event_callback, {
                    "type": "status",
                    "state": "started",
                    "message": f"Analysis starting ({context})"
                })

                # Create LLM instance
                llm = self._create_llm(request, quality_capture)
                request_rag_client = self._rag_client_for_request(request)
                agent_service = None

                if mcp_available:
                    try:
                        self._emit_event(event_callback, {
                            "type": "status",
                            "state": "mcp_initializing",
                            "message": "Initializing repository tools"
                        })
                        rag_mcp_context = self._build_rag_mcp_context(
                            request,
                            request_rag_client,
                        )
                        config = MCPConfigBuilder.build_config(
                            jar_path,
                            self._build_jvm_props(request),
                            rag_mcp_context=rag_mcp_context,
                        )
                        client = self._create_mcp_client(config)
                        # Keep the agent runtime lazy for MCP-disabled reviews and
                        # provider-free prompt capture.
                        from service.agent import AgentExecutionService
                        agent_service = AgentExecutionService(
                            llm=llm,
                            client=client,
                        )
                        optional_errors = await agent_service.initialize(
                            required_server_names=("codecrow-vcs-mcp",),
                            optional_server_names=(
                                ("codecrow-rag-mcp",)
                                if rag_mcp_context is not None
                                else ()
                            ),
                            session_timeout_seconds=(
                                self.MCP_SESSION_INITIALIZATION_TIMEOUT_SECONDS
                            ),
                        )
                        rag_start_error = optional_errors.get(
                            "codecrow-rag-mcp"
                        )
                        if rag_start_error is not None:
                            logger.warning(
                                "Optional on-demand RAG tool failed to start; "
                                "continuing with repository tools and prepared "
                                "RAG context: %s",
                                rag_start_error,
                            )
                            self._emit_event(event_callback, {
                                "type": "status",
                                "state": "rag_mcp_degraded",
                                "message": (
                                    "On-demand RAG search is unavailable; "
                                    "repository tools and prepared context remain "
                                    "available"
                                ),
                            })
                        self._emit_event(event_callback, {
                            "type": "status",
                            "state": "mcp_initialized",
                            "message": "Repository tools are ready"
                        })
                    except Exception as mcp_error:
                        logger.warning(
                            "Optional agentic repository tools failed to "
                            "initialize; continuing with direct Stage 1 prompts: %s",
                            mcp_error,
                            exc_info=True,
                        )
                        if client is not None:
                            try:
                                await client.close_all_sessions()
                            except Exception as close_error:
                                logger.debug(
                                    "Failed to close partially initialized MCP "
                                    "sessions: %s",
                                    close_error,
                                )
                        client = None
                        agent_service = None
                        self._emit_event(event_callback, {
                            "type": "status",
                            "state": "mcp_degraded",
                            "message": (
                                "Repository tools could not start; analysis will "
                                "continue with the diff and prepared context"
                            ),
                        })

                # Use the new pipeline
                self._emit_event(event_callback, {
                    "type": "status",
                    "state": "multi_stage_started",
                    "message": "Starting Multi-Stage Review Pipeline"
                })

                # This replaces the monolithic _execute_review_with_streaming call
                orchestrator = MultiStageReviewOrchestrator(
                    llm=llm,
                    mcp_client=client,
                    rag_client=request_rag_client,
                    event_callback=event_callback,
                    agent_service=agent_service,
                )

                # Check for Branch Analysis / Reconciliation mode
                if request.analysisType == "BRANCH_ANALYSIS":
                     logger.info("Executing Branch Analysis & Reconciliation mode")
                     pr_metadata = self._build_pr_metadata(request)
                     num_issues = len(pr_metadata.get("previousCodeAnalysisIssues", []))
                     logger.info(f"Branch reconciliation: {num_issues} previous issues to process")

                     if num_issues > 0:
                         # Use batched execution — splits large issue sets into
                         # token-safe batches automatically.  Single-batch fast
                         # path is handled inside execute_batched_branch_analysis.
                         result = await orchestrator.execute_batched_branch_analysis(
                             request, pr_metadata
                         )
                     else:
                         # No previous issues to reconcile — this is a fresh
                         # branch analysis (e.g. direct push with no prior
                         # review history).  Run the full multi-stage review
                         # pipeline on the diff instead of short-circuiting.
                         logger.info(
                             "Branch analysis: no previous issues — running "
                             "fresh multi-stage review on the diff"
                         )
                         result = await orchestrator.orchestrate_review(
                             request=request,
                             processed_diff=processed_diff,
                             full_pr_processed_diff=full_pr_processed_diff,
                         )
                else:
                    # Execute review with Multi-Stage Orchestrator
                    # Standard PR Review
                    result = await orchestrator.orchestrate_review(
                        request=request,
                        processed_diff=processed_diff,
                        full_pr_processed_diff=full_pr_processed_diff,
                    )


                # Post-process issues (no-op pass-through — Java handles all processing)
                if result and 'issues' in result:
                    self._emit_event(event_callback, {
                        "type": "status",
                        "state": "post_processing",
                        "message": "Finalizing issues (Java-side post-processing handles line correction, dedup, diff cleanup)..."
                    })
                    
                    result = post_process_analysis_result(result)

                self._emit_event(event_callback, {
                    "type": "status",
                    "state": "completed",
                    "message": "Pull Request analysis completed; the report is being generated..."
                })

                return {"result": result}

        except TimeoutError:
            timeout_msg = f"Review timed out after {self.REVIEW_TIMEOUT_SECONDS} seconds"
            logger.error(timeout_msg)
            self._emit_event(event_callback, {"type": "error", "message": timeout_msg})
            error_response = ResponseParser.create_error_response(
                "Review timed out", timeout_msg
            )
            return {"result": error_response}

        except Exception as e:
            # Log full error for debugging, but sanitize for user display
            logger.error(f"Review processing failed: {str(e)}", exc_info=True)
            sanitized_message = create_user_friendly_error(e)
            
            error_response = ResponseParser.create_error_response(
                "Review execution failed", sanitized_message
            )
            self._emit_event(event_callback, {
                "type": "error",
                "message": sanitized_message
            })
            return {"result": error_response}
        finally:
            # Client ownership begins at construction, so timeout/cancellation
            # during session startup cannot leave MCP child processes behind.
            if client is not None:
                try:
                    await client.close_all_sessions()
                except Exception as close_err:
                    logger.warning(f"Error closing MCP sessions: {close_err}")

    def _build_jvm_props(
            self,
            request: ReviewRequestDto,
    ) -> Dict[str, str]:
        """Build JVM properties from request."""
        return MCPConfigBuilder.build_jvm_props(
            project_id=request.projectId,
            pull_request_id=request.pullRequestId,
            workspace=request.projectVcsWorkspace,
            repo_slug=request.projectVcsRepoSlug,
            oAuthClient=request.oAuthClient,
            oAuthSecret=request.oAuthSecret,
            access_token=request.accessToken,
            max_allowed_tokens=request.maxAllowedTokens,
            vcs_provider=request.vcsProvider,
            vcs_base_url=request.vcsBaseUrl,
            local_repo_path=request.localRepoPath,
            local_repo_target_branch=request.localRepoTargetBranch,
            local_repo_revision=request.localRepoRevision,
        )

    def _build_rag_mcp_context(
            self,
            request: ReviewRequestDto,
            rag_client: Optional[RagClient],
    ) -> Optional[Dict[str, str]]:
        """Return the exact tenant/repository binding for optional RAG search."""
        if rag_client is None:
            return None
        context = {
            "workspace": request.projectWorkspace,
            "project": request.projectNamespace,
            "branch": request.targetBranchName,
            "revision": request.get_target_head_commit_hash(),
            "manifest": request.ragBaseGenerationManifestSha256,
            "collection_target": request.ragCollectionTarget,
        }
        if not all(
            isinstance(value, str) and bool(value.strip())
            for value in context.values()
        ):
            logger.info(
                "Stage 1 RAG search tool is unavailable because the request has "
                "no complete exact-generation binding; preassembled RAG context "
                "and repository tools remain available"
            )
            return None
        return context

    def _rag_client_for_request(
            self,
            request: ReviewRequestDto,
    ) -> Optional[RagClient]:
        """Apply global and project-scoped RAG enablement without shared mutation."""
        if not request.ragEnabled:
            logger.info(
                "RAG disabled for project request: project=%s PR=%s",
                request.projectId,
                request.pullRequestId or "n/a",
            )
            return None
        if not bool(getattr(self.rag_client, "enabled", True)):
            return None
        return self.rag_client

    def _create_mcp_client(self, config: Dict[str, Any]) -> MCPClient:
        """Create MCP client from configuration."""
        try:
            return MCPClient.from_dict(config)
        except Exception as e:
            raise Exception(f"Failed to construct MCPClient: {str(e)}")

    def _create_llm(
        self,
        request: ReviewRequestDto,
        quality_capture: Optional[ReviewQualityCaptureSession] = None,
    ):
        """Create LLM instance from request parameters."""
        try:
            # Log the model being used for this request
            logger.info(
                "Creating LLM for project %s PR %s: provider=%s, model=%s",
                request.projectId,
                request.pullRequestId or "n/a",
                request.aiProvider,
                request.aiModel,
            )
            
            llm = LLMFactory.create_llm(
                request.aiModel,
                request.aiProvider,
                request.aiApiKey,
                ai_base_url=getattr(request, 'aiBaseUrl', None),
                ai_custom_parameters=getattr(request, 'aiCustomParameters', None),
            )
            
            return wrap_quality_capture_llm(llm, quality_capture)
        except Exception as e:
            raise Exception(f"Failed to create LLM instance: {str(e)}")

    def _build_pr_metadata(self, request: ReviewRequestDto) -> Dict[str, Any]:
        """Build pull request metadata dictionary from request."""
        metadata = {
            "branch": request.get_rag_branch(),
            "baseBranch": request.get_rag_base_branch(),
            "commitHash": request.commitHash,
            "pullRequestId": request.pullRequestId,
            "repoSlug": request.projectVcsRepoSlug,
            "workspace": request.projectVcsWorkspace,
            "previousCodeAnalysisIssues": [
                issue.dict(by_alias=True, exclude_none=True)
                for issue in (request.previousCodeAnalysisIssues or [])
            ]
        }
        return metadata

    @staticmethod
    def _emit_event(callback: Optional[Callable[[Dict], None]], event: Dict[str, Any]) -> None:
        """Safely emit an event via the callback."""
        if callback:
            try:
                callback(event)
            except Exception as e:
                # Don't let callback errors break the processing
                logger.warning(f"Event callback failed: {e}")
