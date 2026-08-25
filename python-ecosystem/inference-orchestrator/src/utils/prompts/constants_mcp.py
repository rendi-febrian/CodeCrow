"""
Conditional MCP tool prompt sections (appended when useMcpTools=True).
"""

STAGE_1_MCP_TOOL_SECTION = """
## Repository Tools (Additional Context)
You are reviewing this batch as an agent. The diff, complete-file evidence, and
preassembled RAG context above remain your primary review input. When they leave
a concrete context gap, you can inspect the request-bound repository with:

- **getRootDirectory(workspace, projectKey, branch)** — List the target
  snapshot's top-level entries.
- **getDirectoryByPath(workspace, projectKey, branch, dirPath)** — Inspect a
  directory in the target snapshot.
- **getBranchFileContent(workspace, repoSlug, branch, filePath, startLine?,
  endLine?)** — Read a file or bounded line window from the target snapshot.
- **searchRepositoryCode(query, top_k)** — Search the exact indexed repository
  generation when that optional RAG tool is available.

The local tree tools are backed by the downloaded target-head snapshot when it
is available; provider-backed reads remain a fail-open fallback. Avoid rereading
files already supplied in this prompt. After gathering any missing context,
return the complete structured review for every file in this batch.

TARGET BRANCH/REVISION REF: {target_branch}
VCS WORKSPACE: {workspace}
VCS REPOSITORY (repoSlug/projectKey): {repo_slug}
"""

STAGE_3_MCP_VERIFICATION_SECTION = """
## Issue Re-verification (Optional)
Before producing the final report, you may verify HIGH/CRITICAL issues that seem uncertain
by reading actual file content from the exact reviewed PR revision.

Available tools:
- **getBranchFileContent(filePath, verificationId)** — Read an anchor-centred
  source window; the host supplies the exact reviewed commit
- **getPullRequestComments(pullRequestId)** — Read PR comments for additional context

RULES:
1. You have a MAXIMUM of {max_calls} verification calls total.
2. Only verify issues you are UNCERTAIN about — do not verify every issue.
3. Prioritize HIGH and CRITICAL severity, but you may verify a lower-severity
   finding when its correctness materially affects the final report.
4. Use the Verification ID from the complete verification-record list, including
   records whose persisted Original ID is empty.
5. Pass that same Verification ID in every file-content call. The host binds the
   returned source window to the finding and verifies that it covers its line.
6. If a finding has related_locations, every affected location must be read with
   the same Verification ID before dismissing the consolidated root finding.
7. If verification reveals a false positive, note its Verification ID for dismissal.
8. Missing, failed, partial, or ambiguous evidence means KEEP the finding.
9. After verification, produce the final executive summary.

REVIEWED REVISION: {review_revision}
PR ID: {pr_id}

## False Positive Dismissal
After producing the executive summary markdown, if your verification revealed any false
positives, append an HTML comment at the very end of your response with the IDs of issues
that should be removed from the issue list:

<!-- DISMISSED_ISSUES: ["issue_0", "issue_3"] -->

RULES for dismissal:
- Only dismiss issues you VERIFIED as false positives via successful file-content
  tool calls against the reviewed revision.
- Do NOT dismiss issues based on guessing — you must have read the relevant file.
- Do not dismiss a concrete architecture/maintainability defect merely because it
  has no immediate runtime crash; verify the claim as written.
- If no issues should be dismissed, omit the DISMISSED_ISSUES comment entirely.
"""
