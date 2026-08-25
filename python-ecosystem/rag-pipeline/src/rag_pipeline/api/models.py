"""
Pydantic request/response models for the repository-index API.

All models are defined here to avoid circular imports between routers
and to keep the router files focused on endpoint logic.
"""
import os
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


def _validate_repo_path(path: str) -> str:
    """Validate that a repo path is within the allowed root and contains no traversal."""
    allowed_root = os.environ.get("ALLOWED_REPO_ROOT", "/tmp")
    resolved = os.path.realpath(path)
    if not resolved.startswith(os.path.realpath(allowed_root)):
        raise ValueError(f"Path must be under {allowed_root}, got: {path}")
    return path


def _validate_source_root(path: Optional[str]) -> Optional[str]:
    if path is None or not path.strip() or path.strip() == ".":
        return None
    normalized = path.strip().replace("\\", "/")
    if (
        normalized.startswith("/")
        or normalized.endswith("/")
        or any(segment in {"", ".", ".."} for segment in normalized.split("/"))
    ):
        raise ValueError("source_root must be a normalized repository-relative directory")
    return normalized


# ── Index models ──

class IndexRequest(BaseModel):
    repo_path: str
    workspace: str
    project: str
    branch: str
    commit: str
    source_tree_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    collection_target: Optional[str] = Field(default=None, min_length=1)
    transfer_repo_ownership: bool = False
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    project_type: Optional[str] = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9-]{0,63}$",
    )
    source_root: Optional[str] = None

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        return _validate_repo_path(v)

    @field_validator("project_type", mode="before")
    @classmethod
    def validate_project_type(cls, v: Optional[str]) -> Optional[str]:
        if v is None or not str(v).strip() or str(v).strip().casefold() == "auto":
            return None
        return str(v).strip().casefold()

    @field_validator("source_root")
    @classmethod
    def validate_source_root(cls, v: Optional[str]) -> Optional[str]:
        return _validate_source_root(v)


class RevisionPreflightResponse(BaseModel):
    workspace: str
    project: str
    branch: str
    commit: str
    point_count: int = Field(gt=0)
    repository_revision: str
    repository_facts_sha256: str
    plugin_ids: List[str]
    plugin_fingerprint: str
    plugin_descriptor_fingerprint: str
    plugin_implementation_fingerprint: str
    index_representation_fingerprint: str
    current_index_representation_fingerprint: str
    generation_schema: str
    generation_member_count: int = Field(gt=0)
    generation_members_sha256: str
    generation_manifest_sha256: str
    source_tree_sha256: str
    index_include_patterns: List[str]
    index_exclude_patterns: List[str]
    index_selection_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EstimateRequest(BaseModel):
    repo_path: str
    include_patterns: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None

    @field_validator("repo_path")
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        return _validate_repo_path(v)


class EstimateResponse(BaseModel):
    file_count: int
    estimated_chunks: int
    max_files_allowed: int
    max_chunks_allowed: int
    within_limits: bool
    message: str


# ── Query models ──

class CodeSearchRequest(BaseModel):
    """Exact revision-bound structural code search."""
    query: str = Field(min_length=1, max_length=1000)
    workspace: str
    project: str
    branch: str
    repository_revision: str = Field(
        min_length=1,
        max_length=200,
    )
    repository_generation_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    collection_target: str = Field(min_length=1)
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        le=5000,
        description=(
            "Optional explicit result limit. Omit for complete matching up to "
            "the observable global matching-point safety limit."
        ),
    )


class DeterministicContextRequest(BaseModel):
    """Request for deterministic metadata-based context retrieval."""
    workspace: str
    project: str
    branches: List[str] = Field(min_length=1, max_length=1)
    file_paths: List[str]
    limit_per_file: Optional[int] = Field(
        default=None,
        description=(
            "Optional explicit per-file result limit. Normal review retrieval "
            "is unbounded per file up to the observable global matching-point "
            "safety limit."
        ),
    )
    pr_number: Optional[int] = None
    pr_changed_files: Optional[List[str]] = None
    additional_identifiers: Optional[List[str]] = Field(
        default=None,
        description=(
            "Imported and inherited type names from AST enrichment. "
            "Declarations and call names are excluded because they are not "
            "repository-wide dependency edges."
        ),
    )
    source_revision: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    base_revision: str = Field(
        min_length=1,
        max_length=200,
    )
    base_generation_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    pr_generation_fingerprint: Optional[str] = Field(
        default=None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    pr_overlay_generation_manifest_sha256: Optional[str] = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    collection_target: str = Field(min_length=1)


# ── Parse models ──

class ParseFileRequest(BaseModel):
    """Request to parse a single file and extract AST metadata."""
    path: str
    content: str
    language: Optional[str] = None


class ParseBatchRequest(BaseModel):
    """Request to parse multiple files in batch."""
    files: List[ParseFileRequest]


class ParsedFileMetadata(BaseModel):
    """AST metadata extracted from a file."""
    path: str
    language: Optional[str] = None
    imports: List[str] = []
    extends: List[str] = []
    implements: List[str] = []
    symbol_names: List[str] = []
    parent_class: Optional[str] = None
    namespace: Optional[str] = None
    calls: List[str] = []
    success: bool = True
    error: Optional[str] = None


# ── PR indexing models ──

class PRFileInfo(BaseModel):
    """Info about a single PR file.

    ``partial_diff`` content is review evidence, not a complete repository
    artifact. It must never be parsed as complete source code.
    """
    path: str
    content: str
    change_type: str  # ADDED, MODIFIED, DELETED
    content_state: Literal["complete", "partial_diff"] = "complete"

    @field_validator("change_type")
    @classmethod
    def normalize_change_type(cls, value: str) -> str:
        normalized = str(value or "").strip().upper()
        if normalized not in {
            "ADDED",
            "MODIFIED",
            "DELETED",
            "RENAMED",
            "BINARY",
        }:
            raise ValueError(f"Unsupported PR change type: {value!r}")
        return normalized


class PRIndexRequest(BaseModel):
    """Request to index PR files into main collection with PR metadata."""
    workspace: str
    project: str
    pr_number: int
    branch: str
    base_branch: Optional[str] = None
    source_revision: str = Field(min_length=1, max_length=200)
    base_revision: str = Field(min_length=1, max_length=200)
    repository_plugins: List[str] = Field(default_factory=list)
    plugin_detection_evidence: Dict[str, List[str]] = Field(default_factory=dict)
    plugin_fingerprint: str = "sha256:" + "0" * 64
    plugin_descriptor_fingerprint: str = "sha256:" + "0" * 64
    files: List[PRFileInfo]
    base_generation_manifest_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    collection_target: str = Field(min_length=1)


# ── Repository index inspection models ──

class RepositoryIndexFilters(BaseModel):
    """Bounded filters for structural repository-index inspection.

    These are internal service-to-service filters. The public web app must
    resolve workspace/project access on the Java side before forwarding them.
    """
    branches: List[str] = Field(default_factory=list, max_length=20)
    languages: List[str] = Field(default_factory=list, max_length=20)
    path: Optional[str] = Field(default=None, max_length=500)
    file_query: Optional[str] = Field(default=None, max_length=500)
    text_query: Optional[str] = Field(default=None, max_length=160)
    pr_number: Optional[int] = Field(default=None, ge=1)
    include_pr: bool = True


class RepositoryIndexGraphRequest(BaseModel):
    """Request a bounded graph slice from a project structural collection."""
    collection_target: str = Field(min_length=1)
    filters: RepositoryIndexFilters = Field(default_factory=RepositoryIndexFilters)
    limit: int = Field(default=160, ge=20, le=5000)
    cursor: Optional[str] = Field(default=None, max_length=256)
    scan_limit: int = Field(default=2500, ge=100, le=100000)


class RepositoryIndexNodeRequest(BaseModel):
    """Request a point detail and bounded neighborhood."""
    collection_target: str = Field(min_length=1)
    filters: RepositoryIndexFilters = Field(default_factory=RepositoryIndexFilters)
    neighbor_limit: int = Field(default=80, ge=10, le=160)
