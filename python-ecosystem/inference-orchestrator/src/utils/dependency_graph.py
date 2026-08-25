"""
Dependency graph builder for intelligent file batching.

SMART APPROACH: Leverages RAG's pre-indexed tree-sitter metadata to discover
file relationships instead of re-parsing diffs with regex.

The structural repository index provides exact imports, inheritance, and
framework/plugin relations extracted from source syntax.

This module queries RAG to build a relationship graph, enabling intelligent
batching that keeps related files together for better cross-file context.
"""
import logging
import inspect
from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from service.rag.rag_client import RagClient

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _rag_response_error(response: Any) -> Optional[str]:
    if not isinstance(response, dict):
        return None
    if str(response.get("status", "")).strip().casefold() != "error":
        return None
    detail = str(response.get("error") or "RAG request failed").strip()
    return detail or "RAG request failed"


@dataclass
class FileNode:
    """Represents a file in the dependency graph."""
    path: str
    priority: str
    # Relationships discovered from RAG tree-sitter metadata
    related_files: Set[str] = field(default_factory=set)
    imports_symbols: Set[str] = field(default_factory=set)
    exports_symbols: Set[str] = field(default_factory=set)
    extends: Set[str] = field(default_factory=set)
    focus_areas: List[str] = field(default_factory=list)
    relationship_degree: int = 0


@dataclass
class FileRelationship:
    """Represents a relationship between two files."""
    source_file: str
    target_file: str
    relationship_type: str
    matched_on: str


class DependencyGraphBuilder:
    """
    Builds a dependency graph using RAG's tree-sitter metadata or pre-computed relationships.

    SMART APPROACH: When Java sends enrichment data with pre-computed relationships,
    use those directly instead of querying RAG. This eliminates duplicate work since
    Java already called RAG's /parse endpoint.

    Fallback: When no enrichment data available, query RAG's deterministic context API
    which has the FULL file indexed with tree-sitter metadata.

    Relationships come from explicit enrichment edges, plugin graph groups, or
    uniquely resolved dependency definitions. Co-location alone is not an edge.
    """

    def __init__(self, rag_client: Optional["RAGClient"] = None):
        self.rag_client = rag_client
        self.nodes: Dict[str, FileNode] = {}
        self.relationships: List[FileRelationship] = []
        self._metadata_cache: Dict[str, Dict] = {}

    def _initialize_nodes(self, file_groups: List[Any]) -> None:
        """Initialize nodes and preserve deterministic plugin evidence groups."""
        for group in file_groups:
            paths = []
            for review_file in group.files:
                self.nodes[review_file.path] = FileNode(
                    path=review_file.path,
                    priority=group.priority,
                    focus_areas=(
                        review_file.focus_areas
                        if hasattr(review_file, "focus_areas")
                        else []
                    ),
                )
                paths.append(review_file.path)

            if not str(getattr(group, "group_id", "")).startswith(
                "PLUGIN_EVIDENCE_"
            ):
                continue
            # A deterministic star preserves the connected component without
            # materializing an O(n²) clique for large architecture groups.
            if paths:
                anchor = paths[0]
                for related in paths[1:]:
                    self.nodes[anchor].related_files.add(related)
                    self.nodes[related].related_files.add(anchor)
                    self.relationships.append(FileRelationship(
                        source_file=anchor,
                        target_file=related,
                        relationship_type="PLUGIN_EVIDENCE",
                        matched_on=str(group.group_id),
                    ))
        for path, node in self.nodes.items():
            if node.related_files:
                node.relationship_degree = self._relationship_degree(path)

    def build_graph_from_enrichment(
        self,
        file_groups: List[Any],
        enrichment_data: Any,
    ) -> Dict[str, FileNode]:
        """
        Build dependency graph from pre-computed enrichment data sent by Java.

        This is the preferred method when enrichment_data is available, as it:
        - Eliminates redundant RAG calls (Java already parsed the files)
        - Uses full file content for accurate relationship detection
        - Provides relationships computed with proper AST parsing

        Args:
            file_groups: List of FileGroup objects with files to analyze
            enrichment_data: PrEnrichmentDataDto from Java with relationships and metadata

        Returns:
            Dict of file paths to FileNode objects with relationships populated
        """
        if not enrichment_data or not enrichment_data.has_data():
            logger.info("No enrichment data available; batching without inferred edges")
            return self._build_basic_graph(file_groups)

        # Initialize nodes from file groups and retain graph-derived constraints
        # that were projected before Stage 0.
        self._initialize_nodes(file_groups)

        # Process pre-computed relationships
        relationships_by_file: Dict[str, Set[str]] = defaultdict(set)

        for rel in enrichment_data.relationships:
            source = rel.sourceFile
            target = rel.targetFile
            rel_type = rel.relationshipType.value if hasattr(rel.relationshipType, 'value') else str(rel.relationshipType)

            # Only add relationships between files we're analyzing
            if source in self.nodes and target in self.nodes:
                relationships_by_file[source].add(target)
                relationships_by_file[target].add(source)

                self.relationships.append(FileRelationship(
                    source_file=source,
                    target_file=target,
                    relationship_type=rel_type,
                    matched_on=rel.matchedOn or "",
                ))

        # Process metadata to populate node symbols
        for meta in enrichment_data.fileMetadata:
            if meta.path in self.nodes:
                node = self.nodes[meta.path]
                if meta.imports:
                    node.imports_symbols.update(meta.imports)
                if meta.symbolNames:
                    node.exports_symbols.update(meta.symbolNames)
                if meta.extendsClasses:
                    node.extends.update(meta.extendsClasses)

        # Update nodes with discovered relationships
        for file_path, related in relationships_by_file.items():
            if file_path in self.nodes:
                self.nodes[file_path].related_files.update(related)
                self.nodes[file_path].relationship_degree = self._relationship_degree(file_path)

        logger.info(
            f"Dependency graph built from enrichment: {len(self.nodes)} files, "
            f"{len(self.relationships)} relationships"
        )

        return self.nodes

    def build_graph_from_rag(
        self,
        file_groups: List[Any],
        workspace: str,
        project: str,
        branches: List[str],
    ) -> Dict[str, FileNode]:
        """
        Build dependency graph by querying RAG's deterministic context API.

        This leverages tree-sitter metadata extracted during indexing:
        explicit imports and inheritance whose definitions resolve uniquely.
        """
        if not self.rag_client:
            logger.warning("No RAG client provided; batching without inferred edges")
            return self._build_basic_graph(file_groups)

        # Collect all file paths
        all_file_paths = []
        file_priority_map = {}
        file_info_map = {}

        self._initialize_nodes(file_groups)
        for group in file_groups:
            for f in group.files:
                all_file_paths.append(f.path)
                file_priority_map[f.path] = group.priority
                file_info_map[f.path] = f

        if not all_file_paths:
            return self.nodes

        # Query RAG for deterministic context
        try:
            rag_response = self.rag_client.get_deterministic_context(
                workspace=workspace,
                project=project,
                branches=branches,
                file_paths=all_file_paths,
                limit_per_file=15,
            )
            if inspect.isawaitable(rag_response):
                logger.warning(
                    "RAG deterministic context client is async; use build_graph_from_rag_async "
                    "for relationship-aware batching"
                )
                return self._build_basic_graph(file_groups)
            if error := _rag_response_error(rag_response):
                logger.warning(
                    "RAG batching lookup failed; batching without inferred edges: %s",
                    error,
                )
                return self.nodes
            self._metadata_cache['last_response'] = rag_response
        except Exception as e:
            logger.warning(f"RAG query failed; batching without inferred edges: {e}")
            return self._build_basic_graph(file_groups)

        # Extract relationships from RAG response
        self._extract_relationships_from_rag(
            _unwrap_deterministic_context(rag_response),
            all_file_paths,
        )

        logger.info(
            f"Dependency graph built: {len(self.nodes)} files, "
            f"{len(self.relationships)} relationships"
        )

        return self.nodes

    async def build_graph_from_rag_async(
        self,
        file_groups: List[Any],
        workspace: str,
        project: str,
        branches: List[str],
    ) -> Dict[str, FileNode]:
        """
        Async variant for the inference orchestrator's httpx-based RAG client.

        Keeps smart batching relationship-aware instead of accidentally treating
        an un-awaited coroutine as a response and falling back to simple batches.
        """
        if not self.rag_client:
            logger.warning("No RAG client provided; batching without inferred edges")
            return self._build_basic_graph(file_groups)

        all_file_paths = []
        self._initialize_nodes(file_groups)
        for group in file_groups:
            for f in group.files:
                all_file_paths.append(f.path)

        if not all_file_paths:
            return self.nodes

        try:
            rag_response = self.rag_client.get_deterministic_context(
                workspace=workspace,
                project=project,
                branches=branches,
                file_paths=all_file_paths,
                limit_per_file=15,
            )
            if inspect.isawaitable(rag_response):
                rag_response = await rag_response
            if error := _rag_response_error(rag_response):
                logger.warning(
                    "Async RAG batching lookup failed; batching without inferred edges: %s",
                    error,
                )
                return self.nodes
            self._metadata_cache['last_response'] = rag_response
        except Exception as e:
            logger.warning(f"Async RAG query failed; batching without inferred edges: {e}")
            return self._build_basic_graph(file_groups)

        self._extract_relationships_from_rag(
            _unwrap_deterministic_context(rag_response),
            all_file_paths,
        )

        logger.info(
            f"Async dependency graph built: {len(self.nodes)} files, "
            f"{len(self.relationships)} relationships"
        )

        return self.nodes

    def _extract_relationships_from_rag(
        self,
        rag_response: Dict,
        changed_file_paths: List[str]
    ) -> None:
        """Extract file relationships from RAG deterministic context response."""
        changed_file_set = set(changed_file_paths)
        file_relationships: Dict[str, Set[str]] = defaultdict(set)

        # Process changed_files to extract metadata
        changed_files = rag_response.get('changed_files', {})
        for file_path, chunks in changed_files.items():
            norm_path = file_path.lstrip('/')
            if norm_path in self.nodes:
                for chunk in chunks:
                    metadata = chunk.get('metadata', {})

                    # Retain declarations for diagnostics only. Declarations
                    # are not dependency edges by themselves.
                    if metadata.get('primary_name'):
                        self.nodes[norm_path].exports_symbols.add(metadata['primary_name'])
                    if metadata.get('symbol_names'):
                        self.nodes[norm_path].exports_symbols.update(metadata['symbol_names'])

                    # Extract what this file imports
                    if metadata.get('imports'):
                        for imp in metadata['imports']:
                            if isinstance(imp, str):
                                name = (
                                    imp.replace(';', '')
                                    .replace('::', '.')
                                    .replace('\\', '.')
                                    .replace('/', '.')
                                    .split('.')[-1]
                                    .strip()
                                )
                                if name:
                                    self.nodes[norm_path].imports_symbols.add(name)

                    # Track enclosing-class membership and inheritance.
                    if metadata.get('extends'):
                        for parent in metadata['extends']:
                            if not isinstance(parent, str):
                                continue
                            name = (
                                parent.replace('::', '.')
                                .replace('\\', '.')
                                .split('.')[-1]
                                .strip()
                            )
                            if name:
                                self.nodes[norm_path].extends.add(name)

        # Process related_definitions
        related_definitions = rag_response.get('related_definitions', {})
        for symbol, chunks in related_definitions.items():
            for chunk in chunks:
                metadata = chunk.get('metadata', {})
                related_path = metadata.get('path', '').lstrip('/')

                if related_path and related_path in self.nodes:
                    for file_path in changed_file_set:
                        norm_path = file_path.lstrip('/')
                        if norm_path in self.nodes:
                            node = self.nodes[norm_path]
                            if (
                                symbol in node.imports_symbols
                                or symbol in node.extends
                            ):
                                file_relationships[norm_path].add(related_path)
                                file_relationships[related_path].add(norm_path)
                                self.relationships.append(FileRelationship(
                                    source_file=norm_path,
                                    target_file=related_path,
                                    relationship_type='definition',
                                    matched_on=symbol,
                                ))

        # Update nodes with discovered relationships
        for file_path, related in file_relationships.items():
            if file_path in self.nodes:
                self.nodes[file_path].related_files.update(related)
                self.nodes[file_path].relationship_degree = self._relationship_degree(file_path)

    def _relationship_degree(self, file_path: str) -> int:
        return sum(
            1
            for relationship in self.relationships
            if relationship.source_file == file_path
            or relationship.target_file == file_path
        )

    def _build_basic_graph(self, file_groups: List[Any]) -> Dict[str, FileNode]:
        """Fallback without inventing relationships from file co-location."""
        self._initialize_nodes(file_groups)
        return self.nodes

    def get_connected_components(self) -> List[Set[str]]:
        """Find connected components in the dependency graph."""
        visited = set()
        components = []

        def dfs(node_path: str, component: Set[str]):
            if node_path in visited:
                return
            visited.add(node_path)
            component.add(node_path)

            node = self.nodes.get(node_path)
            if not node:
                return

            for related_path in node.related_files:
                if related_path in self.nodes:
                    dfs(related_path, component)

        for path in self.nodes:
            if path not in visited:
                component: Set[str] = set()
                dfs(path, component)
                if component:
                    components.append(component)

        return components

    def get_smart_batches(
        self,
        file_groups: List[Any],
        workspace: str,
        project: str,
        branches: List[str],
        max_batch_size: int = 15,
        min_batch_size: int = 3,
        enrichment_data: Any = None,
        max_allowed_tokens: int = 200000,
        processed_diff: Any = None,
        token_cost_by_path: Optional[Dict[str, int]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Create intelligent batches that keep related files together.

        Strategy:
        1. If enrichment_data is available, use pre-computed relationships from Java
        2. Otherwise, query RAG to discover file relationships via tree-sitter metadata
        3. Find connected components (files that are related)
        4. Batch files within components together
        5. Split only when the file or token ceiling requires it

        Args:
            file_groups: List of FileGroup objects with files
            workspace: Repository workspace/owner
            project: Repository slug
            branches: Branch names for context
            max_batch_size: Maximum files per batch
            min_batch_size: Minimum files per batch
            enrichment_data: Optional PrEnrichmentDataDto from Java with pre-computed relationships
        """
        # Use enrichment data if available, otherwise fall back to RAG
        if enrichment_data and hasattr(enrichment_data, 'has_data') and enrichment_data.has_data():
            logger.info("Using pre-computed enrichment data for dependency graph")
            self.build_graph_from_enrichment(file_groups, enrichment_data)
        else:
            self.build_graph_from_rag(file_groups, workspace, project, branches)
        return self._build_batches_from_graph(
            file_groups=file_groups,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            max_allowed_tokens=max_allowed_tokens,
            processed_diff=processed_diff,
            token_cost_by_path=token_cost_by_path,
        )

    async def get_smart_batches_async(
        self,
        file_groups: List[Any],
        workspace: str,
        project: str,
        branches: List[str],
        max_batch_size: int = 15,
        min_batch_size: int = 3,
        enrichment_data: Any = None,
        max_allowed_tokens: int = 200000,
        processed_diff: Any = None,
        token_cost_by_path: Optional[Dict[str, int]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Async equivalent of get_smart_batches for async RAG clients."""
        if enrichment_data and hasattr(enrichment_data, 'has_data') and enrichment_data.has_data():
            logger.info("Using pre-computed enrichment data for dependency graph")
            self.build_graph_from_enrichment(file_groups, enrichment_data)
        else:
            await self.build_graph_from_rag_async(file_groups, workspace, project, branches)

        return self._build_batches_from_graph(
            file_groups=file_groups,
            max_batch_size=max_batch_size,
            min_batch_size=min_batch_size,
            max_allowed_tokens=max_allowed_tokens,
            processed_diff=processed_diff,
            token_cost_by_path=token_cost_by_path,
        )

    def _build_batches_from_graph(
        self,
        file_groups: List[Any],
        max_batch_size: int,
        min_batch_size: int,
        max_allowed_tokens: int,
        processed_diff: Any = None,
        token_cost_by_path: Optional[Dict[str, int]] = None,
    ) -> List[List[Dict[str, Any]]]:
        components = self.get_connected_components()

        logger.info(
            f"Dependency analysis: {len(self.nodes)} files, "
            f"{len(components)} connected components, "
            f"{len(self.relationships)} relationships"
        )

        file_priority_map = {}
        file_info_map = {}
        file_token_cost = dict(token_cost_by_path or {})

        # Diff-only estimation is retained only for callers that cannot render
        # their real local prompt. Stage 1 supplies the exact rendered cost.
        if processed_diff and hasattr(processed_diff, 'files'):
            for df in processed_diff.files:
                file_token_cost.setdefault(
                    df.path,
                    (len(df.content) // 4) + 1000,
                )

        for group in file_groups:
            for f in group.files:
                file_priority_map[f.path] = group.priority
                file_info_map[f.path] = f
                # Fallback token estimate
                if f.path not in file_token_cost:
                    file_token_cost[f.path] = 2000

        batches = []
        processed_files = set()
        priority_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']

        def component_sort_key(comp):
            max_priority = min(
                priority_order.index(file_priority_map.get(f, 'LOW'))
                for f in comp
            )
            return (-len(comp), max_priority)

        for component in sorted(components, key=component_sort_key):
            if all(f in processed_files for f in component):
                continue

            component_files = [f for f in component if f not in processed_files]
            if not component_files:
                continue

            component_files_sorted = sorted(
                component_files,
                key=lambda f: (
                    -self.nodes[f].relationship_degree,
                    priority_order.index(file_priority_map.get(f, 'LOW')),
                    f
                )
            )

            current_batch = []
            current_batch_tokens = 0
            for file_path in component_files_sorted:
                file_info = file_info_map.get(file_path)
                if not file_info:
                    continue

                node = self.nodes[file_path]
                file_tokens = file_token_cost.get(file_path, 2000)

                if current_batch and (current_batch_tokens + file_tokens > max_allowed_tokens):
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

                current_batch.append({
                    "file": file_info,
                    "priority": file_priority_map.get(file_path, 'MEDIUM'),
                    "has_relationships": len(node.related_files) > 0,
                    "relationship_degree": node.relationship_degree,
                    "related_files": tuple(sorted(node.related_files)),
                    "related_in_batch": [],
                })
                current_batch_tokens += file_tokens
                processed_files.add(file_path)

                if len(current_batch) >= max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

            if current_batch:
                batches.append(current_batch)

        # Handle orphan files
        orphan_files = []
        for group in file_groups:
            for f in group.files:
                if f.path not in processed_files:
                    orphan_files.append({
                        "file": f,
                        "priority": group.priority,
                        "has_relationships": False,
                        "relationship_degree": 0,
                        "related_in_batch": []
                    })
                    processed_files.add(f.path)

        if orphan_files:
            orphan_files_sorted = sorted(
                orphan_files,
                key=lambda x: (priority_order.index(x['priority']), x['file'].path)
            )

            current_batch = []
            current_batch_tokens = 0
            for orphan in orphan_files_sorted:
                file_tokens = file_token_cost.get(orphan['file'].path, 2000)
                if current_batch and (current_batch_tokens + file_tokens > max_allowed_tokens):
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

                current_batch.append(orphan)
                current_batch_tokens += file_tokens
                if len(current_batch) >= max_batch_size:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_tokens = 0

            if current_batch:
                batches.append(current_batch)

        batches = self._merge_small_batches(batches, min_batch_size, max_batch_size, max_allowed_tokens, file_token_cost)

        for batch in batches:
            batch_paths = {item['file'].path for item in batch}
            for item in batch:
                related_files = tuple(item.get('related_files', ()))
                item['related_in_batch'] = [
                    path for path in related_files if path in batch_paths
                ]
                item['related_outside_batch'] = [
                    path for path in related_files if path not in batch_paths
                ]

        logger.info(f"Smart batching created {len(batches)} batches from {len(self.nodes)} files")
        for i, batch in enumerate(batches):
            paths = [b['file'].path for b in batch]
            rel_count = sum(1 for b in batch if b.get('has_relationships'))
            logger.debug(f"Batch {i+1}: {len(batch)} files ({rel_count} with relationships): {paths}")

        return batches

    def _merge_small_batches(
        self,
        batches: List[List[Dict[str, Any]]],
        min_size: int,
        max_size: int,
        max_allowed_tokens: int = 200000,
        file_token_cost: Dict[str, int] = None
    ) -> List[List[Dict[str, Any]]]:
        """Pack component batches up to the explicit file and token ceilings.

        Priority affects ordering inside a batch, not whether related review
        input is split into separate model calls.
        """
        if not batches:
            return batches
        merged = []
        file_token_cost = file_token_cost or {}
        current_merged = []
        current_merged_tokens = 0
        for batch in batches:
            if not batch:
                continue
            batch_tokens = sum(
                file_token_cost.get(item['file'].path, 2000)
                for item in batch
            )
            if current_merged and (
                len(current_merged) + len(batch) > max_size
                or current_merged_tokens + batch_tokens > max_allowed_tokens
            ):
                merged.append(current_merged)
                current_merged = []
                current_merged_tokens = 0
            current_merged.extend(batch)
            current_merged_tokens += batch_tokens

        if current_merged:
            merged.append(current_merged)

        return merged

    def get_relationship_summary(self) -> Dict[str, Any]:
        """Get a summary of discovered relationships."""
        relationship_types = defaultdict(int)
        for rel in self.relationships:
            relationship_types[rel.relationship_type] += 1

        return {
            "total_files": len(self.nodes),
            "total_relationships": len(self.relationships),
            "relationship_types": dict(relationship_types),
            "files_with_relationships": sum(
                1 for node in self.nodes.values()
                if len(node.related_files) > 0
            ),
            "avg_relationships_per_file": (
                len(self.relationships) * 2 / len(self.nodes)
                if self.nodes else 0
            )
        }


def create_smart_batches(
    file_groups: List[Any],
    workspace: str,
    project: str,
    branches: List[str],
    rag_client: Optional["RAGClient"] = None,
    max_batch_size: int = 15,
    enrichment_data: Any = None,
    max_allowed_tokens: int = 200000,
    processed_diff: Any = None,
    token_cost_by_path: Optional[Dict[str, int]] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Convenience function to create smart batches from file groups.

    Args:
        file_groups: List of FileGroup objects with files
        workspace: Repository workspace/owner
        project: Repository slug
        branches: Branch names for context
        rag_client: Optional RAG client for relationship discovery
        max_batch_size: Maximum files per batch
        enrichment_data: Optional PrEnrichmentDataDto with pre-computed relationships from Java
    """
    builder = DependencyGraphBuilder(rag_client=rag_client)
    return builder.get_smart_batches(
        file_groups,
        workspace,
        project,
        branches,
        max_batch_size,
        enrichment_data=enrichment_data,
        max_allowed_tokens=max_allowed_tokens,
        processed_diff=processed_diff,
        token_cost_by_path=token_cost_by_path,
    )


async def create_smart_batches_async(
    file_groups: List[Any],
    workspace: str,
    project: str,
    branches: List[str],
    rag_client: Optional["RAGClient"] = None,
    max_batch_size: int = 15,
    enrichment_data: Any = None,
    max_allowed_tokens: int = 200000,
    processed_diff: Any = None,
    token_cost_by_path: Optional[Dict[str, int]] = None,
) -> List[List[Dict[str, Any]]]:
    """Async convenience function for async RAG clients."""
    builder = DependencyGraphBuilder(rag_client=rag_client)
    return await builder.get_smart_batches_async(
        file_groups,
        workspace,
        project,
        branches,
        max_batch_size,
        enrichment_data=enrichment_data,
        max_allowed_tokens=max_allowed_tokens,
        processed_diff=processed_diff,
        token_cost_by_path=token_cost_by_path,
    )


def _unwrap_deterministic_context(rag_response: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Accept either raw deterministic context or {'context': context} API response."""
    if not isinstance(rag_response, dict):
        return {}
    context = rag_response.get("context")
    if isinstance(context, dict):
        return context
    return rag_response


def build_dependency_aware_batches(
    changed_files: List[str],
    enrichment_data: Any = None,
    max_batch_token_budget: int = 60_000,
    diff: Optional[str] = None,
) -> List[List[Dict[str, Any]]]:
    """
    Lightweight batching for QA-doc and other pipelines that only have
    a flat list of changed file paths (no FileGroup objects).

    Uses enrichment_data relationships (if available) to keep related
    files together.  Falls back to simple sequential batching otherwise.

    Token budget accounts for BOTH file content AND per-file diff size,
    since the Stage 1 prompt includes both sections.

    Returns batches in the format expected by BaseOrchestrator:
        List[ List[ {"file_info": <obj with .path>, "priority": str} ] ]
    """
    if not changed_files:
        return []

    # Build a quick adjacency map from enrichment relationships
    adjacency: Dict[str, Set[str]] = defaultdict(set)
    changed_set = set(changed_files)

    if enrichment_data and hasattr(enrichment_data, "relationships") and enrichment_data.relationships:
        for rel in enrichment_data.relationships:
            src = rel.sourceFile if hasattr(rel, "sourceFile") else None
            tgt = rel.targetFile if hasattr(rel, "targetFile") else None
            if src and tgt and src in changed_set and tgt in changed_set:
                adjacency[src].add(tgt)
                adjacency[tgt].add(src)

    # Connected components via BFS
    visited: Set[str] = set()
    components: List[List[str]] = []
    for f in changed_files:
        if f in visited:
            continue
        queue = [f]
        comp: List[str] = []
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            # ``queue`` is LIFO. Reverse lexical insertion makes the next
            # visited neighbor stable and lexical instead of hash-seed driven.
            for nb in sorted(adjacency.get(node, set()), reverse=True):
                if nb not in visited:
                    queue.append(nb)
        if comp:
            components.append(comp)

    # ── Per-file diff sizes (parsed from the unified diff) ────────────
    # The Stage 1 prompt includes both file content AND the per-file diff,
    # so both must be counted in the token budget.
    file_diff_chars: Dict[str, int] = {}
    if diff:
        import re
        sections = re.split(r'(?=^diff --git )', diff, flags=re.MULTILINE)
        for section in sections:
            if not section.strip():
                continue
            header_match = re.match(r'diff --git a/(.+?) b/(.+?)(?:\n|$)', section)
            if header_match:
                b_path = header_match.group(2)
                sec_len = len(section)
                file_diff_chars[b_path] = sec_len
                # Index by every suffix so path-format mismatches don't
                # cause the diff size to be invisible to the budget.
                # e.g. "services/pipeline-agent/src/Foo.java" also gets
                # indexed as "pipeline-agent/src/Foo.java", "src/Foo.java", "Foo.java".
                remaining = b_path
                while '/' in remaining:
                    remaining = remaining.split('/', 1)[1]
                    file_diff_chars[remaining] = sec_len

    # Estimate token cost per file: file content + diff size + overhead.
    # Use the REAL content size so batches are split to actually fit the
    # model context window — never truncate, just make smaller batches.
    _PROMPT_OVERHEAD_TOKENS = 1_500  # template text, formatting, etc.

    def _lookup_diff_chars(path: str) -> int:
        """Find diff size for a path using suffix matching."""
        if path in file_diff_chars:
            return file_diff_chars[path]
        # Try suffix matching
        for dp, dc in file_diff_chars.items():
            if dp.endswith(path) or path.endswith(dp):
                return dc
        return 0

    file_token_est: Dict[str, int] = {}
    if enrichment_data and hasattr(enrichment_data, "fileContents") and enrichment_data.fileContents:
        for fc in enrichment_data.fileContents:
            if fc.content and not fc.skipped:
                content_chars = len(fc.content)
                diff_chars = _lookup_diff_chars(fc.path)
                est = (content_chars + diff_chars) // 4 + _PROMPT_OVERHEAD_TOKENS
                file_token_est[fc.path] = est
                # Index by every suffix so path-format mismatches are handled
                # e.g. "app/code/Vendor/File.php" also indexed as
                # "code/Vendor/File.php", "Vendor/File.php", "File.php"
                remaining = fc.path
                while '/' in remaining:
                    remaining = remaining.split('/', 1)[1]
                    if remaining not in file_token_est:
                        file_token_est[remaining] = est

    def _lookup_token_est(path: str) -> int:
        """Find token estimate for a path using suffix matching."""
        if path in file_token_est:
            return file_token_est[path]
        for ep, et in file_token_est.items():
            if ep.endswith(path) or path.endswith(ep):
                return et
        return 0

    # For files not in enrichment but present in the diff, estimate from diff alone
    for fp in changed_files:
        if not _lookup_token_est(fp):
            diff_chars = _lookup_diff_chars(fp)
            file_token_est[fp] = diff_chars // 4 + _PROMPT_OVERHEAD_TOKENS if diff_chars else 2000

    def _make_item(path: str) -> Dict[str, Any]:
        fi = type("FI", (), {"path": path})()
        return {"file_info": fi, "priority": "MEDIUM"}

    max_files_per_batch = 15
    batches: List[List[Dict[str, Any]]] = []

    for comp in components:
        current_batch: List[Dict[str, Any]] = []
        current_tokens = 0
        for path in comp:
            tok = _lookup_token_est(path) or 2000
            if tok > max_batch_token_budget:
                logger.warning(
                    "Single file %s estimated at %dK tokens (budget=%dK) — "
                    "will be content-capped at prompt level",
                    path, tok // 1000, max_batch_token_budget // 1000,
                )
            if current_batch and (
                current_tokens + tok > max_batch_token_budget
                or len(current_batch) >= max_files_per_batch
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(_make_item(path))
            current_tokens += tok
        if current_batch:
            batches.append(current_batch)

    logger.info(
        "build_dependency_aware_batches: %d files → %d batches (budget=%d tokens/batch)",
        len(changed_files), len(batches), max_batch_token_budget,
    )
    return batches
