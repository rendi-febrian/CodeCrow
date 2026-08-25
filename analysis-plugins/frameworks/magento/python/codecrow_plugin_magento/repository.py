from __future__ import annotations

import base64
import gzip
import json
import logging
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from typing import Callable

from codecrow_plugins import (
    ArchitecturePacket,
    FileArtifact,
    GraphFact,
    PluginDiagnostic,
    PluginOutcome,
    RepositoryAnalysis,
    RepositoryContext,
    RepositorySnapshot,
    SymbolDefinition,
)
from codecrow_plugins.graphql import (
    parse_operations,
    parse_schema,
    parse_schema_root_types,
)

from .architecture import (
    MAGENTO_GRAPHQL_CLIENT_SUFFIXES,
    MAGENTO_AREAS,
    MAGENTO_VIEW_SOURCE_SUFFIXES,
    ModuleRecord,
    PacketGraph,
    attrs,
    config_area,
    is_magento_config_xml,
    is_magento_view_xml,
    line,
    safe_xml,
    tag,
    view_area,
)
from .events import (
    EventDispatch,
    EventEntrypoint,
    decode_event_dispatches,
    resolve_dispatch_area,
)
from .frontend_init import extract_frontend_initializers
from .javascript import (
    OptionalJavaScriptEnrichmentError,
    TemplateEventReference,
    TemplateGlobalReference,
    extract_requirejs_relations,
    extract_template_event_references,
    extract_template_global_references,
)
from .layout import LayoutArgument, merge_layout, parse_layout_document
from .requirejs import (
    EffectiveRequireJsConfig,
    RequireJsResolution,
    build_effective_requirejs_config,
    extract_amd_dependencies,
    extract_template_amd_dependencies,
    normalize_amd_dependency,
)

logger = logging.getLogger(__name__)


_MODULE_ENABLED = re.compile(
    r"['\"](?P<name>[A-Za-z][A-Za-z0-9]*_[A-Za-z][A-Za-z0-9]*)['\"]\s*=>\s*(?P<enabled>[01])"
)
_MODULES_SECTION = re.compile(
    r"['\"]modules['\"]\s*=>\s*\[(?P<body>.*?)\]",
    re.DOTALL,
)
_REGISTRATION = re.compile(
    r"ComponentRegistrar::MODULE\s*,\s*['\"](?P<name>[A-Za-z0-9_]+)['\"]"
)
_THEME_REGISTRATION = re.compile(
    r"ComponentRegistrar::THEME\s*,\s*['\"](?P<name>[^'\"]+)['\"]"
)
_PHP_TEMPLATE_CALL_REFERENCE = "php-template-instance-call-reference:"
_DEPLOYMENT_DEFAULT_CONNECTION = "deployment-default"
_BROKER_DEFAULT_EXCHANGE = "broker-default-exchange"
_DEFAULT_MESSAGE_CONSUMER = (
    r"Magento\Framework\MessageQueue\Consumer"
)
_BUILTIN_MESSAGE_CONSUMERS = frozenset({
    _DEFAULT_MESSAGE_CONSUMER,
    r"Magento\Framework\MessageQueue\BatchConsumer",
})
_MASS_MESSAGE_CONSUMER = (
    r"Magento\AsynchronousOperations\Model\MassConsumer"
)


def _enabled(value: object, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().casefold() not in {"1", "true", "yes", "on"}


def _message_topic_matches(pattern: str, topic: str) -> bool:
    """Mirror Magento QueueResolver's AMQP `*`/`#` topic matching."""
    if pattern == topic:
        return True
    if "*" not in pattern and "#" not in pattern:
        return False

    pattern_parts = pattern.split(".")
    topic_parts = topic.split(".")
    pattern_index = 0
    topic_index = 0
    hash_pattern_index = -1
    hash_topic_index = -1

    while topic_index < len(topic_parts):
        part = (
            pattern_parts[pattern_index]
            if pattern_index < len(pattern_parts)
            else None
        )
        if part == "#":
            hash_pattern_index = pattern_index
            hash_topic_index = topic_index
            pattern_index += 1
            continue
        if part is not None and (
            part == "*" or part == topic_parts[topic_index]
        ):
            pattern_index += 1
            topic_index += 1
            continue
        if hash_pattern_index == -1:
            return False
        hash_topic_index += 1
        topic_index = hash_topic_index
        pattern_index = hash_pattern_index + 1

    while (
        pattern_index < len(pattern_parts)
        and pattern_parts[pattern_index] == "#"
    ):
        pattern_index += 1
    return pattern_index == len(pattern_parts)


@dataclass(frozen=True)
class ConfigValue:
    value: str
    path: str
    line: int
    module: str
    order: int
    attributes: tuple[tuple[str, str], ...] = ()
    position: int = 0


@dataclass
class DiState:
    preferences: dict[str, ConfigValue] = field(default_factory=dict)
    virtual_types: dict[str, ConfigValue] = field(default_factory=dict)
    plugins: dict[tuple[str, str], ConfigValue] = field(default_factory=dict)
    arguments: dict[tuple[str, str, str], ConfigValue] = field(default_factory=dict)
    argument_types: dict[tuple[str, str], ConfigValue] = field(default_factory=dict)
    item_types: dict[tuple[str, str, str], ConfigValue] = field(default_factory=dict)
    item_values: dict[tuple[str, str, str], ConfigValue] = field(default_factory=dict)


@dataclass(frozen=True)
class ThemeRecord:
    name: str
    area: str
    root: str
    theme_xml: str
    parent: str = ""


@dataclass(frozen=True, order=True)
class TemplatePhpCall:
    receiver: str
    method: str
    line: int
    literal_arguments: tuple[tuple[int, str], ...] = ()


def _module_root(path: str) -> str:
    suffix = "/etc/module.xml"
    if path == "etc/module.xml":
        return ""
    return path[:-len(suffix)] if path.endswith(suffix) else ""


def _path_under(root: str, relative: str) -> str:
    return f"{root}/{relative}" if root else relative


def _method_subject(method: str) -> tuple[str, str] | None:
    for prefix in ("before", "around", "after"):
        if method.startswith(prefix) and len(method) > len(prefix):
            subject = method[len(prefix):]
            return prefix, subject[:1].casefold() + subject[1:]
    return None


class MagentoRepositoryResolver:
    def __init__(
        self,
        plugin_id: str,
        artifacts: dict[str, str],
        symbols: tuple[SymbolDefinition, ...],
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        deadline: float | None = None,
    ) -> None:
        self.plugin_id = plugin_id
        self.artifacts = artifacts
        self.symbols = symbols
        self.progress_callback = progress_callback
        self.deadline = deadline
        self.symbols_by_name: dict[str, tuple[SymbolDefinition, ...]] = {}
        self.symbols_by_casefold: dict[str, tuple[SymbolDefinition, ...]] = {}
        for symbol in symbols:
            self.symbols_by_name.setdefault(symbol.qualified_name, tuple())
            self.symbols_by_name[symbol.qualified_name] += (symbol,)
            normalized = symbol.qualified_name.lstrip("\\").casefold()
            self.symbols_by_casefold.setdefault(normalized, tuple())
            self.symbols_by_casefold[normalized] += (symbol,)
        # A normal VCS checkout often omits Composer-installed module source,
        # while app/etc/config.php still names the deployed modules. Theme
        # overrides for those modules are runtime inputs and must not disappear
        # merely because vendor/ was not committed.
        self.configured_modules: dict[str, bool] = {}
        self.graph = PacketGraph(plugin_id)
        self._roots: dict[str, object] = {}
        self._diagnostics: list[PluginDiagnostic] = []
        self.invalid_paths: set[str] = set()
        self._optional_enrichment_diagnostics: set[tuple[str, str | None]] = set()
        self._template_metadata_diagnostics: set[tuple[str, str]] = set()
        self._template_php_call_cache: dict[
            tuple[str, str],
            tuple[TemplatePhpCall, ...],
        ] = {}
        self._ui_component_paths_by_name: dict[str, tuple[str, ...]] = {}
        for path in artifacts:
            if "/ui_component/" not in f"/{path}" or not path.endswith(
                ".xml"
            ):
                continue
            name = PurePosixPath(path).stem
            self._ui_component_paths_by_name.setdefault(name, tuple())
            self._ui_component_paths_by_name[name] += (path,)
        self._requirejs_configs: dict[
            tuple[str, str],
            EffectiveRequireJsConfig,
        ] = {}
        self._effective_di_arguments: dict[str, dict[str, dict]] = {}
        self._template_layout_sources: dict[
            str,
            set[tuple[str, str, str]],
        ] = {}
        self._conditional_template_layout_sources: dict[
            str,
            set[tuple[str, str, str]],
        ] = {}
        self._system_config_sources: dict[str, set[str]] = {}
        self._acl_sources: dict[str, set[str]] = {}
        self._admin_controller_sources: dict[str, set[str]] = {}

    def _report_progress(self, event: dict[str, object]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(event)
        except Exception:
            # Repository progress is optional host observability.
            return

    def _check_deadline(self, stage_name: str) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(
                "Magento repository architecture exceeded its time budget "
                f"during {stage_name}"
            )

    def _run_stage(self, stage_name: str, stage):
        self._check_deadline(stage_name)
        started = time.monotonic()
        self._report_progress({
            "pluginId": self.plugin_id,
            "substage": stage_name,
            "status": "started",
            "message": f"Building Magento architecture: {stage_name}",
        })
        try:
            result = stage()
            self._check_deadline(stage_name)
        except Exception as exception:
            duration_ms = round((time.monotonic() - started) * 1000)
            status = "timed_out" if isinstance(exception, TimeoutError) else "failed"
            logger.warning(
                "Magento architecture substage %s status=%s duration_ms=%s: %s",
                stage_name,
                status,
                duration_ms,
                exception,
            )
            self._report_progress({
                "pluginId": self.plugin_id,
                "substage": stage_name,
                "status": status,
                "durationMs": duration_ms,
                "message": (
                    f"Magento architecture {stage_name} {status.replace('_', ' ')} "
                    f"after {duration_ms} ms"
                ),
            })
            raise
        duration_ms = round((time.monotonic() - started) * 1000)
        logger.info(
            "Magento architecture substage %s status=completed duration_ms=%s",
            stage_name,
            duration_ms,
        )
        self._report_progress({
            "pluginId": self.plugin_id,
            "substage": stage_name,
            "status": "completed",
            "durationMs": duration_ms,
            "message": (
                f"Built Magento architecture: {stage_name} in {duration_ms} ms"
            ),
        })
        return result

    def resolve(self) -> tuple[RepositoryAnalysis, tuple[PluginDiagnostic, ...]]:
        started = time.monotonic()
        modules = self._run_stage("module discovery", self._modules)
        if not modules:
            themes = self._run_stage(
                "theme discovery",
                lambda: self._themes(modules),
            )
            if not themes:
                return RepositoryAnalysis(), tuple(self._diagnostics)
            theme_stages = (
                ("routes/layouts", lambda: self._routes_and_layouts(
                    modules,
                    themes,
                )),
                ("template globals", self._template_globals),
                ("template events", lambda: self._template_events(themes)),
                ("UI components", lambda: self._ui_components(
                    modules,
                    themes,
                )),
                ("RequireJS", lambda: self._requirejs(modules, themes)),
                ("frontend initializers", lambda: (
                    self._frontend_initializers(modules, themes)
                )),
                ("AMD consumers", lambda: self._amd_consumers(
                    modules,
                    themes,
                )),
            )
            for stage_name, stage in theme_stages:
                try:
                    self._run_stage(stage_name, stage)
                except TimeoutError:
                    raise
                except OptionalJavaScriptEnrichmentError as exception:
                    self._record_optional_enrichment_failure(
                        stage_name,
                        exception,
                    )
                    continue
                except Exception as exception:
                    raise RuntimeError(
                        f"Magento {stage_name} enrichment failed: "
                        f"{type(exception).__name__}: {exception}"
                    ) from exception
            packets = self._run_stage("packet materialization", self.graph.build)
            logger.info(
                "Magento theme repository resolution: themes=%s packets=%s elapsed=%.3fs",
                len(themes),
                len(packets),
                time.monotonic() - started,
            )
            return RepositoryAnalysis(packets=packets), tuple(self._diagnostics)
        try:
            self._run_stage(
                "module packets",
                lambda: self._module_packets(modules),
            )
            themes = self._run_stage(
                "theme discovery",
                lambda: self._themes(modules),
            )
            di_states = self._run_stage(
                "dependency injection",
                lambda: self._di(modules),
            )
            stages = (
                ("constructor/DI", lambda: self._constructor_packets(
                    modules,
                    di_states,
                )),
                ("generated factories", lambda: (
                    self._generated_factory_packets(modules, di_states)
                )),
                ("generated proxies", lambda: (
                    self._generated_proxy_packets(modules, di_states)
                )),
                ("events", lambda: self._events(modules, di_states)),
                ("system configuration", lambda: (
                    self._system_configuration(modules)
                )),
                ("routes/layouts", lambda: self._routes_and_layouts(
                    modules,
                    themes,
                    di_states,
                )),
                ("Admin menu", lambda: self._admin_menu(modules)),
                ("template globals", self._template_globals),
                ("template events", lambda: self._template_events(themes)),
                ("UI components", lambda: self._ui_components(
                    modules,
                    themes,
                )),
                ("email templates", lambda: self._email_templates(
                    modules,
                    themes,
                )),
                ("RequireJS", lambda: self._requirejs(modules, themes)),
                ("frontend initializers", lambda: (
                    self._frontend_initializers(modules, themes)
                )),
                ("AMD consumers", lambda: self._amd_consumers(
                    modules,
                    themes,
                )),
                ("Web API/ACL", lambda: self._webapi_and_acl(
                    modules,
                    di_states,
                )),
                ("cron", lambda: self._cron(modules)),
                ("message queues", lambda: self._message_queues(modules)),
                ("indexers/mview", lambda: (
                    self._indexers_and_materialized_views(modules)
                )),
                ("declarative schema", lambda: self._schema(modules)),
                ("GraphQL", lambda: self._graphql(modules)),
                ("GraphQL clients", lambda: self._graphql_clients(modules)),
                ("extension attributes", lambda: (
                    self._extension_attributes(modules)
                )),
                ("generic config references", lambda: (
                    self._generic_config_references(modules)
                )),
            )
            for stage_name, stage in stages:
                try:
                    self._run_stage(stage_name, stage)
                except TimeoutError:
                    raise
                except OptionalJavaScriptEnrichmentError as exception:
                    self._record_optional_enrichment_failure(
                        stage_name,
                        exception,
                    )
                    continue
                except Exception as exception:
                    raise RuntimeError(
                        f"Magento {stage_name} enrichment failed: "
                        f"{type(exception).__name__}: {exception}"
                    ) from exception
        except RuntimeError:
            raise
        packets = self._run_stage("packet materialization", self.graph.build)
        logger.info(
            "Magento repository resolution: modules=%s packets=%s elapsed=%.3fs",
            len(modules),
            len(packets),
            time.monotonic() - started,
        )
        return RepositoryAnalysis(packets=packets), tuple(self._diagnostics)

    def _record_optional_enrichment_failure(
        self,
        stage_name: str,
        exception: OptionalJavaScriptEnrichmentError,
        path: str | None = None,
    ) -> None:
        """Record an expected optional frontend degradation once per source."""

        diagnostic_path = path if exception.source_specific else None
        identity = (exception.diagnostic_code, diagnostic_path)
        if identity in self._optional_enrichment_diagnostics:
            return
        self._optional_enrichment_diagnostics.add(identity)
        location = f" for {path}" if path else ""
        self._diagnostics.append(PluginDiagnostic(
            exception.diagnostic_code,
            (
                f"Magento {stage_name} enrichment skipped{location}: "
                f"{exception}"
            ),
            self.plugin_id,
            diagnostic_path,
            recoverable=True,
        ))

    def _optional_frontend_source(
        self,
        stage_name: str,
        path: str,
        extractor,
    ):
        """Skip only malformed sources; propagate stage-wide unavailability."""

        try:
            return extractor(self.artifacts[path])
        except OptionalJavaScriptEnrichmentError as exception:
            if not exception.source_specific:
                raise
            self._record_optional_enrichment_failure(
                stage_name,
                exception,
                path,
            )
            return ()

    def _xml(self, path: str):
        if path in self._roots:
            return self._roots[path]
        root, diagnostic = safe_xml(self.plugin_id, path, self.artifacts[path])
        if diagnostic:
            self._diagnostics.append(diagnostic)
            self.invalid_paths.add(path)
        self._roots[path] = root
        return root

    def _modules(self) -> tuple[ModuleRecord, ...]:
        enabled_order: dict[str, tuple[bool, int]] = {}
        config_content = self.artifacts.get("app/etc/config.php", "")
        modules_section = _MODULES_SECTION.search(config_content)
        if modules_section is not None:
            for index, match in enumerate(
                _MODULE_ENABLED.finditer(modules_section.group("body"))
            ):
                enabled_order[match.group("name")] = (
                    match.group("enabled") == "1",
                    index,
                )
            self.configured_modules = {
                name: enabled
                for name, (enabled, _) in enabled_order.items()
            }
        elif config_content:
            self._diagnostics.append(PluginDiagnostic(
                "magento-config-modules-unreadable",
                "app/etc/config.php does not contain a statically readable modules array",
                self.plugin_id,
            ))

        discovered: dict[str, tuple[str, str, tuple[str, ...]]] = {}
        for path in sorted(self.artifacts):
            if not (path == "etc/module.xml" or path.endswith("/etc/module.xml")):
                continue
            root = self._xml(path)
            if root is None:
                continue
            module_node = next(
                (element for element in root.iter() if tag(element) == "module" and element.get("name")),
                None,
            )
            if module_node is None:
                continue
            name = module_node.get("name")
            sequence = tuple(sorted({
                element.get("name")
                for element in module_node.iter()
                if element is not module_node and tag(element) == "module" and element.get("name")
            }))
            discovered[name] = (_module_root(path), path, sequence)

        ordered_names = self._sort_modules(discovered, enabled_order)
        records: list[ModuleRecord] = []
        for order, name in enumerate(ordered_names):
            root, module_xml, sequence = discovered[name]
            # Once a readable deployment module list exists it is authoritative:
            # discovered code absent from the list is installed but not enabled.
            enabled = (
                enabled_order.get(name, (False, order))[0]
                if modules_section is not None
                else True
            )
            records.append(ModuleRecord(name, root, module_xml, sequence, enabled, order))
        return tuple(records)

    def _sort_modules(
        self,
        discovered: dict[str, tuple[str, str, tuple[str, ...]]],
        enabled_order: dict[str, tuple[bool, int]],
    ) -> tuple[str, ...]:
        names = set(discovered)
        if enabled_order:
            # Magento materializes the effective component order into
            # app/etc/config.php. Runtime XML merging follows this order, even
            # when module.xml was edited without regenerating the component
            # list. Keep unlisted installed modules for disabled-module facts,
            # but never let them perturb enabled merge order.
            configured = [
                name
                for name, _ in sorted(
                    enabled_order.items(),
                    key=lambda item: item[1][1],
                )
                if name in names
            ]
            return tuple((*configured, *sorted(names - set(configured))))

        base = sorted(
            names,
            key=lambda name: (
                0 if "Magento_" in name else 1,
                name,
            ),
        )

        sequence_cache: dict[str, tuple[str, ...]] = {}

        def expand(name: str, stack: tuple[str, ...] = ()) -> tuple[str, ...]:
            if name in stack:
                cycle_start = stack.index(name)
                cycle = (*stack[cycle_start:], name)
                raise ValueError(" -> ".join(cycle))
            if name in sequence_cache:
                return sequence_cache[name]
            direct = discovered.get(name, ("", "", ()))[2]
            expanded: list[str] = []
            for dependency in direct:
                expanded.extend(expand(dependency, (*stack, name)))
            expanded.extend(direct)
            sequence_cache[name] = tuple(dict.fromkeys(expanded))
            return sequence_cache[name]

        try:
            expanded = [
                [name, set(expand(name))]
                for name in base
            ]
        except ValueError as exception:
            self._diagnostics.append(PluginDiagnostic(
                "magento-module-sequence-cycle",
                f"Module sequence cycle detected: {exception}",
                self.plugin_id,
            ))
            return tuple(base)

        # Mirror Magento's pairwise sequence ordering. A normal topological sort
        # changes the order of otherwise-unrelated modules and therefore changes
        # last-wins XML merge results.
        total = len(expanded)
        for left in range(total - 1):
            for right in range(left, total):
                if expanded[right][0] in expanded[left][1]:
                    expanded[left], expanded[right] = expanded[right], expanded[left]
        return tuple(item[0] for item in expanded)

    def _module_for_path(self, path: str, modules: tuple[ModuleRecord, ...]) -> ModuleRecord | None:
        candidates = [
            module for module in modules
            if not module.root or path == module.root or path.startswith(module.root + "/")
        ]
        return max(candidates, key=lambda module: len(module.root), default=None)

    def _module_packets(self, modules: tuple[ModuleRecord, ...]) -> None:
        enabled_modules = tuple(module for module in modules if module.enabled)
        enabled_positions = {
            module.name: index
            for index, module in enumerate(enabled_modules)
        }
        deployment_order_path = (
            "app/etc/config.php"
            if _MODULES_SECTION.search(self.artifacts.get("app/etc/config.php", ""))
            else ""
        )
        for module in modules:
            packet = self.graph.packet(
                "magento-module",
                module.name,
                enabled=str(module.enabled).lower(),
                order=str(module.order),
                root=module.root or ".",
            )
            packet.add(GraphFact(
                "magento-module",
                module.name,
                "enabled" if module.enabled else "disabled",
                module.root or ".",
                module.module_xml,
                line(self.artifacts[module.module_xml], module.name),
                attrs(order=module.order),
            ))
            registration = _path_under(module.root, "registration.php")
            if registration in self.artifacts:
                match = _REGISTRATION.search(self.artifacts[registration])
                if match:
                    packet.add(GraphFact(
                        "magento-module-registration",
                        registration,
                        "registers-module",
                        match.group("name"),
                        registration,
                        line(self.artifacts[registration], match.group(0)),
                    ))
            for dependency in module.sequence:
                dependency_module = next((item for item in modules if item.name == dependency), None)
                packet.add(GraphFact(
                    "magento-module-sequence",
                    module.name,
                    "loads-after",
                    dependency,
                    module.module_xml,
                    line(self.artifacts[module.module_xml], dependency),
                    attrs(
                        effectiveOrder=module.order,
                        dependencyOrder=(
                            dependency_module.order
                            if dependency_module is not None
                            else ""
                        ),
                    ),
                ), dependency_module.module_xml if dependency_module else "")
                if (
                    module.enabled
                    and dependency_module is not None
                    and dependency_module.enabled
                    and enabled_positions[dependency] > enabled_positions[module.name]
                ):
                    packet.add(GraphFact(
                        "magento-module-sequence-mismatch",
                        module.name,
                        "configured-before-required-module",
                        dependency,
                        module.module_xml,
                        line(self.artifacts[module.module_xml], dependency),
                        attrs(
                            effectiveOrder=module.order,
                            dependencyOrder=dependency_module.order,
                        ),
                    ), dependency_module.module_xml, "app/etc/config.php")

            position = enabled_positions.get(module.name)
            if deployment_order_path and position is not None and position:
                previous = enabled_modules[position - 1]
                packet.add(GraphFact(
                    "magento-module-effective-order",
                    module.name,
                    "configured-after",
                    previous.name,
                    deployment_order_path,
                    line(self.artifacts[deployment_order_path], module.name),
                    attrs(
                        effectiveOrder=module.order,
                        previousOrder=previous.order,
                    ),
                ), module.module_xml, previous.module_xml)

    def _themes(self, modules: tuple[ModuleRecord, ...]) -> tuple[ThemeRecord, ...]:
        themes: list[ThemeRecord] = []
        for path in sorted(self.artifacts):
            if PurePosixPath(path).name != "theme.xml":
                continue
            root = self._xml(path)
            if root is None:
                continue
            theme_root = path.removesuffix("/theme.xml") if path != "theme.xml" else ""
            inferred = re.match(
                r"app/design/(?P<area>frontend|adminhtml)/(?P<vendor>[^/]+)/(?P<theme>[^/]+)/theme\.xml$",
                path,
            )
            registration_path = _path_under(theme_root, "registration.php")
            registered = ""
            if registration_path in self.artifacts:
                match = _THEME_REGISTRATION.search(self.artifacts[registration_path])
                registered = match.group("name") if match else ""
            if registered and registered.count("/") >= 2:
                area, vendor, theme = registered.split("/", 2)
                name = f"{vendor}/{theme}"
            elif inferred:
                area = inferred.group("area")
                name = f"{inferred.group('vendor')}/{inferred.group('theme')}"
            else:
                # Composer-installed themes must be registered to establish area/name.
                continue
            parent_node = next(
                (node for node in root.iter() if tag(node) == "parent" and node.text),
                None,
            )
            parent = parent_node.text.strip() if parent_node is not None else ""
            record = ThemeRecord(name, area, theme_root, path, parent)
            themes.append(record)
            packet = self.graph.packet(
                "magento-theme",
                f"{area}:{name}",
                area=area,
                theme=name,
            )
            packet.add(GraphFact(
                "magento-theme",
                name,
                "inherits" if parent else "declared-in",
                parent or path,
                path,
                line(self.artifacts[path], parent or "<theme"),
                attrs(area=area),
            ), registration_path if registration_path in self.artifacts else "")
            if parent:
                parent_theme = next(
                    (
                        item for item in themes
                        if item.area == area and item.name == parent
                    ),
                    None,
                )
                if parent_theme:
                    packet.add(GraphFact(
                        "magento-theme-parent",
                        name,
                        "inherits-files-from",
                        parent,
                        path,
                        line(self.artifacts[path], parent),
                        attrs(area=area),
                    ), parent_theme.theme_xml)

        by_identity = {(theme.area, theme.name): theme for theme in themes}
        for theme in themes:
            if not theme.parent:
                continue
            parent_theme = by_identity.get((theme.area, theme.parent))
            if parent_theme is None:
                continue
            packet = self.graph.packet("magento-theme", f"{theme.area}:{theme.name}")
            packet.add(GraphFact(
                "magento-theme-parent",
                theme.name,
                "inherits-files-from",
                theme.parent,
                theme.theme_xml,
                line(self.artifacts[theme.theme_xml], theme.parent),
                attrs(area=theme.area),
            ), parent_theme.theme_xml)
        return tuple(sorted(themes, key=lambda item: (item.area, item.name, item.root)))

    def _ordered_configs(
        self,
        filename: str,
        modules: tuple[ModuleRecord, ...],
        area: str,
    ) -> tuple[tuple[str, ModuleRecord | None, int], ...]:
        result: list[tuple[str, ModuleRecord | None, int]] = []
        initial = f"app/etc/{filename}"
        if initial in self.artifacts:
            result.append((initial, None, -1))
        for module in modules:
            if not module.enabled:
                continue
            global_path = _path_under(module.root, f"etc/{filename}")
            if global_path in self.artifacts:
                result.append((global_path, module, module.order))
        if area not in {"global", "initial"}:
            for module in modules:
                if not module.enabled:
                    continue
                scoped_path = _path_under(module.root, f"etc/{area}/{filename}")
                if scoped_path in self.artifacts:
                    result.append((scoped_path, module, module.order))
        return tuple(result)

    def _di(self, modules: tuple[ModuleRecord, ...]) -> dict[str, DiState]:
        discovered_areas = {
            area
            for path in self.artifacts
            if (area := config_area(path, "di.xml")) not in {None, "initial", "global"}
        }
        states = {
            area: self._di_state(self._ordered_configs("di.xml", modules, area))
            for area in ("global", *sorted(discovered_areas))
        }
        global_state = states.get("global", DiState())
        descendants = self._descendants()

        for area, state in states.items():
            plugin_positions = self._plugin_priority_positions(state, area)
            effective_arguments = self._effective_argument_objects(
                state,
                area,
            )
            self._effective_di_arguments[area] = effective_arguments
            for interface, preference in sorted(state.preferences.items()):
                packet = self.graph.packet("magento-di", f"{area}:preference:{interface}", area=area)
                packet.add(GraphFact(
                    "magento-di-effective-preference",
                    interface,
                    "resolves-to",
                    preference.value,
                    preference.path,
                    preference.line,
                    attrs(area=area, module=preference.module, order=preference.order),
                ), self._symbol_path(interface), self._symbol_path(preference.value))

            for virtual_name, virtual_type in sorted(
                state.virtual_types.items()
            ):
                resolved = self._resolve_virtual_type(
                    virtual_name,
                    state,
                )
                inherited_objects = effective_arguments.get(
                    virtual_name,
                    {},
                )
                packet = self.graph.packet(
                    "magento-di-virtual-type",
                    f"{area}:{virtual_name}",
                    area=area,
                )
                packet.add(GraphFact(
                    "magento-di-virtual-type",
                    virtual_name,
                    "instantiates",
                    resolved,
                    virtual_type.path,
                    virtual_type.line,
                    attrs(
                        area=area,
                        configuredType=virtual_type.value,
                        module=virtual_type.module,
                        order=virtual_type.order,
                    ),
                ),
                    self._symbol_path(resolved),
                    *self._resolution_paths(virtual_name, state),
                    *(
                        configured.path
                        for configured, _ in inherited_objects.values()
                    ),
                )

            for (target, plugin_name), plugin in sorted(state.plugins.items()):
                plugin_attrs = dict(plugin.attributes)
                disabled = plugin_attrs.get("disabled", "false").casefold() in {"1", "true"}
                plugin_class = plugin.value
                relation = "disables-interceptor" if disabled else "intercepted-by"
                packet = self.graph.packet(
                    "magento-interception",
                    f"{area}:{target}:{plugin_name}",
                    area=area,
                    plugin=plugin_name,
                )
                packet.add(GraphFact(
                    "magento-di-effective-plugin",
                    target,
                    relation,
                    plugin_class,
                    plugin.path,
                    plugin.line,
                    attrs(**{
                        "area": area,
                        "module": plugin.module,
                        **plugin_attrs,
                        "name": plugin_name,
                        "effectivePriorityPosition": plugin_positions.get(
                            (target, plugin_name),
                            "",
                        ),
                    }),
                ), self._symbol_path(target), self._symbol_path(plugin_class))

                affected = {target, *descendants.get(target, set())}
                preferred = self._resolve_type(target, state)
                affected.add(preferred)
                affected.update(descendants.get(preferred, set()))
                for affected_type in sorted(affected)[:200]:
                    affected_path = self._symbol_path(affected_type)
                    if affected_type != target:
                        # Magento merges inherited plugin configuration by plugin
                        # name. A direct declaration on the concrete type is the
                        # effective override (including a disable) and must not
                        # coexist with a contradictory inherited fact.
                        if (affected_type, plugin_name) in state.plugins:
                            continue
                        packet.add(GraphFact(
                            "magento-di-inherited-plugin",
                            affected_type,
                            relation,
                            plugin_class,
                            plugin.path,
                            plugin.line,
                            attrs(area=area, declaredFor=target, name=plugin_name),
                        ), affected_path)
                plugin_symbol = self._symbol(plugin_class)
                if not disabled and plugin_symbol:
                    for method in plugin_symbol.methods:
                        subject = _method_subject(method)
                        if subject:
                            phase, intercepted_method = subject
                            applicability, reason, target_path = (
                                self._interception_applicability(
                                    target,
                                    intercepted_method,
                                    state,
                                    plugin_symbol,
                                    method,
                                )
                            )
                            if applicability is True:
                                packet.add(GraphFact(
                                    "magento-intercepted-method",
                                    plugin_class,
                                    phase,
                                    f"{target}::{intercepted_method}",
                                    plugin_symbol.path,
                                    plugin_symbol.line,
                                    attrs(area=area, plugin=plugin_name),
                                ), target_path)
                            elif applicability is False:
                                packet.add(GraphFact(
                                    "magento-interceptor-inapplicable",
                                    plugin_class,
                                    "cannot-intercept",
                                    f"{target}::{intercepted_method}",
                                    plugin.path,
                                    plugin.line,
                                    attrs(
                                        area=area,
                                        plugin=plugin_name,
                                        phase=phase,
                                        reason=reason,
                                        semanticRole="diagnostic",
                                    ),
                                ), plugin_symbol.path, target_path)

            by_target: dict[str, list[tuple[str, ConfigValue]]] = {}
            for (target, plugin_name), plugin in state.plugins.items():
                if (target, plugin_name) not in plugin_positions:
                    continue
                by_target.setdefault(target, []).append((plugin_name, plugin))
            for target, plugins in sorted(by_target.items()):
                ordered = sorted(
                    plugins,
                    key=lambda item: plugin_positions[(target, item[0])],
                )
                packet = self.graph.packet(
                    "magento-interception-order",
                    f"{area}:{target}",
                    area=area,
                    observedType=target,
                )
                for (
                    (plugin_name, plugin),
                    (next_name, next_plugin),
                ) in zip(ordered, ordered[1:]):
                    packet.add(GraphFact(
                        "magento-di-plugin-priority",
                        plugin.value,
                        "prioritized-before",
                        next_plugin.value,
                        plugin.path,
                        plugin.line,
                        attrs(
                            area=area,
                            observedType=target,
                            plugin=plugin_name,
                            nextPlugin=next_name,
                            position=plugin_positions[(target, plugin_name)],
                            nextPosition=plugin_positions[(target, next_name)],
                            sortOrder=dict(plugin.attributes).get("sortOrder", "0"),
                            nextSortOrder=dict(next_plugin.attributes).get(
                                "sortOrder",
                                "0",
                            ),
                        ),
                    ), next_plugin.path, self._symbol_path(plugin.value),
                       self._symbol_path(next_plugin.value))

            for owner, objects in sorted(effective_arguments.items()):
                inheritance_paths = self._argument_inheritance_paths(
                    owner,
                    state,
                )
                for (
                    (argument_name, item_name),
                    (argument, declared_for),
                ) in sorted(objects.items()):
                    resolved = self._resolve_type(argument.value, state)
                    resolution_paths = self._resolution_paths(
                        argument.value,
                        state,
                    )
                    packet = self.graph.packet(
                        "magento-di",
                        (
                            f"{area}:argument:{owner}:"
                            f"{argument_name}:{item_name}"
                        ),
                        area=area,
                    )
                    packet.add(GraphFact(
                        "magento-di-argument",
                        owner,
                        "injects",
                        resolved,
                        argument.path,
                        argument.line,
                        attrs(
                            area=area,
                            argument=argument_name,
                            item=item_name,
                            configured=argument.value,
                            declaredFor=declared_for,
                            inherited=(
                                "true"
                                if declared_for != owner
                                else ""
                            ),
                            virtualType=(
                                "true"
                                if owner in state.virtual_types
                                else ""
                            ),
                        ),
                    ),
                        self._symbol_path(owner),
                        self._symbol_path(resolved),
                        *inheritance_paths,
                        *resolution_paths,
                    )

            self._price_pool_packets(
                state,
                area,
                global_state if area != "global" else None,
            )

        # Area state that equals global does not need duplicate constructor packets.
        if "global" not in states:
            states["global"] = global_state
        return states

    def _di_state(self, documents: tuple[tuple[str, ModuleRecord | None, int], ...]) -> DiState:
        state = DiState()
        plugin_position = 0
        argument_items: dict[tuple[str, str], set[str]] = {}

        def clear_argument(owner: str, argument_name: str) -> None:
            argument_key = (owner, argument_name)
            for item_name in argument_items.pop(argument_key, set()):
                key = (owner, argument_name, item_name)
                state.arguments.pop(key, None)
                state.item_types.pop(key, None)
                state.item_values.pop(key, None)

        def clear_item(owner: str, argument_name: str, item_name: str) -> None:
            prefix = item_name + "/"
            argument_key = (owner, argument_name)
            names = argument_items.setdefault(argument_key, set())
            removed = {
                candidate
                for candidate in names
                if candidate == item_name or candidate.startswith(prefix)
            }
            for candidate in removed:
                key = (owner, argument_name, candidate)
                state.arguments.pop(key, None)
                state.item_types.pop(key, None)
                state.item_values.pop(key, None)
            names.difference_update(removed)

        def xsi_type(element) -> str:
            return next(
                (
                    value
                    for key, value in element.attrib.items()
                    if key.endswith("}type") or key == "xsi:type"
                ),
                "",
            )

        def merge_array_items(
            element,
            owner: str,
            argument_name: str,
            item_path: str,
            source_path: str,
            content: str,
            module_name: str,
            order: int,
        ) -> None:
            for item in element:
                if tag(item) != "item" or not item.get("name"):
                    continue
                item_name = (
                    f"{item_path}/{item.get('name')}"
                    if item_path
                    else item.get("name")
                )
                key = (owner, argument_name, item_name)
                item_type = xsi_type(item)
                prior_type = state.item_types.get(key)
                if prior_type is None or prior_type.value != item_type:
                    clear_item(owner, argument_name, item_name)
                state.item_types[key] = ConfigValue(
                    item_type,
                    source_path,
                    line(content, item.get("name")),
                    module_name,
                    order,
                )
                argument_items.setdefault(
                    (owner, argument_name),
                    set(),
                ).add(item_name)
                if item_type == "array":
                    merge_array_items(
                        item,
                        owner,
                        argument_name,
                        item_name,
                        source_path,
                        content,
                        module_name,
                        order,
                    )
                    continue
                if item_type != "object":
                    clear_item(owner, argument_name, item_name)
                    state.item_types[key] = ConfigValue(
                        item_type,
                        source_path,
                        line(content, item.get("name")),
                        module_name,
                        order,
                    )
                    argument_items.setdefault(
                        (owner, argument_name),
                        set(),
                    ).add(item_name)
                    configured = (item.text or "").strip().lstrip("\\")
                    if configured:
                        state.item_values[key] = ConfigValue(
                            configured,
                            path=source_path,
                            line=line(content, configured),
                            module=module_name,
                            order=order,
                        )
                    continue
                configured = (item.text or "").strip().lstrip("\\")
                if configured:
                    state.arguments[key] = ConfigValue(
                        configured,
                        path=source_path,
                        line=line(content, configured),
                        module=module_name,
                        order=order,
                    )

        for path, module, order in documents:
            root = self._xml(path)
            if root is None:
                continue
            module_name = module.name if module else "application"
            content = self.artifacts[path]
            for element in root.iter():
                element_tag = tag(element)
                if element_tag == "preference" and element.get("for") and element.get("type"):
                    state.preferences[element.get("for").lstrip("\\")] = ConfigValue(
                        element.get("type").lstrip("\\"), path,
                        line(content, element.get("for")), module_name, order,
                    )
                elif element_tag == "virtualType" and element.get("name") and element.get("type"):
                    state.virtual_types[element.get("name").lstrip("\\")] = ConfigValue(
                        element.get("type").lstrip("\\"), path,
                        line(content, element.get("name")), module_name, order,
                    )
                elif element_tag == "plugin" and element.get("name"):
                    parent = next((candidate for candidate in root.iter() if element in list(candidate)), None)
                    target = parent.get("name") if parent is not None else None
                    prior = state.plugins.get((target or "", element.get("name")))
                    plugin_class = element.get("type") or (prior.value if prior else "")
                    if target and plugin_class:
                        merged_attrs = dict(prior.attributes) if prior else {}
                        merged_attrs.update({
                            key: value for key, value in element.attrib.items()
                            if key not in {"name", "type"}
                        })
                        state.plugins[(target.lstrip("\\"), element.get("name"))] = ConfigValue(
                            plugin_class.lstrip("\\"), path,
                            line(content, element.get("name")), module_name, order,
                            tuple(sorted(merged_attrs.items())),
                            plugin_position,
                        )
                        plugin_position += 1

            for owner in root.iter():
                if tag(owner) not in {"type", "virtualType"} or not owner.get("name"):
                    continue
                owner_name = owner.get("name").lstrip("\\")
                for arguments in owner:
                    if tag(arguments) != "arguments":
                        continue
                    for argument in arguments:
                        if tag(argument) != "argument" or not argument.get("name"):
                            continue
                        argument_name = argument.get("name")
                        argument_key = (owner_name, argument_name)
                        argument_type = xsi_type(argument)
                        prior_type = state.argument_types.get(argument_key)
                        if prior_type is None or prior_type.value != argument_type:
                            clear_argument(owner_name, argument_name)
                        state.argument_types[argument_key] = ConfigValue(
                            argument_type,
                            path,
                            line(content, argument_name),
                            module_name,
                            order,
                        )
                        if argument_type == "array":
                            merge_array_items(
                                argument,
                                owner_name,
                                argument_name,
                                "",
                                path,
                                content,
                                module_name,
                                order,
                            )
                            continue
                        clear_argument(owner_name, argument_name)
                        state.argument_types[argument_key] = ConfigValue(
                            argument_type,
                            path,
                            line(content, argument_name),
                            module_name,
                            order,
                        )
                        if argument_type != "object":
                            continue
                        configured = (argument.text or "").strip().lstrip("\\")
                        if configured:
                            argument_items.setdefault(
                                argument_key,
                                set(),
                            ).add("")
                            state.arguments[
                                (owner_name, argument_name, "")
                            ] = ConfigValue(
                                configured,
                                path,
                                line(content, configured),
                                module_name,
                                order,
                            )
        return state

    def _price_pool_packets(
        self,
        state: DiState,
        area: str,
        global_state: DiState | None = None,
    ) -> None:
        """Join exact PHP price-code reads to Magento's effective price pool."""
        canonical_pool = r"Magento\Catalog\Pricing\Price\Pool"
        framework_pool = r"Magento\Framework\Pricing\Price\Pool"
        registrations: dict[
            tuple[str, str],
            tuple[str, ConfigValue, SymbolDefinition],
        ] = {}

        for (
            owner,
            argument_name,
            item_name,
        ), configured in sorted(state.item_values.items()):
            state_key = (owner, argument_name, item_name)
            item_type = state.item_types.get(state_key)
            if (
                global_state is not None
                and configured == global_state.item_values.get(state_key)
                and item_type == global_state.item_types.get(state_key)
            ):
                # A global registration applies to every area. Only publish an
                # area packet when scoped DI actually changes the effective
                # entry, otherwise prompt selection would repeat the same fact.
                continue
            if (
                argument_name != "prices"
                or not item_name
                or "/" in item_name
                or item_type is None
                or item_type.value != "string"
            ):
                continue
            resolved_owner = (
                self._resolve_virtual_type(owner, state)
                if owner in state.virtual_types
                else ""
            )
            if owner != canonical_pool and resolved_owner != framework_pool:
                continue
            provider = self._unique_symbol_casefold(configured.value)
            if provider is None:
                continue
            declared = None
            for key, value in provider.attributes:
                if not key.startswith("php-class-constant:"):
                    continue
                try:
                    candidate = json.loads(value)
                except (TypeError, ValueError):
                    continue
                if (
                    candidate.get("name") == "PRICE_CODE"
                    and candidate.get("value") == item_name
                ):
                    declared = candidate
                    break
            if declared is None:
                continue
            registrations[(provider.qualified_name, item_name)] = (
                owner,
                configured,
                provider,
            )
            packet = self.graph.packet(
                "magento-price-pool",
                f"{area}:registration:{owner}:{item_name}",
                area=area,
                pool=owner,
            )
            packet.add(GraphFact(
                "magento-price-pool-registration",
                owner,
                "registers-price-model",
                provider.qualified_name,
                configured.path,
                configured.line,
                attrs(
                    area=area,
                    argument="prices",
                    priceCode=item_name,
                ),
            ), provider.path)

        if not registrations:
            return
        registration_by_provider = {
            provider_name: (price_code, registration)
            for (
                provider_name,
                price_code,
            ), registration in registrations.items()
        }
        for consumer in self.symbols:
            for key, value in consumer.attributes:
                if not key.startswith("php-class-constant-reference:"):
                    continue
                try:
                    reference = json.loads(value)
                except (TypeError, ValueError):
                    continue
                if (
                    str(reference.get("argumentOf", "")).casefold()
                    != "getprice"
                    or reference.get("constant") != "PRICE_CODE"
                ):
                    continue
                provider_name = str(
                    reference.get("target", "")
                ).lstrip("\\")
                resolved_registration = registration_by_provider.get(
                    provider_name
                )
                if resolved_registration is None:
                    continue
                price_code, registration = resolved_registration
                identity = (provider_name, price_code)
                owner, configured, provider = registration
                try:
                    reference_line = max(1, int(reference.get("line", 1)))
                except (TypeError, ValueError):
                    reference_line = consumer.line
                packet = self.graph.packet(
                    "magento-price-pool-reference",
                    (
                        f"{area}:{consumer.qualified_name}:"
                        f"{provider.qualified_name}:{identity[1]}"
                    ),
                    area=area,
                    pool=owner,
                )
                packet.add(GraphFact(
                    "magento-price-pool-reference",
                    consumer.qualified_name,
                    "requests-registered-price",
                    provider.qualified_name,
                    consumer.path,
                    reference_line,
                    attrs(
                        area=area,
                        configPath=configured.path,
                        pool=owner,
                        priceCode=identity[1],
                    ),
                ), configured.path, provider.path)

    def _plugin_priority_positions(
        self,
        state: DiState,
        area: str,
    ) -> dict[tuple[str, str], int]:
        by_target: dict[str, list[tuple[str, ConfigValue, int]]] = {}
        invalid_targets: set[str] = set()
        for (target, plugin_name), plugin in sorted(state.plugins.items()):
            plugin_attrs = dict(plugin.attributes)
            if plugin_attrs.get("disabled", "false").casefold() in {"1", "true"}:
                continue
            raw_sort_order = plugin_attrs.get("sortOrder", "0")
            try:
                sort_order = int(raw_sort_order)
            except ValueError:
                invalid_targets.add(target)
                self._diagnostics.append(PluginDiagnostic(
                    "magento-plugin-sort-order-invalid",
                    (
                        f"{area} plugin {plugin_name!r} for {target} has "
                        f"non-integer sortOrder {raw_sort_order!r}"
                    ),
                    self.plugin_id,
                ))
                continue
            by_target.setdefault(target, []).append(
                (plugin_name, plugin, sort_order)
            )

        positions: dict[tuple[str, str], int] = {}
        for target, plugins in sorted(by_target.items()):
            # A single invalid member makes the complete priority chain
            # unknowable; keep each raw plugin fact but emit no ordering claim.
            if target in invalid_targets:
                continue
            ordered = sorted(
                plugins,
                key=lambda item: (
                    item[2],
                    item[1].order,
                    item[1].position,
                    item[0],
                ),
            )
            positions.update({
                (target, plugin_name): position
                for position, (plugin_name, _, _) in enumerate(ordered)
            })
        return positions

    def _php_argument_bases(self, owner: str) -> tuple[str, ...]:
        """Mirror ClassReader parent order recorded by the PHP repository plugin."""
        symbol = self._symbol(owner)
        if symbol is None or symbol.kind != "class":
            return ()
        parent_class = next(
            (
                value
                for key, value in symbol.attributes
                if key == "php-parent-class"
            ),
            "",
        )

        def declared_interfaces(
            candidate: SymbolDefinition | None,
        ) -> tuple[str, ...]:
            if candidate is None:
                return ()
            return tuple(
                value
                for key, value in sorted(candidate.attributes)
                if key.startswith("php-interface:")
            )

        def interface_closure(
            interface: str,
            seen: set[str],
        ) -> tuple[str, ...]:
            if interface in seen:
                return ()
            seen.add(interface)
            result = [interface]
            interface_symbol = self._symbol(interface)
            if interface_symbol is not None:
                for key, parent in sorted(interface_symbol.attributes):
                    if key.startswith("php-parent-interface:"):
                        result.extend(interface_closure(parent, seen))
            return tuple(result)

        inherited_interfaces: set[str] = set()
        current_parent = self._symbol(parent_class)
        while current_parent is not None and current_parent.kind == "class":
            for interface in declared_interfaces(current_parent):
                inherited_interfaces.update(
                    interface_closure(interface, set())
                )
            next_parent = next(
                (
                    value
                    for key, value in current_parent.attributes
                    if key == "php-parent-class"
                ),
                "",
            )
            current_parent = self._symbol(next_parent)

        interfaces: list[str] = []
        seen_interfaces: set[str] = set()
        for interface in declared_interfaces(symbol):
            for candidate in interface_closure(
                interface,
                seen_interfaces,
            ):
                if candidate not in inherited_interfaces:
                    interfaces.append(candidate)

        return tuple(
            value
            for value in (parent_class, *interfaces)
            if value
        )

    def _argument_inheritance_paths(
        self,
        owner: str,
        state: DiState,
    ) -> tuple[str, ...]:
        """Return only sources that can contribute effective DI arguments."""
        paths: set[str] = set()
        visiting: set[str] = set()

        def collect(candidate: str) -> None:
            if candidate in visiting:
                return
            visiting.add(candidate)
            try:
                virtual_type = state.virtual_types.get(candidate)
                if virtual_type is not None:
                    paths.add(virtual_type.path)
                    instance_path = self._symbol_path(virtual_type.value)
                    if instance_path:
                        paths.add(instance_path)
                    collect(virtual_type.value)
                    return
                for base in self._php_argument_bases(candidate):
                    base_path = self._symbol_path(base)
                    if base_path:
                        paths.add(base_path)
                    collect(base)
            finally:
                visiting.remove(candidate)

        collect(owner)
        return tuple(sorted(paths))

    @staticmethod
    def _remove_argument_subtree(
        values: dict[tuple[str, str], object],
        argument_name: str,
        item_name: str = "",
    ) -> None:
        prefix = item_name + "/" if item_name else ""
        for key in tuple(values):
            argument, item = key
            if argument != argument_name:
                continue
            if not item_name or item == item_name or item.startswith(prefix):
                values.pop(key, None)

    def _effective_argument_objects(
        self,
        state: DiState,
        area: str,
    ) -> dict[
        str,
        dict[tuple[str, str], tuple[ConfigValue, str]],
    ]:
        """Reproduce Config::_collectConfiguration for object-valued leaves.

        Parent relations replace complete top-level arguments in their runtime
        order. A type's own arguments then use array_replace_recursive, so only
        same-named nested array items merge; scalar/object type changes remove
        inherited descendants.
        """
        memo: dict[
            str,
            tuple[
                dict[tuple[str, str], ConfigValue],
                dict[tuple[str, str], tuple[ConfigValue, str]],
            ],
        ] = {}
        visiting: list[str] = []

        def local(owner: str):
            types = {
                (argument, ""): configured
                for (configured_owner, argument), configured
                in state.argument_types.items()
                if configured_owner == owner
            }
            types.update({
                (argument, item): configured
                for (configured_owner, argument, item), configured
                in state.item_types.items()
                if configured_owner == owner
            })
            objects = {
                (argument, item): (configured, owner)
                for (configured_owner, argument, item), configured
                in state.arguments.items()
                if configured_owner == owner
            }
            return types, objects

        def replace_argument(
            destination_types,
            destination_objects,
            source_types,
            source_objects,
            argument_name: str,
        ) -> None:
            self._remove_argument_subtree(
                destination_types,
                argument_name,
            )
            self._remove_argument_subtree(
                destination_objects,
                argument_name,
            )
            destination_types.update({
                key: value
                for key, value in source_types.items()
                if key[0] == argument_name
            })
            destination_objects.update({
                key: value
                for key, value in source_objects.items()
                if key[0] == argument_name
            })

        def overlay_local(
            inherited_types,
            inherited_objects,
            local_types,
            local_objects,
        ) -> None:
            local_arguments = sorted({
                argument
                for argument, item in local_types
                if item == ""
            })
            for argument in local_arguments:
                local_root = local_types[(argument, "")]
                inherited_root = inherited_types.get((argument, ""))
                if (
                    inherited_root is None
                    or inherited_root.value != "array"
                    or local_root.value != "array"
                ):
                    replace_argument(
                        inherited_types,
                        inherited_objects,
                        local_types,
                        local_objects,
                        argument,
                    )
                    continue

                # array_replace_recursive keeps inherited named items unless a
                # local item with the same path replaces or recursively merges it.
                inherited_types[(argument, "")] = local_root
                local_items = sorted(
                    (
                        (item, configured)
                        for (candidate_argument, item), configured
                        in local_types.items()
                        if candidate_argument == argument and item
                    ),
                    key=lambda item: (
                        item[0].count("/"),
                        item[0],
                    ),
                )
                for item_name, configured in local_items:
                    key = (argument, item_name)
                    inherited = inherited_types.get(key)
                    if (
                        inherited is None
                        or inherited.value != "array"
                        or configured.value != "array"
                    ):
                        self._remove_argument_subtree(
                            inherited_types,
                            argument,
                            item_name,
                        )
                        self._remove_argument_subtree(
                            inherited_objects,
                            argument,
                            item_name,
                        )
                    inherited_types[key] = configured
                    if key in local_objects:
                        inherited_objects[key] = local_objects[key]

        def collect(owner: str):
            if owner in memo:
                types, objects = memo[owner]
                return dict(types), dict(objects)
            if owner in visiting:
                cycle = " -> ".join((*visiting, owner))
                raise ValueError(
                    f"DI argument inheritance cycle in {area}: {cycle}"
                )
            visiting.append(owner)
            try:
                if owner in state.virtual_types:
                    base = state.virtual_types[owner].value
                    inherited_types, inherited_objects = collect(base)
                else:
                    inherited_types = {}
                    inherited_objects = {}
                    for base in self._php_argument_bases(owner):
                        base_types, base_objects = collect(base)
                        # Config::_collectConfiguration uses array_replace
                        # between parent/interface relations.
                        for argument in sorted({
                            name
                            for name, item in base_types
                            if item == ""
                        }):
                            replace_argument(
                                inherited_types,
                                inherited_objects,
                                base_types,
                                base_objects,
                                argument,
                            )
                local_types, local_objects = local(owner)
                overlay_local(
                    inherited_types,
                    inherited_objects,
                    local_types,
                    local_objects,
                )
                memo[owner] = (
                    dict(inherited_types),
                    dict(inherited_objects),
                )
                return inherited_types, inherited_objects
            finally:
                visiting.pop()

        owners = {
            owner for owner, _ in state.argument_types
        } | set(state.virtual_types)
        result = {}
        try:
            for owner in sorted(owners):
                _, objects = collect(owner)
                result[owner] = objects
        except ValueError as exception:
            self._diagnostics.append(PluginDiagnostic(
                "magento-di-argument-inheritance-cycle",
                str(exception),
                self.plugin_id,
            ))
            return {}
        return result

    @staticmethod
    def _resolve_virtual_type(
        requested: str,
        state: DiState,
    ) -> str:
        current = requested.lstrip("\\")
        seen: set[str] = set()
        while current not in seen and current in state.virtual_types:
            seen.add(current)
            current = state.virtual_types[current].value
        return current

    def _resolve_type(self, requested: str, state: DiState) -> str:
        current = requested.lstrip("\\")
        # ObjectManager resolves the complete preference chain before the
        # factory resolves the requested virtual type. It does not re-apply a
        # preference to the concrete base reached inside getInstanceType().
        for mapping in (state.preferences, state.virtual_types):
            seen: set[str] = set()
            while current not in seen and current in mapping:
                seen.add(current)
                current = mapping[current].value
        return current

    def _resolution_paths(self, requested: str, state: DiState) -> tuple[str, ...]:
        """Return every configuration source participating in type resolution."""
        current = requested.lstrip("\\")
        paths: set[str] = set()
        for mapping in (state.preferences, state.virtual_types):
            seen: set[str] = set()
            while current not in seen and current in mapping:
                seen.add(current)
                configured = mapping[current]
                paths.add(configured.path)
                current = configured.value
        return tuple(sorted(paths))

    def _constructor_packets(
        self,
        modules: tuple[ModuleRecord, ...],
        states: dict[str, DiState],
    ) -> None:
        global_state = states.get("global", DiState())
        for symbol in self.symbols:
            if not symbol.constructor_types:
                continue
            module = self._module_for_path(symbol.path, modules)
            for area, state in states.items():
                resolutions = tuple(
                    (requested, self._resolve_type(requested, state))
                    for requested in symbol.constructor_types
                )
                global_resolutions = tuple(
                    (requested, self._resolve_type(requested, global_state))
                    for requested in symbol.constructor_types
                )
                if area != "global" and resolutions == global_resolutions:
                    continue
                packet = self.graph.packet(
                    "magento-object-graph",
                    f"{area}:{symbol.qualified_name}",
                    area=area,
                    module=module.name if module else "",
                )
                for requested, resolved in resolutions:
                    resolution_paths = self._resolution_paths(requested, state)
                    packet.add(GraphFact(
                        "php-constructor-dependency",
                        symbol.qualified_name,
                        "requests",
                        requested,
                        symbol.path,
                        symbol.line,
                        attrs(area=area),
                    ), self._symbol_path(requested), *resolution_paths)
                    # An identity result is not a Magento DI resolution.  In
                    # particular, an unconfigured interface cannot be created
                    # by the object manager, so claiming that it resolves to
                    # itself would leave false architecture context after a
                    # preference is removed.  Keep the language-level request
                    # edge above and emit this framework edge only when the
                    # effective Magento configuration actually maps the type.
                    if resolved != requested:
                        packet.add(GraphFact(
                            "magento-object-resolution",
                            requested,
                            "resolves-to",
                            resolved,
                            symbol.path,
                            symbol.line,
                            attrs(area=area, consumer=symbol.qualified_name),
                        ), self._symbol_path(resolved), *resolution_paths)

    def _generated_factory_packets(
        self,
        modules: tuple[ModuleRecord, ...],
        states: dict[str, DiState],
    ) -> None:
        """Resolve only Magento's exact, absent ``<type>Factory`` convention.

        PHP cannot link a constructor dependency to a class that is intentionally
        absent from source. Magento owns the missing-class semantics: its object
        manager generates the factory, while a factory targeting an interface
        follows the effective DI preference. Keep that framework knowledge here
        and abstain whenever source identity or deployment state is ambiguous.
        """
        global_state = states.get("global", DiState())
        for consumer in self.symbols:
            if consumer.kind != "class" or not consumer.constructor_types:
                continue
            consumer_module = self._module_for_path(consumer.path, modules)
            if consumer_module is None or not consumer_module.enabled:
                continue
            for requested_factory in consumer.constructor_types:
                factory_type = requested_factory.lstrip("\\")
                if (
                    not factory_type.endswith("Factory")
                    or len(factory_type) <= len("Factory")
                ):
                    continue
                # A declared class is a custom factory. Its create() semantics
                # belong to PHP source analysis and must not be guessed from its
                # name. Ambiguous declarations are equally unsafe.
                if self.symbols_by_casefold.get(factory_type.casefold()):
                    continue

                requested_type = factory_type[:-len("Factory")]
                target_candidates = self.symbols_by_casefold.get(
                    requested_type.casefold(),
                    (),
                )
                if len(target_candidates) != 1:
                    continue
                target_symbol = target_candidates[0]
                if target_symbol.kind not in {"class", "interface"}:
                    continue
                target_module = self._module_for_path(
                    target_symbol.path,
                    modules,
                )
                if target_module is None or not target_module.enabled:
                    continue

                requested_type = target_symbol.qualified_name
                packet = self.graph.packet(
                    "magento-generated-factory",
                    f"{consumer.qualified_name}:{factory_type}",
                    consumerModule=consumer_module.name,
                    factoryType=factory_type,
                    targetModule=target_module.name,
                )
                packet.add(GraphFact(
                    "magento-generated-factory",
                    consumer.qualified_name,
                    "uses-generated-factory-for",
                    requested_type,
                    consumer.path,
                    consumer.line,
                    attrs(
                        consumerModule=consumer_module.name,
                        factoryType=factory_type,
                        generated="true",
                        requestedKind=target_symbol.kind,
                        targetModule=target_module.name,
                    ),
                ), target_symbol.path)

                global_resolution = self._resolve_type(
                    requested_type,
                    global_state,
                )
                for area, state in states.items():
                    resolved = self._resolve_type(requested_type, state)
                    if (
                        area != "global"
                        and resolved == global_resolution
                    ):
                        continue
                    # An interface without a preference is not constructible.
                    # Keep the exact factory-target edge above, but do not claim
                    # a runtime-created implementation that Magento cannot prove.
                    if (
                        target_symbol.kind == "interface"
                        and resolved == requested_type
                    ):
                        continue
                    resolved_symbol = self._unique_symbol_casefold(resolved)
                    canonical_resolved = (
                        resolved_symbol.qualified_name
                        if resolved_symbol is not None
                        else resolved.lstrip("\\")
                    )
                    resolution_paths = self._resolution_paths(
                        requested_type,
                        state,
                    )
                    packet.add(GraphFact(
                        "magento-generated-factory-resolution",
                        consumer.qualified_name,
                        "creates-via-generated-factory",
                        canonical_resolved,
                        consumer.path,
                        consumer.line,
                        attrs(
                            area=area,
                            factoryType=factory_type,
                            requestedType=requested_type,
                        ),
                    ),
                        target_symbol.path,
                        resolved_symbol.path if resolved_symbol else "",
                        *resolution_paths,
                    )

    def _generated_proxy_packets(
        self,
        modules: tuple[ModuleRecord, ...],
        states: dict[str, DiState],
    ) -> None:
        """Resolve exact, absent ``<type>\\Proxy`` DI object arguments.

        Magento proxies are configured object values rather than constructor
        declarations. The generated class lazily delegates to the suffix-free
        original type, whose effective runtime implementation still follows the
        area's DI preference. Preserve the configured proxy edge and abstain
        whenever source identity or deployment state is ambiguous.
        """
        global_state = states.get("global", DiState())
        global_objects = self._effective_di_arguments.get("global", {})

        def proxy_target(configured: ConfigValue):
            proxy_type = configured.value.lstrip("\\")
            suffix = "\\Proxy"
            if (
                not proxy_type.endswith(suffix)
                or len(proxy_type) <= len(suffix)
                or self.symbols_by_casefold.get(proxy_type.casefold())
            ):
                return None
            requested_type = proxy_type[:-len(suffix)]
            candidates = self.symbols_by_casefold.get(
                requested_type.casefold(),
                (),
            )
            if len(candidates) != 1:
                return None
            target_symbol = candidates[0]
            if target_symbol.kind not in {"class", "interface"}:
                return None
            target_module = self._module_for_path(
                target_symbol.path,
                modules,
            )
            if target_module is None or not target_module.enabled:
                return None
            return (
                proxy_type,
                target_symbol.qualified_name,
                target_symbol,
                target_module,
            )

        for area, state in states.items():
            area_objects = self._effective_di_arguments.get(area, {})
            for owner, objects in sorted(area_objects.items()):
                global_owner_objects = global_objects.get(owner, {})
                for (
                    (argument_name, item_name),
                    (argument, declared_for),
                ) in sorted(objects.items()):
                    target = proxy_target(argument)
                    if target is None:
                        continue
                    (
                        proxy_type,
                        requested_type,
                        target_symbol,
                        target_module,
                    ) = target

                    global_argument_entry = global_owner_objects.get(
                        (argument_name, item_name)
                    )
                    global_target = (
                        proxy_target(global_argument_entry[0])
                        if global_argument_entry is not None
                        else None
                    )
                    dependency_changed = (
                        area == "global"
                        or global_argument_entry is None
                        or global_target is None
                        or (
                            proxy_type,
                            requested_type,
                            declared_for,
                        ) != (
                            global_target[0],
                            global_target[1],
                            global_argument_entry[1],
                        )
                    )

                    resolved = self._resolve_type(requested_type, state)
                    global_resolved = self._resolve_type(
                        requested_type,
                        global_state,
                    )
                    resolution_changed = (
                        area == "global"
                        or dependency_changed
                        or resolved != global_resolved
                    )
                    if not dependency_changed and not resolution_changed:
                        continue

                    packet = self.graph.packet(
                        "magento-generated-proxy",
                        (
                            f"{area}:{owner}:{argument_name}:"
                            f"{item_name}:{proxy_type}"
                        ),
                        area=area,
                        proxyType=proxy_type,
                    )
                    owner_path = self._symbol_path(owner)
                    inheritance_paths = self._argument_inheritance_paths(
                        owner,
                        state,
                    )
                    if dependency_changed:
                        packet.add(GraphFact(
                            "magento-generated-proxy",
                            owner,
                            "injects-generated-proxy-for",
                            requested_type,
                            argument.path,
                            argument.line,
                            attrs(
                                area=area,
                                argument=argument_name,
                                configured=argument.value,
                                declaredFor=declared_for,
                                generated="true",
                                inherited=(
                                    "true"
                                    if declared_for != owner
                                    else ""
                                ),
                                item=item_name,
                                module=argument.module,
                                proxyType=proxy_type,
                                requestedKind=target_symbol.kind,
                                targetModule=target_module.name,
                                virtualType=(
                                    "true"
                                    if owner in state.virtual_types
                                    else ""
                                ),
                            ),
                        ),
                            owner_path,
                            target_symbol.path,
                            *inheritance_paths,
                        )

                    if not resolution_changed:
                        continue
                    if (
                        target_symbol.kind == "interface"
                        and resolved == requested_type
                    ):
                        continue
                    resolved_symbol = self._unique_symbol_casefold(resolved)
                    canonical_resolved = (
                        resolved_symbol.qualified_name
                        if resolved_symbol is not None
                        else resolved.lstrip("\\")
                    )
                    packet.add(GraphFact(
                        "magento-generated-proxy-resolution",
                        owner,
                        "lazy-loads-via-generated-proxy",
                        canonical_resolved,
                        argument.path,
                        argument.line,
                        attrs(
                            area=area,
                            argument=argument_name,
                            item=item_name,
                            proxyType=proxy_type,
                            requestedType=requested_type,
                        ),
                    ),
                        owner_path,
                        target_symbol.path,
                        resolved_symbol.path if resolved_symbol else "",
                        *inheritance_paths,
                        *self._resolution_paths(requested_type, state),
                    )

    def _descendants(self) -> dict[str, set[str]]:
        direct: dict[str, set[str]] = {}
        for symbol in self.symbols:
            for parent in symbol.parents:
                direct.setdefault(parent, set()).add(symbol.qualified_name)
        result: dict[str, set[str]] = {}
        for parent in direct:
            queue = list(direct[parent])
            descendants: set[str] = set()
            while queue:
                child = queue.pop()
                if child in descendants:
                    continue
                descendants.add(child)
                queue.extend(direct.get(child, ()))
            result[parent] = descendants
        return result

    def _event_entrypoints(
        self,
        modules: tuple[ModuleRecord, ...],
        states: dict[str, DiState],
        dispatches: tuple[EventDispatch, ...],
    ) -> tuple[EventEntrypoint, ...]:
        """Collect exact runtime-area evidence for dispatching callables."""
        entrypoints: set[EventEntrypoint] = set()
        global_state = states.get("global", DiState())

        routed_modules: dict[tuple[str, str], set[str]] = {}
        for area in ("adminhtml", "frontend"):
            for path, _, _ in self._ordered_configs(
                "routes.xml",
                modules,
                area,
            ):
                root = self._xml(path)
                if root is None:
                    continue
                for route_module in (
                    node
                    for route in root.iter()
                    if tag(route) == "route"
                    for node in route
                    if tag(node) == "module" and node.get("name")
                ):
                    routed_modules.setdefault(
                        (area, route_module.get("name")),
                        set(),
                    ).add(path)

        for dispatch in dispatches:
            if dispatch.caller.casefold() != "execute":
                continue
            module = self._module_for_path(dispatch.path, modules)
            if module is None or not module.enabled:
                continue
            relative = (
                dispatch.path[len(module.root):].lstrip("/")
                if module.root
                else dispatch.path
            )
            normalized = "/" + relative.casefold().strip("/")
            candidate_area = ""
            if "/controller/adminhtml/" in normalized:
                candidate_area = "adminhtml"
            elif "/controller/" in normalized:
                candidate_area = "frontend"
            for path in sorted(routed_modules.get(
                (candidate_area, module.name),
                (),
            )):
                entrypoints.add(EventEntrypoint(
                    candidate_area,
                    dispatch.owner,
                    dispatch.caller,
                    path,
                ))

        for path, _, _ in self._ordered_configs(
            "crontab.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            state = states.get("crontab", global_state)
            for job in (
                node
                for node in root.iter()
                if tag(node) == "job" and node.get("instance")
            ):
                target = self._resolve_type(job.get("instance"), state)
                entrypoints.add(EventEntrypoint(
                    "crontab",
                    target,
                    job.get("method", "execute"),
                    path,
                ))

        for path, content in sorted(self.artifacts.items()):
            if not path.casefold().endswith(".graphqls"):
                continue
            module = self._module_for_path(path, modules)
            if module is None or not module.enabled:
                continue
            state = states.get("graphql", global_state)
            for declaration in parse_schema(content):
                directives = [
                    *declaration.directives,
                    *(
                        directive
                        for field in declaration.fields
                        for directive in field.directives
                    ),
                ]
                for directive in directives:
                    target = directive.argument("class")
                    if (
                        directive.name not in {"resolver", "typeResolver"}
                        or not target
                    ):
                        continue
                    entrypoints.add(EventEntrypoint(
                        "graphql",
                        self._resolve_type(target, state),
                        (
                            "resolveType"
                            if directive.name == "typeResolver"
                            else "resolve"
                        ),
                        path,
                    ))

        for path, _, _ in self._ordered_configs(
            "webapi.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            for service in (
                node
                for node in root.iter()
                if tag(node) == "service"
                and node.get("class")
                and node.get("method")
            ):
                for area in ("webapi_rest", "webapi_soap"):
                    state = states.get(area, global_state)
                    entrypoints.add(EventEntrypoint(
                        area,
                        self._resolve_type(service.get("class"), state),
                        service.get("method"),
                        path,
                    ))
        return tuple(sorted(entrypoints))

    def _events(
        self,
        modules: tuple[ModuleRecord, ...],
        states: dict[str, DiState],
    ) -> None:
        areas = {"global", *MAGENTO_AREAS}
        effective_by_area: dict[
            str,
            dict[tuple[str, str], ConfigValue],
        ] = {}
        provenance_by_area: dict[
            str,
            dict[tuple[str, str], tuple[ConfigValue, ...]],
        ] = {}
        global_state = states.get("global", DiState())
        for area in sorted(areas):
            observers: dict[tuple[str, str], ConfigValue] = {}
            provenance: dict[tuple[str, str], list[ConfigValue]] = {}
            for path, module, order in self._ordered_configs("events.xml", modules, area):
                root = self._xml(path)
                if root is None:
                    continue
                content = self.artifacts[path]
                module_name = module.name if module else "application"
                for event in (node for node in root.iter() if tag(node) == "event" and node.get("name")):
                    for observer in (node for node in event if tag(node) == "observer" and node.get("name")):
                        prior = observers.get((event.get("name"), observer.get("name")))
                        instance = observer.get("instance") or (prior.value if prior else "")
                        merged = dict(prior.attributes) if prior else {}
                        merged.update({
                            key: value for key, value in observer.attrib.items()
                            if key not in {"instance", "name"}
                        })
                        key = (event.get("name"), observer.get("name"))
                        declaration = ConfigValue(
                            observer.get("instance", ""),
                            path,
                            line(content, observer.get("name")),
                            module_name,
                            order,
                            tuple(sorted({
                                name: value
                                for name, value in observer.attrib.items()
                                if name not in {"instance", "name"}
                            }.items())),
                        )
                        provenance.setdefault(key, []).append(declaration)
                        observers[key] = ConfigValue(
                            instance, path, line(content, observer.get("name")), module_name, order,
                            tuple(sorted(merged.items())),
                        )
            effective_by_area[area] = observers
            provenance_by_area[area] = {
                key: tuple(values)
                for key, values in provenance.items()
            }
            state = states.get(area, global_state)
            for (event_name, observer_name), observer in sorted(observers.items()):
                observer_attrs = dict(observer.attributes)
                disabled = observer_attrs.get("disabled", "false").casefold() in {"1", "true"}
                resolved_instance = (
                    self._resolve_type(observer.value, state)
                    if observer.value
                    else ""
                )
                declaration_paths = tuple(sorted({
                    declaration.path
                    for declaration in provenance[(event_name, observer_name)]
                }))
                packet = self.graph.packet("magento-event", f"{area}:{event_name}", area=area)
                packet.add(GraphFact(
                    "magento-effective-observer",
                    event_name,
                    "disables-observer" if disabled else "observed-by",
                    observer.value or observer_name,
                    observer.path,
                    observer.line,
                    attrs(**{
                        "area": area,
                        "module": observer.module,
                        **observer_attrs,
                        "name": observer_name,
                        "resolvedInstance": resolved_instance,
                        "declarationCount": len(
                            provenance[(event_name, observer_name)]
                        ),
                    }),
                ),
                    self._symbol_path(observer.value),
                    self._symbol_path(resolved_instance),
                    *self._resolution_paths(observer.value, state),
                    *declaration_paths,
                )
                for declaration in provenance[(event_name, observer_name)][:-1]:
                    packet.add(GraphFact(
                        "magento-observer-override",
                        f"{event_name}:{observer_name}",
                        "overridden-by-observer-config",
                        observer.path,
                        declaration.path,
                        declaration.line,
                        attrs(
                            area=area,
                            disabled=disabled,
                            effectiveInstance=observer.value,
                            effectiveModule=observer.module,
                        ),
                    ), observer.path)

        dispatches = decode_event_dispatches(self.symbols)
        entrypoints = self._event_entrypoints(
            modules,
            states,
            dispatches,
        )
        for dispatch in dispatches:
            area_resolution = resolve_dispatch_area(dispatch, entrypoints)
            packet = self.graph.packet(
                "magento-event-dispatch",
                f"{dispatch.path}:{dispatch.line}:{dispatch.event_name}",
                event=dispatch.event_name,
            )
            packet.add(GraphFact(
                "magento-event-dispatch",
                dispatch.callable,
                "dispatches-event",
                dispatch.event_name,
                dispatch.path,
                dispatch.line,
                attrs(
                    area=area_resolution.area,
                    areaCandidates=",".join(area_resolution.candidate_areas),
                    areaResolution=area_resolution.status,
                    literalResolution=dispatch.literal_resolution,
                    receiverResolution=dispatch.receiver_resolution,
                    receiverType=dispatch.receiver_type,
                    semanticRole="topology",
                ),
            ), *area_resolution.paths)

            observer_area = (
                area_resolution.area
                if area_resolution.status == "proven"
                else "global"
            )
            possible_areas = (
                area_resolution.candidate_areas
                if area_resolution.candidate_areas
                else tuple(sorted(MAGENTO_AREAS))
            )
            for key, observer in sorted(
                effective_by_area.get(observer_area, {}).items()
            ):
                event_name, observer_name = key
                if event_name != dispatch.event_name:
                    continue
                observer_attrs = dict(observer.attributes)
                if observer_attrs.get("disabled", "false").casefold() in {
                    "1",
                    "true",
                }:
                    continue

                candidate_observers = (
                    (observer,)
                    if area_resolution.status == "proven"
                    else tuple(
                        effective_by_area.get(area, {}).get(key)
                        for area in possible_areas
                    )
                )
                if (
                    any(candidate is None for candidate in candidate_observers)
                    or len(set(candidate_observers)) != 1
                ):
                    continue
                resolved_candidates = {
                    self._resolve_type(
                        observer.value,
                        states.get(area, global_state),
                    )
                    for area in (
                        (observer_area,)
                        if area_resolution.status == "proven"
                        else possible_areas
                    )
                    if observer.value
                }
                if len(resolved_candidates) != 1:
                    continue
                resolved_instance = next(iter(resolved_candidates))
                configured_symbol = self._unique_symbol_casefold(
                    observer.value
                )
                resolved_symbol = self._unique_symbol_casefold(
                    resolved_instance
                )
                execution = (
                    self._method_symbol(resolved_symbol, "execute")
                    if resolved_symbol is not None
                    else None
                )
                if execution is not None:
                    declaring_symbol, execute_method = execution
                    target = (
                        f"{declaring_symbol.qualified_name}::{execute_method}"
                    )
                    relation = "dispatches-to-observer-execute"
                    execution_path = declaring_symbol.path
                else:
                    target = resolved_instance
                    relation = "dispatches-to-observer-class"
                    execution_path = (
                        resolved_symbol.path if resolved_symbol else ""
                    )
                declaration_paths = tuple(sorted({
                    declaration.path
                    for declaration in provenance_by_area[
                        observer_area
                    ][key]
                }))
                resolution_areas = (
                    (observer_area,)
                    if area_resolution.status == "proven"
                    else possible_areas
                )
                resolution_paths = tuple(sorted({
                    path
                    for area in resolution_areas
                    for path in self._resolution_paths(
                        observer.value,
                        states.get(area, global_state),
                    )
                }))
                packet.add(GraphFact(
                    "magento-event-dispatch-observer",
                    dispatch.callable,
                    relation,
                    target,
                    dispatch.path,
                    dispatch.line,
                    attrs(
                        area=area_resolution.area,
                        areaResolution=area_resolution.status,
                        configuredObserver=observer.value,
                        event=dispatch.event_name,
                        observerArea=observer_area,
                        observerName=observer_name,
                        resolvedObserver=resolved_instance,
                        semanticRole="topology",
                    ),
                ),
                    observer.path,
                    configured_symbol.path if configured_symbol else "",
                    resolved_symbol.path if resolved_symbol else "",
                    execution_path,
                    *resolution_paths,
                    *declaration_paths,
                    *area_resolution.paths,
                )

    def _routes_and_layouts(
        self,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
        di_states: dict[str, DiState] | None = None,
    ) -> None:
        route_modules: dict[tuple[str, str], dict[str, ConfigValue]] = {}
        route_front_names: dict[tuple[str, str], str] = {}
        route_position = 0
        for area in ("adminhtml", "frontend"):
            for path, module, order in self._ordered_configs("routes.xml", modules, area):
                root = self._xml(path)
                if root is None:
                    continue
                content = self.artifacts[path]
                module_name = module.name if module else "application"
                for router in (node for node in root.iter() if tag(node) == "router" and node.get("id")):
                    for route in (node for node in router if tag(node) == "route" and node.get("id")):
                        key = (area, route.get("id"))
                        if route.get("frontName"):
                            route_front_names[key] = route.get("frontName")
                        for route_module in (node for node in route if tag(node) == "module" and node.get("name")):
                            name = route_module.get("name")
                            prior = route_modules.setdefault(key, {}).get(name)
                            merged_attributes = (
                                dict(prior.attributes) if prior else {}
                            )
                            merged_attributes.update(dict(attrs(
                                router=router.get("id"),
                                before=route_module.get("before", ""),
                                after=route_module.get("after", ""),
                            )))
                            value = ConfigValue(
                                name, path, line(content, name), module_name,
                                order, tuple(sorted(merged_attributes.items())),
                                route_position,
                            )
                            route_modules[key][name] = value
                            route_position += 1

        layout_by_handle: dict[tuple[str, str], list[str]] = {}
        for path in sorted(self.artifacts):
            theme = self._theme_for_path(path, themes)
            if not self._is_deployed_view_source(
                path,
                modules,
                themes,
            ):
                continue
            area = view_area(path, "layout") or (theme.area if theme else None)
            if area is None or not path.endswith(".xml"):
                continue
            handle = PurePosixPath(path).stem
            layout_by_handle.setdefault((area, handle), []).append(path)

        def layout_identity(
            path: str,
            directory: str = "layout",
        ) -> tuple[str, str, str, ThemeRecord | None] | None:
            """Mirror Magento's View File identifier for either layout kind."""
            filename = PurePosixPath(path).name
            theme = self._theme_for_path(path, themes)
            if theme is not None:
                relative = path[len(theme.root):].lstrip("/")
                parts = relative.split("/")
                if len(parts) < 3 or parts[1] != directory:
                    return None
                module_name = parts[0]
                if (
                    len(parts) >= 5
                    and parts[2:4] == ["override", "base"]
                ):
                    return (
                        f"module:{module_name}:{filename}",
                        "override-base",
                        "",
                        theme,
                    )
                if (
                    len(parts) >= 7
                    and parts[2:4] == ["override", "theme"]
                ):
                    ancestor = f"{parts[4]}/{parts[5]}"
                    return (
                        f"theme:{ancestor}:{module_name}:{filename}",
                        "override-theme",
                        ancestor,
                        theme,
                    )
                return (
                    f"theme:{theme.name}:{module_name}:{filename}",
                    "theme",
                    theme.name,
                    theme,
                )

            module = self._module_for_path(path, modules)
            if module is None:
                return None
            relative = (
                path[len(module.root):].lstrip("/")
                if module.root
                else path
            )
            match = re.match(
                rf"view/(?P<area>[^/]+)/{re.escape(directory)}/.+\.xml$",
                relative,
            )
            if match is None:
                return None
            collected_area = match.group("area")
            identity_prefix = (
                "base" if collected_area == "base" else "module"
            )
            return (
                f"{identity_prefix}:{module.name}:{filename}",
                f"module-{collected_area}",
                module.name,
                None,
            )

        layout_identities = {
            path: identity
            for paths in layout_by_handle.values()
            for path in paths
            if (identity := layout_identity(path)) is not None
        }

        def effective_layout_paths(
            area: str,
            handle: str,
            source_theme: ThemeRecord | None = None,
        ) -> tuple[str, ...]:
            # Magento's base collector loads module `view/base/layout` files
            # before the current design area's files. For a theme-owned source,
            # Magento\Framework\View\Design\Fallback\Rule\Theme walks only that
            # theme and its parent chain; sibling themes are not runtime
            # fallback candidates. For module-owned sources, keep every
            # installed theme variant explicit because the active store theme
            # is repository-external database state.
            areas = ("base", area) if area != "base" else ("base",)
            allowed_theme_roots = (
                {
                    candidate.root
                    for candidate in self._theme_chain(source_theme, themes)
                }
                if source_theme is not None
                else None
            )
            candidates = {
                candidate
                for candidate_area in areas
                for candidate in layout_by_handle.get(
                    (candidate_area, handle),
                    (),
                )
                if (
                    (candidate_theme := self._theme_for_path(candidate, themes))
                    is None
                    or allowed_theme_roots is None
                    or candidate_theme.root in allowed_theme_roots
                )
            }
            if source_theme is None:
                return tuple(sorted(candidates))

            # Magento's Aggregated collector starts with module base/area
            # files, then walks inherited themes root-to-child. Regular theme
            # files are added; override/base and override/theme files replace
            # the existing View\File identity. Preserve exactly the resulting
            # paths instead of forwarding both a replacement and suppressed
            # source as executable context.
            effective_by_identity = {
                identity[0]: candidate
                for candidate in sorted(candidates)
                if (
                    (identity := layout_identities.get(candidate))
                    is not None
                    and identity[3] is None
                )
            }
            for current_theme in reversed(
                self._theme_chain(source_theme, themes)
            ):
                theme_candidates = sorted(
                    candidate
                    for candidate in candidates
                    if (
                        (identity := layout_identities.get(candidate))
                        is not None
                        and identity[3] == current_theme
                    )
                )
                for candidate in theme_candidates:
                    identity = layout_identities[candidate]
                    if not identity[1].startswith("override-"):
                        effective_by_identity[identity[0]] = candidate
                for candidate in theme_candidates:
                    identity = layout_identities[candidate]
                    if identity[1].startswith("override-"):
                        if identity[0] in effective_by_identity:
                            effective_by_identity[identity[0]] = candidate
            return tuple(sorted(effective_by_identity.values()))

        for path in sorted(self.artifacts):
            theme = self._theme_for_path(path, themes)
            if not self._is_deployed_view_source(
                path,
                modules,
                themes,
            ):
                continue
            area = view_area(path, "layout") or (theme.area if theme else None)
            if area is None or not path.endswith(".xml"):
                continue
            handle = PurePosixPath(path).stem
            root = self._xml(path)
            if root is None:
                continue
            theme_module = self._theme_module(path, theme)
            identity = layout_identities.get(path)
            override_kind = (
                identity[1]
                if identity is not None
                and identity[1].startswith("override-")
                else ""
            )
            packet = self.graph.packet(
                "magento-layout",
                f"{area}:{handle}",
                area=area,
                handle=handle,
            )
            packet.add(GraphFact(
                "magento-layout-handle",
                handle,
                "declared-in",
                path,
                path,
                1,
                attrs(
                    area=area,
                    theme=theme.name if theme else "",
                    themeModule=theme_module,
                    themeSelection=(
                        theme.name if theme else "runtime-selected"
                    ),
                    overrideKind=override_kind,
                ),
            ), theme.theme_xml if theme else "", *(
                candidate
                for candidate in effective_layout_paths(area, handle, theme)
                if candidate != path
            ))
            if identity is not None and override_kind and theme is not None:
                allowed_roots = {
                    candidate.root
                    for candidate in self._theme_chain(theme, themes)
                }
                replacement_candidates = tuple(sorted(
                    candidate
                    for candidate, candidate_identity in layout_identities.items()
                    if candidate != path
                    and candidate_identity[0] == identity[0]
                    and (
                        candidate_identity[3] is None
                        or candidate_identity[3].root in allowed_roots
                    )
                ))
                packet.add(GraphFact(
                    (
                        "magento-layout-override"
                        if replacement_candidates
                        else "magento-layout-override-unresolved"
                    ),
                    path,
                    (
                        "replaces-layout-identity"
                        if replacement_candidates
                        else "has-no-replaceable-layout-identity"
                    ),
                    identity[0],
                    path,
                    1,
                    attrs(
                        area=area,
                        theme=theme.name,
                        overrideKind=override_kind,
                        ancestorTheme=identity[2],
                    ),
                ), *replacement_candidates)
            content = self.artifacts[path]
            parents = {
                id(child): parent
                for parent in root.iter()
                for child in parent
            }
            for element in root.iter():
                element_tag = tag(element)
                config_condition = element.get("ifconfig", "").strip()
                if config_condition:
                    element_identity = (
                        element.get("name")
                        or element.get("class")
                        or element_tag
                    )
                    packet.add(GraphFact(
                        "magento-layout-config-condition",
                        element_identity,
                        "visible-when-config-enabled",
                        config_condition,
                        path,
                        line(content, config_condition),
                        attrs(
                            area=area,
                            element=element_tag,
                            handle=handle,
                        ),
                    ), *sorted(
                        self._system_config_sources.get(
                            config_condition,
                            (),
                        )
                    ))
                acl_condition = element.get("aclResource", "").strip()
                if acl_condition:
                    element_identity = (
                        element.get("name")
                        or element.get("class")
                        or element_tag
                    )
                    packet.add(GraphFact(
                        "magento-layout-acl-condition",
                        element_identity,
                        "visible-to-resource",
                        acl_condition,
                        path,
                        line(content, acl_condition),
                        attrs(
                            area=area,
                            element=element_tag,
                            handle=handle,
                        ),
                    ), *sorted(
                        self._acl_sources.get(acl_condition, ())
                    ))
                if element_tag == "update" and element.get("handle"):
                    related = effective_layout_paths(
                        area,
                        element.get("handle"),
                        theme,
                    )
                    packet.add(GraphFact(
                        "magento-layout-update",
                        handle,
                        "includes-handle",
                        element.get("handle"),
                        path,
                        line(content, element.get("handle")),
                        attrs(area=area),
                    ), *related)
                elif element_tag in {"block", "referenceBlock"}:
                    target = element.get("class") or element.get("name")
                    if target:
                        parent = parents.get(id(element))
                        while (
                            parent is not None
                            and tag(parent) not in {"block", "referenceBlock"}
                        ):
                            parent = parents.get(id(parent))
                        template_paths = self._template_paths(
                            element.get("template", ""),
                            area,
                            modules,
                            themes,
                            theme,
                        )
                        selected_template_path = (
                            self._selected_template_path(
                                element.get("template", ""),
                                area,
                                modules,
                                themes,
                                theme,
                            )
                        )
                        packet.add(GraphFact(
                            "magento-layout-block",
                            handle,
                            "declares-block",
                            target,
                            path,
                            line(content, target),
                            attrs(
                                area=area,
                                alias=element.get("as", ""),
                                name=element.get("name", ""),
                                parentName=(
                                    parent.get("name", "")
                                    if parent is not None
                                    else ""
                                ),
                                template=element.get("template", ""),
                                selectedTemplatePath=(
                                    selected_template_path or ""
                                ),
                                theme=theme.name if theme else "",
                            ),
                        ),
                            self._symbol_path(element.get("class", "")),
                            *template_paths,
                        )
                        block_class = element.get("class", "").strip()
                        block_symbol = self._unique_symbol_casefold(block_class)
                        if selected_template_path and block_class:
                            related = tuple(sorted(filter(None, (
                                path,
                                block_symbol.path if block_symbol else "",
                            ))))
                            packet.add(GraphFact(
                                "magento-template-block-binding",
                                selected_template_path,
                                "declared-with-block-class",
                                block_class,
                                selected_template_path,
                                1,
                                attrs(
                                    area=area,
                                    handle=handle,
                                    layoutPath=path,
                                ),
                                related_paths=related,
                            ))
                            for call in self._template_php_calls(
                                selected_template_path,
                                "block",
                            ):
                                declaration = (
                                    self._method_symbol(
                                        block_symbol,
                                        call.method,
                                    )
                                    if block_symbol is not None
                                    else None
                                )
                                if declaration is not None:
                                    declaring_symbol, declared_method = declaration
                                    packet.add(GraphFact(
                                        "magento-template-block-method-call",
                                        selected_template_path,
                                        "declares-block-method-call",
                                        f"{declaring_symbol.qualified_name}::{declared_method}",
                                        selected_template_path,
                                        call.line,
                                        attrs(
                                            area=area,
                                            blockClass=block_class,
                                            handle=handle,
                                            layoutPath=path,
                                        ),
                                        related_paths=tuple(sorted({
                                            path,
                                            declaring_symbol.path,
                                        })),
                                    ))
                            for arguments_node in (
                                child for child in element
                                if tag(child) == "arguments"
                            ):
                                for argument in (
                                    child for child in arguments_node
                                    if tag(child) == "argument"
                                ):
                                    argument_type = next((
                                        value
                                        for key, value in argument.attrib.items()
                                        if key == "type" or key.endswith("}type")
                                    ), "")
                                    object_class = (argument.text or "").strip()
                                    if argument_type != "object" or not object_class:
                                        continue
                                    object_symbol = self._unique_symbol_casefold(
                                        object_class
                                    )
                                    packet.add(GraphFact(
                                        "magento-template-view-model-binding",
                                        selected_template_path,
                                        "declares-layout-object",
                                        object_class,
                                        selected_template_path,
                                        1,
                                        attrs(
                                            argument=argument.get("name", ""),
                                            area=area,
                                            handle=handle,
                                            layoutPath=path,
                                        ),
                                        related_paths=tuple(sorted(filter(None, (
                                            path,
                                            object_symbol.path if object_symbol else "",
                                        )))),
                                    ))

        self._effective_layouts(
            modules,
            themes,
            layout_by_handle,
            effective_layout_paths,
            layout_identity,
            di_states,
        )

        for (area, route_id), entries_by_module in sorted(route_modules.items()):
            entries, route_order_complete = self._ordered_route_modules(
                area,
                route_id,
                tuple(entries_by_module.values()),
            )
            front_name = route_front_names.get((area, route_id), route_id)
            packet = self.graph.packet(
                "magento-route",
                f"{area}:{route_id}",
                area=area,
                route=route_id,
            )
            for position, entry in enumerate(entries):
                route_module = next(
                    (module for module in modules if module.name == entry.value),
                    None,
                )
                packet.add(GraphFact(
                    "magento-effective-route",
                    route_id,
                    "handled-by-module",
                    entry.value,
                    entry.path,
                    entry.line,
                    attrs(**{
                        "area": area,
                        "module": entry.module,
                        **dict(entry.attributes),
                        "frontName": front_name,
                        "priorityPosition": (
                            position if route_order_complete else ""
                        ),
                    }),
                ), route_module.module_xml if route_module else "")
                if route_order_complete and position:
                    previous = entries[position - 1]
                    packet.add(GraphFact(
                        "magento-route-priority",
                        previous.value,
                        "searched-before",
                        entry.value,
                        previous.path,
                        previous.line,
                        attrs(
                            area=area,
                            routeId=route_id,
                            frontName=front_name,
                            position=position - 1,
                            nextPosition=position,
                        ),
                    ), entry.path)

            if not route_order_complete:
                continue

            controllers: dict[
                str,
                list[tuple[int, ConfigValue, SymbolDefinition, str]],
            ] = {}
            for position, entry in enumerate(entries):
                route_module = next(
                    (module for module in modules if module.name == entry.value),
                    None,
                )
                if route_module is None:
                    continue
                controller_prefix = _path_under(
                    route_module.root,
                    (
                        "Controller/Adminhtml/"
                        if area == "adminhtml"
                        else "Controller/"
                    ),
                )
                for symbol in self.symbols:
                    if not symbol.path.startswith(controller_prefix):
                        continue
                    relative = symbol.path[
                        len(controller_prefix):
                    ].removesuffix(".php")
                    controllers.setdefault(
                        relative.casefold(),
                        [],
                    ).append((position, entry, symbol, relative))

            for _, candidates in sorted(controllers.items()):
                candidates.sort(key=lambda item: item[0])
                _, winner_entry, winner, relative = candidates[0]
                controller_action = relative.replace("/", "_").casefold()
                handle = f"{route_id}_{controller_action}"
                layout_paths = effective_layout_paths(area, handle)
                request_path = (
                    f"{front_name}/"
                    f"{relative.removesuffix('/Index').casefold()}"
                )
                normalized_request_path = request_path.strip("/").casefold()
                if area == "adminhtml":
                    self._admin_controller_sources.setdefault(
                        normalized_request_path,
                        set(),
                    ).update((winner.path, winner_entry.path))
                    if relative.casefold().endswith("/index"):
                        self._admin_controller_sources.setdefault(
                            (
                                f"{front_name}/{relative}"
                                .strip("/")
                                .casefold()
                            ),
                            set(),
                        ).update((winner.path, winner_entry.path))
                packet.add(GraphFact(
                    "magento-route-controller",
                    request_path,
                    "dispatches-to",
                    winner.qualified_name,
                    winner.path,
                    winner.line,
                    attrs(
                        area=area,
                        layoutHandle=handle,
                        routeId=route_id,
                        routeModule=winner_entry.value,
                    ),
                ), winner.path, *layout_paths)
                for _, shadowed_entry, shadowed, _ in candidates[1:]:
                    packet.add(GraphFact(
                        "magento-route-controller-shadowed",
                        shadowed.qualified_name,
                        "shadowed-by",
                        winner.qualified_name,
                        shadowed.path,
                        shadowed.line,
                        attrs(
                            area=area,
                            requestPath=request_path,
                            routeId=route_id,
                            routeModule=shadowed_entry.value,
                        ),
                    ), winner.path)

    def _effective_layouts(
        self,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
        layout_by_handle: dict[tuple[str, str], list[str]],
        effective_layout_paths: Callable[
            [str, str, ThemeRecord | None], tuple[str, ...]
        ],
        layout_identity: Callable[
            [str, str],
            tuple[str, str, str, ThemeRecord | None] | None,
        ],
        di_states: dict[str, DiState] | None,
    ) -> None:
        """Emit the effective Magento layout for every provable theme selection.

        Module/theme file collection remains repository-aware here. Instruction
        parsing and mutation semantics live in ``layout.py`` so the large
        repository resolver only supplies ordered, runtime-compatible inputs.
        """

        module_order = {
            module.name: module.order
            for module in modules
            if module.enabled
        }
        parsed_documents = {}

        def parse_document(path: str, area: str, handle: str):
            identity = (path, area, handle)
            if identity in parsed_documents:
                return parsed_documents[identity]
            root = self._xml(path)
            if root is None:
                parsed_documents[identity] = None
                return None
            document = parse_layout_document(
                path=path,
                area=area,
                handle=handle,
                content=self.artifacts[path],
                root=root,
            )
            parsed_documents[identity] = document
            return document

        def selection_load_key(
            path: str,
            area: str,
            selected_theme: ThemeRecord | None,
        ) -> tuple[int, int, int, str]:
            theme = self._theme_for_path(path, themes)
            if theme is None:
                module = self._module_for_path(path, modules)
                source_area = view_area(path, "layout") or view_area(
                    path,
                    "page_layout",
                )
                return (
                    0 if source_area == "base" else 1,
                    module.order if module is not None else -1,
                    0,
                    path,
                )
            chain = (
                tuple(reversed(self._theme_chain(selected_theme, themes)))
                if selected_theme is not None
                else ()
            )
            theme_position = next(
                (
                    index
                    for index, candidate in enumerate(chain)
                    if candidate == theme
                ),
                len(chain),
            )
            theme_module = self._theme_module(path, theme)
            return (
                2,
                theme_position,
                module_order.get(theme_module, -1),
                path,
            )

        def selection_layout_paths(
            area: str,
            handle: str,
            selected_theme: ThemeRecord | None,
        ) -> tuple[str, ...]:
            candidates = effective_layout_paths(
                area,
                handle,
                selected_theme,
            )
            if selected_theme is None:
                candidates = tuple(
                    path
                    for path in candidates
                    if self._theme_for_path(path, themes) is None
                )
            return tuple(sorted(
                candidates,
                key=lambda path: selection_load_key(
                    path,
                    area,
                    selected_theme,
                ),
            ))

        page_layout_by_id: dict[tuple[str, str], list[str]] = {}
        page_layout_identities: dict[
            str,
            tuple[str, str, str, ThemeRecord | None],
        ] = {}
        layout_declarations: dict[tuple[str, str], set[str]] = {}
        for path in sorted(self.artifacts):
            theme = self._theme_for_path(path, themes)
            if not self._is_deployed_view_source(path, modules, themes):
                continue
            normalized = f"/{path}"
            if "/page_layout/" in normalized and path.endswith(".xml"):
                area = view_area(path, "page_layout") or (
                    theme.area if theme else None
                )
                if area is not None:
                    page_layout_by_id.setdefault(
                        (area, PurePosixPath(path).stem),
                        [],
                    ).append(path)
                    identity = layout_identity(path, "page_layout")
                    if identity is not None:
                        page_layout_identities[path] = identity
                continue
            if (
                PurePosixPath(path).name != "layouts.xml"
                or "/page_layout/" in normalized
            ):
                continue
            area_match = re.search(
                r"/view/(base|frontend|adminhtml)/layouts\.xml$",
                normalized,
            )
            area = (
                area_match.group(1)
                if area_match is not None
                else (theme.area if theme else None)
            )
            if area is None:
                continue
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for element in root.iter():
                if tag(element) != "layout" or not element.get("id"):
                    continue
                layout_id = element.get("id", "").strip()
                if not layout_id:
                    continue
                layout_declarations.setdefault((area, layout_id), set()).add(
                    path
                )
                packet = self.graph.packet(
                    "magento-page-layout",
                    f"{area}:{layout_id}",
                    area=area,
                    pageLayout=layout_id,
                )
                packet.add(GraphFact(
                    "magento-page-layout-declaration",
                    layout_id,
                    "declared-in",
                    path,
                    path,
                    line(content, layout_id),
                    attrs(
                        area=area,
                        label=next(
                            (
                                (child.text or "").strip()
                                for child in element
                                if tag(child) == "label"
                            ),
                            element.get("label", ""),
                        ),
                        theme=theme.name if theme else "",
                    ),
                ), theme.theme_xml if theme else "")

        def selection_page_layout_paths(
            area: str,
            layout_id: str,
            selected_theme: ThemeRecord | None,
        ) -> tuple[str, ...]:
            allowed_theme_roots = (
                {
                    candidate.root
                    for candidate in self._theme_chain(
                        selected_theme,
                        themes,
                    )
                }
                if selected_theme is not None
                else set()
            )
            paths = {
                path
                for candidate_area in (
                    ("base", area) if area != "base" else ("base",)
                )
                for path in page_layout_by_id.get(
                    (candidate_area, layout_id),
                    (),
                )
                if (
                    (candidate_theme := self._theme_for_path(path, themes))
                    is None
                    or (
                        selected_theme is not None
                        and candidate_theme.root in allowed_theme_roots
                    )
                )
            }
            if selected_theme is not None:
                effective_by_identity = {
                    identity[0]: path
                    for path in sorted(paths)
                    if (
                        (identity := page_layout_identities.get(path))
                        is not None
                        and identity[3] is None
                    )
                }
                for current_theme in reversed(
                    self._theme_chain(selected_theme, themes)
                ):
                    theme_paths = tuple(sorted(
                        path
                        for path in paths
                        if (
                            (identity := page_layout_identities.get(path))
                            is not None
                            and identity[3] == current_theme
                        )
                    ))
                    for path in theme_paths:
                        identity = page_layout_identities[path]
                        if not identity[1].startswith("override-"):
                            effective_by_identity[identity[0]] = path
                    for path in theme_paths:
                        identity = page_layout_identities[path]
                        if (
                            identity[1].startswith("override-")
                            and identity[0] in effective_by_identity
                        ):
                            effective_by_identity[identity[0]] = path
                paths = set(effective_by_identity.values())
            return tuple(sorted(
                paths,
                key=lambda path: selection_load_key(
                    path,
                    area,
                    selected_theme,
                ),
            ))

        runtime_areas = {
            area
            for area, _ in layout_by_handle
            if area != "base"
        }
        runtime_areas.update(theme.area for theme in themes)
        if not runtime_areas and any(
            area == "base" for area, _ in layout_by_handle
        ):
            runtime_areas.add("base")

        emitted_diagnostics: set[tuple[str, str, int]] = set()
        for area in sorted(runtime_areas):
            global_di_state = (di_states or {}).get("global", DiState())
            area_di_state = (di_states or {}).get(area, global_di_state)
            area_themes = tuple(
                theme for theme in themes if theme.area == area
            )
            selections: tuple[ThemeRecord | None, ...] = (
                (None, *area_themes)
                if modules
                else area_themes
            )
            for selected_theme in selections:
                selection_name = (
                    selected_theme.name
                    if selected_theme is not None
                    else "module-only"
                )
                selection_theme_roots = (
                    {
                        candidate.root
                        for candidate in self._theme_chain(
                            selected_theme,
                            themes,
                        )
                    }
                    if selected_theme is not None
                    else set()
                )

                def selection_compatible_path(path: str) -> bool:
                    candidate_theme = self._theme_for_path(path, themes)
                    return (
                        candidate_theme is None
                        or (
                            selected_theme is not None
                            and candidate_theme.root in selection_theme_roots
                        )
                    )

                handles = sorted({
                    handle
                    for candidate_area in (
                        ("base", area) if area != "base" else ("base",)
                    )
                    for handle in (
                        candidate_handle
                        for source_area, candidate_handle in layout_by_handle
                        if source_area == candidate_area
                    )
                }, key=lambda candidate: (
                    candidate != "default",
                    candidate,
                ))
                documents_by_handle = {}
                default_nodes_by_name = {}
                default_missing_parents: set[str] = set()
                default_assets_by_src = {}
                default_unresolved_operations = set()
                for handle in handles:
                    documents = tuple(filter(None, (
                        parse_document(path, area, handle)
                        for path in selection_layout_paths(
                            area,
                            handle,
                            selected_theme,
                        )
                    )))
                    if documents:
                        documents_by_handle[handle] = documents

                page_documents_by_handle = {}
                page_layout_ids = sorted({
                    layout_id
                    for candidate_area in (
                        ("base", area) if area != "base" else ("base",)
                    )
                    for source_area, layout_id in page_layout_by_id
                    if source_area == candidate_area
                })
                for layout_id in page_layout_ids:
                    page_handle = f"page_layout:{layout_id}"
                    page_documents = []
                    for path in selection_page_layout_paths(
                        area,
                        layout_id,
                        selected_theme,
                    ):
                        document = parse_document(
                            path,
                            area,
                            page_handle,
                        )
                        if document is None:
                            continue
                        # Page-layout <update> instructions inherit another
                        # page-layout wireframe, not an ordinary page
                        # configuration handle with the same basename.
                        page_documents.append(replace(
                            document,
                            operations=tuple(
                                replace(
                                    operation,
                                    name=f"page_layout:{operation.name}",
                                )
                                if operation.kind == "update"
                                else operation
                                for operation in document.operations
                            ),
                        ))
                    if page_documents:
                        page_documents_by_handle[page_handle] = tuple(
                            page_documents
                        )

                for handle in handles:
                    if handle not in documents_by_handle:
                        continue
                    generic_layout = all(
                        document.root_kind == "layout"
                        for document in documents_by_handle[handle]
                    )
                    requested_handles = (
                        (handle,)
                        if handle == "default" or generic_layout
                        else ("default", handle)
                    )
                    effective = merge_layout(
                        documents_by_handle,
                        requested_handles,
                    )
                    page_layout_id = effective.root_layout
                    page_layout_paths: tuple[str, ...] = ()
                    if page_layout_id:
                        page_handle = f"page_layout:{page_layout_id}"
                        page_documents = page_documents_by_handle.get(
                            page_handle,
                            (),
                        )
                        if page_documents:
                            effective = merge_layout(
                                {
                                    **documents_by_handle,
                                    **page_documents_by_handle,
                                },
                                (page_handle, *requested_handles),
                            )
                            page_layout_paths = tuple(sorted({
                                path
                                for path in effective.document_paths
                                if path in page_layout_identities
                            }))

                    if not effective.document_paths:
                        continue
                    current_nodes_by_name = {
                        node.name: node for node in effective.nodes
                    }
                    suppressed_inherited_names = (
                        set()
                        if generic_layout
                        else (
                            set(default_nodes_by_name)
                            - set(current_nodes_by_name)
                        )
                    )
                    inherited_node_changed = any(
                        current_nodes_by_name.get(name) != baseline
                        for name, baseline in default_nodes_by_name.items()
                    )
                    newly_resolved_parent = bool(
                        (
                            set(current_nodes_by_name)
                            - set(default_nodes_by_name)
                        ).intersection(default_missing_parents)
                    )
                    emit_full_projection = (
                        handle == "default"
                        or generic_layout
                        or not default_nodes_by_name
                        or inherited_node_changed
                        or newly_resolved_parent
                    )
                    packet = self.graph.packet(
                        "magento-effective-layout",
                        f"{area}:{selection_name}:{handle}",
                        area=area,
                        handle=handle,
                        themeSelection=selection_name,
                    )
                    primary_path = effective.document_paths[-1]
                    related_documents = tuple(sorted(
                        set(effective.document_paths) - {primary_path}
                    ))
                    packet.add(GraphFact(
                        "magento-effective-layout",
                        handle,
                        "composes-handles",
                        ",".join(effective.expanded_handles),
                        primary_path,
                        1,
                        attrs(
                            area=area,
                            documentKind=(
                                "generic-layout"
                                if generic_layout
                                else "page-configuration"
                            ),
                            rootLayout=effective.root_layout,
                            emissionMode=(
                                "full"
                                if emit_full_projection
                                else "delta-from-default"
                            ),
                            semanticRole="topology",
                            themeSelection=selection_name,
                        ),
                    ), *related_documents)

                    if page_layout_id:
                        page_source = next(
                            (
                                document
                                for source_handle in reversed(
                                    requested_handles
                                )
                                for document in reversed(
                                    documents_by_handle.get(
                                        source_handle,
                                        (),
                                    )
                                )
                                if document.root_layout == page_layout_id
                            ),
                            None,
                        )
                        page_source_path = (
                            page_source.path if page_source else primary_path
                        )
                        declaration_paths = tuple(sorted({
                            path
                            for candidate_area in {"base", area}
                            for path in layout_declarations.get(
                                (candidate_area, page_layout_id),
                                (),
                            )
                            if selection_compatible_path(path)
                        }))
                        packet.add(GraphFact(
                            "magento-page-layout-selection",
                            handle,
                            "selects-page-layout",
                            page_layout_id,
                            page_source_path,
                            line(
                                self.artifacts[page_source_path],
                                page_layout_id,
                            ),
                            attrs(
                                area=area,
                                resolved=str(bool(page_layout_paths)).lower(),
                                themeSelection=selection_name,
                            ),
                        ), *page_layout_paths, *declaration_paths)

                    nodes_by_name = current_nodes_by_name
                    rendering_state_by_name = {}
                    for candidate_name in nodes_by_name:
                        current_name = candidate_name
                        visited: set[str] = set()
                        rendering_state = "rendered"
                        has_ifconfig = False
                        has_acl = False
                        while current_name:
                            if current_name in visited:
                                rendering_state = "unresolved-parent-cycle"
                                break
                            visited.add(current_name)
                            candidate = nodes_by_name.get(current_name)
                            if candidate is None:
                                rendering_state = "unresolved-parent"
                                break
                            if candidate.removed:
                                rendering_state = (
                                    "removed"
                                    if current_name == candidate_name
                                    else "ancestor-removed"
                                )
                                break
                            if candidate.display is False:
                                rendering_state = (
                                    "display-disabled"
                                    if current_name == candidate_name
                                    else "ancestor-display-disabled"
                                )
                                break
                            candidate_attributes = dict(
                                candidate.attributes
                            )
                            has_ifconfig = has_ifconfig or bool(
                                candidate_attributes.get("ifconfig", "")
                            )
                            has_acl = has_acl or bool(
                                candidate_attributes.get("aclResource", "")
                            )
                            current_name = candidate.parent
                        if rendering_state == "rendered":
                            if has_ifconfig and has_acl:
                                rendering_state = "conditional-ifconfig-acl"
                            elif has_ifconfig:
                                rendering_state = "conditional-ifconfig"
                            elif has_acl:
                                rendering_state = "conditional-acl"
                        rendering_state_by_name[candidate_name] = (
                            rendering_state
                        )

                    for suppressed_name in sorted(
                        suppressed_inherited_names
                    ):
                        baseline = default_nodes_by_name[suppressed_name]
                        suppression_path = primary_path
                        suppression_line = 1
                        ancestor_name = baseline.parent
                        visited_ancestors: set[str] = set()
                        while (
                            ancestor_name
                            and ancestor_name not in visited_ancestors
                        ):
                            visited_ancestors.add(ancestor_name)
                            current_ancestor = current_nodes_by_name.get(
                                ancestor_name
                            )
                            baseline_ancestor = default_nodes_by_name.get(
                                ancestor_name
                            )
                            if current_ancestor != baseline_ancestor:
                                if current_ancestor is not None:
                                    suppression_path = (
                                        current_ancestor.source.path
                                    )
                                    suppression_line = (
                                        current_ancestor.source.line
                                    )
                                break
                            ancestor_name = (
                                baseline_ancestor.parent
                                if baseline_ancestor is not None
                                else ""
                            )
                        baseline_paths = tuple(sorted({
                            source.path for source in baseline.provenance
                        }))
                        packet.add(GraphFact(
                            "magento-layout-effective-node",
                            handle,
                            "suppresses-inherited-node",
                            suppressed_name,
                            suppression_path,
                            suppression_line,
                            attrs(
                                area=area,
                                baselineHandle="default",
                                blockClass=baseline.block_class,
                                handle=handle,
                                nodeKind=baseline.node_kind,
                                parent=baseline.parent,
                                renderingState=(
                                    "suppressed-by-redeclaration"
                                ),
                                semanticRole="topology",
                                template=baseline.template,
                                themeSelection=selection_name,
                            ),
                        ), *baseline_paths)

                        configured_class = baseline.block_class
                        suppressed_class = (
                            self._resolve_type(
                                configured_class,
                                area_di_state,
                            )
                            if configured_class
                            else ""
                        )
                        if suppressed_class:
                            packet.add(GraphFact(
                                "magento-layout-effective-block-class",
                                suppressed_name,
                                "suppresses-inherited-block-class",
                                suppressed_class,
                                suppression_path,
                                suppression_line,
                                attrs(
                                    area=area,
                                    baselineHandle="default",
                                    configuredBlockClass=configured_class,
                                    handle=handle,
                                    renderingState=(
                                        "suppressed-by-redeclaration"
                                    ),
                                    themeSelection=selection_name,
                                ),
                            ), self._symbol_path(suppressed_class),
                                *baseline_paths)

                        baseline_property_sources = {
                            item.property: item.source
                            for item in baseline.property_sources
                        }
                        conditional_templates = []
                        for baseline_action in baseline.actions:
                            if (
                                baseline_action.method.casefold()
                                != "settemplate"
                                or not baseline_action.arguments
                            ):
                                continue
                            template_argument = (
                                baseline_action.arguments[0]
                            )
                            if (
                                not template_argument.value
                                or template_argument.value_type
                                not in {"", "string"}
                            ):
                                continue
                            if baseline_action.ifconfig:
                                conditional_templates.append((
                                    template_argument.value,
                                    baseline_action.ifconfig,
                                    template_argument.source.path,
                                ))
                            else:
                                conditional_templates.clear()
                        template_variants = []
                        if baseline.template:
                            template_source = baseline_property_sources.get(
                                "template",
                                baseline.source,
                            )
                            template_variants.append((
                                baseline.template,
                                "",
                                template_source.path,
                            ))
                        template_variants.extend(conditional_templates)
                        selection_themes = (
                            themes if selected_theme is not None else ()
                        )
                        for (
                            suppressed_template,
                            template_ifconfig,
                            template_declaration_path,
                        ) in dict.fromkeys(template_variants):
                            template_paths = self._template_paths(
                                suppressed_template,
                                area,
                                modules,
                                selection_themes,
                                selected_theme,
                            )
                            suppressed_template_path = (
                                self._selected_template_path(
                                    suppressed_template,
                                    area,
                                    modules,
                                    selection_themes,
                                    selected_theme,
                                )
                            )
                            packet.add(GraphFact(
                                "magento-layout-effective-template",
                                suppressed_name,
                                "suppresses-inherited-template",
                                (
                                    suppressed_template_path
                                    or suppressed_template
                                ),
                                suppression_path,
                                suppression_line,
                                attrs(
                                    area=area,
                                    baselineHandle="default",
                                    handle=handle,
                                    ifconfig=template_ifconfig,
                                    renderingState=(
                                        "suppressed-by-redeclaration"
                                    ),
                                    template=suppressed_template,
                                    themeSelection=selection_name,
                                ),
                            ), *template_paths, *baseline_paths,
                                template_declaration_path)
                            if suppressed_template_path and suppressed_class:
                                packet.add(GraphFact(
                                    "magento-template-effective-block-binding",
                                    suppressed_template_path,
                                    "suppressed-in-effective-layout",
                                    suppressed_class,
                                    suppressed_template_path,
                                    1,
                                    attrs(
                                        activationCertainty="inactive",
                                        area=area,
                                        baselineHandle="default",
                                        blockName=suppressed_name,
                                        handle=handle,
                                        ifconfig=template_ifconfig,
                                        renderingState=(
                                            "suppressed-by-redeclaration"
                                        ),
                                        themeSelection=selection_name,
                                    ),
                                    related_paths=tuple(sorted(set(filter(None, (
                                        suppression_path,
                                        template_declaration_path,
                                        *baseline_paths,
                                        self._symbol_path(suppressed_class),
                                    ))))),
                                ))

                        for argument in baseline.arguments:
                            if (
                                argument.value_type != "object"
                                or not argument.value
                            ):
                                continue
                            object_class = argument.value.lstrip("\\")
                            packet.add(GraphFact(
                                "magento-layout-effective-object-argument",
                                suppressed_name,
                                "suppresses-inherited-layout-object",
                                object_class,
                                suppression_path,
                                suppression_line,
                                attrs(
                                    area=area,
                                    argument=argument.name,
                                    baselineHandle="default",
                                    handle=handle,
                                    renderingState=(
                                        "suppressed-by-redeclaration"
                                    ),
                                    themeSelection=selection_name,
                                ),
                            ), self._symbol_path(object_class),
                                argument.source.path,
                                *baseline_paths)

                        for action in baseline.actions:
                            action_target = (
                                f"{suppressed_class}::{action.method}"
                                if suppressed_class
                                else action.method
                            )
                            packet.add(GraphFact(
                                "magento-layout-effective-action",
                                suppressed_name,
                                "suppresses-inherited-layout-action",
                                action_target,
                                suppression_path,
                                suppression_line,
                                attrs(
                                    area=area,
                                    baselineHandle="default",
                                    handle=handle,
                                    method=action.method,
                                    renderingState=(
                                        "suppressed-by-redeclaration"
                                    ),
                                    themeSelection=selection_name,
                                ),
                            ), action.source.path, *baseline_paths)

                        if baseline.node_kind == "uiComponent":
                            ui_paths = tuple(sorted(
                                path
                                for path in self._ui_component_paths_by_name.get(
                                    suppressed_name,
                                    (),
                                )
                                if self._is_deployed_view_source(
                                    path,
                                    modules,
                                    themes,
                                )
                                and (
                                    (
                                        self._theme_for_path(path, themes)
                                        is not None
                                        and selection_compatible_path(path)
                                    )
                                    or (
                                        self._theme_for_path(path, themes)
                                        is None
                                        and view_area(
                                            path,
                                            "ui_component",
                                        ) in {"base", area}
                                    )
                                )
                            ))
                            packet.add(GraphFact(
                                "magento-layout-ui-component-activation",
                                handle,
                                "suppresses-inherited-ui-component",
                                suppressed_name,
                                suppression_path,
                                suppression_line,
                                attrs(
                                    area=area,
                                    baselineHandle="default",
                                    renderingState=(
                                        "suppressed-by-redeclaration"
                                    ),
                                    resolved=str(bool(ui_paths)).lower(),
                                    themeSelection=selection_name,
                                ),
                            ), *ui_paths, *baseline_paths)

                    for node in effective.nodes:
                        property_sources = {
                            item.property: item.source
                            for item in node.property_sources
                        }
                        provenance_paths = tuple(sorted({
                            source.path for source in node.provenance
                        }))
                        if (
                            not emit_full_projection
                            and node.name in default_nodes_by_name
                        ):
                            continue
                        rendering_state = rendering_state_by_name[node.name]
                        renders = rendering_state == "rendered"
                        potentially_renders = renders or (
                            rendering_state.startswith("conditional-")
                        ) or (
                            rendering_state == "unresolved-parent"
                        )
                        node_configuration = dict(node.attributes)
                        configured_block_class = node.block_class
                        effective_block_class = (
                            self._resolve_type(
                                configured_block_class,
                                area_di_state,
                            )
                            if configured_block_class
                            else ""
                        )
                        block_resolution_paths = (
                            self._resolution_paths(
                                configured_block_class,
                                area_di_state,
                            )
                            if configured_block_class
                            else ()
                        )
                        block_class_defaulted = (
                            node_configuration.get(
                                "blockClassDefaulted",
                                "",
                            ).casefold()
                            == "true"
                        )
                        node_attributes = attrs(
                            after=node.after,
                            alias=node.alias,
                            area=area,
                            before=node.before,
                            blockClass=effective_block_class,
                            blockClassDefaulted=str(
                                block_class_defaulted
                            ).lower(),
                            configuredBlockClass=configured_block_class,
                            display=(
                                str(node.display).lower()
                                if node.display is not None
                                else ""
                            ),
                            handle=handle,
                            ifconfig=node_configuration.get("ifconfig", ""),
                            nodeKind=node.node_kind,
                            order=node.order,
                            parent=node.parent,
                            removed=str(node.removed).lower(),
                            renderingState=rendering_state,
                            aclResource=node_configuration.get(
                                "aclResource",
                                "",
                            ),
                            semanticRole=(
                                "uncertainty"
                                if rendering_state == "unresolved-parent"
                                else "topology"
                            ),
                            template=node.template,
                            themeSelection=selection_name,
                        )
                        packet.add(GraphFact(
                            "magento-layout-effective-node",
                            handle,
                            (
                                "removes-node"
                                if node.removed
                                else "contains-node"
                            ),
                            node.name,
                            node.source.path,
                            node.source.line,
                            node_attributes,
                        ), *(
                            path for path in provenance_paths
                            if path != node.source.path
                        ))

                        if node.parent:
                            parent = nodes_by_name.get(node.parent)
                            parent_paths = (
                                tuple(
                                    source.path
                                    for source in parent.provenance
                                )
                                if parent is not None
                                else ()
                            )
                            parent_source = property_sources.get(
                                "parent",
                                node.source,
                            )
                            packet.add(GraphFact(
                                "magento-layout-effective-parent",
                                node.name,
                                "placed-in",
                                node.parent,
                                parent_source.path,
                                parent_source.line,
                                attrs(
                                    alias=node.alias,
                                    area=area,
                                    handle=handle,
                                    order=node.order,
                                    resolved=str(parent is not None).lower(),
                                    semanticRole=(
                                        "topology"
                                        if parent is not None
                                        else "uncertainty"
                                    ),
                                    themeSelection=selection_name,
                                ),
                            ), *provenance_paths, *parent_paths)

                        if effective_block_class:
                            class_source = property_sources.get(
                                "blockClass",
                                node.source,
                            )
                            packet.add(GraphFact(
                                "magento-layout-effective-block-class",
                                node.name,
                                (
                                    "uses-block-class"
                                    if rendering_state in {
                                        "rendered",
                                        "display-disabled",
                                        "ancestor-display-disabled",
                                    }
                                    else (
                                        "conditionally-uses-block-class"
                                        if potentially_renders
                                        else "declares-inactive-block-class"
                                    )
                                ),
                                effective_block_class,
                                class_source.path,
                                class_source.line,
                                attrs(
                                    area=area,
                                    blockClassDefaulted=str(
                                        block_class_defaulted
                                    ).lower(),
                                    configuredBlockClass=(
                                        configured_block_class
                                    ),
                                    handle=handle,
                                    renderingState=rendering_state,
                                    themeSelection=selection_name,
                                ),
                            ), self._symbol_path(effective_block_class),
                                *block_resolution_paths,
                                *provenance_paths)

                        selected_template_path = ""
                        template_paths: tuple[str, ...] = ()
                        template_binding_candidates: list[
                            tuple[str, bool]
                        ] = []
                        selection_themes = (
                            themes if selected_theme is not None else ()
                        )
                        conditional_template_actions = []
                        for candidate_action in node.actions:
                            if (
                                candidate_action.method.casefold()
                                != "settemplate"
                                or not candidate_action.arguments
                            ):
                                continue
                            candidate_argument = (
                                candidate_action.arguments[0]
                            )
                            if (
                                not candidate_argument.value
                                or candidate_argument.value_type
                                not in {"", "string"}
                            ):
                                continue
                            if candidate_action.ifconfig:
                                conditional_template_actions.append((
                                    candidate_action,
                                    candidate_argument,
                                ))
                            else:
                                # A later unconditional setTemplate call wins
                                # over every earlier conditional candidate.
                                conditional_template_actions.clear()
                        has_conditional_template = bool(
                            conditional_template_actions
                        )
                        if node.template:
                            template_paths = self._template_paths(
                                node.template,
                                area,
                                modules,
                                selection_themes,
                                selected_theme,
                            )
                            selected_template_path = self._selected_template_path(
                                node.template,
                                area,
                                modules,
                                selection_themes,
                                selected_theme,
                            )
                            if selected_template_path and potentially_renders:
                                template_binding_candidates.append((
                                    selected_template_path,
                                    renders and not has_conditional_template,
                                ))
                            template_source = property_sources.get(
                                "template",
                                node.source,
                            )
                            packet.add(GraphFact(
                                "magento-layout-effective-template",
                                node.name,
                                (
                                    "renders-template"
                                    if renders and not has_conditional_template
                                    else (
                                        "conditionally-renders-template"
                                        if potentially_renders
                                        else "declares-inactive-template"
                                    )
                                ),
                                selected_template_path or node.template,
                                template_source.path,
                                template_source.line,
                                attrs(
                                    area=area,
                                    handle=handle,
                                    template=node.template,
                                    themeSelection=selection_name,
                                    renderingState=rendering_state,
                                    conditionalTemplateCandidate=(
                                        node_configuration.get(
                                            "conditionalTemplateCandidate",
                                            "",
                                        )
                                    ),
                                    conditionalTemplateIfconfig=(
                                        node_configuration.get(
                                            "conditionalTemplateIfconfig",
                                            "",
                                        )
                                    ),
                                ),
                            ), *template_paths, *provenance_paths)
                            if (
                                selected_template_path
                                and renders
                                and not has_conditional_template
                            ):
                                self._template_layout_sources.setdefault(
                                    selected_template_path,
                                    set(),
                                ).add((template_source.path, area, handle))
                            elif selected_template_path and potentially_renders:
                                self._conditional_template_layout_sources.setdefault(
                                    selected_template_path,
                                    set(),
                                ).add((template_source.path, area, handle))

                        for (
                            candidate_action,
                            candidate_argument,
                        ) in conditional_template_actions:
                            conditional_template = candidate_argument.value
                            candidate_paths = self._template_paths(
                                conditional_template,
                                area,
                                modules,
                                selection_themes,
                                selected_theme,
                            )
                            candidate_path = self._selected_template_path(
                                conditional_template,
                                area,
                                modules,
                                selection_themes,
                                selected_theme,
                            )
                            if candidate_path and potentially_renders:
                                template_binding_candidates.append((
                                    candidate_path,
                                    False,
                                ))
                                self._conditional_template_layout_sources.setdefault(
                                    candidate_path,
                                    set(),
                                ).add((
                                    candidate_argument.source.path,
                                    area,
                                    handle,
                                ))
                            packet.add(GraphFact(
                                "magento-layout-effective-template",
                                node.name,
                                (
                                    "conditionally-renders-template"
                                    if potentially_renders
                                    else "declares-inactive-template"
                                ),
                                candidate_path or conditional_template,
                                candidate_argument.source.path,
                                candidate_argument.source.line,
                                attrs(
                                    area=area,
                                    handle=handle,
                                    ifconfig=candidate_action.ifconfig,
                                    renderingState=rendering_state,
                                    template=conditional_template,
                                    themeSelection=selection_name,
                                ),
                            ), *candidate_paths, *provenance_paths)

                        for (
                            binding_template_path,
                            binding_is_exact,
                        ) in dict.fromkeys(template_binding_candidates):
                            if not effective_block_class:
                                continue
                            block_symbol = self._unique_symbol_casefold(
                                effective_block_class
                            )
                            packet.add(GraphFact(
                                "magento-template-effective-block-binding",
                                binding_template_path,
                                (
                                    "rendered-by-effective-block"
                                    if binding_is_exact
                                    else (
                                        "conditionally-rendered-by-"
                                        "effective-block"
                                    )
                                ),
                                effective_block_class,
                                binding_template_path,
                                1,
                                attrs(
                                    activationCertainty=(
                                        "exact"
                                        if binding_is_exact
                                        else "conditional"
                                    ),
                                    area=area,
                                    configuredBlockClass=(
                                        configured_block_class
                                    ),
                                    blockName=node.name,
                                    handle=handle,
                                    themeSelection=selection_name,
                                ),
                                related_paths=tuple(sorted(filter(None, (
                                    *provenance_paths,
                                    *block_resolution_paths,
                                    block_symbol.path if block_symbol else "",
                                )))),
                            ))
                            for call in self._template_php_calls(
                                binding_template_path,
                                "block",
                            ):
                                argument_name = self._layout_argument_for_call(
                                    call,
                                    node.arguments,
                                )
                                if argument_name:
                                    argument = next(
                                        candidate
                                        for candidate in node.arguments
                                        if candidate.name == argument_name
                                    )
                                    packet.add(GraphFact(
                                        "magento-template-effective-layout-argument-read",
                                        binding_template_path,
                                        (
                                            "reads-effective-layout-argument"
                                            if binding_is_exact
                                            else (
                                                "conditionally-reads-effective-"
                                                "layout-argument"
                                            )
                                        ),
                                        argument.value or argument.name,
                                        binding_template_path,
                                        call.line,
                                        attrs(
                                            activationCertainty=(
                                                "exact"
                                                if binding_is_exact
                                                else "conditional"
                                            ),
                                            argument=argument.name,
                                            area=area,
                                            blockName=node.name,
                                            handle=handle,
                                            method=call.method,
                                            valueType=argument.value_type,
                                            themeSelection=selection_name,
                                        ),
                                        related_paths=tuple(sorted({
                                            *provenance_paths,
                                            argument.source.path,
                                        })),
                                    ))
                                declaration = (
                                    self._method_symbol(
                                        block_symbol,
                                        call.method,
                                    )
                                    if block_symbol is not None
                                    else None
                                )
                                if declaration is not None:
                                    declaring_symbol, declared_method = declaration
                                    packet.add(GraphFact(
                                        "magento-template-effective-block-method-call",
                                        binding_template_path,
                                        (
                                            "calls-effective-block-method"
                                            if binding_is_exact
                                            else (
                                                "conditionally-calls-effective-"
                                                "block-method"
                                            )
                                        ),
                                        (
                                            f"{declaring_symbol.qualified_name}::"
                                            f"{declared_method}"
                                        ),
                                        binding_template_path,
                                        call.line,
                                        attrs(
                                            activationCertainty=(
                                                "exact"
                                                if binding_is_exact
                                                else "conditional"
                                            ),
                                            area=area,
                                            blockClass=effective_block_class,
                                            blockName=node.name,
                                            handle=handle,
                                            themeSelection=selection_name,
                                        ),
                                        related_paths=tuple(sorted({
                                            *provenance_paths,
                                            *block_resolution_paths,
                                            declaring_symbol.path,
                                        })),
                                    ))
                                    continue
                                packet.add(GraphFact(
                                    "magento-template-effective-block-method-call-unresolved",
                                    binding_template_path,
                                    (
                                        "calls-unresolved-effective-block-method"
                                        if binding_is_exact
                                        else (
                                            "conditionally-calls-unresolved-"
                                            "effective-block-method"
                                        )
                                    ),
                                    f"{effective_block_class}::{call.method}",
                                    binding_template_path,
                                    call.line,
                                    attrs(
                                        activationCertainty=(
                                            "exact"
                                            if binding_is_exact
                                            else "conditional"
                                        ),
                                        area=area,
                                        blockClass=effective_block_class,
                                        blockName=node.name,
                                        handle=handle,
                                        semanticRole="uncertainty",
                                        themeSelection=selection_name,
                                    ),
                                    related_paths=tuple(sorted({
                                        *provenance_paths,
                                        *block_resolution_paths,
                                    })),
                                ))

                        for argument in node.arguments:
                            if argument.value_type != "object" or not argument.value:
                                continue
                            object_class = argument.value.lstrip("\\")
                            object_symbol = self._unique_symbol_casefold(
                                object_class
                            )
                            object_inactive = rendering_state in {
                                "removed",
                                "ancestor-removed",
                                "unresolved-parent-cycle",
                            }
                            packet.add(GraphFact(
                                "magento-layout-effective-object-argument",
                                node.name,
                                (
                                    "declares-inactive-layout-object"
                                    if object_inactive
                                    else (
                                        "conditionally-receives-layout-object"
                                        if potentially_renders and not renders
                                        else "receives-layout-object"
                                    )
                                ),
                                object_class,
                                argument.source.path,
                                argument.source.line,
                                attrs(
                                    area=area,
                                    argument=argument.name,
                                    handle=handle,
                                    renderingState=rendering_state,
                                    themeSelection=selection_name,
                                ),
                            ), *provenance_paths, (
                                object_symbol.path if object_symbol else ""
                            ))
                            if selected_template_path and potentially_renders:
                                template_relation = (
                                    "receives-layout-object"
                                    if renders
                                    else "conditionally-receives-layout-object"
                                )
                                packet.add(GraphFact(
                                    "magento-template-effective-object-binding",
                                    selected_template_path,
                                    template_relation,
                                    object_class,
                                    selected_template_path,
                                    1,
                                    attrs(
                                        argument=argument.name,
                                        area=area,
                                        blockName=node.name,
                                        handle=handle,
                                        renderingState=rendering_state,
                                        themeSelection=selection_name,
                                    ),
                                    related_paths=tuple(sorted(filter(None, (
                                        *provenance_paths,
                                        object_symbol.path if object_symbol else "",
                                    )))),
                                ))
                            if (
                                selected_template_path
                                and potentially_renders
                                and self._is_view_model_argument(
                                    argument.name,
                                    object_symbol,
                                )
                            ):
                                packet.add(GraphFact(
                                    "magento-template-effective-view-model-binding",
                                    selected_template_path,
                                    (
                                        "receives-view-model"
                                        if renders
                                        else "conditionally-receives-view-model"
                                    ),
                                    object_class,
                                    selected_template_path,
                                    1,
                                    attrs(
                                        argument=argument.name,
                                        area=area,
                                        blockName=node.name,
                                        handle=handle,
                                        renderingState=rendering_state,
                                        themeSelection=selection_name,
                                    ),
                                    related_paths=tuple(sorted(filter(None, (
                                        *provenance_paths,
                                        object_symbol.path if object_symbol else "",
                                    )))),
                                ))

                        for action in node.actions:
                            target = (
                                f"{effective_block_class}::{action.method}"
                                if effective_block_class
                                else action.method
                            )
                            method_paths: tuple[str, ...] = ()
                            if effective_block_class:
                                block_symbol = self._unique_symbol_casefold(
                                    effective_block_class
                                )
                                declaration = (
                                    self._method_symbol(
                                        block_symbol,
                                        action.method,
                                    )
                                    if block_symbol is not None
                                    else None
                                )
                                if declaration is not None:
                                    target = (
                                        f"{declaration[0].qualified_name}::"
                                        f"{declaration[1]}"
                                    )
                                    method_paths = (declaration[0].path,)
                            action_inactive = rendering_state in {
                                "removed",
                                "ancestor-removed",
                                "unresolved-parent-cycle",
                            }
                            action_conditional = bool(
                                action.ifconfig
                            ) or (potentially_renders and not renders)
                            packet.add(GraphFact(
                                "magento-layout-effective-action",
                                node.name,
                                (
                                    "declares-inactive-layout-action"
                                    if action_inactive
                                    else (
                                        "conditionally-calls-layout-action"
                                        if action_conditional
                                        else "calls-layout-action"
                                    )
                                ),
                                target,
                                action.source.path,
                                action.source.line,
                                attrs(
                                    area=area,
                                    handle=handle,
                                    ifconfig=action.ifconfig,
                                    method=action.method,
                                    renderingState=rendering_state,
                                    themeSelection=selection_name,
                                ),
                            ), *provenance_paths, *block_resolution_paths,
                                *method_paths)

                        if (
                            node.node_kind == "uiComponent"
                            and potentially_renders
                        ):
                            ui_paths = tuple(sorted(
                                path
                                for path in self._ui_component_paths_by_name.get(
                                    node.name,
                                    (),
                                )
                                if self._is_deployed_view_source(
                                    path,
                                    modules,
                                    themes,
                                )
                                and (
                                    (
                                        self._theme_for_path(path, themes)
                                        is not None
                                        and selection_compatible_path(path)
                                    )
                                    or (
                                        self._theme_for_path(path, themes)
                                        is None
                                        and view_area(
                                            path,
                                            "ui_component",
                                        ) in {"base", area}
                                    )
                                )
                            ))
                            packet.add(GraphFact(
                                "magento-layout-ui-component-activation",
                                handle,
                                (
                                    "activates-ui-component"
                                    if renders
                                    else "conditionally-activates-ui-component"
                                ),
                                node.name,
                                node.source.path,
                                node.source.line,
                                attrs(
                                    area=area,
                                    renderingState=rendering_state,
                                    resolved=str(bool(ui_paths)).lower(),
                                    themeSelection=selection_name,
                                ),
                            ), *ui_paths, *provenance_paths)

                    for asset in effective.assets:
                        if (
                            not emit_full_projection
                            and default_assets_by_src.get(asset.src) == asset
                        ):
                            continue
                        packet.add(GraphFact(
                            "magento-layout-effective-asset",
                            handle,
                            (
                                "removes-asset"
                                if asset.removed
                                else "loads-asset"
                            ),
                            asset.src,
                            asset.source.path,
                            asset.source.line,
                            attrs(
                                area=area,
                                assetKind=asset.asset_kind,
                                themeSelection=selection_name,
                            ),
                        ), *(
                            source.path for source in asset.provenance
                            if source.path != asset.source.path
                        ))

                    for operation in effective.unresolved_operations:
                        if (
                            not emit_full_projection
                            and operation in default_unresolved_operations
                        ):
                            continue
                        packet.add(GraphFact(
                            "magento-layout-unresolved-instruction",
                            operation.name or handle,
                            "has-unresolved-layout-instruction",
                            operation.kind,
                            operation.source.path,
                            operation.source.line,
                            attrs(
                                area=area,
                                semanticRole="uncertainty",
                                themeSelection=selection_name,
                            ),
                        ))

                    for diagnostic in effective.diagnostics:
                        identity = (
                            diagnostic.code,
                            diagnostic.source.path,
                            diagnostic.source.line,
                        )
                        if identity in emitted_diagnostics:
                            continue
                        emitted_diagnostics.add(identity)
                        self._diagnostics.append(PluginDiagnostic(
                            diagnostic.code,
                            diagnostic.message,
                            self.plugin_id,
                            diagnostic.source.path,
                            recoverable=True,
                        ))
                    if handle == "default":
                        default_nodes_by_name = current_nodes_by_name
                        default_missing_parents = {
                            node.parent
                            for node in effective.nodes
                            if node.parent
                            and node.parent not in current_nodes_by_name
                        }
                        default_assets_by_src = {
                            asset.src: asset for asset in effective.assets
                        }
                        default_unresolved_operations = set(
                            effective.unresolved_operations
                        )

    def _admin_menu(
        self,
        modules: tuple[ModuleRecord, ...],
    ) -> None:
        """Resolve Magento Admin menu commands into effective topology.

        Magento chains ``add``, ``update``, and ``remove`` commands by item ID
        in module merge order. ``update`` replaces named attributes, ``add``
        fills only attributes not already supplied by an earlier command, and
        ``remove`` suppresses the item. Duplicate ``add`` commands and missing
        parents are exact configuration failures, so they are retained as
        diagnostics instead of being guessed into a navigable menu.
        """
        commands: dict[str, list[dict[str, object]]] = {}
        for path, module, order in self._ordered_configs(
            "menu.xml",
            modules,
            "adminhtml",
        ):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for menu in (
                node for node in root.iter() if tag(node) == "menu"
            ):
                for position, node in enumerate(menu):
                    operation = tag(node)
                    item_id = (node.get("id") or "").strip()
                    if operation not in {"add", "update", "remove"} or not item_id:
                        continue
                    values = {
                        key.rsplit("}", 1)[-1]: value.strip()
                        for key, value in node.attrib.items()
                        if value is not None and value.strip()
                    }
                    commands.setdefault(item_id, []).append({
                        "operation": operation,
                        "values": values,
                        "path": path,
                        "line": line(content, item_id),
                        "module": (
                            module.name if module else "application"
                        ),
                        "order": order,
                        "position": position,
                    })

        states: dict[str, dict[str, object]] = {}
        required = {"id", "title", "module", "resource"}
        for item_id, item_commands in sorted(commands.items()):
            item_commands.sort(key=lambda command: (
                int(command["order"]),
                int(command["position"]),
                str(command["path"]),
            ))
            values: dict[str, str] = {}
            value_sources: dict[str, tuple[str, int]] = {}
            all_paths: set[str] = set()
            add_count = 0
            removed = False
            removal_source: tuple[str, int] | None = None
            for command in item_commands:
                operation = str(command["operation"])
                path = str(command["path"])
                command_line = int(command["line"])
                all_paths.add(path)
                command_values = dict(command["values"])
                if operation == "add":
                    add_count += 1
                    for key, value in command_values.items():
                        if key not in values:
                            values[key] = str(value)
                            value_sources[key] = (path, command_line)
                elif operation == "update":
                    for key, value in command_values.items():
                        values[key] = str(value)
                        value_sources[key] = (path, command_line)
                else:
                    removed = True
                    removal_source = (path, command_line)
            states[item_id] = {
                "add_count": add_count,
                "all_paths": all_paths,
                "removed": removed,
                "removal_source": removal_source,
                "required_missing": required.difference(values),
                "values": values,
                "value_sources": value_sources,
            }

        visibility: dict[str, tuple[bool, str]] = {}

        def visible(
            item_id: str,
            stack: tuple[str, ...] = (),
        ) -> tuple[bool, str]:
            if item_id in visibility:
                return visibility[item_id]
            state = states.get(item_id)
            if state is None:
                return False, "missing"
            if item_id in stack:
                result = (False, "parent-cycle")
            elif int(state["add_count"]) != 1:
                result = (
                    False,
                    (
                        "duplicate-add"
                        if int(state["add_count"]) > 1
                        else "missing-add"
                    ),
                )
            elif state["required_missing"]:
                result = (False, "missing-required-attributes")
            elif bool(state["removed"]):
                result = (False, "removed")
            else:
                parent = str(dict(state["values"]).get("parent", ""))
                if not parent:
                    result = (True, "")
                elif parent not in states:
                    result = (False, "missing-parent")
                else:
                    parent_visible, parent_reason = visible(
                        parent,
                        (*stack, item_id),
                    )
                    result = (
                        (True, "")
                        if parent_visible
                        else (False, f"parent-{parent_reason}")
                    )
            visibility[item_id] = result
            return result

        modules_by_name = {module.name: module for module in modules}
        for item_id, state in sorted(states.items()):
            values = dict(state["values"])
            value_sources = dict(state["value_sources"])
            all_paths = set(state["all_paths"])
            is_visible, reason = visible(item_id)
            packet = self.graph.packet(
                "magento-admin-menu",
                item_id,
                itemId=item_id,
            )

            if not is_visible:
                source = (
                    state["removal_source"]
                    or value_sources.get("id")
                    or (sorted(all_paths)[0], 1)
                )
                kind = (
                    "magento-admin-menu-removed"
                    if reason == "removed"
                    else (
                        "magento-admin-menu-suppressed"
                        if reason.startswith("parent-")
                        and reason not in {
                            "parent-missing",
                            "parent-parent-cycle",
                        }
                        else "magento-admin-menu-invalid"
                    )
                )
                packet.add(GraphFact(
                    kind,
                    item_id,
                    (
                        "removed-by-config"
                        if reason == "removed"
                        else "not-in-effective-menu"
                    ),
                    reason,
                    str(source[0]),
                    int(source[1]),
                    attrs(
                        reason=reason,
                        semanticRole=(
                            "topology"
                            if kind == "magento-admin-menu-suppressed"
                            else "diagnostic"
                        ),
                    ),
                ), *sorted(all_paths))
                continue

            action = values.get("action", "").strip("/").casefold()
            resource = values.get("resource", "")
            config_path = values.get("dependsOnConfig", "")
            dependency_module = values.get("dependsOnModule", "")
            parent = values.get("parent", "")
            controller_paths = self._admin_controller_sources.get(
                action,
                set(),
            )
            resource_paths = self._acl_sources.get(resource, set())
            config_paths = self._system_config_sources.get(
                config_path,
                set(),
            )
            dependency = modules_by_name.get(dependency_module)
            parent_paths = (
                set(states[parent]["all_paths"])
                if parent in states
                else set()
            )
            related = set(all_paths)
            related.update(controller_paths)
            related.update(resource_paths)
            related.update(config_paths)
            related.update(parent_paths)
            if dependency is not None:
                related.add(dependency.module_xml)

            primary_key = "action" if action else "id"
            primary_source = value_sources.get(
                primary_key,
                value_sources.get("id", (sorted(all_paths)[0], 1)),
            )
            packet.add(GraphFact(
                "magento-admin-menu-item",
                item_id,
                "navigates-to" if action else "declares-container",
                action or parent or item_id,
                primary_source[0],
                primary_source[1],
                attrs(
                    module=values.get("module", ""),
                    parent=parent,
                    sortOrder=values.get("sortOrder", ""),
                    target=values.get("target", ""),
                    title=values.get("title", ""),
                ),
            ), *sorted(related))

            relation_specs = (
                (
                    "magento-admin-menu-parent",
                    "child-of-menu-item",
                    parent,
                    parent_paths,
                    "parent",
                ),
                (
                    "magento-admin-menu-acl",
                    "requires-resource",
                    resource,
                    resource_paths,
                    "resource",
                ),
                (
                    "magento-admin-menu-config-condition",
                    "visible-when-config-enabled",
                    config_path,
                    config_paths,
                    "dependsOnConfig",
                ),
                (
                    "magento-admin-menu-module-condition",
                    "visible-when-module-enabled",
                    dependency_module,
                    (
                        {dependency.module_xml}
                        if dependency is not None
                        else set()
                    ),
                    "dependsOnModule",
                ),
                (
                    "magento-admin-menu-action",
                    "dispatches-admin-action",
                    action,
                    controller_paths,
                    "action",
                ),
            )
            for (
                kind,
                relation,
                target,
                target_paths,
                attribute_name,
            ) in relation_specs:
                if not target:
                    continue
                source = value_sources.get(
                    attribute_name,
                    primary_source,
                )
                packet.add(GraphFact(
                    kind,
                    item_id,
                    relation,
                    target,
                    source[0],
                    source[1],
                    attrs(
                        exactTarget=bool(target_paths),
                        moduleEnabled=(
                            dependency.enabled
                            if (
                                attribute_name == "dependsOnModule"
                                and dependency is not None
                            )
                            else None
                        ),
                    ),
                ), *sorted(target_paths))

    def _template_globals(self) -> None:
        """Link PHTML browser globals only through one co-declaring layout.

        A repository-wide name match is not runtime evidence. Both templates
        must be referenced by the same concrete layout XML source, and that
        source must expose exactly one defining template for the global name.
        """

        references_by_path = {
            path: self._optional_frontend_source(
                "template globals",
                path,
                extract_template_global_references,
            )
            for path in sorted(self._template_layout_sources)
            if path.casefold().endswith(".phtml")
            and path in self.artifacts
        }
        definitions_by_name: dict[
            str,
            list[tuple[str, TemplateGlobalReference]],
        ] = {}
        calls: list[tuple[str, TemplateGlobalReference]] = []
        for path, references in references_by_path.items():
            for reference in references:
                if reference.relation == "defines":
                    definitions_by_name.setdefault(
                        reference.name,
                        [],
                    ).append((path, reference))
                elif reference.relation == "calls":
                    calls.append((path, reference))

        definition_callers: dict[
            tuple[str, str, int],
            set[str],
        ] = {}
        for caller_path, call in sorted(
            calls,
            key=lambda item: (item[0], item[1]),
        ):
            caller_layouts = self._template_layout_sources[caller_path]
            candidates: dict[
                str,
                tuple[
                    TemplateGlobalReference,
                    set[tuple[str, str, str]],
                ],
            ] = {}
            for definition_path, definition in definitions_by_name.get(
                call.name,
                (),
            ):
                shared_layouts = caller_layouts.intersection(
                    self._template_layout_sources[definition_path]
                )
                if not shared_layouts:
                    continue
                prior = candidates.get(definition_path)
                if prior is None or definition.line < prior[0].line:
                    candidates[definition_path] = (
                        definition,
                        set(shared_layouts),
                    )
                else:
                    prior[1].update(shared_layouts)
            if len(candidates) != 1:
                continue

            definition_path, (
                definition,
                shared_layouts,
            ) = next(iter(candidates.items()))
            global_name = f"window.{call.name}"
            layout_sources = ",".join(
                source_path
                for source_path, _, _ in sorted(shared_layouts)
            )
            packet = self.graph.packet(
                "magento-template-global",
                f"{global_name}:{definition_path}",
                globalName=global_name,
            )
            packet.add(GraphFact(
                "magento-template-global-call",
                global_name,
                "calls-unique-co-declared-definition",
                global_name,
                caller_path,
                call.line,
                attrs(
                    definitionLine=definition.line,
                    definitionPath=definition_path,
                    layoutSources=layout_sources,
                    resolution="exact-layout-source",
                    semanticRole="topology",
                    **{"retrievalIdentifier:0000": global_name},
                ),
            ), definition_path)
            definition_callers.setdefault(
                (call.name, definition_path, definition.line),
                set(),
            ).add(caller_path)

        for (
            name,
            definition_path,
            definition_line,
        ), caller_paths in sorted(definition_callers.items()):
            packet = self.graph.packet(
                "magento-template-global",
                f"window.{name}:{definition_path}",
                globalName=f"window.{name}",
            )
            packet.add(GraphFact(
                "magento-template-global-definition",
                f"window.{name}",
                "defined-in-layout-template",
                definition_path,
                definition_path,
                definition_line,
                attrs(
                    resolution="exact-layout-source",
                    semanticRole="topology",
                ),
            ), *sorted(caller_paths))

    def _template_events(
        self,
        themes: tuple[ThemeRecord, ...],
    ) -> None:
        """Link exact browser events only across co-active layout handles.

        A shared concrete layout source proves co-activation directly.
        Magento's ``default`` handle is active on every page in its area, so it
        can also prove co-activation with one page-specific handle. Theme-owned
        sources must belong to the same inheritance chain; sibling themes are
        never joined.
        """

        references_by_path = {
            path: self._optional_frontend_source(
                "template events",
                path,
                extract_template_event_references,
            )
            for path in sorted(self._template_layout_sources)
            if path.casefold().endswith(".phtml")
            and path in self.artifacts
        }
        listeners_by_event: dict[
            tuple[str, str],
            list[tuple[str, TemplateEventReference]],
        ] = {}
        dispatchers: list[tuple[str, TemplateEventReference]] = []
        for path, references in references_by_path.items():
            for reference in references:
                key = (reference.owner, reference.name)
                if reference.relation == "listens":
                    listeners_by_event.setdefault(key, []).append(
                        (path, reference)
                    )
                elif reference.relation == "dispatches":
                    dispatchers.append((path, reference))

        def theme_compatible(
            first_layout: str,
            second_layout: str,
        ) -> bool:
            first = self._theme_for_path(first_layout, themes)
            second = self._theme_for_path(second_layout, themes)
            if first is None or second is None:
                return True
            first_chain = {
                candidate.name
                for candidate in self._theme_chain(first, themes)
            }
            second_chain = {
                candidate.name
                for candidate in self._theme_chain(second, themes)
            }
            return first.name in second_chain or second.name in first_chain

        def coactive_layouts(
            first_path: str,
            second_path: str,
        ) -> tuple[tuple[str, str, str, str, str, str], ...]:
            proofs = set()
            for (
                first_layout,
                first_area,
                first_handle,
            ) in self._template_layout_sources[first_path]:
                for (
                    second_layout,
                    second_area,
                    second_handle,
                ) in self._template_layout_sources[second_path]:
                    if (
                        first_area != second_area
                        or not theme_compatible(
                            first_layout,
                            second_layout,
                        )
                    ):
                        continue
                    if first_layout == second_layout:
                        resolution = "shared-layout-source"
                    elif "default" in {first_handle, second_handle}:
                        resolution = "default-handle-coactivation"
                    else:
                        continue
                    proofs.add((
                        first_layout,
                        second_layout,
                        first_area,
                        first_handle,
                        second_handle,
                        resolution,
                    ))
            return tuple(sorted(proofs))

        listener_dispatchers: dict[
            tuple[str, str, str, int],
            set[tuple[str, int, tuple[str, ...]]],
        ] = {}
        for dispatcher_path, dispatch in sorted(
            dispatchers,
            key=lambda item: (item[0], item[1]),
        ):
            candidates: dict[
                str,
                tuple[
                    TemplateEventReference,
                    set[tuple[str, str, str, str, str, str]],
                ],
            ] = {}
            for listener_path, listener in listeners_by_event.get(
                (dispatch.owner, dispatch.name),
                (),
            ):
                if listener_path == dispatcher_path:
                    continue
                proofs = coactive_layouts(
                    dispatcher_path,
                    listener_path,
                )
                if not proofs:
                    continue
                prior = candidates.get(listener_path)
                if prior is None or listener.line < prior[0].line:
                    candidates[listener_path] = (
                        listener,
                        set(proofs),
                    )
                else:
                    prior[1].update(proofs)
            if len(candidates) != 1:
                continue

            listener_path, (listener, proofs) = next(iter(candidates.items()))
            event_identity = f"{dispatch.owner}:{dispatch.name}"
            layout_paths = tuple(sorted({
                path
                for proof in proofs
                for path in proof[:2]
            }))
            resolutions = ",".join(sorted({
                proof[5] for proof in proofs
            }))
            handles = ",".join(sorted({
                f"{proof[3]}->{proof[4]}" for proof in proofs
            }))
            packet = self.graph.packet(
                "magento-template-event",
                f"{event_identity}:{listener_path}",
                eventName=dispatch.name,
                eventOwner=dispatch.owner,
            )
            packet.add(GraphFact(
                "magento-template-event-dispatch",
                event_identity,
                "dispatches-to-unique-layout-listener",
                event_identity,
                dispatcher_path,
                dispatch.line,
                attrs(
                    handles=handles,
                    listenerLine=listener.line,
                    listenerPath=listener_path,
                    resolution=resolutions,
                    semanticRole="topology",
                ),
            ), listener_path, *layout_paths)
            listener_dispatchers.setdefault(
                (
                    dispatch.owner,
                    dispatch.name,
                    listener_path,
                    listener.line,
                ),
                set(),
            ).add((
                dispatcher_path,
                dispatch.line,
                layout_paths,
            ))

        for (
            owner,
            event_name,
            listener_path,
            listener_line,
        ), matched_dispatchers in sorted(listener_dispatchers.items()):
            event_identity = f"{owner}:{event_name}"
            dispatcher_paths = tuple(sorted({
                path
                for path, _, _ in matched_dispatchers
            }))
            layout_paths = tuple(sorted({
                layout_path
                for _, _, paths in matched_dispatchers
                for layout_path in paths
            }))
            packet = self.graph.packet(
                "magento-template-event",
                f"{event_identity}:{listener_path}",
                eventName=event_name,
                eventOwner=owner,
            )
            packet.add(GraphFact(
                "magento-template-event-listener",
                event_identity,
                "listens-to-layout-dispatchers",
                event_identity,
                listener_path,
                listener_line,
                attrs(
                    dispatcherCount=len(dispatcher_paths),
                    semanticRole="topology",
                ),
            ), *dispatcher_paths, *layout_paths)

    def _ordered_route_modules(
        self,
        area: str,
        route_id: str,
        entries: tuple[ConfigValue, ...],
    ) -> tuple[tuple[ConfigValue, ...], bool]:
        del area, route_id
        base = tuple(sorted(
            entries,
            key=lambda entry: (
                entry.order,
                entry.position,
                entry.value,
            ),
        ))
        # Match Magento\Framework\App\Route\Config\Converter::_sortModulesList.
        # This is an insertion algorithm, not a topological sort: an unresolved
        # `before` target inserts at the front, an unresolved `after` target
        # appends, and a self-reference is resolved before the current module
        # has been inserted. Core Magento route declarations rely on these
        # semantics (for example Magento_Reports before Magento_Reports).
        ordered: list[ConfigValue] = []
        for entry in base:
            attributes = dict(entry.attributes)
            if "before" in attributes:
                target = attributes["before"]
                position = next(
                    (
                        index
                        for index, candidate in enumerate(ordered)
                        if candidate.value == target
                    ),
                    0,
                )
                ordered.insert(position, entry)
            elif "after" in attributes:
                target = attributes["after"]
                position = next(
                    (
                        index
                        for index, candidate in enumerate(ordered)
                        if candidate.value == target
                    ),
                    len(base),
                )
                ordered.insert(position + 1, entry)
            else:
                ordered.append(entry)
        return tuple(ordered), True

    def _ui_components(
        self,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
    ) -> None:
        ui_by_component: dict[tuple[str, str], set[str]] = {}
        for path in sorted(self.artifacts):
            if "/ui_component/" not in f"/{path}" or not path.endswith(".xml"):
                continue
            theme = self._theme_for_path(path, themes)
            if not self._is_deployed_view_source(
                path,
                modules,
                themes,
            ):
                continue
            area = view_area(path, "ui_component") or (
                theme.area if theme else None
            )
            if area is None:
                continue
            ui_by_component.setdefault(
                (area, PurePosixPath(path).stem),
                set(),
            ).add(path)

        def effective_ui_paths(
            area: str,
            component: str,
            source_theme: ThemeRecord | None = None,
        ) -> tuple[str, ...]:
            areas = ("base", area) if area != "base" else ("base",)
            allowed_theme_roots = (
                {
                    candidate.root
                    for candidate in self._theme_chain(source_theme, themes)
                }
                if source_theme is not None
                else None
            )
            return tuple(sorted({
                candidate
                for candidate_area in areas
                for candidate in ui_by_component.get(
                    (candidate_area, component),
                    (),
                )
                if (
                    (candidate_theme := self._theme_for_path(candidate, themes))
                    is None
                    or allowed_theme_roots is None
                    or candidate_theme.root in allowed_theme_roots
                )
            }))

        for path in sorted(self.artifacts):
            if "/ui_component/" not in f"/{path}" or not path.endswith(".xml"):
                continue
            theme = self._theme_for_path(path, themes)
            if not self._is_deployed_view_source(
                path,
                modules,
                themes,
            ):
                continue
            area = view_area(path, "ui_component") or (theme.area if theme else None)
            if area is None:
                continue
            root = self._xml(path)
            if root is None:
                continue
            component_name = PurePosixPath(path).stem
            packet = self.graph.packet(
                "magento-ui-component",
                f"{area}:{component_name}",
                area=area,
                component=component_name,
            )
            packet.add(GraphFact(
                "magento-ui-component",
                component_name,
                "declared-in",
                path,
                path,
                1,
                attrs(area=area, theme=theme.name if theme else ""),
            ), theme.theme_xml if theme else "", *(
                candidate
                for candidate in effective_ui_paths(
                    area,
                    component_name,
                    theme,
                )
                if candidate != path
            ))
            content = self.artifacts[path]
            for element in root.iter():
                class_name = element.get("class", "").lstrip("\\")
                if "\\" in class_name:
                    packet.add(GraphFact(
                        "magento-ui-php-class",
                        component_name,
                        "uses-class",
                        class_name,
                        path,
                        line(content, class_name),
                        attrs(area=area, element=tag(element), name=element.get("name", "")),
                    ), self._symbol_path(class_name))

                values: list[tuple[str, str]] = []
                if element.get("component"):
                    values.append(("component", element.get("component")))
                if element.get("template"):
                    values.append(("template", element.get("template")))
                if tag(element) in {"item", "param"} and element.get("name") in {
                    "component", "template", "provider", "deps",
                } and element.text:
                    values.append((element.get("name"), element.text.strip()))
                if tag(element) in {"provider", "dep", "aclResource"} and element.text:
                    values.append((tag(element), element.text.strip()))

                for value_kind, value in values:
                    if not value:
                        continue
                    relation = {
                        "component": "uses-js-component",
                        "template": "uses-ui-template",
                        "provider": "depends-on-provider",
                        "deps": "depends-on-component",
                        "dep": "depends-on-component",
                        "aclResource": "requires-resource",
                    }[value_kind]
                    asset_paths = (
                        tuple(sorted(self._acl_sources.get(value, ())))
                        if value_kind == "aclResource"
                        else self._ui_asset_paths(
                            value,
                            area,
                            value_kind == "template",
                            modules,
                            themes,
                            theme,
                        )
                    )
                    packet.add(GraphFact(
                        "magento-ui-relationship",
                        component_name,
                        relation,
                        value,
                        path,
                        line(content, value),
                        attrs(area=area),
                    ), *asset_paths)

    def _email_templates(
        self,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
    ) -> None:
        """Join merged email declarations to files, overrides, and exact consumers."""
        enabled_modules = {
            module.name: module
            for module in modules
            if module.enabled
        }
        declarations: dict[str, dict[str, object]] = {}
        for path, owner, order in self._ordered_configs(
            "email_templates.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for element in root.iter():
                if tag(element) != "template":
                    continue
                identifier = element.get("id", "").strip()
                filename = element.get("file", "").strip()
                module_name = (
                    element.get("module", "").strip()
                    or (owner.name if owner else "")
                )
                area = element.get("area", "").strip()
                relative = PurePosixPath(filename)
                if (
                    not identifier
                    or not filename
                    or not module_name
                    or not area
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or "\\" in filename
                ):
                    continue
                declarations[identifier] = {
                    "area": area,
                    "file": filename,
                    "label": element.get("label", "").strip(),
                    "module": module_name,
                    "order": order,
                    "path": path,
                    "line": line(content, identifier),
                    "type": element.get("type", "").strip(),
                }

        if not declarations:
            return

        resolved_paths: dict[str, tuple[str, ...]] = {}
        for identifier, declaration in sorted(declarations.items()):
            area = str(declaration["area"])
            filename = str(declaration["file"])
            module_name = str(declaration["module"])
            declaration_path = str(declaration["path"])
            module = enabled_modules.get(module_name)
            module_path = (
                _path_under(
                    module.root,
                    f"view/{area}/email/{filename}",
                )
                if module is not None
                else ""
            )
            if module_path not in self.artifacts:
                module_path = ""

            packet = self.graph.packet(
                "magento-email-template",
                identifier,
                area=area,
                module=module_name,
            )
            packet.add(GraphFact(
                "magento-email-template",
                identifier,
                "renders-email-file",
                f"{module_name}::{filename}",
                declaration_path,
                int(declaration["line"]),
                attrs(
                    area=area,
                    file=filename,
                    label=declaration["label"],
                    module=module_name,
                    order=declaration["order"],
                    type=declaration["type"],
                ),
            ), module_path)

            paths = [path for path in (declaration_path, module_path) if path]
            for theme in themes:
                if theme.area != area:
                    continue
                override_path = _path_under(
                    theme.root,
                    f"{module_name}/email/{filename}",
                )
                if override_path not in self.artifacts:
                    continue
                paths.append(override_path)
                inherited_override_paths = tuple(
                    candidate
                    for ancestor in self._theme_chain(theme, themes)[1:]
                    if (
                        candidate := _path_under(
                            ancestor.root,
                            f"{module_name}/email/{filename}",
                        )
                    ) in self.artifacts
                )
                override_packet = self.graph.packet(
                    "magento-email-template-override",
                    f"{area}:{theme.name}:{identifier}",
                    area=area,
                    module=module_name,
                    theme=theme.name,
                )
                override_packet.add(GraphFact(
                    "magento-email-template-override",
                    f"{theme.name}:{identifier}",
                    "overrides-email-template",
                    identifier,
                    override_path,
                    1,
                    attrs(
                        area=area,
                        file=filename,
                        module=module_name,
                        theme=theme.name,
                    ),
                ), declaration_path, module_path, theme.theme_xml,
                    *inherited_override_paths)
            resolved_paths[identifier] = tuple(sorted(set(paths)))

        for path, _, _ in self._ordered_configs(
            "config.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]

            def visit(element, ancestors: tuple[str, ...]) -> None:
                current = (*ancestors, tag(element))
                value = (element.text or "").strip()
                if value in declarations:
                    packet = self.graph.packet(
                        "magento-email-template-selection",
                        f"{path}:{'/'.join(current)}",
                    )
                    packet.add(GraphFact(
                        "magento-email-config-default",
                        "/".join(current),
                        "selects-email-template",
                        value,
                        path,
                        line(content, value),
                    ), *resolved_paths[value])
                for child in element:
                    visit(child, current)

            visit(root, ())

        transport_builder = (
            r"Magento\Framework\Mail\Template\TransportBuilder"
        ).casefold()
        for consumer in self.symbols:
            for key, encoded in consumer.attributes:
                if not key.startswith(
                    "php-literal-instance-call-reference:"
                ):
                    continue
                try:
                    reference = json.loads(encoded)
                except (TypeError, ValueError):
                    continue
                if (
                    str(reference.get("method", "")).casefold()
                    != "settemplateidentifier"
                    or str(reference.get("target", "")).lstrip(
                        "\\"
                    ).casefold() != transport_builder
                ):
                    continue
                literal_arguments = reference.get(
                    "literalStringArguments",
                    {},
                )
                if not isinstance(literal_arguments, dict):
                    continue
                identifier = literal_arguments.get("0")
                if identifier not in declarations:
                    continue
                try:
                    reference_line = max(
                        1,
                        int(reference.get("line", consumer.line)),
                    )
                except (TypeError, ValueError):
                    reference_line = consumer.line
                packet = self.graph.packet(
                    "magento-email-template-consumer",
                    f"{consumer.qualified_name}:{identifier}",
                )
                packet.add(GraphFact(
                    "magento-email-template-consumer",
                    consumer.qualified_name,
                    "selects-email-template",
                    identifier,
                    consumer.path,
                    reference_line,
                    attrs(
                        caller=reference.get("caller", ""),
                        receiver=reference.get("target", ""),
                    ),
                ), *resolved_paths[identifier])

    def _requirejs(
        self,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
    ) -> None:
        records: dict[str, dict[str, object]] = {}
        for path, content in sorted(self.artifacts.items()):
            if PurePosixPath(path).name != "requirejs-config.js":
                continue
            theme = self._theme_for_path(path, themes)
            if not self._is_deployed_view_source(
                path,
                modules,
                themes,
            ):
                continue
            module = self._module_for_path(path, modules)
            theme_module = self._theme_module(path, theme)
            if theme is not None:
                relative = path[len(theme.root):].lstrip("/")
                if relative not in {
                    "requirejs-config.js",
                    (
                        f"{theme_module}/requirejs-config.js"
                        if theme_module
                        else ""
                    ),
                }:
                    # A same-named JavaScript file below `web/` is a static
                    # asset, not an input to RequireJs\Config\File\Collector.
                    continue
                area = theme.area
            else:
                if module is None:
                    continue
                relative = (
                    path[len(module.root):].lstrip("/")
                    if module.root
                    else path
                )
                area_match = re.fullmatch(
                    r"view/([^/]+)/requirejs-config\.js",
                    relative,
                )
                if area_match is None:
                    continue
                area = area_match.group(1)

            try:
                relations = extract_requirejs_relations(content)
            except OptionalJavaScriptEnrichmentError as exception:
                if not exception.source_specific:
                    raise
                self._record_optional_enrichment_failure(
                    "RequireJS",
                    exception,
                    path,
                )
                continue
            records[path] = {
                "path": path,
                "area": area,
                "theme": theme,
                "theme_module": theme_module,
                "module": module,
                "relations": relations,
            }
            packet = self.graph.packet(
                "magento-requirejs",
                f"{area}:{path}",
                area=area,
            )
            for relation in relations:
                source_paths = self._ui_asset_paths(
                    relation.source,
                    area,
                    False,
                    modules,
                    themes,
                    theme,
                )
                target_paths = self._ui_asset_paths(
                    relation.target,
                    area,
                    False,
                    modules,
                    themes,
                    theme,
                )
                packet.add(GraphFact(
                    f"magento-requirejs-{relation.kind}",
                    relation.source,
                    relation.relation,
                    relation.target,
                    path,
                    relation.line,
                    attrs(
                        area=area,
                        theme=theme.name if theme else "",
                        mapScope=relation.scope,
                        fallbackPosition=relation.position,
                    ),
                ), *source_paths, *target_paths, theme.theme_xml if theme else "")

        if not records:
            return

        def relation_identity(relation) -> tuple[str, ...] | None:
            if relation.kind == "path":
                return ("path", relation.source)
            if relation.kind == "map":
                return ("map", relation.scope, relation.source)
            if relation.kind == "mixin":
                return ("mixin", relation.source, relation.target)
            if relation.kind == "shim":
                return ("shim", relation.source)
            return None

        def signature(relations) -> tuple[tuple[str, str, int], ...]:
            return tuple(sorted(
                (
                    relation.relation,
                    relation.target,
                    relation.position,
                )
                for relation in relations
            ))

        module_order = {
            module.name: module.order
            for module in modules
            if module.enabled
        }
        runtime_areas = {
            str(record["area"])
            for record in records.values()
            if record["area"] != "base"
        }
        runtime_areas.update(theme.area for theme in themes)
        # Retain a base-only effective selection even when another area is
        # present so consumers in an otherwise unconfigured area can still
        # inherit Magento's base RequireJS configuration exactly.
        runtime_areas.add("base")

        selections = [
            (area, theme)
            for area in sorted(runtime_areas)
            for theme in (
                tuple(
                    candidate
                    for candidate in themes
                    if candidate.area == area
                )
                or (None,)
            )
        ]
        emitted_precedence: set[GraphFact] = set()
        for area, selected_theme in selections:
            ordered_records = sorted(
                (
                    record
                    for record in records.values()
                    if record["theme"] is None
                    and record["area"] in {"base", area}
                ),
                key=lambda record: (
                    (
                        record["module"].order
                        if record["module"] is not None
                        else -1
                    ),
                    0 if record["area"] == "base" else 1,
                    str(record["path"]),
                ),
            )
            if selected_theme is not None:
                for current_theme in reversed(
                    self._theme_chain(selected_theme, themes)
                ):
                    theme_records = [
                        record
                        for record in records.values()
                        if record["theme"] == current_theme
                    ]
                    ordered_records.extend(sorted(
                        (
                            record
                            for record in theme_records
                            if record["theme_module"]
                        ),
                        key=lambda record: (
                            module_order.get(
                                str(record["theme_module"]),
                                -1,
                            ),
                            str(record["path"]),
                        ),
                    ))
                    ordered_records.extend(sorted(
                        (
                            record
                            for record in theme_records
                            if not record["theme_module"]
                        ),
                        key=lambda record: str(record["path"]),
                    ))

            effective_theme = (
                selected_theme.name if selected_theme is not None else ""
            )
            self._requirejs_configs[(area, effective_theme)] = (
                build_effective_requirejs_config(
                    area,
                    effective_theme,
                    tuple(
                        (
                            str(record["path"]),
                            tuple(record["relations"]),
                        )
                        for record in ordered_records
                    ),
                )
            )

            declarations: dict[
                tuple[str, ...],
                list[tuple[dict[str, object], tuple[object, ...]]],
            ] = {}
            for record in ordered_records:
                grouped: dict[tuple[str, ...], list[object]] = {}
                for relation in record["relations"]:
                    identity = relation_identity(relation)
                    if identity is not None:
                        grouped.setdefault(identity, []).append(relation)
                for identity, relations in sorted(grouped.items()):
                    declarations.setdefault(identity, []).append((
                        record,
                        tuple(sorted(relations)),
                    ))

            for identity, values in sorted(declarations.items()):
                for position, (
                    (previous, previous_relations),
                    (current, current_relations),
                ) in enumerate(zip(values, values[1:]), start=1):
                    previous_signature = signature(previous_relations)
                    current_signature = signature(current_relations)
                    if previous_signature == current_signature:
                        continue
                    theme_specific = (
                        previous["theme"] is not None
                        or current["theme"] is not None
                    )
                    selection_theme = (
                        selected_theme.name
                        if theme_specific and selected_theme is not None
                        else ""
                    )
                    current_path = str(current["path"])
                    previous_path = str(previous["path"])
                    fact = GraphFact(
                        "magento-requirejs-override",
                        previous_path,
                        "overridden-by-config",
                        current_path,
                        current_path,
                        min(
                            relation.line
                            for relation in current_relations
                        ),
                        attrs(
                            area=area,
                            theme=selection_theme,
                            requireJsKind=identity[0],
                            identity=":".join(identity[1:]),
                            precedencePosition=position,
                            previousValue="|".join(
                                f"{relation}:{target}:{offset}"
                                for relation, target, offset
                                in previous_signature
                            ),
                            effectiveValue="|".join(
                                f"{relation}:{target}:{offset}"
                                for relation, target, offset
                                in current_signature
                            ),
                        ),
                        related_paths=(previous_path,),
                    )
                    if fact in emitted_precedence:
                        continue
                    emitted_precedence.add(fact)
                    packet = self.graph.packet(
                        "magento-requirejs-precedence",
                        (
                            f"{area}:{selection_theme or 'module'}:"
                            f"{':'.join(identity)}"
                        ),
                        area=area,
                        theme=selection_theme,
                    )
                    packet.add(fact)

    def _frontend_source_areas(
        self,
        path: str,
        themes: tuple[ThemeRecord, ...],
    ) -> tuple[str, ...]:
        activated_areas = {
            area
            for sources in (
                self._template_layout_sources,
                self._conditional_template_layout_sources,
            )
            for _, area, _ in sources.get(path, ())
        }
        if activated_areas:
            return tuple(sorted(activated_areas))
        theme = self._theme_for_path(path, themes)
        if theme is not None:
            return (theme.area,)
        match = re.search(
            r"/view/([^/]+)/(?:templates|web)/",
            f"/{path}",
        )
        return (match.group(1),) if match is not None else ()

    def _requirejs_configs_for_source(
        self,
        area: str,
        source_theme: ThemeRecord | None,
    ) -> tuple[EffectiveRequireJsConfig, ...]:
        if source_theme is not None:
            selected = self._requirejs_configs.get((
                area,
                source_theme.name,
            ))
            if selected is not None:
                return (selected,)
        candidates = tuple(sorted(
            (
                config
                for (config_area_name, _), config
                in self._requirejs_configs.items()
                if config_area_name == area
            ),
            key=lambda config: config.theme,
        ))
        if candidates:
            return candidates
        return tuple(sorted(
            (
                config
                for (config_area_name, _), config
                in self._requirejs_configs.items()
                if config_area_name == "base"
            ),
            key=lambda config: config.theme,
        ))

    def _effective_frontend_dependency(
        self,
        requested: str,
        consumer: str,
        area: str,
        source_theme: ThemeRecord | None,
    ) -> tuple[
        tuple[RequireJsResolution, ...],
        EffectiveRequireJsConfig | None,
        str,
    ]:
        normalized = normalize_amd_dependency(requested, consumer)
        if not normalized:
            return (), None, "dynamic-or-relative-unresolved"
        configs = self._requirejs_configs_for_source(area, source_theme)
        if not configs:
            return (
                (RequireJsResolution(normalized),),
                None,
                "literal-module-id",
            )
        candidate_resolutions = tuple(
            config.resolve(requested, consumer)
            for config in configs
        )
        if len(candidate_resolutions) == 1:
            return (
                candidate_resolutions[0],
                configs[0],
                "effective-requirejs",
            )
        first = candidate_resolutions[0]
        if all(candidate == first for candidate in candidate_resolutions[1:]):
            # The effective identifier and its provenance are invariant across
            # every repository-known theme selection. Mixin activation still
            # abstains because a sibling theme may change only its mixin set.
            return first, None, "stable-across-theme-configs"
        return (
            (RequireJsResolution(normalized),),
            None,
            "theme-dependent-requirejs-abstained",
        )

    def _selected_frontend_asset_paths(
        self,
        identifier: str,
        area: str,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
        source_theme: ThemeRecord | None,
    ) -> tuple[str, ...]:
        loader, marker, resource = identifier.partition("!")
        if marker and loader != "text":
            return ()
        asset_identifier = resource if marker else loader
        candidates = self._ui_asset_paths(
            asset_identifier,
            area,
            marker and loader == "text",
            modules,
            themes,
            source_theme,
        )
        if not candidates:
            return ()
        if source_theme is not None:
            for theme in self._theme_chain(source_theme, themes):
                selected = tuple(
                    path
                    for path in candidates
                    if path == theme.root or path.startswith(theme.root + "/")
                )
                if selected:
                    return tuple(sorted(selected))
        elif any(
            self._theme_for_path(path, themes) is not None
            for path in candidates
        ):
            # A module-owned consumer can execute under any configured store
            # theme. Do not connect it to sibling theme overrides without an
            # active-theme selection.
            return ()

        module_candidates = tuple(
            path
            for path in candidates
            if self._theme_for_path(path, themes) is None
        )
        area_marker = f"/view/{area}/web/"
        selected = tuple(
            path for path in module_candidates
            if area_marker in f"/{path}"
        )
        if selected:
            return tuple(sorted(selected))
        base = tuple(
            path for path in module_candidates
            if "/view/base/web/" in f"/{path}"
        )
        return tuple(sorted(base or module_candidates))

    def _emit_frontend_dependency(
        self,
        *,
        packet_kind: str,
        packet_key: str,
        fact_kind: str,
        source: str,
        relation: str,
        requested: str,
        consumer: str,
        path: str,
        source_line: int,
        position: int,
        area: str,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
        source_theme: ThemeRecord | None,
        activation_paths: tuple[str, ...] = (),
        extra_attributes: dict[str, object] | None = None,
    ) -> None:
        resolutions, config, resolution_kind = (
            self._effective_frontend_dependency(
                requested,
                consumer,
                area,
                source_theme,
            )
        )
        if not resolutions:
            return
        packet = self.graph.packet(
            packet_kind,
            packet_key,
            area=area,
            theme=source_theme.name if source_theme else "",
        )
        normalized_requested = normalize_amd_dependency(
            requested,
            consumer,
        )
        for resolution in resolutions:
            target_paths = self._selected_frontend_asset_paths(
                resolution.identifier,
                area,
                modules,
                themes,
                source_theme,
            )
            loader, marker, _ = resolution.identifier.partition("!")
            packet.add(GraphFact(
                fact_kind,
                source,
                relation,
                resolution.identifier,
                path,
                source_line,
                attrs(**{
                    "area": area,
                    "fallbackPosition": resolution.fallback_position,
                    "loaderPlugin": loader if marker else "",
                    "mappedIdentifier": resolution.mapped_identifier,
                    "position": position,
                    "requestedIdentifier": requested,
                    "requireJsConfigKinds": ",".join(
                        resolution.config_kinds
                    ),
                    "requireJsConfigPaths": ",".join(
                        resolution.config_paths
                    ),
                    "resolution": resolution_kind,
                    "theme": source_theme.name if source_theme else "",
                    **(extra_attributes or {}),
                }),
            ),
                *target_paths,
                *resolution.config_paths,
                *activation_paths,
                source_theme.theme_xml if source_theme else "",
            )

            if config is None or marker:
                continue
            activation_certainty = str(
                (extra_attributes or {}).get(
                    "activationCertainty",
                    "exact",
                )
            )
            for mixin in config.mixins_for(
                normalized_requested,
                resolution.mapped_identifier,
                resolution.identifier,
            ):
                mixin_paths = self._selected_frontend_asset_paths(
                    mixin.target,
                    area,
                    modules,
                    themes,
                    source_theme,
                )
                packet.add(GraphFact(
                    "magento-requirejs-consumer-mixin",
                    resolution.identifier,
                    (
                        "loads-effective-mixin"
                        if activation_certainty == "exact"
                        else "conditionally-loads-effective-mixin"
                    ),
                    mixin.target,
                    path,
                    source_line,
                    attrs(
                        activationCertainty=activation_certainty,
                        area=area,
                        configLine=mixin.line,
                        configPath=mixin.path,
                        position=position,
                        theme=source_theme.name if source_theme else "",
                    ),
                ),
                    mixin.path,
                    *target_paths,
                    *mixin_paths,
                    *activation_paths,
                )

    def _frontend_initializers(
        self,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
    ) -> None:
        sources = {
            path
            for path in self._template_layout_sources
            if path.casefold().endswith(".phtml")
            and path in self.artifacts
        }
        sources.update(
            path
            for path in self._conditional_template_layout_sources
            if path.casefold().endswith(".phtml")
            and path in self.artifacts
        )
        sources.update(
            path
            for path in self.artifacts
            if path.casefold().endswith(".html")
            and "/web/" in f"/{path}"
            and self._is_deployed_view_source(path, modules, themes)
        )
        for path in sorted(sources):
            references = self._optional_frontend_source(
                "frontend initializers",
                path,
                extract_frontend_initializers,
            )
            source_theme = self._theme_for_path(path, themes)
            for area in self._frontend_source_areas(path, themes):
                exact_layout_entries = tuple(sorted(
                    entry
                    for entry in self._template_layout_sources.get(path, ())
                    if entry[1] == area
                ))
                conditional_layout_entries = tuple(sorted(
                    entry
                    for entry in self._conditional_template_layout_sources.get(
                        path,
                        (),
                    )
                    if entry[1] == area
                ))
                layout_entries = tuple(sorted({
                    *exact_layout_entries,
                    *conditional_layout_entries,
                }))
                activation_certainty = (
                    "exact" if exact_layout_entries else "conditional"
                )
                activation_paths = tuple(sorted({
                    layout_path for layout_path, _, _ in layout_entries
                }))
                handles = ",".join(sorted({
                    handle for _, _, handle in layout_entries
                }))
                for reference in references:
                    self._emit_frontend_dependency(
                        packet_kind="magento-frontend-init",
                        packet_key=f"{area}:{path}",
                        fact_kind="magento-frontend-init",
                        source=(
                            f"{reference.source_kind}:{reference.selector}"
                        ),
                        relation=(
                            "initializes-component"
                            if activation_certainty == "exact"
                            else "conditionally-initializes-component"
                        ),
                        requested=reference.component,
                        consumer="",
                        path=path,
                        source_line=reference.line,
                        position=reference.position,
                        area=area,
                        modules=modules,
                        themes=themes,
                        source_theme=source_theme,
                        activation_paths=activation_paths,
                        extra_attributes={
                            "activationCertainty": activation_certainty,
                            "handles": handles,
                            "initKind": reference.source_kind,
                            "selector": reference.selector,
                        },
                    )

    def _amd_source_identifier(
        self,
        path: str,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
    ) -> str:
        theme = self._theme_for_path(path, themes)
        if theme is not None:
            relative = path[len(theme.root):].lstrip("/")
            parts = relative.split("/", 2)
            if len(parts) != 3 or "_" not in parts[0] or parts[1] != "web":
                return ""
            module_name = parts[0]
            asset = parts[2]
        else:
            module = self._module_for_path(path, modules)
            if module is None:
                return ""
            relative = (
                path[len(module.root):].lstrip("/")
                if module.root
                else path
            )
            match = re.fullmatch(r"view/[^/]+/web/(.+)", relative)
            if match is None:
                return ""
            module_name = module.name
            asset = match.group(1)
        for suffix in (".jsx", ".mjs", ".js"):
            if asset.casefold().endswith(suffix):
                asset = asset[:-len(suffix)]
                break
        return f"{module_name}/{asset}" if asset else ""

    def _amd_consumers(
        self,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
    ) -> None:
        sources = {
            path
            for path in self.artifacts
            if path.casefold().endswith((".js", ".mjs", ".jsx"))
            and "/web/" in f"/{path}"
            and PurePosixPath(path).name != "requirejs-config.js"
            and self._is_deployed_view_source(path, modules, themes)
        }
        sources.update(
            path
            for path in self._template_layout_sources
            if path.casefold().endswith(".phtml")
            and path in self.artifacts
        )
        sources.update(
            path
            for path in self._conditional_template_layout_sources
            if path.casefold().endswith(".phtml")
            and path in self.artifacts
        )
        for path in sorted(sources):
            is_template = path.casefold().endswith(".phtml")
            dependencies = self._optional_frontend_source(
                "AMD consumers",
                path,
                (
                    extract_template_amd_dependencies
                    if is_template
                    else extract_amd_dependencies
                ),
            )
            source_theme = self._theme_for_path(path, themes)
            source_identifier = (
                ""
                if is_template
                else self._amd_source_identifier(path, modules, themes)
            )
            for area in self._frontend_source_areas(path, themes):
                exact_layout_entries = tuple(sorted(
                    entry
                    for entry in self._template_layout_sources.get(path, ())
                    if entry[1] == area
                ))
                conditional_layout_entries = tuple(sorted(
                    entry
                    for entry in self._conditional_template_layout_sources.get(
                        path,
                        (),
                    )
                    if entry[1] == area
                ))
                layout_entries = tuple(sorted({
                    *exact_layout_entries,
                    *conditional_layout_entries,
                }))
                activation_certainty = (
                    "exact" if exact_layout_entries else "conditional"
                )
                activation_paths = tuple(sorted({
                    layout_path for layout_path, _, _ in layout_entries
                }))
                for dependency in dependencies:
                    consumer = dependency.named_module or source_identifier
                    normalized = normalize_amd_dependency(
                        dependency.dependency,
                        consumer,
                    )
                    if not normalized:
                        continue
                    self._emit_frontend_dependency(
                        packet_kind="magento-amd-consumer",
                        packet_key=f"{area}:{path}",
                        fact_kind="magento-amd-dependency",
                        source=consumer or path,
                        relation=(
                            (
                                "declares-dependency"
                                if dependency.consumer_kind == "define"
                                else "loads-module"
                            )
                            if activation_certainty == "exact"
                            else (
                                "conditionally-declares-dependency"
                                if dependency.consumer_kind == "define"
                                else "conditionally-loads-module"
                            )
                        ),
                        requested=dependency.dependency,
                        consumer=consumer,
                        path=path,
                        source_line=dependency.line,
                        position=dependency.position,
                        area=area,
                        modules=modules,
                        themes=themes,
                        source_theme=source_theme,
                        activation_paths=activation_paths,
                        extra_attributes={
                            "activationCertainty": activation_certainty,
                            "callKind": dependency.consumer_kind,
                            "namedModule": dependency.named_module,
                        },
                    )

    def _system_configuration(
        self,
        modules: tuple[ModuleRecord, ...],
    ) -> None:
        """Connect Admin configuration fields to their exact runtime inputs.

        Magento merges ``etc/adminhtml/system.xml`` declarations into the
        Stores > Configuration tree. A field can override its normal
        ``section/group/field`` storage path and can delegate rendering,
        option generation, and persistence to PHP models. The corresponding
        default is declared separately in ``etc/config.xml``. These are
        architecture relations, not defect rules: unresolved/external class
        names remain explicit targets and no validity claim is inferred.
        """
        acl_paths: dict[str, set[str]] = {}
        for path, _, _ in self._ordered_configs(
            "acl.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            for resource in (
                node
                for node in root.iter()
                if tag(node) == "resource" and node.get("id")
            ):
                acl_paths.setdefault(resource.get("id"), set()).add(path)
        self._acl_sources = acl_paths

        default_paths: dict[
            str,
            list[tuple[str, str, int, str, int]],
        ] = {}

        def collect_defaults(
            node,
            *,
            path: str,
            module_name: str,
            order: int,
            scope: str,
            prefix: tuple[str, ...] = (),
        ) -> None:
            children = tuple(node)
            current = (*prefix, tag(node))
            if not children:
                value = (node.text or "").strip()
                if value:
                    config_path = "/".join(current)
                    default_paths.setdefault(config_path, []).append((
                        path,
                        scope,
                        line(self.artifacts[path], value),
                        module_name,
                        order,
                    ))
                return
            for child in children:
                collect_defaults(
                    child,
                    path=path,
                    module_name=module_name,
                    order=order,
                    scope=scope,
                    prefix=current,
                )

        for path, module, order in self._ordered_configs(
            "config.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            module_name = module.name if module else "application"
            for scope_node in root:
                scope_kind = tag(scope_node)
                if scope_kind == "default":
                    for child in scope_node:
                        collect_defaults(
                            child,
                            path=path,
                            module_name=module_name,
                            order=order,
                            scope="default",
                        )
                    continue
                if scope_kind not in {"websites", "stores"}:
                    continue
                for scope_code in scope_node:
                    for child in scope_code:
                        collect_defaults(
                            child,
                            path=path,
                            module_name=module_name,
                            order=order,
                            scope=f"{scope_kind}:{tag(scope_code)}",
                        )

        def text_child(node, name: str) -> str:
            child = next(
                (
                    candidate
                    for candidate in node
                    if tag(candidate) == name and candidate.text
                ),
                None,
            )
            return child.text.strip() if child is not None else ""

        def model_relations(
            packet,
            node,
            *,
            config_path: str,
            source_path: str,
            content: str,
            element_kind: str,
        ) -> None:
            for child_name, relation in (
                ("frontend_model", "uses-frontend-model"),
                ("backend_model", "uses-backend-model"),
                ("source_model", "uses-source-model"),
            ):
                class_name = text_child(node, child_name).lstrip("\\")
                if not class_name:
                    continue
                packet.add(GraphFact(
                    "magento-system-config-model",
                    config_path,
                    relation,
                    class_name,
                    source_path,
                    line(content, class_name),
                    attrs(element=element_kind),
                ), self._symbol_path(class_name))

        def extension_relation(
            packet,
            node,
            *,
            identity: str,
            source_path: str,
            content: str,
            element_kind: str,
        ) -> None:
            target = node.get("extends", "").strip()
            if not target:
                return
            packet.add(GraphFact(
                "magento-system-config-extension",
                identity,
                "extends-config-node",
                target,
                source_path,
                line(content, target),
                attrs(element=element_kind),
            ))

        for path, module, order in self._ordered_configs(
            "system.xml",
            modules,
            "adminhtml",
        ):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            module_name = module.name if module else "application"
            system_nodes = (
                (root,)
                if tag(root) == "system"
                else tuple(
                    node for node in root if tag(node) == "system"
                )
            )
            for system in system_nodes:
                for section in (
                    node
                    for node in system
                    if tag(node) == "section" and node.get("id")
                ):
                    section_id = section.get("id")
                    section_packet = self.graph.packet(
                        "magento-system-config",
                        f"section:{section_id}",
                        module=module_name,
                    )
                    section_packet.add(GraphFact(
                        "magento-system-config-section",
                        section_id,
                        "declared-in-admin-configuration",
                        path,
                        path,
                        line(content, section_id),
                        attrs(module=module_name, order=order),
                    ))
                    extension_relation(
                        section_packet,
                        section,
                        identity=section_id,
                        source_path=path,
                        content=content,
                        element_kind="section",
                    )
                    model_relations(
                        section_packet,
                        section,
                        config_path=section_id,
                        source_path=path,
                        content=content,
                        element_kind="section",
                    )
                    resource = text_child(section, "resource")
                    if resource:
                        section_packet.add(GraphFact(
                            "magento-system-config-acl",
                            section_id,
                            "requires-resource",
                            resource,
                            path,
                            line(content, resource),
                        ), *sorted(acl_paths.get(resource, ())))

                    pending_groups = [
                        (group, (section_id,))
                        for group in section
                        if tag(group) == "group" and group.get("id")
                    ]
                    while pending_groups:
                        group, parent_parts = pending_groups.pop(0)
                        group_parts = (*parent_parts, group.get("id"))
                        group_path = "/".join(group_parts)
                        group_packet = self.graph.packet(
                            "magento-system-config",
                            f"group:{group_path}",
                            module=module_name,
                        )
                        group_packet.add(GraphFact(
                            "magento-system-config-group",
                            group_path,
                            "declared-in-admin-configuration",
                            path,
                            path,
                            line(content, group.get("id")),
                            attrs(module=module_name, order=order),
                        ))
                        extension_relation(
                            group_packet,
                            group,
                            identity=group_path,
                            source_path=path,
                            content=content,
                            element_kind="group",
                        )
                        model_relations(
                            group_packet,
                            group,
                            config_path=group_path,
                            source_path=path,
                            content=content,
                            element_kind="group",
                        )

                        pending_groups.extend(
                            (child, group_parts)
                            for child in group
                            if tag(child) == "group" and child.get("id")
                        )
                        for field_node in (
                            child
                            for child in group
                            if tag(child) == "field" and child.get("id")
                        ):
                            declared_path = "/".join(
                                (*group_parts, field_node.get("id"))
                            )
                            effective_path = (
                                text_child(field_node, "config_path")
                                or declared_path
                            ).strip("/") or declared_path
                            self._system_config_sources.setdefault(
                                effective_path,
                                set(),
                            ).add(path)
                            field_packet = self.graph.packet(
                                "magento-system-config",
                                f"field:{declared_path}",
                                module=module_name,
                                configPath=effective_path,
                            )
                            field_packet.add(GraphFact(
                                "magento-system-config-field",
                                effective_path,
                                "declared-by-admin-field",
                                declared_path,
                                path,
                                line(content, field_node.get("id")),
                                attrs(
                                    module=module_name,
                                    order=order,
                                    fieldType=field_node.get("type", ""),
                                ),
                            ))
                            extension_relation(
                                field_packet,
                                field_node,
                                identity=declared_path,
                                source_path=path,
                                content=content,
                                element_kind="field",
                            )
                            model_relations(
                                field_packet,
                                field_node,
                                config_path=effective_path,
                                source_path=path,
                                content=content,
                                element_kind="field",
                            )
                            depends = next(
                                (
                                    child
                                    for child in field_node
                                    if tag(child) == "depends"
                                ),
                                None,
                            )
                            if depends is not None:
                                for dependency in (
                                    child
                                    for child in depends
                                    if tag(child) == "field"
                                    and child.get("id")
                                ):
                                    expected_value = (
                                        dependency.text or ""
                                    ).strip()
                                    field_packet.add(GraphFact(
                                        "magento-system-config-dependency",
                                        effective_path,
                                        "depends-on-config-field",
                                        dependency.get("id"),
                                        path,
                                        line(
                                            content,
                                            dependency.get("id"),
                                        ),
                                        attrs(
                                            expectedValue=expected_value,
                                            separator=dependency.get(
                                                "separator",
                                                "",
                                            ),
                                        ),
                                    ))
                            for (
                                default_path,
                                scope,
                                default_line,
                                default_module,
                                default_order,
                            ) in default_paths.get(effective_path, ()):
                                self._system_config_sources[
                                    effective_path
                                ].add(default_path)
                                field_packet.add(GraphFact(
                                    "magento-system-config-default",
                                    effective_path,
                                    "has-default-declaration",
                                    scope,
                                    default_path,
                                    default_line,
                                    attrs(
                                        module=default_module,
                                        order=default_order,
                                    ),
                                ), path)

        scope_config_types = {
            r"Magento\Framework\App\Config",
            r"Magento\Framework\App\Config\ScopeConfigInterface",
        }
        for symbol in sorted(self.symbols):
            for key, value in symbol.attributes:
                if not key.startswith(
                    "php-literal-instance-call-reference:"
                ):
                    continue
                try:
                    reference = json.loads(value)
                except (TypeError, json.JSONDecodeError) as exception:
                    raise ValueError(
                        "PHP literal-call metadata is invalid JSON"
                    ) from exception
                if not isinstance(reference, dict):
                    raise ValueError(
                        "PHP literal-call metadata must be an object"
                    )
                receiver_type = reference.get("target")
                method = reference.get("method")
                caller = reference.get("caller", "")
                call_line = reference.get("line")
                literal_arguments = reference.get(
                    "literalStringArguments"
                )
                receiver_resolution = reference.get(
                    "receiverResolution",
                    "",
                )
                if (
                    not isinstance(receiver_type, str)
                    or not isinstance(method, str)
                    or not isinstance(caller, str)
                    or not isinstance(call_line, int)
                    or call_line < 1
                    or not isinstance(literal_arguments, dict)
                    or not isinstance(receiver_resolution, str)
                    or any(
                        not isinstance(position, str)
                        or not isinstance(argument, str)
                        for position, argument
                        in literal_arguments.items()
                    )
                ):
                    raise ValueError(
                        "PHP literal-call metadata has invalid fields"
                    )
                if receiver_type.lstrip("\\") not in scope_config_types:
                    continue
                normalized_method = method.casefold()
                relation = {
                    "getvalue": "reads-config-value",
                    "issetflag": "checks-config-flag",
                }.get(normalized_method)
                if relation is None:
                    continue
                config_path = literal_arguments.get("0", "").strip("/")
                related_sources = self._system_config_sources.get(
                    config_path
                )
                if not config_path or not related_sources:
                    continue
                packet = self.graph.packet(
                    "magento-system-config",
                    (
                        f"consumer:{symbol.qualified_name}:"
                        f"{caller or '<class>'}:{method}:{config_path}"
                    ),
                    configPath=config_path,
                )
                packet.add(GraphFact(
                    "magento-system-config-consumer",
                    (
                        f"{symbol.qualified_name}::{caller}"
                        if caller
                        else symbol.qualified_name
                    ),
                    relation,
                    config_path,
                    symbol.path,
                    call_line,
                    attrs(
                        method=method,
                        receiverResolution=receiver_resolution,
                        receiverType=receiver_type.lstrip("\\"),
                    ),
                ), *sorted(related_sources))

    def _webapi_and_acl(self, modules: tuple[ModuleRecord, ...], states: dict[str, DiState]) -> None:
        acl_paths: dict[str, set[str]] = {}
        for path, module, order in self._ordered_configs("acl.xml", modules, "global"):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for resource in (node for node in root.iter() if tag(node) == "resource" and node.get("id")):
                acl_paths.setdefault(resource.get("id"), set()).add(path)
                parent = next(
                    (
                        candidate.get("id")
                        for candidate in root.iter()
                        if resource in list(candidate) and candidate.get("id")
                    ),
                    "",
                )
                packet = self.graph.packet("magento-acl", resource.get("id"))
                packet.add(GraphFact(
                    "magento-acl-resource",
                    resource.get("id"),
                    "child-of" if parent else "declared-in",
                    parent or path,
                    path,
                    line(content, resource.get("id")),
                    attrs(module=module.name if module else "application", title=resource.get("title", "")),
                ))

        rest_state = states.get("webapi_rest", states.get("global", DiState()))
        for path, module, order in self._ordered_configs("webapi.xml", modules, "global"):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for route in (node for node in root.iter() if tag(node) == "route" and node.get("url")):
                service = next((node for node in route if tag(node) == "service"), None)
                if service is None or not service.get("class") or not service.get("method"):
                    continue
                contract = service.get("class").lstrip("\\")
                implementation = self._resolve_type(contract, rest_state)
                key = f"{route.get('method', '')}:{route.get('url')}"
                packet = self.graph.packet("magento-webapi", key, method=route.get("method", ""), url=route.get("url"))
                packet.add(GraphFact(
                    "magento-webapi-route",
                    f"{route.get('method', '')} {route.get('url')}",
                    "invokes",
                    f"{contract}::{service.get('method')}",
                    path,
                    line(content, route.get("url")),
                    attrs(
                        implementation=implementation,
                        module=module.name if module else "application",
                        secure=route.get("secure", ""),
                    ),
                ), self._symbol_path(contract), self._symbol_path(implementation))
                for resource in (node for node in route.iter() if tag(node) == "resource" and node.get("ref")):
                    packet.add(GraphFact(
                        "magento-webapi-acl",
                        key,
                        "requires-resource",
                        resource.get("ref"),
                        path,
                        line(content, resource.get("ref")),
                    ), *acl_paths.get(resource.get("ref"), set()))

    def _cron(self, modules: tuple[ModuleRecord, ...]) -> None:
        for path, module, order in self._ordered_configs("crontab.xml", modules, "global"):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for group in (node for node in root.iter() if tag(node) == "group" and node.get("id")):
                for job in (node for node in group if tag(node) == "job" and node.get("name")):
                    target = job.get("instance", "")
                    method = job.get("method", "execute")
                    schedule = next((node.text.strip() for node in job if tag(node) == "schedule" and node.text), "")
                    config_path = next((node.text.strip() for node in job if tag(node) == "config_path" and node.text), "")
                    packet = self.graph.packet("magento-cron", f"{group.get('id')}:{job.get('name')}")
                    packet.add(GraphFact(
                        "magento-cron-job",
                        job.get("name"),
                        "invokes",
                        f"{target}::{method}",
                        path,
                        line(content, job.get("name")),
                        attrs(group=group.get("id"), schedule=schedule, configPath=config_path),
                    ), self._symbol_path(target))

    def _add_message_consumer_resolution(
        self,
        packet,
        *,
        topic: str,
        topic_record: dict[str, object],
        consumer: str,
        consumer_record: dict[str, object],
        destination: str,
        exchange: str,
        publisher_connection: str,
        topology_connection: str,
        consumer_exact: bool,
        route_paths: tuple[str, ...],
    ) -> None:
        """Attach one queue consumer and its runtime-selected callbacks."""
        consumer_attributes = consumer_record["attributes"]
        consumer_connection = str(
            consumer_attributes.get(
                "connection",
                _DEPLOYMENT_DEFAULT_CONNECTION,
            )
        )
        handler = str(consumer_attributes.get("handler", ""))
        configured_consumer_instance = str(
            consumer_attributes.get("consumerInstance", "")
        ).strip()
        consumer_instance = (
            configured_consumer_instance
            or _DEFAULT_MESSAGE_CONSUMER
        )
        consumer_implementation_resolved = (
            not configured_consumer_instance
            or consumer_instance
            in {
                *_BUILTIN_MESSAGE_CONSUMERS,
                _MASS_MESSAGE_CONSUMER,
            }
        )
        packet.add(GraphFact(
            (
                "magento-message-effective-consumer"
                if consumer_exact
                else "magento-message-consumer-candidate"
            ),
            topic,
            (
                "handled-by-consumer"
                if consumer_exact
                else "may-be-handled-by-consumer"
            ),
            consumer,
            str(consumer_record["path"]),
            int(consumer_record["line"]),
            attrs(
                queue=destination,
                exchange=exchange,
                publisherConnection=publisher_connection,
                topologyConnection=topology_connection,
                consumerConnection=consumer_connection,
                connectionResolved=consumer_exact,
                handler=handler,
                consumerInstance=consumer_instance,
                consumerImplementationResolved=(
                    consumer_implementation_resolved
                ),
            ),
        ), self._symbol_path(consumer_instance),
            self._symbol_path(
                handler.split("::", 1)[0] if handler else ""
            ), *sorted(topic_record["paths"]),
            *route_paths,
            *sorted(consumer_record["paths"]))

        communication_handlers = []
        for handler_name, handler_record in sorted(
            topic_record.get("handlers", {}).items()
        ):
            handler_attributes = handler_record["attributes"]
            if not _enabled(handler_attributes.get("disabled")):
                continue
            target_type = str(
                handler_attributes.get("type", "")
            ).strip()
            target_method = str(
                handler_attributes.get("method", "")
            ).strip()
            communication_handlers.append({
                "target": (
                    f"{target_type}::{target_method}"
                    if target_type and target_method
                    else (target_type or str(handler_name))
                ),
                "type": target_type,
                "method": target_method,
                "source": "communication",
                "selection": (
                    "additive"
                    if consumer_instance == _MASS_MESSAGE_CONSUMER
                    else (
                        "fallback"
                        if consumer_implementation_resolved
                        else "implementation-defined"
                    )
                ),
                "record": handler_record,
            })

        queue_handlers = []
        if handler.strip():
            target_type, separator, target_method = (
                handler.strip().partition("::")
            )
            queue_handlers.append({
                "target": handler.strip(),
                "type": target_type.strip(),
                "method": (
                    target_method.strip()
                    if separator
                    else ""
                ),
                "source": "queue-consumer",
                "selection": (
                    "additive"
                    if consumer_instance == _MASS_MESSAGE_CONSUMER
                    else (
                        "override"
                        if consumer_implementation_resolved
                        else "implementation-defined"
                    )
                ),
                "record": consumer_record,
            })

        if consumer_instance == _MASS_MESSAGE_CONSUMER:
            selected_handlers = [
                *communication_handlers,
                *queue_handlers,
            ]
        elif consumer_instance in _BUILTIN_MESSAGE_CONSUMERS:
            selected_handlers = (
                queue_handlers
                if queue_handlers
                else communication_handlers
            )
        else:
            # Adobe's contract delegates handler semantics to an explicit
            # custom consumer implementation. Preserve every configured source
            # as a candidate without inventing which callback the class invokes.
            selected_handlers = [
                *communication_handlers,
                *queue_handlers,
            ]

        handler_exact = (
            consumer_exact
            and consumer_implementation_resolved
        )
        for selected_handler in selected_handlers:
            handler_record = selected_handler["record"]
            handler_valid = bool(
                selected_handler["type"]
                and selected_handler["method"]
            )
            effective = handler_exact and handler_valid
            packet.add(GraphFact(
                (
                    "magento-message-effective-handler"
                    if effective
                    else (
                        "magento-message-handler-unresolved"
                        if handler_exact
                        else "magento-message-handler-candidate"
                    )
                ),
                topic,
                (
                    "handled-by"
                    if effective
                    else (
                        "has-invalid-handler"
                        if handler_exact
                        else "may-be-handled-by"
                    )
                ),
                str(selected_handler["target"]),
                str(handler_record["path"]),
                int(handler_record["line"]),
                attrs(
                    consumer=consumer,
                    consumerInstance=consumer_instance,
                    consumerImplementationResolved=(
                        consumer_implementation_resolved
                    ),
                    connectionResolved=consumer_exact,
                    handlerSource=selected_handler["source"],
                    handlerSelection=selected_handler[
                        "selection"
                    ],
                    handlerValid=handler_valid,
                ),
            ), self._symbol_path(
                str(selected_handler["type"])
            ), self._symbol_path(consumer_instance),
                *sorted(topic_record["paths"]),
                *route_paths,
                *sorted(consumer_record["paths"]),
                *sorted(handler_record["paths"]))

        if handler_exact and not selected_handlers:
            packet.add(GraphFact(
                "magento-message-handler-unresolved",
                topic,
                "has-no-configured-handler",
                consumer,
                str(consumer_record["path"]),
                int(consumer_record["line"]),
                attrs(
                    consumer=consumer,
                    consumerInstance=consumer_instance,
                    consumerImplementationResolved=True,
                    connectionResolved=True,
                    reason="no-enabled-handler",
                ),
            ), self._symbol_path(consumer_instance),
                *sorted(topic_record["paths"]),
                *route_paths,
                *sorted(consumer_record["paths"]))

        if not consumer_implementation_resolved:
            packet.add(GraphFact(
                "magento-message-handler-resolution-dependent",
                topic,
                "handler-use-defined-by",
                consumer_instance,
                str(consumer_record["path"]),
                int(consumer_record["line"]),
                attrs(
                    consumer=consumer,
                    connectionResolved=consumer_exact,
                    configuredHandlerCount=len(selected_handlers),
                ),
            ), self._symbol_path(consumer_instance),
                *sorted(topic_record["paths"]),
                *route_paths,
                *sorted(consumer_record["paths"]))

    def _message_queues(self, modules: tuple[ModuleRecord, ...]) -> None:
        topics: dict[str, dict[str, object]] = {}
        consumers: dict[str, dict[str, object]] = {}
        publishers: dict[str, dict[str, object]] = {}
        exchanges: dict[tuple[str, str], dict[str, object]] = {}

        def merge_record(
            values: dict,
            key,
            element,
            path: str,
            module: ModuleRecord | None,
            order: int,
        ) -> dict[str, object]:
            record = values.setdefault(
                key,
                {
                    "attributes": {},
                    "path": path,
                    "line": 1,
                    "module": module.name if module else "application",
                    "order": order,
                    "paths": set(),
                },
            )
            record["attributes"].update({
                name.rsplit("}", 1)[-1]: value
                for name, value in element.attrib.items()
            })
            record["path"] = path
            record["line"] = line(
                self.artifacts[path],
                str(
                    element.get("name")
                    or element.get("topic")
                    or element.get("id")
                    or key
                ),
            )
            record["module"] = module.name if module else "application"
            record["order"] = order
            record["paths"].add(path)
            return record

        for path, module, order in self._ordered_configs(
            "communication.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            for topic_node in (
                node for node in root.iter()
                if tag(node) == "topic" and node.get("name")
            ):
                topic = topic_node.get("name")
                record = merge_record(
                    topics,
                    topic,
                    topic_node,
                    path,
                    module,
                    order,
                )
                handlers = record.setdefault("handlers", {})
                for handler_node in (
                    node for node in topic_node
                    if tag(node) == "handler"
                ):
                    target = handler_node.get("type", "")
                    method = handler_node.get("method", "")
                    handler_key = (
                        handler_node.get("name")
                        or f"{target}::{method}"
                    )
                    if not handler_key:
                        continue
                    merge_record(
                        handlers,
                        handler_key,
                        handler_node,
                        path,
                        module,
                        order,
                    )

        for path, module, order in self._ordered_configs(
            "queue_consumer.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            for consumer_node in (
                node for node in root.iter()
                if tag(node) == "consumer" and node.get("name")
            ):
                merge_record(
                    consumers,
                    consumer_node.get("name"),
                    consumer_node,
                    path,
                    module,
                    order,
                )

        for path, module, order in self._ordered_configs(
            "queue_publisher.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            for publisher_node in (
                node for node in root.iter()
                if tag(node) == "publisher" and node.get("topic")
            ):
                topic = publisher_node.get("topic")
                record = merge_record(
                    publishers,
                    topic,
                    publisher_node,
                    path,
                    module,
                    order,
                )
                connections = record.setdefault("connections", {})
                for connection_node in (
                    node for node in publisher_node
                    if tag(node) == "connection"
                ):
                    connection_name = (
                        connection_node.get("name")
                        or _DEPLOYMENT_DEFAULT_CONNECTION
                    )
                    merge_record(
                        connections,
                        connection_name,
                        connection_node,
                        path,
                        module,
                        order,
                    )

        for path, module, order in self._ordered_configs(
            "queue_topology.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            for exchange_node in (
                node for node in root.iter()
                if tag(node) == "exchange"
                and node.get("name") is not None
            ):
                connection = (
                    exchange_node.get("connection")
                    or _DEPLOYMENT_DEFAULT_CONNECTION
                )
                exchange_key = (exchange_node.get("name"), connection)
                record = merge_record(
                    exchanges,
                    exchange_key,
                    exchange_node,
                    path,
                    module,
                    order,
                )
                bindings = record.setdefault("bindings", {})
                for binding_node in (
                    node for node in exchange_node
                    if tag(node) == "binding"
                    and node.get("topic")
                    and node.get("destination")
                ):
                    binding_key = (
                        binding_node.get("destinationType", "queue"),
                        binding_node.get("destination"),
                        binding_node.get("topic"),
                    )
                    merge_record(
                        bindings,
                        binding_key,
                        binding_node,
                        path,
                        module,
                        order,
                    )

        for topic, record in sorted(topics.items()):
            topic_attributes = record["attributes"]
            packet = self.graph.packet("magento-message-queue", topic)
            packet.add(GraphFact(
                "magento-message-topic",
                topic,
                "declared-in",
                str(record["path"]),
                str(record["path"]),
                int(record["line"]),
                attrs(
                    request=topic_attributes.get("request", ""),
                    response=topic_attributes.get("response", ""),
                    schema=topic_attributes.get("schema", ""),
                    module=record["module"],
                    order=record["order"],
                ),
            ), *sorted(record["paths"]))
            for handler_name, handler in sorted(
                record.get("handlers", {}).items()
            ):
                handler_attributes = handler["attributes"]
                target = str(handler_attributes.get("type", ""))
                method = str(handler_attributes.get("method", ""))
                disabled = not _enabled(
                    handler_attributes.get("disabled"),
                )
                packet.add(GraphFact(
                    "magento-message-handler",
                    topic,
                    (
                        "disables-handler"
                        if disabled
                        else "handled-by"
                    ),
                    (
                        f"{target}::{method}"
                        if target and method
                        else (target or handler_name)
                    ),
                    str(handler["path"]),
                    int(handler["line"]),
                    attrs(
                        name=handler_name,
                        module=handler["module"],
                        order=handler["order"],
                    ),
                ), self._symbol_path(target), *sorted(handler["paths"]))

        for consumer, record in sorted(consumers.items()):
            consumer_attributes = record["attributes"]
            queue = str(consumer_attributes.get("queue", consumer))
            handler = str(consumer_attributes.get("handler", ""))
            connection = str(
                consumer_attributes.get(
                    "connection",
                    _DEPLOYMENT_DEFAULT_CONNECTION,
                )
            )
            packet = self.graph.packet(
                "magento-message-consumer",
                consumer,
            )
            if not queue.strip():
                packet.add(GraphFact(
                    "magento-message-consumer-invalid",
                    consumer,
                    "has-empty-queue",
                    consumer,
                    str(record["path"]),
                    int(record["line"]),
                    attrs(
                        handler=handler,
                        connection=connection,
                        module=record["module"],
                        order=record["order"],
                        semanticRole="diagnostic",
                    ),
                ), self._symbol_path(
                    handler.split("::", 1)[0] if handler else ""
                ), *sorted(record["paths"]))
                continue
            packet.add(GraphFact(
                "magento-message-consumer",
                consumer,
                "consumes-queue",
                queue,
                str(record["path"]),
                int(record["line"]),
                attrs(
                    handler=handler,
                    connection=connection,
                    connectionResolved=(
                        connection != _DEPLOYMENT_DEFAULT_CONNECTION
                    ),
                    consumerInstance=consumer_attributes.get(
                        "consumerInstance",
                        "",
                    ),
                    module=record["module"],
                    order=record["order"],
                ),
            ), self._symbol_path(
                handler.split("::", 1)[0] if handler else ""
            ), *sorted(record["paths"]))

        for (exchange_name, connection), record in sorted(exchanges.items()):
            for binding_key, binding in sorted(
                record.get("bindings", {}).items()
            ):
                binding_attributes = binding["attributes"]
                topic_pattern = str(binding_attributes.get("topic", ""))
                destination = str(
                    binding_attributes.get("destination", "")
                )
                packet = self.graph.packet(
                    "magento-message-queue",
                    topic_pattern,
                )
                packet.add(GraphFact(
                    "magento-message-binding",
                    topic_pattern,
                    (
                        "disables-route-to"
                        if not _enabled(binding_attributes.get("disabled"))
                        else "routes-to"
                    ),
                    destination,
                    str(binding["path"]),
                    int(binding["line"]),
                    attrs(
                        exchange=(
                            exchange_name
                            or _BROKER_DEFAULT_EXCHANGE
                        ),
                        connection=connection,
                        connectionResolved=(
                            connection != _DEPLOYMENT_DEFAULT_CONNECTION
                        ),
                        destinationType=binding_key[0],
                        module=binding["module"],
                        order=binding["order"],
                    ),
                ), *sorted(record["paths"]), *sorted(binding["paths"]))

        publisher_resolved_consumers: set[tuple[str, str]] = set()
        for topic, topic_record in sorted(topics.items()):
            publisher = publishers.get(topic)
            packet = self.graph.packet("magento-message-queue", topic)
            if publisher is None:
                packet.add(GraphFact(
                    "magento-message-route-unresolved",
                    topic,
                    "has-no-publisher",
                    topic,
                    str(topic_record["path"]),
                    int(topic_record["line"]),
                    attrs(reason="publisher-configuration-absent"),
                ), *sorted(topic_record["paths"]))
                continue

            publisher_attributes = publisher["attributes"]
            if not _enabled(publisher_attributes.get("disabled")):
                packet.add(GraphFact(
                    "magento-message-publisher",
                    topic,
                    "publisher-disabled",
                    topic,
                    str(publisher["path"]),
                    int(publisher["line"]),
                    attrs(
                        module=publisher["module"],
                        order=publisher["order"],
                    ),
                ), *sorted(publisher["paths"]))
                continue

            connections = publisher.get("connections", {})
            if not connections:
                connections[_DEPLOYMENT_DEFAULT_CONNECTION] = {
                    "attributes": {
                        "name": _DEPLOYMENT_DEFAULT_CONNECTION,
                        "exchange": "magento",
                    },
                    "path": publisher["path"],
                    "line": publisher["line"],
                    "module": publisher["module"],
                    "order": publisher["order"],
                    "paths": set(publisher["paths"]),
                }
            active_connection = next(
                (
                    connection
                    for connection in connections.values()
                    if _enabled(connection["attributes"].get("disabled"))
                ),
                None,
            )
            # Magento adds its deployment default when every configured
            # connection is disabled.
            if active_connection is None:
                active_connection = {
                    "attributes": {
                        "name": _DEPLOYMENT_DEFAULT_CONNECTION,
                        "exchange": "magento",
                    },
                    "path": publisher["path"],
                    "line": publisher["line"],
                    "module": publisher["module"],
                    "order": publisher["order"],
                    "paths": set(publisher["paths"]),
                }

            connection_attributes = active_connection["attributes"]
            connection = str(
                connection_attributes.get(
                    "name",
                    _DEPLOYMENT_DEFAULT_CONNECTION,
                )
            )
            exchange_name = str(
                connection_attributes.get("exchange", "magento")
            )
            display_exchange = (
                exchange_name or _BROKER_DEFAULT_EXCHANGE
            )
            connection_resolved = (
                connection != _DEPLOYMENT_DEFAULT_CONNECTION
            )
            packet.add(GraphFact(
                "magento-message-publisher",
                topic,
                "publishes-through",
                display_exchange,
                str(active_connection["path"]),
                int(active_connection["line"]),
                attrs(
                    connection=connection,
                    connectionResolved=connection_resolved,
                    module=active_connection["module"],
                    order=active_connection["order"],
                ),
            ), *sorted(publisher["paths"]), *sorted(
                active_connection["paths"]
            ))

            exchange_candidates = [
                (key, record)
                for key, record in exchanges.items()
                if key[0] == exchange_name
                and (
                    key[1] == connection
                    or key[1] == _DEPLOYMENT_DEFAULT_CONNECTION
                    or connection == _DEPLOYMENT_DEFAULT_CONNECTION
                )
            ]
            routed = False
            for (candidate_exchange, candidate_connection), exchange in (
                exchange_candidates
            ):
                binding = next(
                    (
                        candidate
                        for candidate in exchange.get(
                            "bindings",
                            {},
                        ).values()
                        if _enabled(
                            candidate["attributes"].get("disabled")
                        )
                        and _message_topic_matches(
                            str(
                                candidate["attributes"].get(
                                    "topic",
                                    "",
                                )
                            ),
                            topic,
                        )
                    ),
                    None,
                )
                if binding is None:
                    continue
                routed = True
                destination = str(
                    binding["attributes"].get("destination", "")
                )
                exact_connection = (
                    connection_resolved
                    and candidate_connection == connection
                ) or (
                    connection == _DEPLOYMENT_DEFAULT_CONNECTION
                    and candidate_connection
                    == _DEPLOYMENT_DEFAULT_CONNECTION
                )
                route_kind = (
                    "magento-message-effective-route"
                    if exact_connection
                    else "magento-message-route-candidate"
                )
                route_relation = (
                    "routes-to-queue"
                    if exact_connection
                    else "may-route-to-queue"
                )
                packet.add(GraphFact(
                    route_kind,
                    topic,
                    route_relation,
                    destination,
                    str(binding["path"]),
                    int(binding["line"]),
                    attrs(
                        exchange=(
                            candidate_exchange
                            or _BROKER_DEFAULT_EXCHANGE
                        ),
                        publisherConnection=connection,
                        topologyConnection=candidate_connection,
                        connectionResolved=exact_connection,
                    ),
                ), *sorted(topic_record["paths"]),
                    *sorted(publisher["paths"]),
                    *sorted(exchange["paths"]),
                    *sorted(binding["paths"]))

                for consumer, consumer_record in sorted(consumers.items()):
                    consumer_attributes = consumer_record["attributes"]
                    consumer_queue = str(
                        consumer_attributes.get("queue", consumer)
                    )
                    if consumer_queue != destination:
                        continue
                    consumer_connection = str(
                        consumer_attributes.get(
                            "connection",
                            _DEPLOYMENT_DEFAULT_CONNECTION,
                        )
                    )
                    consumer_exact = exact_connection and (
                        consumer_connection == candidate_connection
                        or (
                            consumer_connection
                            == _DEPLOYMENT_DEFAULT_CONNECTION
                            and candidate_connection
                            == _DEPLOYMENT_DEFAULT_CONNECTION
                        )
                    )
                    if consumer_exact:
                        publisher_resolved_consumers.add((topic, consumer))
                    handler = str(
                        consumer_attributes.get("handler", "")
                    )
                    configured_consumer_instance = str(
                        consumer_attributes.get(
                            "consumerInstance",
                            "",
                        )
                    ).strip()
                    consumer_instance = (
                        configured_consumer_instance
                        or _DEFAULT_MESSAGE_CONSUMER
                    )
                    consumer_implementation_resolved = (
                        not configured_consumer_instance
                        or consumer_instance
                        in {
                            *_BUILTIN_MESSAGE_CONSUMERS,
                            _MASS_MESSAGE_CONSUMER,
                        }
                    )
                    packet.add(GraphFact(
                        (
                            "magento-message-effective-consumer"
                            if consumer_exact
                            else "magento-message-consumer-candidate"
                        ),
                        topic,
                        (
                            "handled-by-consumer"
                            if consumer_exact
                            else "may-be-handled-by-consumer"
                        ),
                        consumer,
                        str(consumer_record["path"]),
                        int(consumer_record["line"]),
                        attrs(
                            queue=destination,
                            exchange=candidate_exchange,
                            publisherConnection=connection,
                            topologyConnection=candidate_connection,
                            consumerConnection=consumer_connection,
                            connectionResolved=consumer_exact,
                            handler=handler,
                            consumerInstance=consumer_instance,
                            consumerImplementationResolved=(
                                consumer_implementation_resolved
                            ),
                        ),
                    ), self._symbol_path(consumer_instance),
                        self._symbol_path(
                            handler.split("::", 1)[0] if handler else ""
                        ), *sorted(topic_record["paths"]),
                        *sorted(publisher["paths"]),
                        *sorted(exchange["paths"]),
                        *sorted(binding["paths"]),
                        *sorted(consumer_record["paths"]))

                    communication_handlers = []
                    for handler_name, handler_record in sorted(
                        topic_record.get("handlers", {}).items()
                    ):
                        handler_attributes = handler_record["attributes"]
                        if not _enabled(
                            handler_attributes.get("disabled")
                        ):
                            continue
                        target_type = str(
                            handler_attributes.get("type", "")
                        ).strip()
                        target_method = str(
                            handler_attributes.get("method", "")
                        ).strip()
                        communication_handlers.append({
                            "target": (
                                f"{target_type}::{target_method}"
                                if target_type and target_method
                                else (
                                    target_type
                                    or str(handler_name)
                                )
                            ),
                            "type": target_type,
                            "method": target_method,
                            "source": "communication",
                            "selection": (
                                "additive"
                                if consumer_instance
                                == _MASS_MESSAGE_CONSUMER
                                else (
                                    "fallback"
                                    if consumer_implementation_resolved
                                    else "implementation-defined"
                                )
                            ),
                            "record": handler_record,
                        })

                    queue_handlers = []
                    if handler.strip():
                        target_type, separator, target_method = (
                            handler.strip().partition("::")
                        )
                        queue_handlers.append({
                            "target": handler.strip(),
                            "type": target_type.strip(),
                            "method": (
                                target_method.strip()
                                if separator
                                else ""
                            ),
                            "source": "queue-consumer",
                            "selection": (
                                "additive"
                                if consumer_instance
                                == _MASS_MESSAGE_CONSUMER
                                else (
                                    "override"
                                    if consumer_implementation_resolved
                                    else "implementation-defined"
                                )
                            ),
                            "record": consumer_record,
                        })

                    if consumer_instance == _MASS_MESSAGE_CONSUMER:
                        selected_handlers = [
                            *communication_handlers,
                            *queue_handlers,
                        ]
                    elif consumer_instance in _BUILTIN_MESSAGE_CONSUMERS:
                        selected_handlers = (
                            queue_handlers
                            if queue_handlers
                            else communication_handlers
                        )
                    else:
                        # Adobe's contract delegates handler semantics to an
                        # explicit custom consumer implementation. Preserve
                        # every configured source as a candidate without
                        # inventing which callback the class invokes.
                        selected_handlers = [
                            *communication_handlers,
                            *queue_handlers,
                        ]

                    handler_exact = (
                        consumer_exact
                        and consumer_implementation_resolved
                    )
                    for selected_handler in selected_handlers:
                        handler_record = selected_handler["record"]
                        handler_valid = bool(
                            selected_handler["type"]
                            and selected_handler["method"]
                        )
                        effective = handler_exact and handler_valid
                        packet.add(GraphFact(
                            (
                                "magento-message-effective-handler"
                                if effective
                                else (
                                    "magento-message-handler-unresolved"
                                    if handler_exact
                                    else "magento-message-handler-candidate"
                                )
                            ),
                            topic,
                            (
                                "handled-by"
                                if effective
                                else (
                                    "has-invalid-handler"
                                    if handler_exact
                                    else "may-be-handled-by"
                                )
                            ),
                            str(selected_handler["target"]),
                            str(handler_record["path"]),
                            int(handler_record["line"]),
                            attrs(
                                consumer=consumer,
                                consumerInstance=consumer_instance,
                                consumerImplementationResolved=(
                                    consumer_implementation_resolved
                                ),
                                connectionResolved=consumer_exact,
                                handlerSource=selected_handler["source"],
                                handlerSelection=selected_handler[
                                    "selection"
                                ],
                                handlerValid=handler_valid,
                            ),
                        ), self._symbol_path(
                            str(selected_handler["type"])
                        ), self._symbol_path(consumer_instance),
                            *sorted(topic_record["paths"]),
                            *sorted(publisher["paths"]),
                            *sorted(exchange["paths"]),
                            *sorted(binding["paths"]),
                            *sorted(consumer_record["paths"]),
                            *sorted(handler_record["paths"]))

                    if (
                        handler_exact
                        and not selected_handlers
                    ):
                        packet.add(GraphFact(
                            "magento-message-handler-unresolved",
                            topic,
                            "has-no-configured-handler",
                            consumer,
                            str(consumer_record["path"]),
                            int(consumer_record["line"]),
                            attrs(
                                consumer=consumer,
                                consumerInstance=consumer_instance,
                                consumerImplementationResolved=True,
                                connectionResolved=True,
                                reason="no-enabled-handler",
                            ),
                        ), self._symbol_path(consumer_instance),
                            *sorted(topic_record["paths"]),
                            *sorted(publisher["paths"]),
                            *sorted(exchange["paths"]),
                            *sorted(binding["paths"]),
                            *sorted(consumer_record["paths"]))

                    if not consumer_implementation_resolved:
                        packet.add(GraphFact(
                            "magento-message-handler-resolution-dependent",
                            topic,
                            "handler-use-defined-by",
                            consumer_instance,
                            str(consumer_record["path"]),
                            int(consumer_record["line"]),
                            attrs(
                                consumer=consumer,
                                connectionResolved=consumer_exact,
                                configuredHandlerCount=len(
                                    selected_handlers
                                ),
                            ),
                        ), self._symbol_path(consumer_instance),
                            *sorted(topic_record["paths"]),
                            *sorted(consumer_record["paths"]))
                # QueueResolver returns the first enabled matching binding.
                break

            if not routed:
                packet.add(GraphFact(
                    "magento-message-route-unresolved",
                    topic,
                    "has-no-matching-binding",
                    display_exchange,
                    str(active_connection["path"]),
                    int(active_connection["line"]),
                    attrs(
                        connection=connection,
                        connectionResolved=connection_resolved,
                        reason="topology-binding-absent-or-disabled",
                    ),
                ), *sorted(topic_record["paths"]),
                    *sorted(publisher["paths"]),
                    *sorted(active_connection["paths"]))

        # A consumer does not require a local publisher. Adobe explicitly
        # supports topology + consumer configuration for queues populated by a
        # third-party system. Resolve that inbound route directly from the
        # enabled binding, queue name and connection instead of treating the
        # absent local publisher as missing handler evidence.
        inbound_routes: dict[
            tuple[str, str],
            list[dict[str, object]],
        ] = {}
        for (
            (exchange_name, topology_connection),
            exchange_record,
        ) in sorted(exchanges.items()):
            for binding_key, binding in sorted(
                exchange_record.get("bindings", {}).items()
            ):
                binding_attributes = binding["attributes"]
                if (
                    binding_key[0] != "queue"
                    or not _enabled(
                        binding_attributes.get("disabled")
                    )
                ):
                    continue
                destination = str(
                    binding_attributes.get("destination", "")
                )
                topic_pattern = str(
                    binding_attributes.get("topic", "")
                )
                for consumer, consumer_record in sorted(
                    consumers.items()
                ):
                    consumer_attributes = consumer_record["attributes"]
                    consumer_queue = str(
                        consumer_attributes.get("queue", consumer)
                    )
                    if consumer_queue != destination:
                        continue
                    consumer_connection = str(
                        consumer_attributes.get(
                            "connection",
                            _DEPLOYMENT_DEFAULT_CONNECTION,
                        )
                    )
                    connection_exact = (
                        consumer_connection == topology_connection
                        and (
                            consumer_connection
                            != _DEPLOYMENT_DEFAULT_CONNECTION
                            or topology_connection
                            == _DEPLOYMENT_DEFAULT_CONNECTION
                        )
                    )
                    for topic, topic_record in sorted(topics.items()):
                        if not _message_topic_matches(
                            topic_pattern,
                            topic,
                        ):
                            continue
                        inbound_routes.setdefault(
                            (topic, consumer),
                            [],
                        ).append({
                            "connectionExact": connection_exact,
                            "destination": destination,
                            "exchange": exchange_name,
                            "topologyConnection": topology_connection,
                            "topicRecord": topic_record,
                            "consumerRecord": consumer_record,
                            "paths": tuple(sorted({
                                *exchange_record["paths"],
                                *binding["paths"],
                            })),
                        })

        for (topic, consumer), routes in sorted(
            inbound_routes.items()
        ):
            if (topic, consumer) in publisher_resolved_consumers:
                continue
            ordered_routes = sorted(
                routes,
                key=lambda route: (
                    not bool(route["connectionExact"]),
                    str(route["exchange"]),
                    str(route["topologyConnection"]),
                    tuple(route["paths"]),
                ),
            )
            route = ordered_routes[0]
            packet = self.graph.packet(
                "magento-message-queue",
                topic,
            )
            self._add_message_consumer_resolution(
                packet,
                topic=topic,
                topic_record=route["topicRecord"],
                consumer=consumer,
                consumer_record=route["consumerRecord"],
                destination=str(route["destination"]),
                        exchange=(
                            str(route["exchange"])
                            or _BROKER_DEFAULT_EXCHANGE
                        ),
                publisher_connection="",
                topology_connection=str(
                    route["topologyConnection"]
                ),
                consumer_exact=bool(route["connectionExact"]),
                route_paths=route["paths"],
            )

    def _indexers_and_materialized_views(
        self,
        modules: tuple[ModuleRecord, ...],
    ) -> None:
        indexers: dict[str, dict[str, object]] = {}
        for path, module, order in self._ordered_configs("indexer.xml", modules, "global"):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for node in (
                item for item in root
                if tag(item) == "indexer" and item.get("id")
            ):
                identifier = node.get("id")
                prior = indexers.get(identifier, {"attributes": {}, "dependencies": set()})
                merged_attributes = dict(prior["attributes"])
                merged_attributes.update(node.attrib)
                dependencies = set(prior["dependencies"])
                dependencies.update(
                    child.get("id")
                    for container in node
                    if tag(container) == "dependencies"
                    for child in container
                    if tag(child) == "indexer" and child.get("id")
                )
                indexers[identifier] = {
                    "attributes": merged_attributes,
                    "dependencies": dependencies,
                    "path": path,
                    "line": line(content, identifier),
                    "module": module.name if module else "application",
                    "order": order,
                }

        views: dict[str, dict[str, object]] = {}
        for path, module, order in self._ordered_configs("mview.xml", modules, "global"):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for node in (
                item for item in root
                if tag(item) == "view" and item.get("id")
            ):
                identifier = node.get("id")
                prior = views.get(identifier, {"attributes": {}, "tables": {}})
                merged_attributes = dict(prior["attributes"])
                merged_attributes.update(node.attrib)
                tables = dict(prior["tables"])
                for table in (
                    child for container in node
                    if tag(container) == "subscriptions"
                    for child in container
                    if tag(child) == "table" and child.get("name")
                ):
                    table_key = (table.get("name"), table.get("entity_column", ""))
                    table_attributes = dict(tables.get(table_key, {}))
                    table_attributes.update(table.attrib)
                    tables[table_key] = table_attributes
                views[identifier] = {
                    "attributes": merged_attributes,
                    "tables": tables,
                    "path": path,
                    "line": line(content, identifier),
                    "module": module.name if module else "application",
                    "order": order,
                }

        schema_paths: dict[str, set[str]] = {}
        for path, _, _ in self._ordered_configs("db_schema.xml", modules, "global"):
            root = self._xml(path)
            if root is None:
                continue
            for table in (
                node for node in root
                if tag(node) == "table" and node.get("name")
            ):
                schema_paths.setdefault(table.get("name"), set()).add(path)

        for identifier, record in sorted(indexers.items()):
            attributes = record["attributes"]
            packet = self.graph.packet(
                "magento-indexer",
                identifier,
                module=str(record["module"]),
            )
            implementation = str(attributes.get("class", ""))
            if implementation:
                packet.add(GraphFact(
                    "magento-indexer-class", identifier, "executes", implementation,
                    str(record["path"]), int(record["line"]),
                    attrs(module=record["module"], order=record["order"]),
                ), self._symbol_path(implementation))
            view_id = str(attributes.get("view_id", ""))
            if view_id:
                packet.add(GraphFact(
                    "magento-indexer-view", identifier, "materializes-through", view_id,
                    str(record["path"]), int(record["line"]),
                ), str(views.get(view_id, {}).get("path", "")))
            shared_index = str(attributes.get("shared_index", ""))
            if shared_index:
                packet.add(GraphFact(
                    "magento-indexer-shared-index", identifier, "shares-index", shared_index,
                    str(record["path"]), int(record["line"]),
                ))
            for dependency in sorted(record["dependencies"]):
                packet.add(GraphFact(
                    "magento-indexer-dependency", identifier, "depends-on-indexer", dependency,
                    str(record["path"]), int(record["line"]),
                ), str(indexers.get(dependency, {}).get("path", "")))

        for identifier, record in sorted(views.items()):
            attributes = record["attributes"]
            packet = self.graph.packet(
                "magento-materialized-view",
                identifier,
                module=str(record["module"]),
                group=str(attributes.get("group", "")),
            )
            implementation = str(attributes.get("class", ""))
            if implementation:
                packet.add(GraphFact(
                    "magento-mview-class", identifier, "executes", implementation,
                    str(record["path"]), int(record["line"]),
                ), self._symbol_path(implementation))
            walker = str(attributes.get(
                "walker", "Magento\\Framework\\Mview\\View\\ChangeLogBatchWalker"
            ))
            packet.add(GraphFact(
                "magento-mview-walker", identifier, "uses-walker", walker,
                str(record["path"]), int(record["line"]),
            ), self._symbol_path(walker))
            for (table_name, entity_column), table_attributes in sorted(record["tables"].items()):
                packet.add(GraphFact(
                    "magento-mview-subscription",
                    identifier,
                    "subscribes-to-table",
                    table_name,
                    str(record["path"]),
                    int(record["line"]),
                    attrs(
                        entityColumn=entity_column,
                        processor=table_attributes.get("processor", ""),
                        subscriptionModel=table_attributes.get("subscription_model", ""),
                    ),
                ), *sorted(schema_paths.get(table_name, ())))

    def _schema(self, modules: tuple[ModuleRecord, ...]) -> None:
        whitelist: dict[str, object] = {}

        def merge_mapping(
            destination: dict[str, object],
            source: dict[str, object],
        ) -> None:
            for key, value in source.items():
                current = destination.get(key)
                if isinstance(current, dict) and isinstance(value, dict):
                    merge_mapping(current, value)
                else:
                    destination[key] = value

        for path, content in sorted(self.artifacts.items()):
            if PurePosixPath(path).name != "db_schema_whitelist.json":
                continue
            module = self._module_for_path(path, modules)
            if module is None or not module.enabled:
                continue
            try:
                document = json.loads(content)
            except json.JSONDecodeError as exception:
                self._diagnostics.append(PluginDiagnostic(
                    "magento-invalid-schema-whitelist",
                    f"{path}: {exception}",
                    self.plugin_id,
                ))
                continue
            if not isinstance(document, dict):
                self._diagnostics.append(PluginDiagnostic(
                    "magento-invalid-schema-whitelist",
                    f"{path}: top-level value must be an object",
                    self.plugin_id,
                ))
                continue
            merge_mapping(whitelist, document)

        tables: dict[str, dict[str, object]] = {}
        for path, module, order in self._ordered_configs(
            "db_schema.xml",
            modules,
            "global",
        ):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for table in (
                node for node in root
                if tag(node) == "table" and node.get("name")
            ):
                table_name = table.get("name")
                record = tables.setdefault(table_name, {
                    "attributes": {},
                    "path": path,
                    "line": line(content, table_name),
                    "module": module.name if module else "application",
                    "order": order,
                    "paths": set(),
                    "children": {
                        "column": {},
                        "index": {},
                        "constraint": {},
                    },
                })
                record["attributes"].update({
                    key.rsplit("}", 1)[-1]: value
                    for key, value in table.attrib.items()
                    if key != "name"
                })
                record.update({
                    "path": path,
                    "line": line(content, table_name),
                    "module": module.name if module else "application",
                    "order": order,
                })
                record["paths"].add(path)
                for child in table:
                    child_kind = tag(child)
                    if child_kind == "column":
                        identity = child.get("name", "")
                    elif child_kind in {"index", "constraint"}:
                        identity = child.get("referenceId", "")
                    else:
                        continue
                    if not identity:
                        continue
                    child_record = record["children"][child_kind].setdefault(
                        identity,
                        {
                            "attributes": {},
                            "path": path,
                            "line": line(content, identity),
                            "module": (
                                module.name if module else "application"
                            ),
                            "order": order,
                            "paths": set(),
                            "columns": {},
                        },
                    )
                    child_record["attributes"].update({
                        key.rsplit("}", 1)[-1]: value
                        for key, value in child.attrib.items()
                        if key not in {"name", "referenceId"}
                    })
                    child_record.update({
                        "path": path,
                        "line": line(content, identity),
                        "module": (
                            module.name if module else "application"
                        ),
                        "order": order,
                    })
                    child_record["paths"].add(path)
                    for column_ref in (
                        node for node in child
                        if tag(node) == "column" and node.get("name")
                    ):
                        child_record["columns"][column_ref.get("name")] = path

        def whitelist_contains(
            table_name: str,
            element_kind: str = "",
            identity: str = "",
        ) -> bool:
            table_whitelist = whitelist.get(table_name)
            if not isinstance(table_whitelist, dict):
                return False
            if not element_kind:
                return True
            group = table_whitelist.get(element_kind)
            return isinstance(group, dict) and identity in group

        disabled_tables = {
            table_name
            for table_name, record in tables.items()
            if not _enabled(record["attributes"].get("disabled"))
        }
        for table_name, record in sorted(tables.items()):
            packet = self.graph.packet("magento-database", table_name)
            table_attributes = record["attributes"]
            if table_name in disabled_tables:
                packet.add(GraphFact(
                    "magento-db-removal",
                    str(record["module"]),
                    "disables-table",
                    table_name,
                    str(record["path"]),
                    int(record["line"]),
                    attrs(
                        elementType="table",
                        whitelisted=str(
                            whitelist_contains(table_name)
                        ).casefold(),
                        destructiveOperationAllowed=str(
                            whitelist_contains(table_name)
                        ).casefold(),
                    ),
                ), *sorted(record["paths"]))
                continue

            packet.add(GraphFact(
                "magento-db-table",
                str(record["module"]),
                "declares-table",
                table_name,
                str(record["path"]),
                int(record["line"]),
                attrs(
                    resource=table_attributes.get("resource", ""),
                    engine=table_attributes.get("engine", ""),
                    order=record["order"],
                ),
            ), *sorted(record["paths"]))

            for child_kind in ("column", "index", "constraint"):
                children = record["children"][child_kind]
                for identity, child_record in sorted(children.items()):
                    child_attributes = child_record["attributes"]
                    if not _enabled(child_attributes.get("disabled")):
                        allowed = whitelist_contains(
                            table_name,
                            child_kind,
                            identity,
                        )
                        packet.add(GraphFact(
                            "magento-db-removal",
                            table_name,
                            f"disables-{child_kind}",
                            identity,
                            str(child_record["path"]),
                            int(child_record["line"]),
                            attrs(
                                elementType=child_kind,
                                whitelisted=str(allowed).casefold(),
                                destructiveOperationAllowed=str(
                                    allowed
                                ).casefold(),
                                whitelistIdentity="reference-id",
                            ),
                        ), *sorted(child_record["paths"]))
                        continue

                    if child_kind == "column":
                        packet.add(GraphFact(
                            "magento-db-column",
                            table_name,
                            "has-column",
                            identity,
                            str(child_record["path"]),
                            int(child_record["line"]),
                            attrs(
                                dataType=child_attributes.get("type", ""),
                                nullable=child_attributes.get(
                                    "nullable",
                                    "",
                                ),
                                identity=child_attributes.get(
                                    "identity",
                                    "",
                                ),
                                module=child_record["module"],
                            ),
                        ), *sorted(child_record["paths"]))
                        continue

                    if child_kind == "index":
                        packet.add(GraphFact(
                            "magento-db-index",
                            table_name,
                            "indexes-columns",
                            identity,
                            str(child_record["path"]),
                            int(child_record["line"]),
                            attrs(
                                indexType=child_attributes.get(
                                    "indexType",
                                    "",
                                ),
                                columns=",".join(
                                    sorted(child_record["columns"])
                                ),
                                module=child_record["module"],
                            ),
                        ), *sorted(child_record["paths"]))
                        continue

                    reference_table = child_attributes.get(
                        "referenceTable",
                        "",
                    )
                    if reference_table:
                        invalid = reference_table in disabled_tables
                        packet.add(GraphFact(
                            (
                                "magento-db-foreign-key-invalid"
                                if invalid
                                else "magento-db-foreign-key"
                            ),
                            table_name,
                            (
                                "references-disabled-table"
                                if invalid
                                else "references-table"
                            ),
                            reference_table,
                            str(child_record["path"]),
                            int(child_record["line"]),
                            attrs(
                                referenceId=identity,
                                column=child_attributes.get("column", ""),
                                referenceColumn=child_attributes.get(
                                    "referenceColumn",
                                    "",
                                ),
                                module=child_record["module"],
                                semanticRole=(
                                    "diagnostic" if invalid else None
                                ),
                            ),
                        ), *sorted(child_record["paths"]),
                            *sorted(tables.get(
                                reference_table,
                                {},
                            ).get("paths", ())))
                    else:
                        packet.add(GraphFact(
                            "magento-db-constraint",
                            table_name,
                            "constrains-columns",
                            identity,
                            str(child_record["path"]),
                            int(child_record["line"]),
                            attrs(
                                constraintType=child_attributes.get(
                                    "type",
                                    "",
                                ),
                                columns=",".join(
                                    sorted(child_record["columns"])
                                ),
                                module=child_record["module"],
                            ),
                        ), *sorted(child_record["paths"]))

    def _graphql(self, modules: tuple[ModuleRecord, ...]) -> None:
        for path, content in sorted(self.artifacts.items()):
            if not path.endswith(".graphqls"):
                continue
            module = self._module_for_path(path, modules)
            if module is None or not module.enabled:
                continue
            for declaration in parse_schema(content):
                type_name = declaration.name
                packet = self.graph.packet("magento-graphql", type_name, module=module.name if module else "")
                packet.add(GraphFact(
                    "magento-graphql-type",
                    type_name,
                    "declared-in",
                    path,
                    path,
                    declaration.line,
                    attrs(kind=declaration.kind),
                ))
                type_resolver = next((
                    directive for directive in declaration.directives
                    if directive.name in {"resolver", "typeResolver"}
                    and directive.argument("class")
                ), None)
                if type_resolver is not None:
                    resolver_class = type_resolver.argument("class") or ""
                    packet.add(GraphFact(
                        "magento-graphql-type-resolver",
                        type_name,
                        "resolved-by",
                        resolver_class,
                        path,
                        declaration.line,
                        attrs(directive=type_resolver.name),
                    ), self._symbol_path(resolver_class))
                for declared_field in declaration.fields:
                    field_key = f"{type_name}.{declared_field.name}"
                    packet.add(GraphFact(
                        "magento-graphql-field",
                        type_name,
                        "has-field",
                        declared_field.name,
                        path,
                        declared_field.line,
                        attrs(dataType=declared_field.target_type),
                    ))
                    resolver = next((
                        directive for directive in declared_field.directives
                        if directive.name in {"resolver", "typeResolver"}
                        and directive.argument("class")
                    ), None)
                    if resolver is not None:
                        resolver_class = resolver.argument("class") or ""
                        packet.add(GraphFact(
                            "magento-graphql-resolver",
                            field_key,
                            "resolved-by",
                            resolver_class,
                            path,
                            declared_field.line,
                            attrs(directive=resolver.name),
                        ), self._symbol_path(resolver_class))

    def _graphql_clients(self, modules: tuple[ModuleRecord, ...]) -> None:
        """Link embedded operations to the unique schema fields they select.

        Magento's GraphQL boundary is a typed traversal, not a shared-word
        relationship. Each segment must resolve from its current GraphQL owner
        to exactly one enabled schema declaration; ambiguity or a missing field
        makes the plugin abstain from that segment and every deeper segment.
        """
        declarations: dict[
            tuple[str, str],
            list[tuple[str, str, int]],
        ] = {}
        root_types: dict[str, set[str]] = {}
        for schema_path, content in sorted(self.artifacts.items()):
            if not schema_path.casefold().endswith(".graphqls"):
                continue
            module = self._module_for_path(schema_path, modules)
            if module is None or not module.enabled:
                continue
            for operation, type_name in parse_schema_root_types(content):
                root_types.setdefault(operation, set()).add(type_name)
            for definition in parse_schema(content):
                for declared_field in definition.fields:
                    declarations.setdefault(
                        (definition.name, declared_field.name),
                        [],
                    ).append((
                        schema_path,
                        declared_field.target_type,
                        declared_field.line,
                    ))

        resolved_root_types = {
            operation: next(iter(type_names))
            for operation, type_names in root_types.items()
            if len(type_names) == 1
        }

        for client_path, content in sorted(self.artifacts.items()):
            if not client_path.casefold().endswith(
                MAGENTO_GRAPHQL_CLIENT_SUFFIXES
            ):
                continue
            for selection in parse_operations(
                content,
                embedded_only=not client_path.casefold().endswith(
                    (".gql", ".graphql")
                ),
                root_types=resolved_root_types,
            ):
                owner = selection.root
                resolved: tuple[str, str, int] | None = None
                resolved_owner = ""
                for segment in selection.segments:
                    candidates = declarations.get((owner, segment), ())
                    if len(candidates) != 1:
                        resolved = None
                        break
                    resolved_owner = owner
                    resolved = candidates[0]
                    owner = resolved[1]
                if resolved is None:
                    continue
                schema_path, target_type, declaration_line = resolved
                selection_key = ".".join((selection.root, *selection.segments))
                packet = self.graph.packet(
                    "magento-graphql-client",
                    f"{client_path}:{selection_key}",
                    operationRoot=selection.root,
                )
                packet.add(GraphFact(
                    "magento-graphql-operation-field",
                    f"{client_path}::{selection_key}",
                    "selects-schema-field",
                    f"{schema_path}::{resolved_owner}.{selection.segments[-1]}",
                    client_path,
                    selection.line,
                    attrs(
                        schemaPath=schema_path,
                        schemaLine=declaration_line,
                        targetType=target_type,
                        resolution="exact-typed-graphql-traversal",
                        semanticRole="topology",
                    ),
                ), schema_path)

    def _extension_attributes(self, modules: tuple[ModuleRecord, ...]) -> None:
        schema_paths: dict[str, set[str]] = {}
        for schema_path, _, _ in self._ordered_configs(
            "db_schema.xml",
            modules,
            "global",
        ):
            schema_root = self._xml(schema_path)
            if schema_root is None:
                continue
            for table_node in (
                node for node in schema_root
                if tag(node) == "table" and node.get("name")
            ):
                schema_paths.setdefault(
                    table_node.get("name"),
                    set(),
                ).add(schema_path)

        effective: dict[
            tuple[str, str],
            dict[str, object],
        ] = {}
        for path, module, order in self._ordered_configs("extension_attributes.xml", modules, "global"):
            root = self._xml(path)
            if root is None:
                continue
            content = self.artifacts[path]
            for extension in (
                node for node in root.iter()
                if tag(node) == "extension_attributes" and node.get("for")
            ):
                interface = extension.get("for").lstrip("\\")
                packet = self.graph.packet("magento-extension-attributes", interface)
                for attribute in (
                    node for node in extension
                    if tag(node) == "attribute" and node.get("code") and node.get("type")
                ):
                    code = attribute.get("code")
                    record = effective.setdefault(
                        (interface, code),
                        {
                            "attributes": {},
                            "path": path,
                            "line": line(content, code),
                            "module": (
                                module.name
                                if module
                                else "application"
                            ),
                            "order": order,
                            "paths": set(),
                            "resources": set(),
                            "join": None,
                        },
                    )
                    record["attributes"].update({
                        name.rsplit("}", 1)[-1]: value
                        for name, value in attribute.attrib.items()
                    })
                    record["path"] = path
                    record["line"] = line(content, code)
                    record["module"] = (
                        module.name if module else "application"
                    )
                    record["order"] = order
                    record["paths"].add(path)
                    for resource in (
                        node for container in attribute
                        if tag(container) == "resources"
                        for node in container
                        if tag(node) == "resource" and node.get("ref")
                    ):
                        record["resources"].add(resource.get("ref"))
                    join_node = next(
                        (
                            node for node in attribute
                            if tag(node) == "join"
                        ),
                        None,
                    )
                    if join_node is not None:
                        fields = tuple(
                            (
                                (field_node.text or "").strip(),
                                (
                                    field_node.get("column")
                                    or (field_node.text or "").strip()
                                ),
                            )
                            for field_node in join_node
                            if tag(field_node) == "field"
                            and (field_node.text or "").strip()
                        )
                        record["join"] = {
                            "reference_table": join_node.get(
                                "reference_table",
                                "",
                            ),
                            "reference_field": join_node.get(
                                "reference_field",
                                "",
                            ),
                            "join_on_field": join_node.get(
                                "join_on_field",
                                "",
                            ),
                            "fields": fields,
                            "path": path,
                            "line": line(
                                content,
                                join_node.get("reference_table", ""),
                            ),
                        }

        for (interface, code), record in sorted(effective.items()):
            attribute_values = record["attributes"]
            data_type = str(attribute_values.get("type", ""))
            target = data_type.replace("[]", "").lstrip("\\")
            packet = self.graph.packet(
                "magento-extension-attributes",
                interface,
            )
            packet.add(GraphFact(
                "magento-extension-attribute",
                interface,
                "adds-attribute",
                code,
                str(record["path"]),
                int(record["line"]),
                attrs(
                    dataType=data_type,
                    module=record["module"],
                    order=record["order"],
                    resources=",".join(sorted(record["resources"])),
                ),
            ), self._symbol_path(interface), self._symbol_path(target),
                *sorted(record["paths"]))

            join_record = record.get("join")
            if not join_record:
                continue
            if data_type.endswith("[]"):
                # JoinProcessor cannot hydrate an array-typed extension
                # attribute from a joined row. Preserve the declaration, but
                # do not present the join as an effective data relationship.
                packet.add(GraphFact(
                    "magento-extension-attribute-join-inapplicable",
                    f"{interface}.{code}",
                    "cannot-hydrate-array-type",
                    data_type,
                    str(join_record["path"]),
                    int(join_record["line"]),
                    attrs(
                        module=record["module"],
                        semanticRole="diagnostic",
                    ),
                ), self._symbol_path(interface), self._symbol_path(target),
                    *sorted(record["paths"]))
                continue
            reference_table = str(
                join_record.get("reference_table", "")
            )
            reference_field = str(
                join_record.get("reference_field", "")
            )
            join_on_field = str(
                join_record.get("join_on_field", "")
            )
            packet.add(GraphFact(
                "magento-extension-attribute-join",
                f"{interface}.{code}",
                "joins-reference-table",
                reference_table,
                str(join_record["path"]),
                int(join_record["line"]),
                attrs(
                    dataType=data_type,
                    referenceField=reference_field,
                    joinOnField=join_on_field,
                    tableAlias=f"extension_attribute_{code}",
                ),
            ), self._symbol_path(interface), self._symbol_path(target),
                *sorted(record["paths"]),
                *sorted(schema_paths.get(reference_table, ())))
            for property_name, column_name in join_record["fields"]:
                packet.add(GraphFact(
                    "magento-extension-attribute-join-field",
                    f"{interface}.{code}",
                    "maps-table-column-to-property",
                    f"{reference_table}.{column_name}",
                    str(join_record["path"]),
                    int(join_record["line"]),
                    attrs(
                        property=property_name,
                        column=column_name,
                        referenceField=reference_field,
                        joinOnField=join_on_field,
                    ),
                ), self._symbol_path(interface), self._symbol_path(target),
                    *sorted(record["paths"]),
                    *sorted(schema_paths.get(reference_table, ())))

    def _generic_config_references(self, modules: tuple[ModuleRecord, ...]) -> None:
        """Connect additional Magento XML schemas to exact PHP declarations.

        Magento modules define many schema-specific configuration files beyond
        the specialized flows above.  This fallback does not guess semantics:
        it emits an edge only when an XML attribute/text value resolves to a
        PHP symbol present in the indexed repository.
        """
        existing_references = {
            (fact.path, fact.target.split("::", 1)[0].lstrip("\\"))
            for packet in self.graph.build()
            for fact in packet.facts
        }
        for path in sorted(self.artifacts):
            if not is_magento_config_xml(path):
                continue
            module = self._module_for_path(path, modules)
            is_application_config = path.startswith("app/etc/")
            if not is_application_config and (
                module is None or not module.enabled
            ):
                # Composer libraries can contain unrelated XML under `etc/`
                # (including MFTF's intentionally entity-bearing DI fixture).
                # Only deployed Magento module roots and application config
                # participate in Magento's merged configuration.
                continue
            root = self._xml(path)
            if root is None:
                continue
            area = config_area(path, PurePosixPath(path).name) or "global"
            packet = None
            content = self.artifacts[path]
            for element in root.iter():
                candidates = [
                    (name.rsplit("}", 1)[-1], value)
                    for name, value in element.attrib.items()
                ]
                if element.text and element.text.strip():
                    candidates.append(("value", element.text.strip()))
                for attribute_name, raw_value in candidates:
                    class_name = raw_value.split("::", 1)[0].strip().lstrip("\\")
                    symbol = self._symbol(class_name)
                    if symbol is None:
                        continue
                    if (path, class_name) in existing_references:
                        continue
                    if packet is None:
                        packet = self.graph.packet(
                            "magento-config-reference",
                            path,
                            module=module.name if module else "",
                            area=area,
                            schema=PurePosixPath(path).name,
                        )
                    packet.add(GraphFact(
                        "magento-config-class-reference",
                        f"{PurePosixPath(path).name}:{tag(element)}",
                        f"references-via-{attribute_name}",
                        class_name,
                        path,
                        line(content, raw_value),
                        attrs(
                            element=tag(element),
                            attribute=attribute_name,
                            module=module.name if module else "",
                            area=area,
                        ),
                    ), symbol.path)

    def _symbol(self, qualified_name: str) -> SymbolDefinition | None:
        values = self.symbols_by_name.get(qualified_name.lstrip("\\"), ())
        return values[0] if values else None

    def _unique_symbol_casefold(
        self,
        qualified_name: str,
    ) -> SymbolDefinition | None:
        values = self.symbols_by_casefold.get(
            qualified_name.lstrip("\\").casefold(),
            (),
        )
        return values[0] if len(values) == 1 else None

    @staticmethod
    def _method_attributes(
        symbol: SymbolDefinition,
        method: str,
    ) -> tuple[str, dict[str, str]] | None:
        declared = next(
            (
                name for name in symbol.methods
                if name.casefold() == method.casefold()
            ),
            None,
        )
        if declared is None:
            return None
        prefix = f"method:{declared}:"
        return declared, {
            key.removeprefix(prefix): value
            for key, value in symbol.attributes
            if key.startswith(prefix)
        }

    def _interception_applicability(
        self,
        target: str,
        method: str,
        state: DiState,
        plugin_symbol: SymbolDefinition,
        plugin_method: str,
    ) -> tuple[bool | None, str, str]:
        """Prove method interception only when PHP/Magento constraints are known."""
        target = target.lstrip("\\")
        if target in state.virtual_types:
            return False, "virtual-type", self._symbol_path(
                state.virtual_types[target].value
            )
        if method.casefold() in {"__construct", "__destruct"}:
            return False, "lifecycle-method", self._symbol_path(target)

        plugin_declaration = self._method_attributes(plugin_symbol, plugin_method)
        if plugin_declaration is not None:
            _, plugin_attributes = plugin_declaration
            if plugin_attributes.get("visibility", "public") != "public":
                return False, "plugin-method-not-public", plugin_symbol.path
            if plugin_attributes.get("static", "false") == "true":
                return False, "plugin-method-static", plugin_symbol.path

        target_symbol = self._symbol(target)
        if target_symbol is None:
            return None, "target-symbol-unavailable", ""
        if dict(target_symbol.attributes).get("type:final") == "true":
            return False, "final-class", target_symbol.path

        queue = [target_symbol]
        seen: set[str] = set()
        while queue:
            symbol = queue.pop(0)
            if symbol.qualified_name in seen:
                continue
            seen.add(symbol.qualified_name)
            declaration = self._method_attributes(symbol, method)
            if declaration is not None:
                _, method_attributes = declaration
                if method_attributes.get("visibility", "public") != "public":
                    return False, "method-not-public", symbol.path
                if method_attributes.get("static", "false") == "true":
                    return False, "method-static", symbol.path
                if method_attributes.get("final", "false") == "true":
                    return False, "method-final", symbol.path
                return True, "", symbol.path
            parents = [
                self._symbol(parent)
                for parent in symbol.parents
            ]
            queue.extend(sorted(
                (parent for parent in parents if parent is not None),
                key=lambda parent: (
                    0 if parent.kind == "class" else 1,
                    parent.qualified_name,
                ),
            ))

        # Trait methods and unavailable external parents are not represented in
        # the current neutral symbol contract. Absence is therefore unknown,
        # never proof that the configured method cannot be intercepted.
        return None, "method-declaration-unavailable", target_symbol.path

    def _symbol_path(self, qualified_name: str) -> str:
        if not qualified_name:
            return ""
        symbol = self._symbol(qualified_name.split("::", 1)[0])
        return symbol.path if symbol else ""

    def _method_symbol(
        self,
        symbol: SymbolDefinition,
        method: str,
    ) -> tuple[SymbolDefinition, str] | None:
        queue = [symbol]
        seen: set[str] = set()
        while queue:
            candidate = queue.pop(0)
            if candidate.qualified_name in seen:
                continue
            seen.add(candidate.qualified_name)
            declaration = self._method_attributes(candidate, method)
            if (
                declaration is not None
                and declaration[1].get("visibility", "public") == "public"
            ):
                declared, _ = declaration
                return candidate, declared
            queue.extend(
                parent
                for parent_name in candidate.parents
                if (parent := self._unique_symbol_casefold(parent_name)) is not None
            )
        return None

    def _is_view_model_argument(
        self,
        argument_name: str,
        symbol: SymbolDefinition | None,
    ) -> bool:
        leaf_name = argument_name.rsplit(".", 1)[-1]
        if re.sub(r"[^a-z0-9]", "", leaf_name.casefold()) == "viewmodel":
            return True
        expected = (
            r"magento\framework\view\element\block\argumentinterface"
        )
        queue = [symbol] if symbol is not None else []
        seen: set[str] = set()
        while queue:
            candidate = queue.pop(0)
            normalized = candidate.qualified_name.lstrip("\\").casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            if normalized == expected:
                return True
            for parent_name in candidate.parents:
                if parent_name.lstrip("\\").casefold() == expected:
                    return True
                parent = self._unique_symbol_casefold(parent_name)
                if parent is not None:
                    queue.append(parent)
        return False

    @staticmethod
    def _layout_argument_for_call(
        call: TemplatePhpCall,
        arguments: tuple[LayoutArgument, ...],
    ) -> str:
        requested = ""
        if call.method.casefold() in {"getdata", "hasdata"}:
            requested = dict(call.literal_arguments).get(0, "")
        else:
            match = re.fullmatch(r"(?:get|has)([A-Z][A-Za-z0-9]*)", call.method)
            if match is not None:
                requested = re.sub(
                    r"(?<!^)(?=[A-Z])",
                    "_",
                    match.group(1),
                ).casefold()
        if not requested:
            return ""
        matches = tuple(
            argument.name
            for argument in arguments
            if argument.name == requested
        )
        return matches[0] if len(matches) == 1 else ""

    def _template_php_calls(
        self,
        path: str,
        receiver: str,
    ) -> tuple[TemplatePhpCall, ...]:
        """Read syntax-proven PHTML calls published by the neutral PHP plugin."""

        cache_key = (path, receiver)
        cached = self._template_php_call_cache.get(cache_key)
        if cached is not None:
            return cached
        records: set[TemplatePhpCall] = set()
        for symbol in self.symbols:
            if symbol.kind != "template" or symbol.path != path:
                continue
            for key, value in symbol.attributes:
                if not key.startswith(_PHP_TEMPLATE_CALL_REFERENCE):
                    continue
                try:
                    payload = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    self._record_template_metadata_diagnostic(
                        path,
                        key,
                        "invalid JSON",
                    )
                    continue
                if not isinstance(payload, dict):
                    self._record_template_metadata_diagnostic(
                        path,
                        key,
                        "value must be an object",
                    )
                    continue
                call_receiver = payload.get("receiver")
                method = payload.get("method")
                call_line = payload.get("line")
                literal_arguments = payload.get(
                    "literalStringArguments",
                    {},
                )
                if (
                    not isinstance(call_receiver, str)
                    or not isinstance(method, str)
                    or not isinstance(call_line, int)
                    or call_line < 1
                    or not isinstance(literal_arguments, dict)
                    or any(
                        not isinstance(position, str)
                        or not position.isdigit()
                        or not isinstance(argument, str)
                        for position, argument in literal_arguments.items()
                    )
                ):
                    self._record_template_metadata_diagnostic(
                        path,
                        key,
                        "object has invalid fields",
                    )
                    continue
                if call_receiver != receiver:
                    continue
                records.add(TemplatePhpCall(
                    receiver=call_receiver,
                    method=method,
                    line=call_line,
                    literal_arguments=tuple(sorted(
                        (int(position), argument)
                        for position, argument
                        in literal_arguments.items()
                    )),
                ))
        result = tuple(sorted(records))
        self._template_php_call_cache[cache_key] = result
        return result

    def _record_template_metadata_diagnostic(
        self,
        path: str,
        key: str,
        reason: str,
    ) -> None:
        identity = (path, key)
        if identity in self._template_metadata_diagnostics:
            return
        self._template_metadata_diagnostics.add(identity)
        self._diagnostics.append(PluginDiagnostic(
            code="magento-invalid-php-template-call-metadata",
            message=(
                f"{path}: ignored PHP template call metadata {key!r}: "
                f"{reason}"
            ),
            plugin_id=self.plugin_id,
            path=path,
            recoverable=True,
        ))

    def _theme_for_path(
        self,
        path: str,
        themes: tuple[ThemeRecord, ...],
    ) -> ThemeRecord | None:
        return max(
            (
                theme for theme in themes
                if (
                    not theme.root
                    or path == theme.root
                    or path.startswith(theme.root + "/")
                )
            ),
            key=lambda theme: len(theme.root),
            default=None,
        )

    def _is_deployed_view_source(
        self,
        path: str,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
    ) -> bool:
        """Match Magento's enabled-module filter while retaining registered themes."""
        theme = self._theme_for_path(path, themes)
        if theme is not None:
            theme_module = self._theme_module(path, theme)
            if theme_module:
                module = next(
                    (
                        candidate
                        for candidate in modules
                        if candidate.name == theme_module
                    ),
                    None,
                )
                if module is not None:
                    return module.enabled
                if theme_module in self.configured_modules:
                    return self.configured_modules[theme_module]
                # A standalone theme package has no deployment config from
                # which to prove installed modules. Retain its explicitly
                # named override directories without inventing ModuleRecords;
                # exact module-dependent resolution continues to abstain.
                return not modules and not self.configured_modules
            return True
        module = self._module_for_path(path, modules)
        return module is not None and module.enabled

    @staticmethod
    def _theme_chain(
        theme: ThemeRecord,
        themes: tuple[ThemeRecord, ...],
    ) -> tuple[ThemeRecord, ...]:
        """Return the exact child-to-parent fallback chain for one theme."""
        by_identity = {
            (candidate.area, candidate.name): candidate
            for candidate in themes
        }
        chain: list[ThemeRecord] = []
        seen: set[tuple[str, str]] = set()
        current: ThemeRecord | None = theme
        while current is not None:
            identity = (current.area, current.name)
            if identity in seen:
                break
            seen.add(identity)
            chain.append(current)
            current = (
                by_identity.get((current.area, current.parent))
                if current.parent
                else None
            )
        return tuple(chain)

    @staticmethod
    def _theme_module(path: str, theme: ThemeRecord | None) -> str:
        if theme is None:
            return ""
        relative = path[len(theme.root):].lstrip("/")
        first = relative.split("/", 1)[0]
        return first if "_" in first else ""

    def _template_paths(
        self,
        template: str,
        area: str,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
        source_theme: ThemeRecord | None = None,
    ) -> tuple[str, ...]:
        if not template or "::" not in template:
            return ()
        module_name, relative = template.split("::", 1)
        module = next((item for item in modules if item.name == module_name), None)
        theme_only_selection = (
            source_theme is not None
            and not modules
            and not self.configured_modules
        )
        if module is not None and not module.enabled:
            return ()
        if (
            module is None
            and not self.configured_modules.get(module_name, False)
            and not theme_only_selection
        ):
            return ()
        paths: set[str] = set()
        if module is not None:
            for candidate_area in (area, "base"):
                candidate = _path_under(
                    module.root,
                    f"view/{candidate_area}/templates/{relative}",
                )
                if candidate in self.artifacts:
                    paths.add(candidate)
        theme_candidates = (
            self._theme_chain(source_theme, themes)
            if source_theme is not None
            else themes
        )
        for theme in theme_candidates:
            if theme.area != area:
                continue
            candidate = _path_under(theme.root, f"{module_name}/templates/{relative}")
            if candidate in self.artifacts:
                paths.add(candidate)
        return tuple(sorted(paths))

    def _selected_template_path(
        self,
        template: str,
        area: str,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
        source_theme: ThemeRecord | None = None,
    ) -> str:
        """Return one runtime-selected PHTML path or abstain if theme is unknown."""

        if not template or "::" not in template:
            return ""
        module_name, relative = template.split("::", 1)
        module = next(
            (item for item in modules if item.name == module_name),
            None,
        )
        theme_only_selection = (
            source_theme is not None
            and not modules
            and not self.configured_modules
        )
        if module is not None and not module.enabled:
            return ""
        if (
            module is None
            and not self.configured_modules.get(module_name, False)
            and not theme_only_selection
        ):
            return ""

        if source_theme is not None:
            for theme in self._theme_chain(source_theme, themes):
                if theme.area != area:
                    continue
                candidate = _path_under(
                    theme.root,
                    f"{module_name}/templates/{relative}",
                )
                if candidate in self.artifacts:
                    return candidate
        else:
            # A module-owned layout can run under any configured store theme.
            # If any installed theme overrides this exact template identity,
            # repository state cannot select the runtime source.
            if any(
                theme.area == area
                and _path_under(
                    theme.root,
                    f"{module_name}/templates/{relative}",
                ) in self.artifacts
                for theme in themes
            ):
                return ""

        if module is not None:
            for candidate_area in (area, "base"):
                candidate = _path_under(
                    module.root,
                    f"view/{candidate_area}/templates/{relative}",
                )
                if candidate in self.artifacts:
                    return candidate
        return ""

    def _ui_asset_paths(
        self,
        identifier: str,
        area: str,
        is_template: bool,
        modules: tuple[ModuleRecord, ...],
        themes: tuple[ThemeRecord, ...],
        source_theme: ThemeRecord | None = None,
    ) -> tuple[str, ...]:
        if "/" not in identifier:
            return ()
        module_name, separator, relative = identifier.partition("/")
        if not separator or "_" not in module_name:
            return ()
        extension = ".html" if is_template else ".js"
        relative_path = relative if relative.endswith(extension) else relative + extension
        paths: set[str] = set()
        module = next((item for item in modules if item.name == module_name), None)
        theme_only_selection = (
            source_theme is not None
            and not modules
            and not self.configured_modules
        )
        if module is not None and not module.enabled:
            return ()
        if (
            module is None
            and not self.configured_modules.get(module_name, False)
            and not theme_only_selection
        ):
            return ()
        if module is not None:
            for candidate_area in (area, "base"):
                candidate = _path_under(
                    module.root,
                    f"view/{candidate_area}/web/{relative_path}",
                )
                if candidate in self.artifacts:
                    paths.add(candidate)
        theme_candidates = (
            self._theme_chain(source_theme, themes)
            if source_theme is not None
            else themes
        )
        for theme in theme_candidates:
            if theme.area != area:
                continue
            candidate = _path_under(
                theme.root,
                f"{module_name}/web/{relative_path}",
            )
            if candidate in self.artifacts:
                paths.add(candidate)
        return tuple(sorted(paths))


@dataclass
class MagentoRepositorySession:
    plugin_id: str
    revision: str
    artifacts: dict[str, str] = field(default_factory=dict)
    source_root: str | None = None
    progress_callback: Callable[[dict[str, object]], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    analysis_deadline: float | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    @classmethod
    def restore(cls, plugin_id: str, revision: str, snapshots) -> "MagentoRepositorySession":
        snapshot = next(
            (item for item in snapshots if item.kind == "magento-architecture-sources"),
            None,
        )
        if snapshot is None:
            raise ValueError(
                "Magento repository snapshot is missing magento-architecture-sources"
            )
        raw = gzip.decompress(base64.b64decode(snapshot.content.encode("ascii")))
        artifacts = json.loads(raw.decode("utf-8"))
        if not isinstance(artifacts, dict) or any(
            not isinstance(path, str) or not isinstance(content, str)
            for path, content in artifacts.items()
        ):
            raise ValueError("Magento repository snapshot has invalid architecture sources")
        return cls(plugin_id, revision, dict(sorted(artifacts.items())))

    def _snapshot(self) -> RepositorySnapshot:
        raw = json.dumps(
            dict(sorted(self.artifacts.items())),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        content = base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode("ascii")
        return RepositorySnapshot(
            self.plugin_id,
            "magento-architecture-sources",
            content,
        )

    def set_source_root(self, source_root: str | None) -> None:
        self.source_root = source_root

    def set_progress_callback(
        self,
        progress_callback: Callable[[dict[str, object]], None] | None,
    ) -> None:
        self.progress_callback = progress_callback

    def set_analysis_deadline(self, deadline: float | None) -> None:
        self.analysis_deadline = deadline

    def _report_scoped_progress(
        self,
        root: str,
        event: dict[str, object],
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback({
            **event,
            "sourceRoot": root or ".",
        })

    def ingest(self, artifacts: tuple[FileArtifact, ...]) -> None:
        for artifact in artifacts:
            path = artifact.path
            if artifact.deleted:
                self.artifacts.pop(path, None)
                continue
            filename = PurePosixPath(path).name
            is_config = is_magento_config_xml(path)
            is_schema_whitelist = (
                filename == "db_schema_whitelist.json"
                and "/etc/" in f"/{path}"
            )
            is_component = filename in {
                "composer.json", "module.xml", "registration.php", "theme.xml",
            }
            is_view_configuration = is_magento_view_xml(path)
            is_template = (
                "/templates/" in f"/{path}"
            )
            is_email_template = (
                "/email/" in f"/{path}"
                and path.casefold().endswith(".html")
            )
            is_view_asset = (
                "/web/" in f"/{path}"
                and path.casefold().endswith(MAGENTO_VIEW_SOURCE_SUFFIXES)
            )
            is_graphql = path.casefold().endswith(".graphqls")
            is_requirejs = filename == "requirejs-config.js"
            is_app_config = path == "app/etc/config.php" or path.endswith(
                "/app/etc/config.php"
            )
            if (
                is_config
                or is_schema_whitelist
                or is_component
                or is_view_configuration
                or is_template
                or is_email_template
                or is_view_asset
                or is_graphql
                or is_requirejs
                or is_app_config
            ):
                self.artifacts[path] = artifact.content

    def finish(self, dependencies: RepositoryAnalysis):
        started = time.monotonic()
        roots = self._analysis_roots()
        analyses: list[RepositoryAnalysis] = []
        diagnostics: list[PluginDiagnostic] = []
        invalid_paths: set[str] = set()
        for root in roots:
            scoped = self._scoped_artifacts(root)
            scoped_symbols = self._scoped_symbols(root, dependencies.symbols)
            scoped_paths = {*scoped, *(symbol.path for symbol in scoped_symbols)}
            resolver = MagentoRepositoryResolver(
                self.plugin_id,
                scoped,
                scoped_symbols,
                progress_callback=(
                    lambda event, root=root: self._report_scoped_progress(
                        root,
                        event,
                    )
                ),
                deadline=self.analysis_deadline,
            )
            analysis, scoped_diagnostics = resolver.resolve()
            analyses.append(self._prefix_analysis(analysis, root, scoped_paths))
            diagnostics.extend(
                PluginDiagnostic(
                    code=item.code,
                    message=item.message,
                    plugin_id=item.plugin_id,
                    path=(
                        self._prefix_path(root, item.path)
                        if item.path in scoped_paths
                        else item.path
                    ),
                    recoverable=item.recoverable,
                )
                for item in scoped_diagnostics
            )
            invalid_paths.update(
                self._prefix_path(root, path) for path in resolver.invalid_paths
            )
        analysis = RepositoryAnalysis(
            symbols=tuple(sorted({item for part in analyses for item in part.symbols})),
            packets=tuple(sorted({item for part in analyses for item in part.packets})),
            contexts=tuple(sorted({item for part in analyses for item in part.contexts})),
            diagnostics=tuple(
                item for part in analyses for item in part.diagnostics
            ),
        )
        if diagnostics:
            for diagnostic in diagnostics:
                logger.warning(
                    "Skipping invalid Magento repository input "
                    "(code=%s path=%s): %s",
                    diagnostic.code,
                    diagnostic.path or "<repository>",
                    diagnostic.message,
                )
        for path in invalid_paths:
            self.artifacts.pop(path, None)
        related_paths = {
            path for packet in analysis.packets for path in packet.paths
        }
        contexts = tuple(sorted(
            RepositoryContext(
                self.plugin_id,
                (
                    "magento-template-source"
                    if path.casefold().endswith(".phtml")
                    else "magento-view-source"
                ),
                path,
                content,
            )
            for path, content in self.artifacts.items()
            if any(
                path.startswith(f"{root}/vendor/" if root else "vendor/")
                for root in roots
            )
            and path in related_paths
            and content.strip()
            and path.casefold().endswith(MAGENTO_VIEW_SOURCE_SUFFIXES)
        ))
        snapshot_started = time.monotonic()
        snapshot = self._snapshot()
        logger.info(
            "Magento repository snapshot: sources=%s contexts=%s encoded_bytes=%s elapsed=%.3fs total_finish=%.3fs",
            len(self.artifacts),
            len(contexts),
            len(snapshot.content),
            time.monotonic() - snapshot_started,
            time.monotonic() - started,
        )
        return PluginOutcome.handled(RepositoryAnalysis(
            symbols=analysis.symbols,
            packets=analysis.packets,
            snapshots=(snapshot,),
            contexts=contexts,
            diagnostics=tuple(
                PluginDiagnostic(
                    code=diagnostic.code,
                    message=diagnostic.message,
                    plugin_id=diagnostic.plugin_id,
                    path=diagnostic.path,
                    recoverable=True,
                )
                for diagnostic in diagnostics
            ),
        ))

    def _analysis_roots(self) -> tuple[str, ...]:
        if self.source_root is not None:
            return (self.source_root,)
        application_markers = (
            "app/etc/config.php",
            "app/etc/di.xml",
            "app/etc/env.php",
            "bin/magento",
        )
        roots = {
            "" if path == marker else path[: -(len(marker) + 1)]
            for path in self.artifacts
            for marker in application_markers
            if path == marker or path.endswith("/" + marker)
        }
        if roots:
            return tuple(sorted(roots, key=lambda value: (value.count("/"), value)))
        return ("",)

    def _scoped_artifacts(self, root: str) -> dict[str, str]:
        if not root:
            return dict(self.artifacts)
        prefix = root + "/"
        return {
            path[len(prefix):]: content
            for path, content in self.artifacts.items()
            if path.startswith(prefix)
        }

    @staticmethod
    def _scoped_symbols(
        root: str,
        symbols: tuple[SymbolDefinition, ...],
    ) -> tuple[SymbolDefinition, ...]:
        if not root:
            return symbols
        prefix = root + "/"
        return tuple(sorted(
            replace(symbol, path=symbol.path[len(prefix):])
            for symbol in symbols
            if symbol.path.startswith(prefix)
        ))

    @staticmethod
    def _prefix_path(root: str, path: str) -> str:
        return f"{root}/{path}" if root else path

    def _prefix_analysis(
        self,
        analysis: RepositoryAnalysis,
        root: str,
        scoped_paths: set[str],
    ) -> RepositoryAnalysis:
        if not root:
            return analysis

        def path(value: str) -> str:
            return self._prefix_path(root, value) if value in scoped_paths else value

        packets = tuple(sorted(
            ArchitecturePacket(
                plugin_id=packet.plugin_id,
                kind=packet.kind,
                key=packet.key,
                paths=tuple(sorted({path(value) for value in packet.paths})),
                facts=tuple(sorted(
                    GraphFact(
                        kind=fact.kind,
                        source=fact.source,
                        relation=fact.relation,
                        target=fact.target,
                        path=path(fact.path),
                        line=fact.line,
                        attributes=fact.attributes,
                        related_paths=tuple(sorted({
                            path(value) for value in fact.related_paths
                        })),
                    )
                    for fact in packet.facts
                )),
                attributes=packet.attributes,
            )
            for packet in analysis.packets
        ))
        contexts = tuple(sorted(
            RepositoryContext(
                context.plugin_id,
                context.kind,
                path(context.path),
                context.content,
                context.attributes,
            )
            for context in analysis.contexts
        ))
        return RepositoryAnalysis(
            symbols=tuple(sorted(
                replace(symbol, path=path(symbol.path))
                for symbol in analysis.symbols
            )),
            packets=packets,
            contexts=contexts,
            diagnostics=analysis.diagnostics,
        )
