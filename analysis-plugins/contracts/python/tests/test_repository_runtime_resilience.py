from types import SimpleNamespace
import time

from codecrow_plugins import (
    ArchitecturePacket,
    FileArtifact,
    GraphFact,
    PluginDiagnostic,
    PluginOutcome,
    RepositoryAnalysis,
    SymbolDefinition,
)
from codecrow_plugins.runtime import RepositoryAnalysisHandle


class _FileIsolatingSession:
    def __init__(self):
        self.ingested = []

    def ingest(self, artifacts):
        artifact = artifacts[0]
        if artifact.path == "bad.xml":
            raise ValueError("invalid project file")
        self.ingested.append(artifact.path)

    def finish(self, _dependencies):
        return PluginOutcome.handled(RepositoryAnalysis(
            diagnostics=(PluginDiagnostic(
                code="project-warning",
                message="recoverable repository diagnostic",
                plugin_id="test-plugin",
                path="warning.xml",
                recoverable=True,
            ),),
        ))


def test_repository_runtime_quarantines_ingest_failure_and_keeps_session():
    session = _FileIsolatingSession()
    runtime = SimpleNamespace(
        MAX_REPOSITORY_SYMBOLS=10,
        MAX_ARCHITECTURE_PACKETS=10,
    )
    handle = RepositoryAnalysisHandle(
        runtime,
        [("test-plugin", session)],
        [],
    )
    progress_events = []

    handle.ingest((
        FileArtifact("bad.xml", "<invalid>"),
        FileArtifact("good.xml", "<valid />"),
    ), progress_callback=progress_events.append)
    _analysis, diagnostics = handle.finish()

    assert session.ingested == ["good.xml"]
    assert [event["status"] for event in progress_events] == [
        "started",
        "completed",
    ]
    assert all(event["pluginId"] == "test-plugin" for event in progress_events)
    assert all(event["files"] == 2 for event in progress_events)
    assert all(event["firstPath"] == "bad.xml" for event in progress_events)
    assert all(event["lastPath"] == "good.xml" for event in progress_events)
    assert progress_events[-1]["durationMs"] >= 0
    assert [
        (diagnostic.code, diagnostic.path, diagnostic.recoverable)
        for diagnostic in diagnostics
    ] == [
        ("plugin-repository-file-skipped", "bad.xml", True),
        ("project-warning", "warning.xml", True),
    ]


class _TimedRepositorySession:
    def __init__(self):
        self.progress_callback = None
        self.deadline = None

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def set_analysis_deadline(self, deadline):
        self.deadline = deadline

    def finish(self, _dependencies):
        raise TimeoutError("test plugin exhausted the architecture budget")


def test_repository_runtime_reports_timeout_as_recoverable_and_stops():
    session = _TimedRepositorySession()
    later = _FileIsolatingSession()
    runtime = SimpleNamespace(
        MAX_REPOSITORY_SYMBOLS=10,
        MAX_ARCHITECTURE_PACKETS=10,
    )
    events = []
    deadline = time.monotonic() + 60
    handle = RepositoryAnalysisHandle(
        runtime,
        [("timed-plugin", session), ("later-plugin", later)],
        [],
    )

    analysis, diagnostics = handle.finish(
        progress_callback=events.append,
        deadline=deadline,
    )

    assert analysis == RepositoryAnalysis()
    assert session.progress_callback is not None
    assert session.deadline == deadline
    assert [(item.code, item.recoverable) for item in diagnostics] == [
        ("plugin-repository-finalization-timeout", True),
    ]
    assert [event["status"] for event in events] == ["started", "timed_out"]
    assert later.ingested == []


class _StaticRepositorySession:
    def __init__(self, analysis: RepositoryAnalysis):
        self.analysis = analysis
        self.finished = False

    def finish(self, _dependencies):
        self.finished = True
        return PluginOutcome.handled(self.analysis)


def _symbol(name: str) -> SymbolDefinition:
    return SymbolDefinition(name, "class", f"src/{name}.py")


def _packet(key: str) -> ArchitecturePacket:
    path = f"src/{key}.py"
    return ArchitecturePacket(
        "test-plugin",
        "test-architecture",
        key,
        (path,),
        (GraphFact("test-fact", key, "declares", key, path),),
    )


def test_repository_symbol_overflow_is_fatal_and_does_not_publish_a_slice():
    first = _StaticRepositorySession(RepositoryAnalysis(symbols=(_symbol("First"),)))
    overflow = _StaticRepositorySession(RepositoryAnalysis(symbols=(_symbol("Second"),)))
    later = _StaticRepositorySession(RepositoryAnalysis(symbols=(_symbol("Third"),)))
    runtime = SimpleNamespace(
        MAX_REPOSITORY_SYMBOLS=1,
        MAX_ARCHITECTURE_PACKETS=10,
    )
    handle = RepositoryAnalysisHandle(
        runtime,
        [("first-plugin", first), ("overflow-plugin", overflow), ("later-plugin", later)],
        [],
    )

    analysis, diagnostics = handle.finish()

    assert analysis.symbols == (_symbol("First"),)
    assert [(item.code, item.recoverable) for item in diagnostics] == [
        ("plugin-repository-symbol-limit", False),
    ]
    assert later.finished is False


def test_repository_packet_overflow_is_fatal_and_does_not_publish_a_slice():
    first = _StaticRepositorySession(RepositoryAnalysis(packets=(_packet("first"),)))
    overflow = _StaticRepositorySession(RepositoryAnalysis(packets=(_packet("second"),)))
    later = _StaticRepositorySession(RepositoryAnalysis(packets=(_packet("third"),)))
    runtime = SimpleNamespace(
        MAX_REPOSITORY_SYMBOLS=10,
        MAX_ARCHITECTURE_PACKETS=1,
    )
    handle = RepositoryAnalysisHandle(
        runtime,
        [("first-plugin", first), ("overflow-plugin", overflow), ("later-plugin", later)],
        [],
    )

    analysis, diagnostics = handle.finish()

    assert analysis.packets == (_packet("first"),)
    assert [(item.code, item.recoverable) for item in diagnostics] == [
        ("plugin-repository-packet-limit", False),
    ]
    assert later.finished is False


def test_repository_runtime_discards_result_that_returns_after_deadline(
    monkeypatch,
):
    session = _FileIsolatingSession()
    runtime = SimpleNamespace(
        MAX_REPOSITORY_SYMBOLS=10,
        MAX_ARCHITECTURE_PACKETS=10,
    )
    handle = RepositoryAnalysisHandle(
        runtime,
        [("slow-plugin", session)],
        [],
    )
    readings = iter((0.0, 1.0, 10.0, 10.0))
    monkeypatch.setattr(time, "monotonic", lambda: next(readings))

    analysis, diagnostics = handle.finish(deadline=5.0)

    assert analysis == RepositoryAnalysis()
    assert [(item.code, item.recoverable) for item in diagnostics] == [
        ("plugin-repository-finalization-timeout", True),
    ]
