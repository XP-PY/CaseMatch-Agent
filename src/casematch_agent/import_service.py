from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

from casematch_ranker import BGEM3DenseEncoder

from .agent import resolve_case_store_config, resolve_lecard_ranker_config
from .case_ingestion import (
    CaseImportBatch,
    CaseImportReport,
    CriminalCaseStructuredDataExtractor,
    import_raw_cases_batch_from_jsonl,
)
from .llm import OpenAICompatibleClient, OpenAICompatibleConfig
from .models import StructuredCase
from .lancedb_store import LanceDBCaseStore
from .sqlite_store import LeCaRDSQLiteStore

try:
    from langgraph.graph import END, START, StateGraph

    _LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    END = START = None
    StateGraph = None
    _LANGGRAPH_AVAILABLE = False


@dataclass(frozen=True)
class CaseImportRuntimeConfig:
    corpus_path: Path
    db_path: Path
    lancedb_uri: Path
    db_backend: str = "auto"
    bge_model_path: str | None = None
    bge_use_fp16: bool | None = None
    bge_batch_size: int | None = None
    bge_max_length: int | None = None
    device: str | list[str] | None = None
    case_id_prefix: str = "CASE"


@dataclass(frozen=True)
class CaseImportExecutionReport:
    import_report: CaseImportReport
    synced_backend: str
    requested_backend: str


class CaseImportGraphState(TypedDict, total=False):
    input_path: str | Path
    runtime_config: CaseImportRuntimeConfig
    llm_client: OpenAICompatibleClient
    synced_backend: str
    requested_backend: str
    sync_backend: Callable[[list[StructuredCase]], None]
    import_batch: CaseImportBatch
    execution_report: CaseImportExecutionReport


@dataclass
class CaseImportWorkflow:
    runtime_config: CaseImportRuntimeConfig
    llm_client: OpenAICompatibleClient
    _compiled_graph: Any = None

    def __post_init__(self) -> None:
        if _LANGGRAPH_AVAILABLE:
            self._compiled_graph = self._build_graph()

    def run(self, *, input_path: str | Path) -> CaseImportExecutionReport:
        if self._compiled_graph is not None:
            result = self._compiled_graph.invoke(
                {
                    "input_path": input_path,
                    "runtime_config": self.runtime_config,
                    "llm_client": self.llm_client,
                }
            )
            return result["execution_report"]
        return self._run_without_graph(input_path=input_path)

    def _run_without_graph(self, *, input_path: str | Path) -> CaseImportExecutionReport:
        synced_backend, sync_backend, requested_backend = self._resolve_backend_state()
        import_batch = self._ingest_batch(input_path)
        if import_batch.structured_cases:
            sync_backend(import_batch.structured_cases)
        return CaseImportExecutionReport(
            import_report=import_batch.report,
            synced_backend=synced_backend,
            requested_backend=requested_backend,
        )

    def _build_graph(self):
        assert StateGraph is not None
        graph = StateGraph(CaseImportGraphState)
        graph.add_node("resolve_backend", self._resolve_backend_node)
        graph.add_node("ingest_cases", self._ingest_cases_node)
        graph.add_node("refresh_indexes", self._refresh_indexes_node)
        graph.add_node("finalize_report", self._finalize_report_node)
        graph.add_edge(START, "resolve_backend")
        graph.add_edge("resolve_backend", "ingest_cases")
        graph.add_edge("ingest_cases", "refresh_indexes")
        graph.add_edge("refresh_indexes", "finalize_report")
        graph.add_edge("finalize_report", END)
        return graph.compile()

    def _resolve_backend_state(self) -> tuple[str, Callable[[list[StructuredCase]], None], str]:
        store_config = resolve_case_store_config(
            backend=self.runtime_config.db_backend,
            lancedb_uri=str(self.runtime_config.lancedb_uri),
        )
        synced_backend, sync_backend = _resolve_sync_backend(self.runtime_config, resolved_backend=store_config.backend)
        return synced_backend, sync_backend, store_config.backend

    def _ingest_batch(self, input_path: str | Path) -> CaseImportBatch:
        extractor = CriminalCaseStructuredDataExtractor(client=self.llm_client)
        return import_raw_cases_batch_from_jsonl(
            input_path=input_path,
            corpus_path=self.runtime_config.corpus_path,
            extractor=extractor,
            case_id_prefix=self.runtime_config.case_id_prefix,
        )

    def _resolve_backend_node(self, _: CaseImportGraphState) -> CaseImportGraphState:
        synced_backend, sync_backend, requested_backend = self._resolve_backend_state()
        return {
            "synced_backend": synced_backend,
            "sync_backend": sync_backend,
            "requested_backend": requested_backend,
        }

    def _ingest_cases_node(self, state: CaseImportGraphState) -> CaseImportGraphState:
        import_batch = self._ingest_batch(state["input_path"])
        return {"import_batch": import_batch}

    def _refresh_indexes_node(self, state: CaseImportGraphState) -> CaseImportGraphState:
        if state["import_batch"].structured_cases:
            state["sync_backend"](state["import_batch"].structured_cases)
        return {}

    def _finalize_report_node(self, state: CaseImportGraphState) -> CaseImportGraphState:
        execution_report = CaseImportExecutionReport(
            import_report=state["import_batch"].report,
            synced_backend=state["synced_backend"],
            requested_backend=state["requested_backend"],
        )
        return {"execution_report": execution_report}


def build_openai_import_client_from_env() -> OpenAICompatibleClient:
    config = OpenAICompatibleConfig.from_env()
    if not config.is_enabled():
        raise RuntimeError("OPENAI_API_KEY is not configured. Case import requires the OpenAI-compatible LLM API.")
    return OpenAICompatibleClient(config)


def import_cases_with_runtime(
    *,
    input_path: str | Path,
    runtime_config: CaseImportRuntimeConfig,
    llm_client: OpenAICompatibleClient,
) -> CaseImportExecutionReport:
    workflow = CaseImportWorkflow(runtime_config=runtime_config, llm_client=llm_client)
    return workflow.run(input_path=input_path)


def _corpus_has_records(corpus_path: Path) -> bool:
    if not corpus_path.exists():
        return False
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return True
    return False


def _prepare_lancedb_store(config: CaseImportRuntimeConfig) -> LanceDBCaseStore:
    ranker_config = resolve_lecard_ranker_config(
        bge_model_path=config.bge_model_path,
        bge_use_fp16=config.bge_use_fp16,
        bge_batch_size=config.bge_batch_size,
        bge_max_length=config.bge_max_length,
        device=config.device,
    )
    encoder = BGEM3DenseEncoder(
        model_name_or_path=ranker_config.bge_model_path,
        use_fp16=ranker_config.bge_use_fp16,
        batch_size=ranker_config.bge_batch_size,
        max_length=ranker_config.bge_max_length,
        device=ranker_config.device,
    )
    store = LanceDBCaseStore(source_path=config.corpus_path, db_uri=config.lancedb_uri, encoder=encoder)
    store.build(force_rebuild=False)
    return store


def _sync_sqlite(*, corpus_path: Path, db_path: Path) -> Callable[[list[StructuredCase]], None]:
    def _sync(_cases: list[StructuredCase]) -> None:
        store = LeCaRDSQLiteStore(source_path=corpus_path, db_path=db_path)
        try:
            store.ensure_ready()
        finally:
            store.close()

    return _sync


def _force_build_lancedb(config: CaseImportRuntimeConfig) -> Callable[[list[StructuredCase]], None]:
    ranker_config = resolve_lecard_ranker_config(
        bge_model_path=config.bge_model_path,
        bge_use_fp16=config.bge_use_fp16,
        bge_batch_size=config.bge_batch_size,
        bge_max_length=config.bge_max_length,
        device=config.device,
    )
    encoder = BGEM3DenseEncoder(
        model_name_or_path=ranker_config.bge_model_path,
        use_fp16=ranker_config.bge_use_fp16,
        batch_size=ranker_config.bge_batch_size,
        max_length=ranker_config.bge_max_length,
        device=ranker_config.device,
    )
    store = LanceDBCaseStore(source_path=config.corpus_path, db_uri=config.lancedb_uri, encoder=encoder)
    return lambda _cases: store.build(force_rebuild=True)


def _resolve_sync_backend(
    config: CaseImportRuntimeConfig,
    *,
    resolved_backend: str,
) -> tuple[str, Callable[[list[StructuredCase]], None]]:
    if resolved_backend == "sqlite":
        return "sqlite", _sync_sqlite(corpus_path=config.corpus_path, db_path=config.db_path)

    if resolved_backend == "lancedb":
        if _corpus_has_records(config.corpus_path):
            store = _prepare_lancedb_store(config)
            return "lancedb", lambda cases: store.add_cases(cases, assume_ready=True)
        return "lancedb", _force_build_lancedb(config)

    try:
        if _corpus_has_records(config.corpus_path):
            store = _prepare_lancedb_store(config)
            return "lancedb", lambda cases: store.add_cases(cases, assume_ready=True)
        return "lancedb", _force_build_lancedb(config)
    except Exception:
        return "sqlite", _sync_sqlite(corpus_path=config.corpus_path, db_path=config.db_path)
