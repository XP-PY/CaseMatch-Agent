from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from casematch_ranker import BGEM3DenseEncoder

from .agent import resolve_case_store_config, resolve_lecard_ranker_config
from .case_ingestion import CaseImportReport, CriminalCaseStructuredDataExtractor, import_raw_cases_from_jsonl
from .llm import OpenAICompatibleClient, OpenAICompatibleConfig
from .models import StructuredCase
from .lancedb_store import LanceDBCaseStore
from .sqlite_store import LeCaRDSQLiteStore


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
    store_config = resolve_case_store_config(
        backend=runtime_config.db_backend,
        lancedb_uri=str(runtime_config.lancedb_uri),
    )
    extractor = CriminalCaseStructuredDataExtractor(client=llm_client)
    synced_backend, sync_backend = _resolve_sync_backend(runtime_config, resolved_backend=store_config.backend)
    import_report = import_raw_cases_from_jsonl(
        input_path=input_path,
        corpus_path=runtime_config.corpus_path,
        extractor=extractor,
        sync_backend=sync_backend,
        case_id_prefix=runtime_config.case_id_prefix,
    )
    return CaseImportExecutionReport(
        import_report=import_report,
        synced_backend=synced_backend,
        requested_backend=store_config.backend,
    )


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
