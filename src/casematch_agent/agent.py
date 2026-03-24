from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List
import torch

from casematch_ranker import (
    BGEM3CaseRanker,
    BGEM3DenseEncoder,
    BM25CaseRanker,
    DEFAULT_HYBRID_BGE_FIELD_SPECS,
    HybridRanker,
)

from .clarification import HeuristicClarificationJudge, LLMClarificationJudge
from .corpus import load_lecard_corpus
from .extractor import HeuristicStructuredQueryExtractor, LLMStructuredQueryExtractor
from .lancedb_store import LanceDBCandidateRepository
from .llm import JsonLLMClient, OpenAICompatibleClient, OpenAICompatibleConfig
from .memory import LECARD_CLARIFICATION_FIELDS, QueryContextManager
from .models import (
    AgentResponse,
    AgentState,
    ClarificationStatus,
    ConversationMemory,
    RetrievalResult,
    StructuredQuery,
)
from .retriever import CaseRanker, InMemoryCandidateRepository, PipelineCaseRetriever, SimpleCaseRanker, HybridCaseRetriever
from .sample_cases import load_sample_cases
from .sqlite_store import SQLiteLeCaRDCandidateRepository

PROCESS_DATA_DIR = Path(__file__).resolve().parents[2] / "data/process"
DEFAULT_STRUCTURED_CORPUS_PATH = PROCESS_DATA_DIR / "lecard/corpus_merged.jsonl"
DEFAULT_LANCEDB_URI = PROCESS_DATA_DIR / "cases.lancedb"
DEFAULT_CASE_DB_PATH = PROCESS_DATA_DIR / "cases.sqlite3"

# Backward-compatible aliases kept for existing imports.
DEFAULT_LECARD_CORPUS_PATH = DEFAULT_STRUCTURED_CORPUS_PATH
DEFAULT_LECARD_DB_PATH = DEFAULT_CASE_DB_PATH


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _env_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def resolve_devices(devices: Optional[Union[str, List[str]]] = None) -> Optional[Union[str, List[str]]]:
    if devices is not None:
        return devices

    if not torch.cuda.is_available():
        return "cpu"

    cuda_visible_devices = os.getenv("CUDA_DEVICES", None)
    
    if cuda_visible_devices is not None:
        visible_indices = [int(x.strip()) for x in cuda_visible_devices.split(",")]

        if len(visible_indices) == 1:
            return f"cuda:{visible_indices[0]}"
        else:
            return [f"cuda:{i}" for i in visible_indices]
    
    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus == 1:
        return "cuda:0"

    return [f"cuda:{i}" for i in range(num_visible_gpus)]


@dataclass(frozen=True)
class LeCaRDRankerConfig:
    ranker_name: str = "bm25"
    bge_model_path: str = "BAAI/bge-m3"
    bge_use_fp16: bool = True
    bge_batch_size: int = 32
    bge_max_length: int = 4096
    hybrid_bm25_weight: float = 0.45
    hybrid_bge_weight: float = 0.55
    hybrid_fe_weight: float | None = None
    hybrid_lc_weight: float | None = None
    device: Optional[Union[str, List[str]]] = None

    @classmethod
    def from_env(cls) -> "LeCaRDRankerConfig":
        return cls(
            ranker_name=os.getenv("CASEMATCH_RANKER", "bm25").strip().lower() or "bm25",
            bge_model_path=os.getenv("CASEMATCH_BGE_MODEL_PATH", "BAAI/bge-m3").strip() or "BAAI/bge-m3",
            bge_use_fp16=_env_flag("CASEMATCH_BGE_USE_FP16", True),
            bge_batch_size=int(os.getenv("CASEMATCH_BGE_BATCH_SIZE", "32")),
            bge_max_length=int(os.getenv("CASEMATCH_BGE_MAX_LENGTH", "4096")),
            hybrid_bm25_weight=float(os.getenv("CASEMATCH_HYBRID_BM25_WEIGHT", "0.45")),
            hybrid_bge_weight=float(os.getenv("CASEMATCH_HYBRID_BGE_WEIGHT", "0.55")),
            hybrid_fe_weight=_env_optional_float("CASEMATCH_HYBRID_FE_WEIGHT"),
            hybrid_lc_weight=_env_optional_float("CASEMATCH_HYBRID_LC_WEIGHT"),
            device=resolve_devices(),
        )


@dataclass(frozen=True)
class CaseStoreConfig:
    backend: str = "auto"
    lancedb_uri: str | None = None

    @classmethod
    def from_env(cls) -> "CaseStoreConfig":
        return cls(
            backend=(os.getenv("CASEMATCH_DB_BACKEND", "auto").strip().lower() or "auto"),
            lancedb_uri=_env_optional_str("CASEMATCH_LANCEDB_URI"),
        )


def resolve_case_store_config(
    backend: str | None = None,
    lancedb_uri: str | None = None,
) -> CaseStoreConfig:
    base = CaseStoreConfig.from_env()
    return CaseStoreConfig(
        backend=(backend or base.backend).strip().lower(),
        lancedb_uri=lancedb_uri if lancedb_uri is not None else base.lancedb_uri,
    )


def _should_fallback_to_sqlite_auto(exc: Exception) -> bool:
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc)
    return "requires lancedb" in message or "requires FlagEmbedding" in message


def resolve_lecard_ranker_config(
    ranker_name: str | None = None,
    bge_model_path: str | None = None,
    bge_use_fp16: bool | None = None,
    bge_batch_size: int | None = None,
    bge_max_length: int | None = None,
    hybrid_bm25_weight: float | None = None,
    hybrid_bge_weight: float | None = None,
    hybrid_fe_weight: float | None = None,
    hybrid_lc_weight: float | None = None,
    device: str | List[str] | None = None
) -> LeCaRDRankerConfig:
    base = LeCaRDRankerConfig.from_env()
    return LeCaRDRankerConfig(
        ranker_name=(ranker_name or base.ranker_name).strip().lower(),
        bge_model_path=bge_model_path or base.bge_model_path,
        bge_use_fp16=base.bge_use_fp16 if bge_use_fp16 is None else bge_use_fp16,
        bge_batch_size=base.bge_batch_size if bge_batch_size is None else bge_batch_size,
        bge_max_length=base.bge_max_length if bge_max_length is None else bge_max_length,
        hybrid_bm25_weight=base.hybrid_bm25_weight if hybrid_bm25_weight is None else hybrid_bm25_weight,
        hybrid_bge_weight=base.hybrid_bge_weight if hybrid_bge_weight is None else hybrid_bge_weight,
        hybrid_fe_weight=base.hybrid_fe_weight if hybrid_fe_weight is None else hybrid_fe_weight,
        hybrid_lc_weight=base.hybrid_lc_weight if hybrid_lc_weight is None else hybrid_lc_weight,
        device=base.device if device is None else device,
    )


def build_lecard_ranker(config: LeCaRDRankerConfig | None = None, shared_bge_encoder: object | None = None) -> CaseRanker:
    resolved = config or LeCaRDRankerConfig.from_env()
    ranker_name = resolved.ranker_name.replace("-", "_")

    if ranker_name == "simple":
        return SimpleCaseRanker()
    if ranker_name == "bm25":
        return BM25CaseRanker()
    if ranker_name == "bge_m3":
        return BGEM3CaseRanker(
            encoder=shared_bge_encoder,
            model_name_or_path=resolved.bge_model_path,
            use_fp16=resolved.bge_use_fp16,
            batch_size=resolved.bge_batch_size,
            max_length=resolved.bge_max_length,
            device=resolved.device,
        )
    if ranker_name == "hybrid":
        bge_ranker = (
            BGEM3CaseRanker(
                encoder=shared_bge_encoder,
                field_specs=DEFAULT_HYBRID_BGE_FIELD_SPECS,
                model_name_or_path=resolved.bge_model_path,
                use_fp16=resolved.bge_use_fp16,
                batch_size=resolved.bge_batch_size,
                max_length=resolved.bge_max_length,
                device=resolved.device,
            )
            if resolved.hybrid_bge_weight > 0
            else None
        )
        return HybridRanker(
            bge_m3_ranker=bge_ranker,
            bm25_weight=resolved.hybrid_bm25_weight,
            bge_m3_weight=resolved.hybrid_bge_weight,
            bm25_fe_weight=resolved.hybrid_fe_weight,
            bm25_lc_weight=resolved.hybrid_lc_weight,
        )
    raise ValueError(f"Unsupported LeCaRD ranker: {resolved.ranker_name}")


@dataclass
class CaseMatchAgent:
    extractor: object
    retriever: PipelineCaseRetriever | HybridCaseRetriever
    clarification_judge: object
    context_manager: QueryContextManager = field(default_factory=QueryContextManager)

    def respond(self, user_message: str, state: AgentState | None = None, top_k: int = 3) -> AgentResponse:
        previous_memory = state.memory if state else ConversationMemory()
        current_query = self.extractor.extract(user_message)
        merged_query = self._merge_query(state.structured_query, current_query, user_message) if state else current_query
        current_memory = self.context_manager.update_after_user_turn(previous_memory, user_message, current_query)
        retrieval_results = self.retriever.search(merged_query, top_k=top_k)
        decision = self.clarification_judge.decide(merged_query, retrieval_results, memory=current_memory)
        next_memory = self.context_manager.update_after_clarification(current_memory, decision)
        next_state = AgentState(
            structured_query=merged_query,
            turn_count=(state.turn_count + 1) if state else 1,
            waiting_for_clarification=decision.status == ClarificationStatus.NEED_MORE_INFO,
            memory=next_memory,
        )
        narrative = self._narrative(merged_query, decision, retrieval_results)
        return AgentResponse(
            state=next_state,
            structured_query=merged_query,
            decision=decision,
            retrieval_results=retrieval_results,
            narrative=narrative,
        )

    def _narrative(
        self,
        query: StructuredQuery,
        decision,
        retrieval_results: list[RetrievalResult],
    ) -> str:
        summary_parts = []
        if query.charges:
            summary_parts.append(f"罪名: {', '.join(query.charges[:3])}")
        if query.dispute_focus:
            summary_parts.append(f"争点: {query.dispute_focus}")
        if query.four_element_objective_aspect:
            summary_parts.append(f"客观方面: {', '.join(query.four_element_objective_aspect[:3])}")

        if decision.status == ClarificationStatus.NEED_MORE_INFO:
            if retrieval_results:
                top_case = retrieval_results[0].case
                base = f"已先基于当前信息给出一版类案结果，当前最接近的是“{top_case.case_name or top_case.case_id}”。"
            else:
                base = "已先完成一轮检索，但暂未召回到足够接近的类案。"
            if summary_parts:
                base = f"{base} 当前已识别信息包括: {'; '.join(summary_parts)}。"
            return f"{base} 如果你愿意继续补充，我可以进一步缩小范围。"

        if retrieval_results:
            top_case = retrieval_results[0].case
            category = "、".join(top_case.charges[:2]) or "未知类型"
            return (
                f"当前信息已经可以支撑类案检索，优先命中的案例是“{top_case.case_name or top_case.case_id}”，"
                f"案件类别为{category}。"
            )
        return "当前没有召回到合适案例，建议补充更多案情信息。"

    def _merge_query(
        self,
        previous_query: StructuredQuery,
        current_query: StructuredQuery,
        latest_user_message: str,
    ) -> StructuredQuery:
        merge_queries = getattr(self.extractor, "merge_queries", None)
        if callable(merge_queries):
            return merge_queries(previous_query, current_query, latest_user_message)
        return previous_query.merge(current_query)


def _build_agent(client: JsonLLMClient | None, cases) -> CaseMatchAgent:
    heuristic_extractor = HeuristicStructuredQueryExtractor()
    heuristic_judge = HeuristicClarificationJudge()
    llm_client = client
    if llm_client is None:
        config = OpenAICompatibleConfig.from_env()
        if config.is_enabled():
            llm_client = OpenAICompatibleClient(config)

    if llm_client is None:
        extractor = heuristic_extractor
        clarification_judge = heuristic_judge
    else:
        extractor = LLMStructuredQueryExtractor(client=llm_client, fallback=heuristic_extractor)
        clarification_judge = LLMClarificationJudge(client=llm_client, fallback=heuristic_judge)

    return CaseMatchAgent(
        extractor=extractor,
        retriever=PipelineCaseRetriever(
            repository=InMemoryCandidateRepository(cases),
            ranker=SimpleCaseRanker(),
            candidate_limit=len(cases),
        ),
        clarification_judge=clarification_judge,
    )


def build_default_agent(client: JsonLLMClient | None = None) -> CaseMatchAgent:
    return _build_agent(client=client, cases=load_sample_cases())


def build_lecard_agent(
    corpus_path: str | Path = DEFAULT_LECARD_CORPUS_PATH,
    db_path: str | Path = DEFAULT_LECARD_DB_PATH,
    lancedb_uri: str | Path | None = None,
    client: JsonLLMClient | None = None,
    ranker: CaseRanker | None = None,
    candidate_limit: int = 200,
    ranker_name: str | None = None,
    bge_model_path: str | None = None,
    bge_use_fp16: bool | None = None,
    bge_batch_size: int | None = None,
    bge_max_length: int | None = None,
    hybrid_bm25_weight: float | None = None,
    hybrid_bge_weight: float | None = None,
    hybrid_fe_weight: float | None = None,
    hybrid_lc_weight: float | None = None,
    db_backend: str | None = None,
    device: str | List[str] | None = None
) -> CaseMatchAgent:
    heuristic_extractor = HeuristicStructuredQueryExtractor()
    heuristic_judge = HeuristicClarificationJudge(supported_fields=LECARD_CLARIFICATION_FIELDS)
    llm_client = client
    if llm_client is None:
        config = OpenAICompatibleConfig.from_env()
        if config.is_enabled():
            llm_client = OpenAICompatibleClient(config)

    if llm_client is None:
        extractor = heuristic_extractor
        clarification_judge = heuristic_judge
    else:
        extractor = LLMStructuredQueryExtractor(client=llm_client, fallback=heuristic_extractor)
        clarification_judge = LLMClarificationJudge(
            client=llm_client,
            fallback=heuristic_judge,
            supported_fields=LECARD_CLARIFICATION_FIELDS,
        )

    ranker_config = resolve_lecard_ranker_config(
        ranker_name=ranker_name,
        bge_model_path=bge_model_path,
        bge_use_fp16=bge_use_fp16,
        bge_batch_size=bge_batch_size,
        bge_max_length=bge_max_length,
        hybrid_bm25_weight=hybrid_bm25_weight,
        hybrid_bge_weight=hybrid_bge_weight,
        hybrid_fe_weight=hybrid_fe_weight,
        hybrid_lc_weight=hybrid_lc_weight,
        device=device,
    )
    store_config = resolve_case_store_config(
        backend=db_backend,
        lancedb_uri=str(lancedb_uri) if lancedb_uri is not None else None,
    )
    resolved_lancedb_uri = Path(store_config.lancedb_uri or DEFAULT_LANCEDB_URI)

    shared_bge_encoder: object | None = None
    repository: object
    if store_config.backend == "lancedb":
        shared_bge_encoder = BGEM3DenseEncoder(
            model_name_or_path=ranker_config.bge_model_path,
            use_fp16=ranker_config.bge_use_fp16,
            batch_size=ranker_config.bge_batch_size,
            max_length=ranker_config.bge_max_length,
            device=ranker_config.device,
        )
        repository = LanceDBCandidateRepository(
            source_path=Path(corpus_path),
            db_uri=resolved_lancedb_uri,
            encoder=shared_bge_encoder,
        )
    elif store_config.backend == "auto":
        shared_bge_encoder = BGEM3DenseEncoder(
            model_name_or_path=ranker_config.bge_model_path,
            use_fp16=ranker_config.bge_use_fp16,
            batch_size=ranker_config.bge_batch_size,
            max_length=ranker_config.bge_max_length,
            device=ranker_config.device,
        )
        try:
            repository = LanceDBCandidateRepository(
                source_path=Path(corpus_path),
                db_uri=resolved_lancedb_uri,
                encoder=shared_bge_encoder,
            )
        except Exception as exc:
            if _should_fallback_to_sqlite_auto(exc):
                shared_bge_encoder = None
                repository = SQLiteLeCaRDCandidateRepository(source_path=Path(corpus_path), db_path=Path(db_path))
            else:
                raise
    elif store_config.backend == "sqlite":
        repository = SQLiteLeCaRDCandidateRepository(source_path=Path(corpus_path), db_path=Path(db_path))
    else:
        raise ValueError(f"Unsupported database backend: {store_config.backend}")

    resolved_ranker = ranker or build_lecard_ranker(ranker_config, shared_bge_encoder=shared_bge_encoder)

    return CaseMatchAgent(
        extractor=extractor,
        retriever=PipelineCaseRetriever(
            repository=repository,
            ranker=resolved_ranker,
            candidate_limit=candidate_limit,
        ),
        clarification_judge=clarification_judge,
    )
