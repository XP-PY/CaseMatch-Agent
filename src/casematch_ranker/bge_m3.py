from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from casematch_agent.models import RetrievalResult, StructuredCase, StructuredQuery


def _join(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def _section(label: str, text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    return f"{label}: {text}"


def _query_semantic_profile(query: StructuredQuery) -> str:
    return _join(
        [
            _section("案情摘要", query.case_summary or query.raw_query),
            _section("争议焦点", query.dispute_focus),
            _section("裁判理由", query.court_reasoning),
        ]
    )


def _case_semantic_profile(case: StructuredCase) -> str:
    return _join(
        [
            _section("案情摘要", case.case_summary),
            _section("争议焦点", case.dispute_focus),
            _section("裁判理由", case.court_reasoning or case.traceability_quote),
        ]
    )


def _query_behavior_profile(query: StructuredQuery) -> str:
    return _join(
        [
            _section("主体", _join(query.four_element_subject)),
            _section("客体", _join(query.four_element_object)),
            _section("客观方面", _join(query.four_element_objective_aspect)),
            _section("主观方面", _join(query.four_element_subjective_aspect)),
        ]
    )


def _case_behavior_profile(case: StructuredCase) -> str:
    return _join(
        [
            _section("主体", _join(case.four_element_subject)),
            _section("客体", _join(case.four_element_object)),
            _section("客观方面", _join(case.four_element_objective_aspect)),
            _section("主观方面", _join(case.four_element_subjective_aspect)),
        ]
    )


def _query_legal_profile(query: StructuredQuery) -> str:
    return _join(
        [
            _section("罪名", _join(query.charges)),
            _section("法条", _join(query.legal_basis)),
        ]
    )


def _case_legal_profile(case: StructuredCase) -> str:
    return _join(
        [
            _section("罪名", _join(case.charges)),
            _section("法条", _join(case.legal_basis)),
        ]
    )


def _to_float_list(vector: object) -> list[float]:
    if hasattr(vector, "tolist"):
        value = vector.tolist()
        if isinstance(value, list):
            return [float(item) for item in value]
    if isinstance(vector, (list, tuple)):
        return [float(item) for item in vector]
    return []


def _normalize_vector(vector: list[float]) -> list[float]:
    if not vector:
        return []
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return []
    return [value / norm for value in vector]


def _dot(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    return sum(left[idx] * right[idx] for idx in range(size))


@dataclass(frozen=True)
class BGEM3FieldSpec:
    name: str
    label: str
    weight: float
    query_text: Callable[[StructuredQuery], str]
    case_text: Callable[[StructuredCase], str]


DEFAULT_BGEM3_FIELD_SPECS: tuple[BGEM3FieldSpec, ...] = (
    BGEM3FieldSpec(
        name="semantic_profile",
        label="案情与争点语义",
        weight=0.7,
        query_text=_query_semantic_profile,
        case_text=_case_semantic_profile,
    ),
    BGEM3FieldSpec(
        name="behavior_profile",
        label="行为模式与四要件",
        weight=0.2,
        query_text=_query_behavior_profile,
        case_text=_case_behavior_profile,
    ),
    BGEM3FieldSpec(
        name="legal_profile",
        label="罪名与法条",
        weight=0.1,
        query_text=_query_legal_profile,
        case_text=_case_legal_profile,
    ),
)


class BGEM3DenseEncoder:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        max_length: int = 4096,
        device: str | list[str] | None = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None
        self.device = device

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        payload = model.encode(texts, batch_size=self.batch_size, max_length=self.max_length)
        dense_vectors = payload.get("dense_vecs") if isinstance(payload, dict) else payload
        return [_to_float_list(vector) for vector in dense_vectors]

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "BGEM3CaseRanker requires FlagEmbedding. "
                "Install it first, or pass a custom encoder into BGEM3CaseRanker."
            ) from exc
        self._model = BGEM3FlagModel(
            self.model_name_or_path, 
            use_fp16=self.use_fp16,
            devices=self.device
        )
        return self._model


@dataclass
class BGEM3CaseRanker:
    encoder: object | None = None
    field_specs: tuple[BGEM3FieldSpec, ...] = DEFAULT_BGEM3_FIELD_SPECS
    model_name_or_path: str = "BAAI/bge-m3"
    use_fp16: bool = True
    batch_size: int = 32
    max_length: int = 4096
    _embedding_cache: dict[str, dict[str, list[float]]] = field(default_factory=dict, init=False, repr=False)
    device: str | list[str] | None = None

    def __post_init__(self) -> None:
        if self.encoder is None:
            self.encoder = BGEM3DenseEncoder(
                model_name_or_path=self.model_name_or_path,
                use_fp16=self.use_fp16,
                batch_size=self.batch_size,
                max_length=self.max_length,
                device=self.device,
            )

    def rank(self, query: StructuredQuery, candidates: list[StructuredCase], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []

        active_specs: list[BGEM3FieldSpec] = []
        field_scores_by_spec: dict[str, list[float]] = {}

        for spec in self.field_specs:
            query_vector = self._encode_text(spec.query_text(query))
            if not query_vector:
                continue

            candidate_vectors = self._candidate_vectors(candidates, spec)
            if not any(candidate_vectors):
                continue

            similarities = [max(0.0, _dot(query_vector, candidate_vector)) for candidate_vector in candidate_vectors]
            if max(similarities, default=0.0) <= 0.0:
                continue

            field_scores_by_spec[spec.name] = similarities
            active_specs.append(spec)

        if not active_specs:
            return [
                RetrievalResult(
                    case=case,
                    total_score=0.0,
                    field_scores={"bge_m3_total": 0.0},
                    reasons=["BGE-M3 未命中有效字段"],
                )
                for case in candidates[:top_k]
            ]

        total_weight = sum(spec.weight for spec in active_specs)
        normalized_weights = {spec.name: spec.weight / total_weight for spec in active_specs}

        results: list[RetrievalResult] = []
        for index, case in enumerate(candidates):
            field_scores: dict[str, float] = {}
            contributions: list[tuple[float, BGEM3FieldSpec]] = []
            total_score = 0.0

            for spec in active_specs:
                similarity = field_scores_by_spec[spec.name][index]
                weighted_similarity = similarity * normalized_weights[spec.name]
                field_scores[spec.name] = round(similarity, 4)
                field_scores[f"{spec.name}_weighted"] = round(weighted_similarity, 4)
                contributions.append((weighted_similarity, spec))
                total_score += weighted_similarity

            total_score = round(total_score, 4)
            field_scores["bge_m3_total"] = total_score
            results.append(
                RetrievalResult(
                    case=case,
                    total_score=total_score,
                    field_scores=field_scores,
                    reasons=self._build_reasons(case, contributions),
                )
            )

        results.sort(key=lambda item: item.total_score, reverse=True)
        return results[:top_k]

    def _candidate_vectors(self, candidates: list[StructuredCase], spec: BGEM3FieldSpec) -> list[list[float]]:
        vectors: list[list[float]] = []
        missing_entries: list[tuple[str, str]] = []

        for case in candidates:
            cache_key = self._case_cache_key(case)
            cache_slot = self._embedding_cache.setdefault(cache_key, {})
            if spec.name not in cache_slot:
                case_text = spec.case_text(case).strip()
                if case_text:
                    missing_entries.append((cache_key, case_text))
                else:
                    cache_slot[spec.name] = []

        if missing_entries:
            encoded_vectors = self.encoder.encode([text for _, text in missing_entries])
            for (cache_key, _), vector in zip(missing_entries, encoded_vectors):
                self._embedding_cache[cache_key][spec.name] = _normalize_vector(vector)

        for case in candidates:
            cache_key = self._case_cache_key(case)
            vectors.append(self._embedding_cache[cache_key].get(spec.name, []))
        return vectors

    def _encode_text(self, text: str) -> list[float]:
        text = text.strip()
        if not text:
            return []
        vectors = self.encoder.encode([text])
        if not vectors:
            return []
        return _normalize_vector(vectors[0])

    def _case_cache_key(self, case: StructuredCase) -> str:
        return case.case_id

    def _build_reasons(
        self,
        case: StructuredCase,
        contributions: list[tuple[float, BGEM3FieldSpec]],
    ) -> list[str]:
        reasons: list[str] = []
        for contribution, spec in sorted(contributions, key=lambda item: item[0], reverse=True):
            if contribution < 0.06:
                continue
            reason = self._reason_for_field(case, spec)
            if reason and reason not in reasons:
                reasons.append(reason)
            if len(reasons) == 3:
                break
        return reasons or ["BGE-M3 语义相似度较高"]

    def _reason_for_field(self, case: StructuredCase, spec: BGEM3FieldSpec) -> str:
        if spec.name == "semantic_profile" and (case.case_summary or case.dispute_focus or case.court_reasoning):
            return "BGE-M3 命中案情与争点语义"
        if spec.name == "behavior_profile" and (
            case.four_element_subject
            or case.four_element_object
            or case.four_element_objective_aspect
            or case.four_element_subjective_aspect
        ):
            return "BGE-M3 命中行为模式与四要件"
        if spec.name == "legal_profile" and (case.charges or case.legal_basis):
            return "BGE-M3 命中罪名与法条语义"
        return ""
