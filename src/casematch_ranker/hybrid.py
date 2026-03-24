from __future__ import annotations

from dataclasses import dataclass, field

from casematch_agent.models import RetrievalResult, StructuredCase, StructuredQuery
from casematch_agent.search_profiles import (
    case_four_elements_text,
    case_fused_text,
    case_laws_and_charges_text,
    query_four_elements_text,
    query_fused_text,
    query_laws_and_charges_text,
)

from .bge_m3 import BGEM3CaseRanker, BGEM3FieldSpec
from .bm25 import BM25CaseRanker, BM25FieldSpec, bm25_tokenize


def _case_key(case: StructuredCase) -> str:
    return case.case_id


def _jieba_tokenize(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        import jieba
    except ImportError:
        return bm25_tokenize(stripped)
    return [token.strip() for token in jieba.lcut(stripped) if token.strip()]


DEFAULT_HYBRID_BGE_FIELD_SPECS: tuple[BGEM3FieldSpec, ...] = (
    BGEM3FieldSpec(
        name="fused_text",
        label="案情摘要+争点+说理语义",
        weight=1.0,
        query_text=query_fused_text,
        case_text=case_fused_text,
    ),
)

DEFAULT_HYBRID_FE_FIELD_SPECS: tuple[BM25FieldSpec, ...] = (
    BM25FieldSpec(
        name="four_elements_bonus",
        label="四要件补充分",
        weight=1.0,
        query_text=query_four_elements_text,
        case_text=case_four_elements_text,
    ),
)

DEFAULT_HYBRID_LC_FIELD_SPECS: tuple[BM25FieldSpec, ...] = (
    BM25FieldSpec(
        name="laws_and_charges_bonus",
        label="法条罪名补充分",
        weight=1.0,
        query_text=query_laws_and_charges_text,
        case_text=case_laws_and_charges_text,
    ),
)


@dataclass
class HybridRanker:
    bm25_ranker: object | None = None
    bge_m3_ranker: object | None = None
    bm25_fe_ranker: object | None = None
    bm25_lc_ranker: object | None = None
    bm25_weight: float = 0.0
    bge_m3_weight: float = 1.0
    bm25_fe_weight: float | None = None
    bm25_lc_weight: float | None = None
    _resolved_bm25_fe_weight: float = field(init=False, repr=False)
    _resolved_bm25_lc_weight: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._resolved_bm25_fe_weight = self.bm25_weight if self.bm25_fe_weight is None else self.bm25_fe_weight
        self._resolved_bm25_lc_weight = self.bm25_weight if self.bm25_lc_weight is None else self.bm25_lc_weight

        if self.bge_m3_weight < 0 or self._resolved_bm25_fe_weight < 0 or self._resolved_bm25_lc_weight < 0:
            raise ValueError("HybridRanker weights must be non-negative.")
        if self.bge_m3_weight == 0 and self._resolved_bm25_fe_weight == 0 and self._resolved_bm25_lc_weight == 0:
            raise ValueError("HybridRanker requires at least one positive weight.")

        if self.bge_m3_weight > 0 and self.bge_m3_ranker is None:
            self.bge_m3_ranker = BGEM3CaseRanker(field_specs=DEFAULT_HYBRID_BGE_FIELD_SPECS)

        if self._resolved_bm25_fe_weight > 0 and self.bm25_fe_ranker is None:
            self.bm25_fe_ranker = BM25CaseRanker(
                field_specs=DEFAULT_HYBRID_FE_FIELD_SPECS,
                tokenizer=_jieba_tokenize,
                normalization="min_max",
            )

        if self._resolved_bm25_lc_weight > 0 and self.bm25_lc_ranker is None:
            self.bm25_lc_ranker = BM25CaseRanker(
                field_specs=DEFAULT_HYBRID_LC_FIELD_SPECS,
                tokenizer=_jieba_tokenize,
                normalization="min_max",
            )

        if self.bm25_ranker is not None:
            if self.bm25_fe_ranker is None and self._resolved_bm25_fe_weight > 0:
                self.bm25_fe_ranker = self.bm25_ranker
            if self.bm25_lc_ranker is None and self._resolved_bm25_lc_weight > 0:
                self.bm25_lc_ranker = self.bm25_ranker

    def rank(self, query: StructuredQuery, candidates: list[StructuredCase], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []

        bge_results = self._run_ranker(self.bge_m3_ranker, self.bge_m3_weight, query, candidates)
        fe_results = self._run_ranker(self.bm25_fe_ranker, self._resolved_bm25_fe_weight, query, candidates)
        lc_results = self._run_ranker(self.bm25_lc_ranker, self._resolved_bm25_lc_weight, query, candidates)

        results: list[RetrievalResult] = []
        for case in candidates:
            case_key = _case_key(case)
            bge_result = bge_results.get(case_key)
            fe_result = fe_results.get(case_key)
            lc_result = lc_results.get(case_key)

            bge_score = bge_result.total_score if bge_result is not None else 0.0
            fe_score = fe_result.total_score if fe_result is not None else 0.0
            lc_score = lc_result.total_score if lc_result is not None else 0.0
            total_score = round(
                self.bge_m3_weight * bge_score
                + self._resolved_bm25_fe_weight * fe_score
                + self._resolved_bm25_lc_weight * lc_score,
                4,
            )

            field_scores = self._merge_field_scores(
                bge_score=bge_score,
                fe_score=fe_score,
                lc_score=lc_score,
                total_score=total_score,
                bge_result=bge_result,
                fe_result=fe_result,
                lc_result=lc_result,
            )
            reasons = self._merge_reasons(
                bge_score=bge_score,
                fe_score=fe_score,
                lc_score=lc_score,
                bge_result=bge_result,
                fe_result=fe_result,
                lc_result=lc_result,
            )

            results.append(
                RetrievalResult(
                    case=case,
                    total_score=total_score,
                    field_scores=field_scores,
                    reasons=reasons,
                )
            )

        results.sort(key=lambda item: item.total_score, reverse=True)
        return results[:top_k]

    def _run_ranker(
        self,
        ranker: object | None,
        weight: float,
        query: StructuredQuery,
        candidates: list[StructuredCase],
    ) -> dict[str, RetrievalResult]:
        if weight <= 0 or ranker is None:
            return {}
        results = ranker.rank(query, candidates, top_k=len(candidates))
        return {_case_key(result.case): result for result in results}

    def _merge_field_scores(
        self,
        *,
        bge_score: float,
        fe_score: float,
        lc_score: float,
        total_score: float,
        bge_result: RetrievalResult | None,
        fe_result: RetrievalResult | None,
        lc_result: RetrievalResult | None,
    ) -> dict[str, float]:
        field_scores = {
            "hybrid_total": total_score,
            "hybrid_bge_m3": round(bge_score, 4),
            "hybrid_bge_base": round(bge_score, 4),
            "hybrid_bm25": round(fe_score + lc_score, 4),
            "hybrid_bm25_fe_bonus": round(fe_score, 4),
            "hybrid_bm25_lc_bonus": round(lc_score, 4),
        }
        if bge_result is not None:
            field_scores["bge_m3_raw_total"] = bge_result.total_score
            for field_name, value in bge_result.field_scores.items():
                field_scores[f"bge_m3_{field_name}"] = value
        if fe_result is not None:
            field_scores["bm25_fe_raw_total"] = fe_result.total_score
            for field_name, value in fe_result.field_scores.items():
                field_scores[f"bm25_fe_{field_name}"] = value
        if lc_result is not None:
            field_scores["bm25_lc_raw_total"] = lc_result.total_score
            for field_name, value in lc_result.field_scores.items():
                field_scores[f"bm25_lc_{field_name}"] = value
        return field_scores

    def _merge_reasons(
        self,
        *,
        bge_score: float,
        fe_score: float,
        lc_score: float,
        bge_result: RetrievalResult | None,
        fe_result: RetrievalResult | None,
        lc_result: RetrievalResult | None,
    ) -> list[str]:
        reasons: list[str] = []
        if bge_score >= 0.15:
            reasons.append("BGE-M3 命中融合语义主分")
        if fe_score >= 0.08:
            reasons.append("BM25 命中四要件补充分")
        if lc_score >= 0.08:
            reasons.append("BM25 命中法条罪名补充分")

        for result in [bge_result, fe_result, lc_result]:
            if result is None:
                continue
            for reason in result.reasons:
                if reason not in reasons:
                    reasons.append(reason)
                if len(reasons) == 6:
                    return reasons
        return reasons or ["BGE 主分与 BM25 bonus 综合得分较高"]
