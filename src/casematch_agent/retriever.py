from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import RetrievalResult, StructuredCase, StructuredQuery
from .utils import jaccard_similarity, law_name, overlap_ratio, tokenize_text


FIELD_WEIGHTS: dict[str, float] = {
    "charges": 0.24,
    "dispute_focus": 0.14,
    "legal_basis": 0.10,
    "four_element_subject": 0.05,
    "four_element_object": 0.05,
    "four_element_objective_aspect": 0.16,
    "four_element_subjective_aspect": 0.10,
    "court_reasoning": 0.08,
    "text": 0.08,
}


def _text_similarity(query: StructuredQuery, case: StructuredCase) -> float:
    query_terms = tokenize_text(f"{query.raw_query} {query.case_summary} {query.dispute_focus}")
    case_terms = tokenize_text(f"{case.case_summary} {case.dispute_focus} {case.court_reasoning}")
    return overlap_ratio(query_terms, case_terms)


def _text_similarity_from_texts(query_text: str, case_text: str) -> float:
    if not query_text or not case_text:
        return 0.0
    return overlap_ratio(tokenize_text(query_text), tokenize_text(case_text))


def _law_similarity(query_laws: list[str], case_laws: list[str]) -> float:
    exact = jaccard_similarity(query_laws, case_laws)
    if exact > 0:
        return exact
    query_names = [law_name(reference) for reference in query_laws]
    case_names = [law_name(reference) for reference in case_laws]
    return jaccard_similarity(query_names, case_names) * 0.8


class CaseCandidateRepository(Protocol):
    def candidate_cases(self, query: StructuredQuery, limit: int) -> list[StructuredCase]:
        ...


class CaseRanker(Protocol):
    def rank(self, query: StructuredQuery, candidates: list[StructuredCase], top_k: int) -> list[RetrievalResult]:
        ...


@dataclass
class HybridCaseRetriever:
    cases: list[StructuredCase]

    def search(self, query: StructuredQuery, top_k: int = 3) -> list[RetrievalResult]:
        results = [self._score_case(query, case) for case in self.cases]
        results.sort(key=lambda item: item.total_score, reverse=True)
        return results[:top_k]

    def _score_case(self, query: StructuredQuery, case: StructuredCase) -> RetrievalResult:
        field_scores = {
            "charges": jaccard_similarity(query.charges, case.charges),
            "dispute_focus": _text_similarity_from_texts(query.dispute_focus, case.dispute_focus),
            "legal_basis": _law_similarity(query.legal_basis, case.legal_basis),
            "four_element_subject": jaccard_similarity(query.four_element_subject, case.four_element_subject),
            "four_element_object": jaccard_similarity(query.four_element_object, case.four_element_object),
            "four_element_objective_aspect": jaccard_similarity(
                query.four_element_objective_aspect, case.four_element_objective_aspect
            ),
            "four_element_subjective_aspect": jaccard_similarity(
                query.four_element_subjective_aspect, case.four_element_subjective_aspect
            ),
            "court_reasoning": _text_similarity_from_texts(query.court_reasoning, case.court_reasoning),
            "text": _text_similarity(query, case),
        }

        total_score = 0.0
        for field_name, weight in FIELD_WEIGHTS.items():
            total_score += field_scores[field_name] * weight
        total_score = round(total_score, 4)

        reasons = self._build_reasons(query, case, field_scores)
        return RetrievalResult(case=case, total_score=total_score, field_scores=field_scores, reasons=reasons)

    def _build_reasons(
        self,
        query: StructuredQuery,
        case: StructuredCase,
        field_scores: dict[str, float],
    ) -> list[str]:
        reasons: list[str] = []
        if field_scores["charges"] >= 0.25:
            overlap = [item for item in query.charges if item in case.charges]
            reasons.append(f"罪名重合: {', '.join(overlap[:3])}")
        if field_scores["dispute_focus"] >= 0.18 and query.dispute_focus:
            reasons.append("争点表述相近")
        if field_scores["legal_basis"] >= 0.2 and query.legal_basis:
            reasons.append("法条依据存在重合")
        if field_scores["four_element_objective_aspect"] >= 0.25:
            overlap = [item for item in query.four_element_objective_aspect if item in case.four_element_objective_aspect]
            reasons.append(f"客观行为特征重合: {', '.join(overlap[:3])}")
        if field_scores["four_element_subjective_aspect"] >= 0.25:
            overlap = [item for item in query.four_element_subjective_aspect if item in case.four_element_subjective_aspect]
            reasons.append(f"主观方面重合: {', '.join(overlap[:3])}")
        if field_scores["court_reasoning"] >= 0.18 and query.court_reasoning:
            reasons.append("裁判说理相近")
        if field_scores["text"] >= 0.18:
            reasons.append("摘要与争点文本相似度较高")
        return reasons or ["由结构化字段和文本语义综合召回"]


@dataclass
class InMemoryCandidateRepository:
    cases: list[StructuredCase]

    def candidate_cases(self, query: StructuredQuery, limit: int) -> list[StructuredCase]:
        return self.cases[:limit]


@dataclass
class SimpleCaseRanker:
    def rank(self, query: StructuredQuery, candidates: list[StructuredCase], top_k: int) -> list[RetrievalResult]:
        return HybridCaseRetriever(candidates).search(query, top_k=top_k)


@dataclass
class PipelineCaseRetriever:
    repository: CaseCandidateRepository
    ranker: CaseRanker
    candidate_limit: int = 200

    def search(self, query: StructuredQuery, top_k: int = 3) -> list[RetrievalResult]:
        candidates = self.repository.candidate_cases(query, limit=max(self.candidate_limit, top_k * 40))
        if not candidates:
            return []
        return self.ranker.rank(query, candidates, top_k=top_k)
