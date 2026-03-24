from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from casematch_agent.models import RetrievalResult, StructuredCase, StructuredQuery
from casematch_agent.utils import normalize_text


_CHINESE_BLOCK = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM_BLOCK = re.compile(r"[a-z0-9]+")


def bm25_tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens: list[str] = []

    for word in _ALNUM_BLOCK.findall(normalized):
        if len(word) >= 2:
            tokens.append(word)

    for segment in _CHINESE_BLOCK.findall(normalized):
        if 2 <= len(segment) <= 8:
            tokens.append(segment)
        for size in (2, 3):
            if len(segment) < size:
                continue
            for start in range(len(segment) - size + 1):
                tokens.append(segment[start : start + size])
    return tokens


def _join(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def _query_case_summary(query: StructuredQuery) -> str:
    return query.case_summary or query.raw_query


def _query_dispute_focus(query: StructuredQuery) -> str:
    return query.dispute_focus


def _query_charges(query: StructuredQuery) -> str:
    return _join(query.charges)


def _query_legal_basis(query: StructuredQuery) -> str:
    return _join(query.legal_basis)


def _query_four_subject(query: StructuredQuery) -> str:
    return _join(query.four_element_subject)


def _query_four_object(query: StructuredQuery) -> str:
    return _join(query.four_element_object)


def _query_four_objective(query: StructuredQuery) -> str:
    return _join(query.four_element_objective_aspect)


def _query_four_subjective(query: StructuredQuery) -> str:
    return _join(query.four_element_subjective_aspect)


def _query_court_reasoning(query: StructuredQuery) -> str:
    return query.court_reasoning


def _case_case_summary(case: StructuredCase) -> str:
    return case.case_summary


def _case_dispute_focus(case: StructuredCase) -> str:
    return case.dispute_focus


def _case_charges(case: StructuredCase) -> str:
    return _join(case.charges)


def _case_legal_basis(case: StructuredCase) -> str:
    return _join(case.legal_basis)


def _case_four_subject(case: StructuredCase) -> str:
    return _join(case.four_element_subject)


def _case_four_object(case: StructuredCase) -> str:
    return _join(case.four_element_object)


def _case_four_objective(case: StructuredCase) -> str:
    return _join(case.four_element_objective_aspect)


def _case_four_subjective(case: StructuredCase) -> str:
    return _join(case.four_element_subjective_aspect)


def _case_court_reasoning(case: StructuredCase) -> str:
    return case.court_reasoning or case.traceability_quote


@dataclass(frozen=True)
class BM25FieldSpec:
    name: str
    label: str
    weight: float
    query_text: Callable[[StructuredQuery], str]
    case_text: Callable[[StructuredCase], str]


DEFAULT_BM25_FIELD_SPECS: tuple[BM25FieldSpec, ...] = (
    BM25FieldSpec(
        name="case_summary",
        label="案情摘要",
        weight=0.36,
        query_text=_query_case_summary,
        case_text=_case_case_summary,
    ),
    BM25FieldSpec(
        name="charges",
        label="罪名",
        weight=0.16,
        query_text=_query_charges,
        case_text=_case_charges,
    ),
    BM25FieldSpec(
        name="dispute_focus",
        label="争议焦点",
        weight=0.14,
        query_text=_query_dispute_focus,
        case_text=_case_dispute_focus,
    ),
    BM25FieldSpec(
        name="court_reasoning",
        label="裁判理由",
        weight=0.12,
        query_text=_query_court_reasoning,
        case_text=_case_court_reasoning,
    ),
    BM25FieldSpec(
        name="legal_basis",
        label="适用法条",
        weight=0.08,
        query_text=_query_legal_basis,
        case_text=_case_legal_basis,
    ),
    BM25FieldSpec(
        name="four_element_objective_aspect",
        label="客观方面",
        weight=0.07,
        query_text=_query_four_objective,
        case_text=_case_four_objective,
    ),
    BM25FieldSpec(
        name="four_element_subjective_aspect",
        label="主观方面",
        weight=0.03,
        query_text=_query_four_subjective,
        case_text=_case_four_subjective,
    ),
    BM25FieldSpec(
        name="four_element_subject",
        label="主体",
        weight=0.02,
        query_text=_query_four_subject,
        case_text=_case_four_subject,
    ),
    BM25FieldSpec(
        name="four_element_object",
        label="客体",
        weight=0.02,
        query_text=_query_four_object,
        case_text=_case_four_object,
    ),
)


@dataclass
class _BM25Index:
    documents: list[list[str]]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_term_freqs = [Counter(doc) for doc in self.documents]
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.doc_freqs: Counter[str] = Counter()
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}

        for doc_idx, term_freqs in enumerate(self.doc_term_freqs):
            for term, frequency in term_freqs.items():
                self.doc_freqs[term] += 1
                self.inverted_index.setdefault(term, []).append((doc_idx, frequency))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        if not self.documents or not query_tokens or self.avg_doc_length <= 0:
            return [0.0 for _ in self.documents]

        scores = [0.0 for _ in self.documents]
        unique_terms = list(dict.fromkeys(token for token in query_tokens if token))
        for term in unique_terms:
            postings = self.inverted_index.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            for doc_idx, frequency in postings:
                doc_length = self.doc_lengths[doc_idx]
                denominator = frequency + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                scores[doc_idx] += idf * (frequency * (self.k1 + 1) / denominator)
        return scores

    def _idf(self, term: str) -> float:
        document_frequency = self.doc_freqs.get(term, 0)
        if document_frequency <= 0:
            return 0.0
        return math.log(1.0 + (len(self.documents) - document_frequency + 0.5) / (document_frequency + 0.5))


@dataclass
class BM25CaseRanker:
    field_specs: tuple[BM25FieldSpec, ...] = DEFAULT_BM25_FIELD_SPECS
    k1: float = 1.5
    b: float = 0.75
    tokenizer: Callable[[str], list[str]] = bm25_tokenize
    normalization: str = "max"
    _token_cache: dict[str, dict[str, list[str]]] = field(default_factory=dict, init=False, repr=False)

    def rank(self, query: StructuredQuery, candidates: list[StructuredCase], top_k: int) -> list[RetrievalResult]:
        if not candidates:
            return []

        active_specs: list[BM25FieldSpec] = []
        normalized_field_scores: dict[str, list[float]] = {}

        for spec in self.field_specs:
            query_text = spec.query_text(query).strip()
            query_tokens = self.tokenizer(query_text)
            if not query_tokens:
                continue

            documents = [self._case_tokens(case, spec) for case in candidates]
            if not any(documents):
                continue

            raw_scores = _BM25Index(documents, k1=self.k1, b=self.b).get_scores(query_tokens)
            normalized_scores = self._normalize_scores(raw_scores)
            if max(normalized_scores, default=0.0) <= 0.0:
                continue

            normalized_field_scores[spec.name] = normalized_scores
            active_specs.append(spec)

        if not active_specs:
            return [
                RetrievalResult(
                    case=case,
                    total_score=0.0,
                    field_scores={"bm25_total": 0.0},
                    reasons=["BM25 未命中有效字段"],
                )
                for case in candidates[:top_k]
            ]

        total_weight = sum(spec.weight for spec in active_specs)
        normalized_weights = {spec.name: spec.weight / total_weight for spec in active_specs}

        results: list[RetrievalResult] = []
        for idx, case in enumerate(candidates):
            field_scores: dict[str, float] = {}
            contributions: list[tuple[float, BM25FieldSpec]] = []
            total_score = 0.0

            for spec in active_specs:
                raw_score = normalized_field_scores[spec.name][idx]
                weighted_score = raw_score * normalized_weights[spec.name]
                field_scores[spec.name] = round(raw_score, 4)
                field_scores[f"{spec.name}_weighted"] = round(weighted_score, 4)
                contributions.append((weighted_score, spec))
                total_score += weighted_score

            total_score = round(total_score, 4)
            field_scores["bm25_total"] = total_score
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

    def _case_tokens(self, case: StructuredCase, spec: BM25FieldSpec) -> list[str]:
        cache_key = self._case_cache_key(case)
        if cache_key not in self._token_cache:
            self._token_cache[cache_key] = {}
        if spec.name not in self._token_cache[cache_key]:
            self._token_cache[cache_key][spec.name] = self.tokenizer(spec.case_text(case))
        return self._token_cache[cache_key][spec.name]

    def _normalize_scores(self, raw_scores: list[float]) -> list[float]:
        if not raw_scores:
            return []

        max_score = max(raw_scores)
        if self.normalization == "max":
            if max_score <= 0.0:
                return [0.0 for _ in raw_scores]
            return [score / max_score for score in raw_scores]

        if self.normalization == "min_max":
            min_score = min(raw_scores)
            if max_score <= min_score:
                return [0.0 for _ in raw_scores]
            scale = max_score - min_score
            return [(score - min_score) / scale for score in raw_scores]

        raise ValueError(f"Unsupported BM25 normalization strategy: {self.normalization}")

    def _case_cache_key(self, case: StructuredCase) -> str:
        return case.case_id

    def _build_reasons(
        self,
        case: StructuredCase,
        contributions: list[tuple[float, BM25FieldSpec]],
    ) -> list[str]:
        reasons: list[str] = []
        for contribution, spec in sorted(contributions, key=lambda item: item[0], reverse=True):
            if contribution < 0.08:
                continue
            reason = self._reason_for_field(case, spec)
            if reason and reason not in reasons:
                reasons.append(reason)
            if len(reasons) == 3:
                break
        return reasons or ["BM25 综合命中多个结构化字段"]

    def _reason_for_field(self, case: StructuredCase, spec: BM25FieldSpec) -> str:
        if spec.name == "charges" and case.charges:
            return f"BM25 命中罪名: {', '.join(case.charges[:2])}"
        if spec.name == "legal_basis" and case.legal_basis:
            return "BM25 命中适用法条"
        if spec.name == "four_element_objective_aspect" and case.four_element_objective_aspect:
            return "BM25 命中客观行为特征"
        if spec.name == "four_element_subjective_aspect" and case.four_element_subjective_aspect:
            return "BM25 命中主观方面"
        if spec.name == "four_element_subject" and case.four_element_subject:
            return "BM25 命中主体特征"
        if spec.name == "four_element_object" and case.four_element_object:
            return "BM25 命中客体法益"
        if spec.name == "dispute_focus" and case.dispute_focus:
            return "BM25 命中争议焦点"
        if spec.name == "court_reasoning" and case.court_reasoning:
            return "BM25 命中裁判理由"
        if spec.name == "case_summary" and case.case_summary:
            return "BM25 命中案情摘要"
        return ""
