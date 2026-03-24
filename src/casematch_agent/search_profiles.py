from __future__ import annotations

from .models import StructuredCase, StructuredQuery


def join_non_empty(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def clean_optional_text(text: str) -> str:
    normalized = text.strip()
    if normalized in {"无", "未提及", "空"}:
        return ""
    return normalized


def query_fused_text(query: StructuredQuery) -> str:
    parts = [
        query.case_summary or query.raw_query,
        clean_optional_text(query.dispute_focus),
        clean_optional_text(query.court_reasoning),
    ]
    return join_non_empty([part.strip() for part in parts if part and part.strip()])


def case_fused_text(case: StructuredCase) -> str:
    parts = [
        case.case_summary,
        clean_optional_text(case.dispute_focus),
        clean_optional_text(case.court_reasoning or case.traceability_quote),
    ]
    return join_non_empty([part.strip() for part in parts if part and part.strip()])


def query_four_elements_text(query: StructuredQuery) -> str:
    return join_non_empty(
        query.four_element_subject
        + query.four_element_object
        + query.four_element_objective_aspect
        + query.four_element_subjective_aspect
    )


def case_four_elements_text(case: StructuredCase) -> str:
    return join_non_empty(
        case.four_element_subject
        + case.four_element_object
        + case.four_element_objective_aspect
        + case.four_element_subjective_aspect
    )


def query_laws_and_charges_text(query: StructuredQuery) -> str:
    return join_non_empty(query.charges + query.legal_basis)


def case_laws_and_charges_text(case: StructuredCase) -> str:
    return join_non_empty(case.charges + case.legal_basis)
