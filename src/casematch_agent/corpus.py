from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import StructuredCase


def _clean_string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = [item for item in value if isinstance(item, str)]
    else:
        items = []

    cleaned: list[str] = []
    for item in items:
        text = item.strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def structured_case_from_components(
    *,
    case_id: str,
    structured_data: dict[str, Any],
    raw_data: dict[str, Any] | None = None,
) -> StructuredCase:
    raw_payload = raw_data if isinstance(raw_data, dict) else {}
    four_elements = structured_data.get("four_elements") if isinstance(structured_data.get("four_elements"), dict) else {}
    laws_and_charges = (
        structured_data.get("laws_and_charges") if isinstance(structured_data.get("laws_and_charges"), dict) else {}
    )
    traceability = structured_data.get("traceability") if isinstance(structured_data.get("traceability"), dict) else {}

    return StructuredCase(
        case_id=str(case_id).strip(),
        case_name=_clean_string(raw_payload.get("case_name")),
        document_name=_clean_string(raw_payload.get("document_name")),
        fact_text=_clean_string(raw_payload.get("fact_text")),
        judgment_text=_clean_string(raw_payload.get("judgment_text")),
        full_text=_clean_string(raw_payload.get("full_text")),
        charges=_clean_string_list(laws_and_charges.get("charges")),
        case_summary=_clean_string(structured_data.get("case_summary")),
        dispute_focus=_clean_string(structured_data.get("dispute_focus")),
        legal_basis=_clean_string_list(laws_and_charges.get("applicable_laws")),
        four_element_subject=_clean_string_list(four_elements.get("subject")),
        four_element_object=_clean_string_list(four_elements.get("object")),
        four_element_objective_aspect=_clean_string_list(four_elements.get("objective_aspect")),
        four_element_subjective_aspect=_clean_string_list(four_elements.get("subjective_aspect")),
        court_reasoning=_clean_string(structured_data.get("court_reasoning")),
        traceability_quote=_clean_string(traceability.get("reasoning_quote")),
    )


def merged_payload_to_structured_case(payload: dict[str, Any]) -> StructuredCase | None:
    if payload.get("error"):
        return None
    structured_data = payload.get("structured_data") if isinstance(payload.get("structured_data"), dict) else {}
    if not structured_data:
        return None
    raw_data = payload.get("raw_data") if isinstance(payload.get("raw_data"), dict) else {}
    case_id = str(payload.get("case_id", "")).strip()
    if not case_id:
        return None
    return structured_case_from_components(case_id=case_id, structured_data=structured_data, raw_data=raw_data)


def load_lecard_corpus(path: str | Path) -> list[StructuredCase]:
    corpus_path = Path(path)
    cases: list[StructuredCase] = []
    with corpus_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            case = merged_payload_to_structured_case(payload)
            if case is not None:
                cases.append(case)
    return cases
