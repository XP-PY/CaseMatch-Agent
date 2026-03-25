from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .corpus import merged_payload_to_structured_case
from .llm import JsonLLMClient
from .models import StructuredCase

REQUIRED_RAW_FIELDS = (
    "case_name",
    "document_name",
    "fact_text",
    "judgment_text",
    "full_text",
)


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


def _normalize_structured_payload(payload: dict[str, Any]) -> dict[str, Any]:
    four_elements = payload.get("four_elements") if isinstance(payload.get("four_elements"), dict) else {}
    laws_and_charges = payload.get("laws_and_charges") if isinstance(payload.get("laws_and_charges"), dict) else {}
    traceability = payload.get("traceability") if isinstance(payload.get("traceability"), dict) else {}
    return {
        "case_summary": _clean_string(payload.get("case_summary")),
        "dispute_focus": _clean_string(payload.get("dispute_focus")),
        "four_elements": {
            "subject": _clean_string_list(four_elements.get("subject")),
            "object": _clean_string_list(four_elements.get("object")),
            "objective_aspect": _clean_string_list(four_elements.get("objective_aspect")),
            "subjective_aspect": _clean_string_list(four_elements.get("subjective_aspect")),
        },
        "court_reasoning": _clean_string(payload.get("court_reasoning")),
        "laws_and_charges": {
            "charges": _clean_string_list(laws_and_charges.get("charges")),
            "applicable_laws": _clean_string_list(laws_and_charges.get("applicable_laws")),
        },
        "traceability": {
            "reasoning_quote": _clean_string(traceability.get("reasoning_quote")),
        },
    }


def _normalize_raw_data(payload: dict[str, Any], *, line_number: int) -> dict[str, Any]:
    raw_data = payload.get("raw_data") if isinstance(payload.get("raw_data"), dict) else payload
    if not isinstance(raw_data, dict):
        raise ValueError(f"Line {line_number}: expected a JSON object or a raw_data object.")

    missing_fields = [field_name for field_name in REQUIRED_RAW_FIELDS if field_name not in raw_data]
    if missing_fields:
        missing_text = ", ".join(missing_fields)
        raise ValueError(f"Line {line_number}: missing required raw_data fields: {missing_text}")

    normalized = dict(raw_data)
    for field_name in REQUIRED_RAW_FIELDS:
        normalized[field_name] = _clean_string(raw_data.get(field_name))
    return normalized


def _document_text(raw_data: dict[str, Any]) -> str:
    full_text = _clean_string(raw_data.get("full_text"))
    if full_text:
        return full_text
    fact_text = _clean_string(raw_data.get("fact_text"))
    judgment_text = _clean_string(raw_data.get("judgment_text"))
    return "\n".join(part for part in [fact_text, judgment_text] if part)


def load_existing_case_ids(corpus_path: str | Path) -> set[str]:
    path = Path(corpus_path)
    if not path.exists():
        return set()

    existing_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            case_id = str(payload.get("case_id", "")).strip()
            if case_id:
                existing_ids.add(case_id)
    return existing_ids


def generate_unique_case_id(existing_ids: set[str], prefix: str = "CASE") -> str:
    normalized_prefix = prefix.strip().upper() or "CASE"
    while True:
        candidate = f"{normalized_prefix}-{secrets.token_hex(8).upper()}"
        if candidate not in existing_ids:
            return candidate


def build_merged_record(*, case_id: str, raw_data: dict[str, Any], structured_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": str(case_id).strip(),
        "structured_data": _normalize_structured_payload(structured_data),
        "raw_data": dict(raw_data),
    }


def append_merged_records(corpus_path: str | Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path = Path(corpus_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class CriminalCaseStructuredDataExtractor:
    client: JsonLLMClient

    def extract(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        document_text = _document_text(raw_data)
        if not document_text:
            raise ValueError("raw_data.full_text and fallback document text are both empty.")

        payload = self.client.chat_json(
            system_prompt=self._system_prompt(),
            user_prompt=self._user_prompt(raw_data=raw_data, document_text=document_text),
            temperature=0.1,
        )
        return _normalize_structured_payload(payload)

    def _system_prompt(self) -> str:
        return (
            "你是中国刑事裁判文书结构化提取器。"
            "请严格基于输入文书原文抽取结构化案件信息，不要编造。"
            "你必须只输出一个 JSON 对象，不要输出 Markdown。"
            "JSON schema: {"
            '"case_summary": string, '
            '"dispute_focus": string, '
            '"four_elements": {"subject": string[], "object": string[], "objective_aspect": string[], "subjective_aspect": string[]}, '
            '"court_reasoning": string, '
            '"laws_and_charges": {"charges": string[], "applicable_laws": string[]}, '
            '"traceability": {"reasoning_quote": string}'
            "}."
            "不确定时返回空字符串或空数组。"
        )

    def _user_prompt(self, *, raw_data: dict[str, Any], document_text: str) -> str:
        case_name = _clean_string(raw_data.get("case_name"))
        document_name = _clean_string(raw_data.get("document_name"))
        return (
            "请阅读以下刑事裁判文书，并将其解构为结构化案件信息。\n"
            "提取要求：\n"
            "1. 案情摘要(case_summary): 基于法院查明事实进行高密度语义重构，保留核心实体、具体行为、金额数量、损害后果，避免“公诉机关指控”“本院认为”等程序性套话。\n"
            "2. 争议焦点(dispute_focus): 提取或概括控辩争议、行为定性争议、量刑情节争议；若无明显争议填“无”。\n"
            "3. 四要件(four_elements): 输出法律术语数组，避免长句。\n"
            "4. 裁判理由(court_reasoning): 提取法院对定性、证据采信、辩护意见处理的核心说理。\n"
            "5. 适用法条与罪名(laws_and_charges): 输出最终认定罪名和具体法条。\n"
            "6. 可追溯映射(traceability.reasoning_quote): 摘录最能支撑裁判理由的 1 句原文。\n"
            f"案件名称: {case_name or '未提供'}\n"
            f"文书名称: {document_name or '未提供'}\n"
            f"文书全文:\n{document_text}"
        )


@dataclass
class CaseImportReport:
    input_path: Path
    corpus_path: Path
    imported_count: int
    case_ids: list[str]


@dataclass
class CaseImportBatch:
    report: CaseImportReport
    structured_cases: list[StructuredCase]


def import_raw_cases_batch_from_jsonl(
    *,
    input_path: str | Path,
    corpus_path: str | Path,
    extractor: CriminalCaseStructuredDataExtractor,
    case_id_prefix: str = "CASE",
) -> CaseImportBatch:
    source_path = Path(input_path)
    target_corpus = Path(corpus_path)
    existing_ids = load_existing_case_ids(target_corpus)

    merged_records: list[dict[str, Any]] = []
    structured_cases: list[StructuredCase] = []
    created_ids: list[str] = []

    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            raw_data = _normalize_raw_data(payload, line_number=line_number)
            structured_data = extractor.extract(raw_data)
            case_id = generate_unique_case_id(existing_ids, prefix=case_id_prefix)
            existing_ids.add(case_id)

            merged_record = build_merged_record(case_id=case_id, raw_data=raw_data, structured_data=structured_data)
            structured_case = merged_payload_to_structured_case(merged_record)
            if structured_case is None:
                raise ValueError(f"Line {line_number}: failed to map merged record into StructuredCase.")

            merged_records.append(merged_record)
            structured_cases.append(structured_case)
            created_ids.append(case_id)

    append_merged_records(target_corpus, merged_records)
    return CaseImportBatch(
        report=CaseImportReport(
            input_path=source_path,
            corpus_path=target_corpus,
            imported_count=len(created_ids),
            case_ids=created_ids,
        ),
        structured_cases=structured_cases,
    )


def import_raw_cases_from_jsonl(
    *,
    input_path: str | Path,
    corpus_path: str | Path,
    extractor: CriminalCaseStructuredDataExtractor,
    sync_backend: Callable[[list[StructuredCase]], None] | None = None,
    case_id_prefix: str = "CASE",
) -> CaseImportReport:
    batch = import_raw_cases_batch_from_jsonl(
        input_path=input_path,
        corpus_path=corpus_path,
        extractor=extractor,
        case_id_prefix=case_id_prefix,
    )
    if sync_backend is not None and batch.structured_cases:
        sync_backend(batch.structured_cases)
    return batch.report
