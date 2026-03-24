from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .llm import JsonLLMClient
from .models import StructuredQuery
from .utils import extract_legal_references


CHARGE_PATTERN = re.compile(r"([\u4e00-\u9fff]{2,15}罪)")
CHARGE_PREFIX_PATTERN = re.compile(r"(?:是否构成|构成|涉嫌|犯有|犯)\s*([\u4e00-\u9fff]{2,15}罪)")
SUBJECTIVE_ASPECT_KEYWORDS = ["直接故意", "间接故意", "过于自信的过失", "疏忽大意的过失"]
FOUR_ELEMENT_OBJECTIVE_KEYWORDS = [
    "醉酒驾驶机动车",
    "在道路上行驶",
    "秘密窃取",
    "多次盗窃",
    "入户盗窃",
    "持刀抢劫",
    "暴力夺取财物",
    "非法持有毒品",
    "贩卖毒品",
    "逃避公安机关查处而丢弃毒品",
]


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


def _clean_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _extract_charges(text: str) -> list[str]:
    charges: list[str] = []
    matches = CHARGE_PREFIX_PATTERN.findall(text) + CHARGE_PATTERN.findall(text)
    for match in matches:
        cleaned = re.sub(r"^(是否构成|构成|涉嫌|犯有|犯)", "", match).strip("，。；;：: ")
        if cleaned.endswith("罪") and cleaned not in charges:
            charges.append(cleaned)
    return charges


def _extract_subjective_aspects(text: str) -> list[str]:
    matched: list[str] = []
    for keyword in SUBJECTIVE_ASPECT_KEYWORDS:
        if keyword in text and keyword not in matched:
            matched.append(keyword)
    return matched


def _extract_objective_aspects(text: str) -> list[str]:
    matched: list[str] = []
    for keyword in FOUR_ELEMENT_OBJECTIVE_KEYWORDS:
        if keyword in text and keyword not in matched:
            matched.append(keyword)
    return matched


def _extract_dispute_focus(text: str) -> str:
    if "争议焦点" in text:
        parts = re.split(r"争议焦点[:：]", text, maxsplit=1)
        if len(parts) == 2:
            return parts[1].split("。")[0].strip()
    for sentence in re.split(r"[。；;]", text):
        snippet = sentence.strip()
        if "是否" in snippet and 6 <= len(snippet) <= 80:
            return snippet
    return ""


@dataclass
class HeuristicStructuredQueryExtractor:
    def extract(self, user_query: str) -> StructuredQuery:
        text = user_query.strip()
        charges = _extract_charges(text)
        dispute_focus = _extract_dispute_focus(text)
        legal_basis = extract_legal_references(text)
        objective_aspects = _extract_objective_aspects(text)
        subjective_aspects = _extract_subjective_aspects(text)
        confidence = self._confidence(text, charges, dispute_focus, legal_basis, objective_aspects, subjective_aspects)
        return StructuredQuery(
            raw_query=text,
            case_summary=text,
            charges=charges,
            dispute_focus=dispute_focus,
            legal_basis=legal_basis,
            four_element_objective_aspect=objective_aspects,
            four_element_subjective_aspect=subjective_aspects,
            confidence=confidence,
        )

    def merge_queries(
        self,
        previous_query: StructuredQuery,
        current_query: StructuredQuery,
        latest_user_message: str,
    ) -> StructuredQuery:
        return previous_query.merge(current_query)

    def _confidence(
        self,
        text: str,
        charges: list[str],
        dispute_focus: str,
        legal_basis: list[str],
        objective_aspects: list[str],
        subjective_aspects: list[str],
    ) -> float:
        signal_score = 0.0
        signal_score += min(len(charges), 2) * 0.22
        signal_score += 0.18 if dispute_focus else 0.0
        signal_score += min(len(legal_basis), 2) * 0.10
        signal_score += min(len(objective_aspects), 2) * 0.12
        signal_score += min(len(subjective_aspects), 2) * 0.08
        signal_score += min(len(text) / 80, 1.0) * 0.20
        return round(min(signal_score, 0.95), 2)


@dataclass
class LLMStructuredQueryExtractor:
    client: JsonLLMClient
    fallback: HeuristicStructuredQueryExtractor

    def extract(self, user_query: str) -> StructuredQuery:
        try:
            payload = self.client.chat_json(
                system_prompt=self._system_prompt(),
                user_prompt=self._user_prompt(user_query),
                temperature=0.1,
            )
        except Exception:
            return self.fallback.extract(user_query)

        llm_query = self._normalize_query(user_query, self._from_payload(user_query, payload))
        if self._should_fallback(llm_query):
            return self.fallback.extract(user_query)
        return llm_query

    def merge_queries(
        self,
        previous_query: StructuredQuery,
        current_query: StructuredQuery,
        latest_user_message: str,
    ) -> StructuredQuery:
        merged_raw_query = f"{previous_query.raw_query} {latest_user_message.strip()}".strip()
        try:
            payload = self.client.chat_json(
                system_prompt=self._merge_system_prompt(),
                user_prompt=self._merge_user_prompt(previous_query, current_query, latest_user_message),
                temperature=0.1,
            )
        except Exception:
            return previous_query.merge(current_query)

        merged_query = self._normalize_query(merged_raw_query, self._from_payload(merged_raw_query, payload))
        if self._should_fallback(merged_query):
            return previous_query.merge(current_query)
        return merged_query

    def _from_payload(self, user_query: str, payload: dict[str, Any]) -> StructuredQuery:
        text = user_query.strip()
        four_elements = payload.get("four_elements") if isinstance(payload.get("four_elements"), dict) else {}
        return StructuredQuery(
            raw_query=text,
            case_summary=_clean_string(payload.get("case_summary")) or text,
            charges=_clean_string_list(payload.get("charges")),
            dispute_focus=_clean_string(payload.get("dispute_focus")),
            legal_basis=_clean_string_list(payload.get("legal_basis")),
            four_element_subject=_clean_string_list(four_elements.get("subject")),
            four_element_object=_clean_string_list(four_elements.get("object")),
            four_element_objective_aspect=_clean_string_list(four_elements.get("objective_aspect")),
            four_element_subjective_aspect=_clean_string_list(four_elements.get("subjective_aspect")),
            court_reasoning=_clean_string(payload.get("court_reasoning")),
            confidence=_clean_confidence(payload.get("confidence")),
        )

    def _normalize_query(self, user_query: str, query: StructuredQuery) -> StructuredQuery:
        text = user_query.strip()
        return StructuredQuery(
            raw_query=text,
            case_summary=query.case_summary or text,
            charges=query.charges,
            dispute_focus=query.dispute_focus,
            legal_basis=query.legal_basis or extract_legal_references(text),
            four_element_subject=query.four_element_subject,
            four_element_object=query.four_element_object,
            four_element_objective_aspect=query.four_element_objective_aspect,
            four_element_subjective_aspect=query.four_element_subjective_aspect,
            court_reasoning=query.court_reasoning,
            confidence=query.confidence,
        )

    def _should_fallback(self, query: StructuredQuery) -> bool:
        if query.confidence <= 0.0:
            return True
        return not self._has_semantic_signal(query)

    def _has_semantic_signal(self, query: StructuredQuery) -> bool:
        return any(
            [
                bool(query.charges),
                bool(query.dispute_focus),
                bool(query.legal_basis),
                bool(query.four_element_subject),
                bool(query.four_element_object),
                bool(query.four_element_objective_aspect),
                bool(query.four_element_subjective_aspect),
                bool(query.court_reasoning),
            ]
        )

    def _query_snapshot(self, query: StructuredQuery) -> dict[str, Any]:
        return {
            "case_summary": query.case_summary,
            "charges": query.charges,
            "dispute_focus": query.dispute_focus,
            "legal_basis": query.legal_basis,
            "four_element_subject": query.four_element_subject,
            "four_element_object": query.four_element_object,
            "four_element_objective_aspect": query.four_element_objective_aspect,
            "four_element_subjective_aspect": query.four_element_subjective_aspect,
            "court_reasoning": query.court_reasoning,
            "confidence": query.confidence,
        }

    def _base_schema_prompt(self) -> str:
        return (
            "你必须只输出一个 JSON 对象，不要输出 Markdown。"
            "JSON schema: {"
            '"case_summary": string, '
            '"charges": string[], '
            '"dispute_focus": string, '
            '"legal_basis": string[], '
            '"four_elements": {"subject": string[], "object": string[], "objective_aspect": string[], "subjective_aspect": string[]}, '
            '"court_reasoning": string, '
            '"confidence": number'
            "}."
            "如果不确定，请返回空字符串或空数组。"
        )

    def _system_prompt(self) -> str:
        return (
            "你是中国刑事类案检索系统的结构化提取器。"
            "请仅根据用户本轮输入提取检索字段，不要编造信息。"
            "当前只抽取刑事检索字段。"
            + self._base_schema_prompt()
        )

    def _merge_system_prompt(self) -> str:
        return (
            "你是中国刑事类案检索系统的多轮查询合并器。"
            "你会看到上一轮已合并的结构化查询、本轮新抽取结果和用户本轮原话。"
            "如果本轮是在补充信息，就保留上一轮有效信息并补充新信息。"
            "如果本轮是在纠正上一轮，例如“不是A，是B”，应以本轮纠正后的值为准。"
            + self._base_schema_prompt()
        )

    def _user_prompt(self, user_query: str) -> str:
        return f"以下是刑事类案检索需求，请提取结构化检索字段并返回 JSON：\n{user_query.strip()}"

    def _merge_user_prompt(
        self,
        previous_query: StructuredQuery,
        current_query: StructuredQuery,
        latest_user_message: str,
    ) -> str:
        return (
            f"上一轮已合并结构化查询: {self._query_snapshot(previous_query)}\n"
            f"本轮新抽取结构化结果: {self._query_snapshot(current_query)}\n"
            f"用户本轮原话: {latest_user_message.strip()}\n"
            "请输出新的最终结构化查询 JSON。"
        )
