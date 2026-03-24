from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .llm import JsonLLMClient
from .models import ClarificationDecision, ClarificationStatus, ConversationMemory, RetrievalResult, StructuredQuery


FIELD_PRIORITY = [
    "charges",
    "dispute_focus",
    "four_element_objective_aspect",
    "four_element_subjective_aspect",
    "legal_basis",
]

FOLLOW_UP_QUESTIONS: dict[str, str] = {
    "charges": "如果你已经知道最终认定或争议罪名，请补充，例如危险驾驶罪、盗窃罪、诈骗罪、贩卖毒品罪等。",
    "dispute_focus": "请补充案件争点，例如是否构成自首、罪名定性是否准确、是否属于共同犯罪、是否属于正当防卫或量刑是否过重。",
    "four_element_objective_aspect": "请补充客观行为特征，例如是否醉酒驾驶、是否秘密窃取、是否持刀抢劫、是否贩卖毒品、是否在道路上行驶。",
    "four_element_subjective_aspect": "请补充主观方面，例如直接故意、间接故意、过于自信的过失或疏忽大意的过失。",
    "legal_basis": "如果你关注具体法条，请补充相关法条，例如《中华人民共和国刑法》第二百六十四条。",
}

ALLOWED_MISSING_FIELDS = set(FIELD_PRIORITY)
DEFAULT_SUPPORTED_FIELDS = tuple(FIELD_PRIORITY)


def _clean_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_items = [value]
    elif isinstance(value, list):
        raw_items = [item for item in value if isinstance(item, str)]
    else:
        raw_items = []
    cleaned: list[str] = []
    for item in raw_items:
        text = item.strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


@dataclass
class HeuristicClarificationJudge:
    supported_fields: tuple[str, ...] = DEFAULT_SUPPORTED_FIELDS

    def decide(
        self,
        query: StructuredQuery,
        results: list[RetrievalResult],
        memory: ConversationMemory | None = None,
    ) -> ClarificationDecision:
        missing_fields = self._missing_fields(query, memory)
        reasons: list[str] = []
        top_score = results[0].total_score if results else 0.0
        second_score = results[1].total_score if len(results) > 1 else 0.0
        score_gap = top_score - second_score

        if query.confidence < 0.45:
            reasons.append("首轮查询中可用于检索的结构化信息偏少")
        if len(query.raw_query) < 18:
            reasons.append("用户输入较短，关键信息可能不充分")
        if missing_fields:
            reasons.append(f"高价值检索字段仍有缺失: {', '.join(missing_fields)}")
        if top_score < 0.34:
            reasons.append("当前召回案例的整体匹配度偏低")
        if len(results) > 1 and score_gap < 0.05 and top_score < 0.6:
            reasons.append("候选案例之间区分度不足，需要进一步缩小范围")

        need_more_info = self._need_more_info(query, missing_fields, top_score, score_gap)
        questions = self._questions(missing_fields) if need_more_info else []
        if need_more_info and not questions:
            need_more_info = False

        return ClarificationDecision(
            status=ClarificationStatus.NEED_MORE_INFO if need_more_info else ClarificationStatus.READY,
            reasons=reasons,
            questions=questions,
            missing_fields=missing_fields,
        )

    def _missing_fields(self, query: StructuredQuery, memory: ConversationMemory | None = None) -> list[str]:
        already_asked = set(memory.asked_fields) if memory else set()
        supported_fields = set(self.supported_fields)
        missing: list[str] = []
        if not query.charges:
            missing.append("charges")
        if not query.dispute_focus:
            missing.append("dispute_focus")
        if not query.four_element_objective_aspect:
            missing.append("four_element_objective_aspect")
        if not query.four_element_subjective_aspect:
            missing.append("four_element_subjective_aspect")
        if not query.legal_basis:
            missing.append("legal_basis")
        return [field for field in missing if field in supported_fields and field not in already_asked]

    def _need_more_info(
        self,
        query: StructuredQuery,
        missing_fields: list[str],
        top_score: float,
        score_gap: float,
    ) -> bool:
        high_value_missing = sum(1 for field in missing_fields if field in {"charges", "dispute_focus", "four_element_objective_aspect"})
        if top_score >= 0.45 and high_value_missing <= 1 and query.confidence >= 0.45:
            return False
        if top_score < 0.34:
            return True
        if query.confidence < 0.45 and high_value_missing >= 2:
            return True
        if high_value_missing >= 2:
            return True
        if 0 <= score_gap < 0.05 and high_value_missing >= 1:
            return True
        return False

    def _questions(self, missing_fields: list[str]) -> list[str]:
        ordered_fields = [field for field in FIELD_PRIORITY if field in missing_fields]
        questions: list[str] = []
        for field in ordered_fields:
            question = FOLLOW_UP_QUESTIONS[field]
            if question not in questions:
                questions.append(question)
            if len(questions) == 3:
                break
        return questions[:3]


@dataclass
class LLMClarificationJudge:
    client: JsonLLMClient
    fallback: HeuristicClarificationJudge
    supported_fields: tuple[str, ...] = DEFAULT_SUPPORTED_FIELDS

    def decide(
        self,
        query: StructuredQuery,
        results: list[RetrievalResult],
        memory: ConversationMemory | None = None,
    ) -> ClarificationDecision:
        try:
            payload = self.client.chat_json(
                system_prompt=self._system_prompt(memory),
                user_prompt=self._user_prompt(query, results, memory),
                temperature=0.1,
            )
        except Exception:
            return self.fallback.decide(query, results, memory=memory)

        decision = self._normalize_decision(payload, memory)
        if self._should_fallback(decision):
            return self.fallback.decide(query, results, memory=memory)
        return decision

    def _normalize_decision(
        self,
        payload: dict[str, Any],
        memory: ConversationMemory | None = None,
    ) -> ClarificationDecision:
        status = ClarificationStatus.NEED_MORE_INFO if bool(payload.get("need_more_info")) else ClarificationStatus.READY
        reasons = _clean_string_list(payload.get("reasons"))
        already_asked = set(memory.asked_fields) if memory else set()
        supported_fields = set(self.supported_fields)
        missing_fields = [
            field
            for field in _clean_string_list(payload.get("missing_fields"))
            if field in ALLOWED_MISSING_FIELDS and field in supported_fields and field not in already_asked
        ]
        questions = _clean_string_list(payload.get("questions"))[:3]

        if status == ClarificationStatus.READY:
            questions = []
        if status == ClarificationStatus.NEED_MORE_INFO and not missing_fields:
            questions = []

        return ClarificationDecision(
            status=status,
            reasons=reasons,
            questions=questions,
            missing_fields=missing_fields,
        )

    def _should_fallback(self, decision: ClarificationDecision) -> bool:
        return decision.status == ClarificationStatus.NEED_MORE_INFO and not decision.questions

    def _base_schema_prompt(self) -> str:
        supported_field_text = ", ".join(self.supported_fields)
        return (
            "你必须只输出一个 JSON 对象，不要输出 Markdown。"
            "JSON schema: {"
            '"need_more_info": boolean, '
            '"reasons": string[], '
            '"missing_fields": string[], '
            '"questions": string[]'
            "}."
            "questions 最多 3 个，必须具体，且只在 need_more_info=true 时输出。"
            f"missing_fields 只能从 {supported_field_text} 中选择。"
        )

    def _system_prompt(self, memory: ConversationMemory | None = None) -> str:
        asked_fields = ", ".join(memory.asked_fields) if memory and memory.asked_fields else "无"
        return (
            "你是中国刑事类案检索 agent 的澄清决策器。"
            "系统已经基于当前信息返回了一轮类案结果，你需要判断是否值得继续追问用户以提高精度。"
            "请不要阻止结果输出；这里只判断要不要追加追问。"
            "不要重复询问已经问过的字段。"
            f"当前只允许围绕这些字段追问: {', '.join(self.supported_fields)}。"
            f"已经问过的字段: {asked_fields}。"
            + self._base_schema_prompt()
        )

    def _user_prompt(
        self,
        query: StructuredQuery,
        results: list[RetrievalResult],
        memory: ConversationMemory | None = None,
    ) -> str:
        top_results = []
        for result in results[:3]:
            top_results.append(
                {
                    "case_id": result.case.case_id,
                    "case_name": result.case.case_name,
                    "score": result.total_score,
                    "reasons": result.reasons,
                    "charges": result.case.charges,
                    "dispute_focus": result.case.dispute_focus,
                }
            )
        asked_fields = memory.asked_fields if memory else []
        answered_fields = memory.answered_fields if memory else []
        declined_fields = memory.declined_fields if memory else []
        return (
            "请按刑事检索场景，根据当前结构化查询和检索结果，判断是否需要继续向用户追问。\n"
            f"当前结构化查询: {self._query_snapshot(query)}\n"
            f"允许追问字段: {list(self.supported_fields)}\n"
            f"已经问过字段: {asked_fields}\n"
            f"已经回答字段: {answered_fields}\n"
            f"已明确不知道的字段: {declined_fields}\n"
            f"Top 检索结果: {top_results}\n"
            "如果当前结果已经足够有参考价值，可以返回 need_more_info=false；"
            "如果继续补充当前仍未问过的字段能明显提高区分度，则返回 need_more_info=true 并给出问题。"
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
