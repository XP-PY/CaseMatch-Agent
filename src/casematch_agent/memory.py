from __future__ import annotations

from dataclasses import dataclass

from .models import ClarificationDecision, ConversationMemory, StructuredQuery


LECARD_CLARIFICATION_FIELDS = (
    "charges",
    "dispute_focus",
    "four_element_objective_aspect",
    "four_element_subjective_aspect",
    "legal_basis",
)

UNKNOWN_PATTERNS = (
    "不知道",
    "不清楚",
    "不确定",
    "不太清楚",
    "说不清",
    "不记得",
    "没有印象",
    "不了解",
)


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for item in left + right:
        if item and item not in merged:
            merged.append(item)
    return merged


def query_field_has_value(query: StructuredQuery, field_name: str) -> bool:
    value = getattr(query, field_name, None)
    if isinstance(value, list):
        return any(bool(item) for item in value)
    return bool(value)


@dataclass
class QueryContextManager:
    def update_after_user_turn(
        self,
        memory: ConversationMemory,
        user_message: str,
        current_query: StructuredQuery,
    ) -> ConversationMemory:
        history = list(memory.turn_history)
        if user_message.strip():
            history.append(user_message.strip())

        answered_fields = list(memory.answered_fields)
        declined_fields = list(memory.declined_fields)
        pending_fields = list(memory.pending_fields)

        newly_answered = [field for field in pending_fields if query_field_has_value(current_query, field)]
        answered_fields = _merge_unique(answered_fields, newly_answered)

        unresolved_pending = [field for field in pending_fields if field not in newly_answered]
        if unresolved_pending and self._looks_unknown_response(user_message):
            declined_fields = _merge_unique(declined_fields, unresolved_pending)

        return ConversationMemory(
            turn_history=history,
            asked_fields=list(memory.asked_fields),
            answered_fields=answered_fields,
            declined_fields=declined_fields,
            pending_fields=[],
        )

    def update_after_clarification(
        self,
        memory: ConversationMemory,
        decision: ClarificationDecision,
    ) -> ConversationMemory:
        pending_fields = decision.missing_fields if decision.missing_fields and decision.questions else []
        return ConversationMemory(
            turn_history=list(memory.turn_history),
            asked_fields=_merge_unique(memory.asked_fields, pending_fields),
            answered_fields=list(memory.answered_fields),
            declined_fields=list(memory.declined_fields),
            pending_fields=list(pending_fields),
        )

    def _looks_unknown_response(self, user_message: str) -> bool:
        text = user_message.strip()
        if not text:
            return False
        return any(pattern in text for pattern in UNKNOWN_PATTERNS)
