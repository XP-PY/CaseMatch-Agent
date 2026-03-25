from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ClarificationStatus(str, Enum):
    READY = "ready"
    NEED_MORE_INFO = "need_more_info"


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for item in left + right:
        if item and item not in merged:
            merged.append(item)
    return merged


@dataclass
class StructuredQuery:
    raw_query: str
    case_summary: str = ""
    charges: list[str] = field(default_factory=list)
    dispute_focus: str = ""
    legal_basis: list[str] = field(default_factory=list)
    four_element_subject: list[str] = field(default_factory=list)
    four_element_object: list[str] = field(default_factory=list)
    four_element_objective_aspect: list[str] = field(default_factory=list)
    four_element_subjective_aspect: list[str] = field(default_factory=list)
    court_reasoning: str = ""
    confidence: float = 0.0

    def merge(self, other: "StructuredQuery") -> "StructuredQuery":
        raw_query = self.raw_query.strip()
        if other.raw_query.strip():
            raw_query = f"{raw_query} {other.raw_query.strip()}".strip()
        return StructuredQuery(
            raw_query=raw_query,
            case_summary=other.case_summary or self.case_summary or raw_query,
            charges=_merge_unique(self.charges, other.charges),
            dispute_focus=other.dispute_focus or self.dispute_focus,
            legal_basis=_merge_unique(self.legal_basis, other.legal_basis),
            four_element_subject=_merge_unique(self.four_element_subject, other.four_element_subject),
            four_element_object=_merge_unique(self.four_element_object, other.four_element_object),
            four_element_objective_aspect=_merge_unique(
                self.four_element_objective_aspect, other.four_element_objective_aspect
            ),
            four_element_subjective_aspect=_merge_unique(
                self.four_element_subjective_aspect, other.four_element_subjective_aspect
            ),
            court_reasoning=other.court_reasoning or self.court_reasoning,
            confidence=max(self.confidence, other.confidence),
        )


@dataclass
class StructuredCase:
    case_id: str
    case_name: str = ""
    document_name: str = ""
    fact_text: str = ""
    judgment_text: str = ""
    full_text: str = ""
    charges: list[str] = field(default_factory=list)
    case_summary: str = ""
    dispute_focus: str = ""
    legal_basis: list[str] = field(default_factory=list)
    four_element_subject: list[str] = field(default_factory=list)
    four_element_object: list[str] = field(default_factory=list)
    four_element_objective_aspect: list[str] = field(default_factory=list)
    four_element_subjective_aspect: list[str] = field(default_factory=list)
    court_reasoning: str = ""
    traceability_quote: str = ""


@dataclass
class RetrievalResult:
    case: StructuredCase
    total_score: float
    field_scores: dict[str, float]
    reasons: list[str]


@dataclass
class ClarificationDecision:
    status: ClarificationStatus
    reasons: list[str]
    questions: list[str]
    missing_fields: list[str]


@dataclass
class ConversationMemory:
    turn_history: list[str] = field(default_factory=list)
    asked_fields: list[str] = field(default_factory=list)
    answered_fields: list[str] = field(default_factory=list)
    declined_fields: list[str] = field(default_factory=list)
    pending_fields: list[str] = field(default_factory=list)


@dataclass
class AgentState:
    structured_query: StructuredQuery
    turn_count: int = 1
    waiting_for_clarification: bool = False
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    thread_id: str | None = None


@dataclass
class AgentResponse:
    state: AgentState
    structured_query: StructuredQuery
    decision: ClarificationDecision
    retrieval_results: list[RetrievalResult]
    narrative: str
