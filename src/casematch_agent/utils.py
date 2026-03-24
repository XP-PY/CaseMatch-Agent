from __future__ import annotations

import re
from typing import Iterable


_CHINESE_BLOCK = re.compile(r"[\u4e00-\u9fff]+")
_ALNUM_BLOCK = re.compile(r"[a-z0-9]+")
_NON_TEXT = re.compile(r"[^\u4e00-\u9fffa-z0-9]+")
_AMOUNT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(亿|万|千|元)")


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    return _NON_TEXT.sub(" ", lowered)


def tokenize_text(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens: list[str] = []

    for word in _ALNUM_BLOCK.findall(normalized):
        if len(word) >= 2 and word not in tokens:
            tokens.append(word)

    for segment in _CHINESE_BLOCK.findall(normalized):
        if 2 <= len(segment) <= 8 and segment not in tokens:
            tokens.append(segment)
        for size in (2, 3):
            if len(segment) < size:
                continue
            for start in range(len(segment) - size + 1):
                token = segment[start : start + size]
                if token not in tokens:
                    tokens.append(token)
    return tokens


def jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = {item for item in left if item}
    right_set = {item for item in right if item}
    if not left_set or not right_set:
        return 0.0
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return intersection / union if union else 0.0


def overlap_ratio(query_terms: Iterable[str], candidate_terms: Iterable[str]) -> float:
    query_set = {item for item in query_terms if item}
    candidate_set = {item for item in candidate_terms if item}
    if not query_set or not candidate_set:
        return 0.0
    return len(query_set & candidate_set) / len(query_set)


def extract_legal_references(text: str) -> list[str]:
    references = re.findall(r"《[^》]+》第[^，。；;、\s]+条", text)
    deduped: list[str] = []
    for reference in references:
        if reference not in deduped:
            deduped.append(reference)
    return deduped


def normalize_amount_range(text: str) -> str:
    match = _AMOUNT_PATTERN.search(text)
    if not match:
        return ""

    value = float(match.group(1))
    unit = match.group(2)
    multiplier = {"元": 1, "千": 1_000, "万": 10_000, "亿": 100_000_000}[unit]
    amount = value * multiplier

    if amount < 100_000:
        return "10万元以下"
    if amount < 1_000_000:
        return "10万-100万元"
    if amount < 10_000_000:
        return "100万-1000万元"
    return "1000万元以上"


def law_name(reference: str) -> str:
    match = re.match(r"(《[^》]+》)", reference)
    return match.group(1) if match else reference
