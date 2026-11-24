"""Deterministic grounding checks for a drafted answer, run before anyone sees it.

Cheaper and stricter than asking another model to judge: every cited id must be
one of the retrieved quotes, and every percentage must be one the statistics
produced. Failures go back to the writer as feedback.
"""

import re
from typing import Any

_CITATION = re.compile(r"\[([^\[\]]{1,40})\]")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")
_TOLERANCE = 0.6  # percentage points: allows 66% for 65.6%
# Some models cite with other brackets: gpt-oss uses lenticular ones (U+3010, U+3011),
# others full-width ones (U+FF3B, U+FF3D). They are rewritten to plain [ ] before checking.
_BRACKETS = str.maketrans({chr(0x3010): "[", chr(0x3011): "]", chr(0xFF3B): "[", chr(0xFF3D): "]"})


def normalize_citations(answer: str) -> str:
    """Rewrite citation brackets models use instead of [ ] to plain square brackets."""
    return answer.translate(_BRACKETS)


def check_grounding(
    answer: str, stats: dict[str, Any] | None, evidence: list[dict[str, Any]]
) -> list[str]:
    """Problems with `answer`; an empty list means it is grounded."""
    issues = []
    retrieved = {quote["comment_id"] for quote in evidence}
    cited = set(_CITATION.findall(answer))
    if unknown := sorted(cited - retrieved):
        issues.append(f"cites answers that were not provided: {unknown}")
    if evidence and not cited:
        issues.append("quotes no answer: cite at least one comment id in square brackets")

    allowed = _allowed_percentages(stats)
    invented = [value for value in _PERCENT.findall(answer) if not _is_close(value, allowed)]
    if invented:
        issues.append(f"percentages not in the statistics: {', '.join(v + '%' for v in invented)}")
    return issues


def _allowed_percentages(stats: dict[str, Any] | None) -> list[float]:
    if not stats:
        return []
    return [
        100 * theme[key]
        for group in stats["groups"]
        for theme in group["themes"]
        for key in ("share", "low", "high")
    ]


def _is_close(value: str, allowed: list[float]) -> bool:
    return any(abs(float(value) - number) <= _TOLERANCE for number in allowed)
