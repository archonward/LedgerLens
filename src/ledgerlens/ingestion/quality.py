"""Reusable, conservative checks for obvious PDF text-extraction defects."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

CID_TOKEN_PATTERN = re.compile(r"\(cid:\d+\)")
NO_TEXT_WARNING = "No text was extracted; this page may be blank or may require OCR."


@dataclass(frozen=True)
class TextQuality:
    """Findings from checks that do not establish whether text is correct."""

    status: str
    unresolved_character_code_count: int
    requires_review: bool
    automatic_indexing_eligible: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def assess_text_quality(text: str) -> TextQuality:
    """Detect blank text and unresolved ``(cid:<digits>)`` character tokens."""
    unresolved_count = len(CID_TOKEN_PATTERN.findall(text))
    warnings: list[str] = []

    if not text.strip():
        warnings.append(NO_TEXT_WARNING)
    if unresolved_count:
        warnings.append(
            f"Detected {unresolved_count} unresolved character-code tokens matching "
            "'(cid:<digits>)'; this page requires review and is unsuitable for automatic "
            "indexing until resolved."
        )

    has_detected_issues = bool(warnings)
    return TextQuality(
        status="issues_detected" if has_detected_issues else "no_detected_issues",
        unresolved_character_code_count=unresolved_count,
        requires_review=has_detected_issues,
        automatic_indexing_eligible=not has_detected_issues,
        warnings=tuple(warnings),
    )
