"""Reusable, conservative checks for obvious PDF text-extraction defects."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

CID_TOKEN_PATTERN = re.compile(r"\(cid:\d+\)")
NO_TEXT_WARNING = "No text was extracted; this page may be blank or may require OCR."


@dataclass(frozen=True)
class TextQuality:
    """Findings from checks that do not establish whether text is correct."""

    status: str
    unresolved_character_code_count: int
    replacement_character_count: int
    unexpected_control_character_count: int
    requires_review: bool
    automatic_indexing_eligible: bool
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        result = asdict(self)
        result["warnings"] = list(self.warnings)
        return result


def assess_text_quality(text: str) -> TextQuality:
    """Run limited checks for blank, unresolved, replacement, and control characters."""
    unresolved_count = len(CID_TOKEN_PATTERN.findall(text))
    replacement_count = text.count("\ufffd")
    unexpected_control_count = sum(
        1
        for character in text
        if character not in "\n\r\t" and unicodedata.category(character) == "Cc"
    )
    warnings: list[str] = []

    if not text.strip():
        warnings.append(NO_TEXT_WARNING)
    if unresolved_count:
        warnings.append(
            f"Detected {unresolved_count} unresolved character-code tokens matching "
            "'(cid:<digits>)'; this page requires review and is unsuitable for automatic "
            "indexing until resolved."
        )
    if replacement_count:
        warnings.append(
            f"Detected {replacement_count} Unicode replacement characters (U+FFFD); this page "
            "requires review and is unsuitable for automatic indexing until resolved."
        )
    if unexpected_control_count:
        warnings.append(
            f"Detected {unexpected_control_count} unexpected Unicode control characters; this "
            "page requires review and is unsuitable for automatic indexing until resolved."
        )

    has_detected_issues = bool(warnings)
    return TextQuality(
        status="issues_detected" if has_detected_issues else "no_detected_issues",
        unresolved_character_code_count=unresolved_count,
        replacement_character_count=replacement_count,
        unexpected_control_character_count=unexpected_control_count,
        requires_review=has_detected_issues,
        automatic_indexing_eligible=not has_detected_issues,
        warnings=tuple(warnings),
    )
