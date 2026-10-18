"""Validated evidence loading and page-level BM25 retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

K1_DEFAULT = 1.5
B_DEFAULT = 0.75
EXCERPT_CHARACTERS_DEFAULT = 500
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[.,][^\W_]+)*", re.UNICODE)
_SUPPORTED_METHODS = {"canonical_extraction", "reviewed_ocr_recovery"}


class RetrievalError(Exception):
    """Raised when selected evidence cannot be loaded or searched safely."""


@dataclass(frozen=True)
class EvidencePage:
    """One validated physical page from a selected-evidence export."""

    document_id: str
    company: str
    report_year: int
    physical_page_number: int
    text: str
    text_sha256: str
    method: str
    provenance: dict[str, Any]
    screening_findings: dict[str, Any]
    review_provenance: dict[str, Any] | None
    known_limitations: tuple[str, ...]

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return warning strings present anywhere in screening metadata."""
        found: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "warnings" and isinstance(child, list):
                        found.extend(item for item in child if isinstance(item, str))
                    else:
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(self.screening_findings)
        return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class EvidenceCorpus:
    """A validated selected-evidence export and its integrity references."""

    document_id: str
    source_sha256: str
    pages: tuple[EvidencePage, ...]
    excluded_pages: tuple[int, ...]
    manifest: dict[str, Any]
    evidence_jsonl_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class SearchResult:
    """A ranked page with its selected text excerpt and provenance."""

    rank: int
    score: float
    page: EvidencePage
    excerpt: str
    evidence_jsonl_sha256: str
    manifest_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable result without dropping provenance."""
        return {
            "rank": self.rank,
            "score": self.score,
            "document_id": self.page.document_id,
            "company": self.page.company,
            "report_year": self.page.report_year,
            "physical_page_number": self.page.physical_page_number,
            "method": self.page.method,
            "excerpt": self.excerpt,
            "text_sha256": self.page.text_sha256,
            "source_sha256": self.page.provenance.get("source_sha256"),
            "evidence_jsonl_sha256": self.evidence_jsonl_sha256,
            "manifest_sha256": self.manifest_sha256,
            "provenance": self.page.provenance,
            "screening_findings": self.page.screening_findings,
            "warnings": list(self.page.warnings),
            "review_provenance": self.page.review_provenance,
            "known_limitations": list(self.page.known_limitations),
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require(value: dict[str, Any], key: str, expected: type, context: str) -> Any:
    item = value.get(key)
    if not isinstance(item, expected) or (expected is str and not item):
        raise RetrievalError(f"{context}: {key!r} must be a non-empty {expected.__name__}")
    return item


def _string_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RetrievalError(f"{context} must be a list of strings")
    return tuple(value)


def _integer_list(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value
    ):
        raise RetrievalError(f"{context} must be a list of positive physical page numbers")
    if len(value) != len(set(value)):
        raise RetrievalError(f"{context} contains duplicate physical page numbers")
    return tuple(value)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RetrievalError(f"Could not read {label} at {path}: {exc}") from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Malformed {label} at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RetrievalError(f"Malformed {label} at {path}: expected a JSON object")
    return value, raw


def load_evidence_export(directory: str | Path) -> EvidenceCorpus:
    """Load and cross-check a selected-evidence export.

    Validation establishes structural consistency and file identity, not financial
    correctness or source-document completeness.
    """
    root = Path(directory)
    if not root.is_dir():
        raise RetrievalError(f"Selected evidence directory does not exist: {root}")
    manifest_path = root / "manifest.json"
    evidence_path = root / "evidence.jsonl"
    if not manifest_path.is_file():
        raise RetrievalError(f"Selected evidence manifest is missing: {manifest_path}")
    if not evidence_path.is_file():
        raise RetrievalError(f"Selected evidence records are missing: {evidence_path}")

    manifest, manifest_raw = _read_json(manifest_path, "evidence manifest")
    if manifest.get("schema_version") != "1.0":
        raise RetrievalError("Evidence manifest uses an unsupported schema_version")
    document_id = _require(manifest, "document_id", str, "Evidence manifest")
    source_sha256 = _require(manifest, "source_sha256", str, "Evidence manifest")
    decisions = manifest.get("selection_decisions")
    if not isinstance(decisions, list):
        raise RetrievalError("Evidence manifest: 'selection_decisions' must be a list")
    excluded_pages = _integer_list(manifest.get("excluded_pages"), "excluded_pages")

    try:
        evidence_raw = evidence_path.read_bytes()
        evidence_text = evidence_raw.decode("utf-8")
    except OSError as exc:
        raise RetrievalError(f"Could not read selected evidence at {evidence_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise RetrievalError(f"Selected evidence is not valid UTF-8: {evidence_path}") from exc

    pages: list[EvidencePage] = []
    for line_number, line in enumerate(evidence_text.splitlines(), start=1):
        if not line.strip():
            raise RetrievalError(f"Malformed evidence.jsonl line {line_number}: blank record")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalError(f"Malformed evidence.jsonl line {line_number}: {exc}") from exc
        context = f"evidence.jsonl line {line_number}"
        if not isinstance(record, dict):
            raise RetrievalError(f"{context}: expected a JSON object")
        if record.get("schema_version") != "1.0":
            raise RetrievalError(f"{context}: unsupported schema_version")
        record_document_id = _require(record, "document_id", str, context)
        if record_document_id != document_id:
            raise RetrievalError(f"{context}: document_id does not match the manifest")
        page_number = record.get("physical_page_number")
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise RetrievalError(f"{context}: physical_page_number must be a positive integer")
        text = record.get("text")
        if not isinstance(text, str):
            raise RetrievalError(f"{context}: text must be a string")
        text_sha256 = _require(record, "text_sha256", str, context)
        if _sha256(text.encode("utf-8")) != text_sha256:
            raise RetrievalError(f"{context}: text_sha256 does not match selected text")
        method = _require(record, "method", str, context)
        if method not in _SUPPORTED_METHODS:
            raise RetrievalError(f"{context}: unsupported evidence method {method!r}")
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            raise RetrievalError(f"{context}: provenance must be an object")
        if provenance.get("source_sha256") != source_sha256:
            raise RetrievalError(f"{context}: provenance source_sha256 does not match the manifest")
        screening = record.get("screening_findings")
        if not isinstance(screening, dict):
            raise RetrievalError(f"{context}: screening_findings must be an object")
        review = record.get("review_provenance")
        if review is not None and not isinstance(review, dict):
            raise RetrievalError(f"{context}: review_provenance must be an object or null")
        if method == "reviewed_ocr_recovery" and not isinstance(review, dict):
            raise RetrievalError(f"{context}: reviewed OCR evidence requires review_provenance")
        limitations = _string_list(record.get("known_limitations"), f"{context} limitations")
        company = _require(record, "company", str, context)
        report_year = record.get("report_year")
        if not isinstance(report_year, int) or isinstance(report_year, bool):
            raise RetrievalError(f"{context}: report_year must be an integer")
        pages.append(
            EvidencePage(
                document_id=record_document_id,
                company=company,
                report_year=report_year,
                physical_page_number=page_number,
                text=text,
                text_sha256=text_sha256,
                method=method,
                provenance=provenance,
                screening_findings=screening,
                review_provenance=review,
                known_limitations=limitations,
            )
        )

    page_numbers = [page.physical_page_number for page in pages]
    if page_numbers != sorted(page_numbers) or len(page_numbers) != len(set(page_numbers)):
        raise RetrievalError("evidence.jsonl pages must be unique and in physical page order")
    if set(page_numbers).intersection(excluded_pages):
        raise RetrievalError("Evidence records include a page declared excluded by the manifest")

    decision_by_page: dict[int, dict[str, Any]] = {}
    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            raise RetrievalError(f"selection_decisions entry {index} must be an object")
        page_number = decision.get("physical_page_number")
        if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
            raise RetrievalError(f"selection_decisions entry {index} has an invalid page number")
        if page_number in decision_by_page:
            raise RetrievalError(f"selection_decisions contains duplicate page {page_number}")
        decision_by_page[page_number] = decision
    if set(decision_by_page) != set(page_numbers):
        raise RetrievalError("Manifest selection decisions do not match evidence.jsonl pages")
    for page in pages:
        decision = decision_by_page[page.physical_page_number]
        if decision.get("method") != page.method:
            raise RetrievalError(
                f"Manifest method mismatch for physical page {page.physical_page_number}"
            )
        if decision.get("selected_text_sha256") != page.text_sha256:
            raise RetrievalError(
                f"Manifest text hash mismatch for physical page {page.physical_page_number}"
            )
        decision_limitations = _string_list(
            decision.get("known_limitations"),
            f"selection decision page {page.physical_page_number} limitations",
        )
        if decision_limitations != page.known_limitations:
            raise RetrievalError(
                f"Manifest limitations mismatch for physical page {page.physical_page_number}"
            )

    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        raise RetrievalError("Evidence manifest: 'counts' must be an object")
    expected_counts = {
        "selected_pages": len(pages),
        "canonical_pages": sum(page.method == "canonical_extraction" for page in pages),
        "reviewed_recoveries": sum(page.method == "reviewed_ocr_recovery" for page in pages),
        "excluded_pages": len(excluded_pages),
        "pages_with_limitations": sum(bool(page.known_limitations) for page in pages),
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise RetrievalError(f"Evidence manifest count {key!r} does not match selected records")
    limited_pages = _integer_list(manifest.get("pages_with_limitations"), "pages_with_limitations")
    actual_limited = tuple(page.physical_page_number for page in pages if page.known_limitations)
    if limited_pages != actual_limited:
        raise RetrievalError("Manifest pages_with_limitations does not match selected records")

    return EvidenceCorpus(
        document_id=document_id,
        source_sha256=source_sha256,
        pages=tuple(pages),
        excluded_pages=excluded_pages,
        manifest=manifest,
        evidence_jsonl_sha256=_sha256(evidence_raw),
        manifest_sha256=_sha256(manifest_raw),
    )


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize case-insensitively without stemming, synonyms, or interpretation."""
    tokens: list[str] = []
    for match in _TOKEN_PATTERN.finditer(text):
        token = match.group().casefold()
        if re.fullmatch(r"[\d,.]+", token):
            token = token.replace(",", "")
        tokens.append(token)
    return tuple(tokens)


def _excerpt(text: str, query_terms: set[str], max_characters: int) -> str:
    matches = []
    for match in _TOKEN_PATTERN.finditer(text):
        matched_tokens = tokenize(match.group())
        if matched_tokens and matched_tokens[0] in query_terms:
            matches.append(match)
    if not matches:
        return ""
    center = matches[0].start()
    start = max(0, center - max_characters // 3)
    end = min(len(text), start + max_characters)
    if start:
        next_space = text.find(" ", start)
        if next_space != -1 and next_space < center:
            start = next_space + 1
    if end < len(text):
        previous_space = text.rfind(" ", start, end)
        if previous_space > start:
            end = previous_space
    return text[start:end].strip()


class BM25Retriever:
    """An in-memory BM25 index with one document per selected physical page."""

    def __init__(
        self,
        corpus: EvidenceCorpus,
        *,
        k1: float = K1_DEFAULT,
        b: float = B_DEFAULT,
    ) -> None:
        if k1 <= 0:
            raise RetrievalError("BM25 k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise RetrievalError("BM25 b must be between zero and one")
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self._term_counts = tuple(Counter(tokenize(page.text)) for page in corpus.pages)
        self._lengths = tuple(sum(counts.values()) for counts in self._term_counts)
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self._term_counts:
            document_frequency.update(counts.keys())
        total = len(corpus.pages)
        self._idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        excerpt_characters: int = EXCERPT_CHARACTERS_DEFAULT,
    ) -> tuple[SearchResult, ...]:
        """Rank positive page matches; empty or unmatched queries return no results."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise RetrievalError("top-k must be a positive integer")
        if excerpt_characters < 1:
            raise RetrievalError("excerpt characters must be a positive integer")
        query_tokens = tokenize(query)
        if not query_tokens or not self.corpus.pages or self._average_length == 0:
            return ()

        scores: list[tuple[float, EvidencePage]] = []
        for page, counts, length in zip(
            self.corpus.pages, self._term_counts, self._lengths, strict=True
        ):
            score = 0.0
            normalization = 1 - self.b + self.b * length / self._average_length
            for term in query_tokens:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                score += (
                    self._idf[term]
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * normalization)
                )
            if score > 0:
                scores.append((score, page))
        scores.sort(key=lambda item: (-item[0], item[1].physical_page_number))
        query_terms = set(query_tokens)
        return tuple(
            SearchResult(
                rank=rank,
                score=score,
                page=page,
                excerpt=_excerpt(page.text, query_terms, excerpt_characters),
                evidence_jsonl_sha256=self.corpus.evidence_jsonl_sha256,
                manifest_sha256=self.corpus.manifest_sha256,
            )
            for rank, (score, page) in enumerate(scores[:top_k], start=1)
        )
