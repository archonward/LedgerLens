"""Select provenance-bound page text for retrieval prototypes."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ledgerlens.ingestion.pdf import IngestionError, _publish

REVIEW_SCHEMA_VERSION = "1.0"
REVIEW_MANIFEST_SCHEMA_VERSION = "1.0"
EVIDENCE_SCHEMA_VERSION = "1.0"


class EvidenceSelectionError(IngestionError):
    """A review or evidence input failed provenance validation."""


@dataclass(frozen=True)
class ReviewedRecoveryResult:
    """Paths and hashes for one reviewed recovery derivative."""

    output_directory: Path
    corrected_text_sha256: str
    review_record_sha256: str


@dataclass(frozen=True)
class EvidenceSelection:
    """Selected evidence and the decisions that produced it."""

    document_id: str
    source_sha256: str
    pages: tuple[dict[str, object], ...]
    excluded_pages: tuple[dict[str, object], ...]
    canonical_count: int
    reviewed_recovery_count: int
    pages_with_limitations: tuple[int, ...]
    input_hashes: dict[str, str]


@dataclass(frozen=True)
class EvidenceExportResult:
    """Summary of a published evidence export."""

    document_id: str
    canonical_count: int
    reviewed_recovery_count: int
    excluded_pages: tuple[int, ...]
    pages_with_limitations: tuple[int, ...]
    output_directory: Path


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_bytes(path: Path, description: str) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise EvidenceSelectionError(f"{description} does not exist: '{path}'.") from None
    except OSError as exc:
        raise EvidenceSelectionError(f"Could not read {description} '{path}': {exc}") from None


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(_read_bytes(path, description))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceSelectionError(
            f"{description} is not valid UTF-8 JSON: '{path}' ({exc})."
        ) from None
    if not isinstance(value, dict):
        raise EvidenceSelectionError(f"{description} must contain a JSON object: '{path}'.")
    return value


def _resolve_manifest_path(manifest_path: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvidenceSelectionError(f"Review selection field '{field}' must be a non-empty path.")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise EvidenceSelectionError(f"{label} mismatch: expected {expected!r}, found {actual!r}.")


def create_reviewed_recovery(
    candidate_path: Path,
    recovery_metadata_path: Path,
    *,
    output_directory: Path,
    expected_phrase: str,
    corrected_phrase: str,
    correction_reason: str,
    review_provenance: dict[str, object],
    review_scope: list[str],
    limitations: list[str],
    overwrite: bool = False,
) -> ReviewedRecoveryResult:
    """Create a separately stored, single-correction reviewed derivative."""
    candidate_path = Path(candidate_path).resolve()
    recovery_metadata_path = Path(recovery_metadata_path).resolve()
    output_directory = Path(output_directory)
    if output_directory.exists() and not overwrite:
        raise EvidenceSelectionError(
            f"Reviewed recovery already exists at '{output_directory}'. "
            "Use overwrite=True to replace it."
        )
    if not expected_phrase or not corrected_phrase:
        raise EvidenceSelectionError("Expected and corrected phrases must be non-empty.")
    if expected_phrase == corrected_phrase:
        raise EvidenceSelectionError("Expected and corrected phrases must differ.")

    candidate_bytes = _read_bytes(candidate_path, "recovery candidate")
    try:
        candidate_text = candidate_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceSelectionError(
            f"Recovery candidate is not valid UTF-8: '{candidate_path}' ({exc})."
        ) from None
    occurrence_count = candidate_text.count(expected_phrase)
    if occurrence_count != 1:
        raise EvidenceSelectionError(
            f"Expected correction phrase must occur exactly once; found {occurrence_count}."
        )

    recovery_bytes = _read_bytes(recovery_metadata_path, "recovery metadata")
    recovery = _read_json(recovery_metadata_path, "recovery metadata")
    if recovery.get("text") != candidate_text:
        raise EvidenceSelectionError(
            "Recovery metadata text does not exactly match the candidate text file."
        )
    if recovery.get("engine") != "tesseract":
        raise EvidenceSelectionError(
            "Reviewed OCR recovery metadata must use the tesseract engine."
        )

    corrected_text = candidate_text.replace(expected_phrase, corrected_phrase, 1)
    corrected_bytes = corrected_text.encode("utf-8")
    record: dict[str, object] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "document_id": recovery.get("document_id"),
        "source_sha256": recovery.get("source_sha256"),
        "physical_page_number": recovery.get("physical_page_number"),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "recovery_engine": "tesseract",
        "artifacts": {
            "original_candidate_sha256": _sha256_bytes(candidate_bytes),
            "recovery_metadata_sha256": _sha256_bytes(recovery_bytes),
            "corrected_text_filename": "corrected.txt",
            "corrected_text_sha256": _sha256_bytes(corrected_bytes),
        },
        "correction": {
            "expected_phrase": expected_phrase,
            "corrected_phrase": corrected_phrase,
            "occurrence_count": 1,
            "reason": correction_reason,
            "other_text_changed": False,
        },
        "review_provenance": review_provenance,
        "review_scope": review_scope,
        "permission_for_retrieval_prototype": True,
        "human_independent_verification_claimed": False,
        "limitations": limitations,
        "approval_exclusions": [
            "unattended numerical interpretation",
            "validated structured financial calculations",
        ],
    }

    try:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temp_directory = Path(
            tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
        )
    except OSError as exc:
        raise EvidenceSelectionError(f"Could not prepare reviewed recovery output: {exc}") from None
    try:
        (temp_directory / "corrected.txt").write_bytes(corrected_bytes)
        (temp_directory / "review.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _publish(temp_directory, output_directory, overwrite)
    except OSError as exc:
        raise EvidenceSelectionError(f"Could not publish reviewed recovery: {exc}") from None
    finally:
        if temp_directory.exists():
            shutil.rmtree(temp_directory, ignore_errors=True)

    review_path = output_directory / "review.json"
    return ReviewedRecoveryResult(
        output_directory=output_directory.resolve(),
        corrected_text_sha256=_sha256_bytes(corrected_bytes),
        review_record_sha256=_sha256_bytes(_read_bytes(review_path, "review record")),
    )


def _load_canonical_pages(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                rows.append(value)
    except FileNotFoundError:
        raise EvidenceSelectionError(f"Canonical pages file does not exist: '{path}'.") from None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceSelectionError(f"Could not read canonical pages '{path}': {exc}") from None
    return rows


def _load_page_limitations(manifest: dict[str, Any], *, page_count: int) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    records = manifest.get("page_limitations", [])
    if not isinstance(records, list):
        raise EvidenceSelectionError("Review manifest 'page_limitations' must be a list.")
    for record in records:
        if not isinstance(record, dict):
            raise EvidenceSelectionError("Each page limitation must be an object.")
        page_number = record.get("physical_page_number")
        limitations = record.get("limitations")
        if not isinstance(page_number, int) or not 1 <= page_number <= page_count:
            raise EvidenceSelectionError(f"Invalid limitation page number: {page_number!r}.")
        if page_number in result:
            raise EvidenceSelectionError(
                f"Duplicate page limitation for physical page {page_number}."
            )
        if not isinstance(limitations, list) or not all(
            isinstance(item, str) and item for item in limitations
        ):
            raise EvidenceSelectionError(
                f"Limitations for physical page {page_number} must be non-empty strings."
            )
        result[page_number] = limitations
    return result


def _validate_review_selection(
    selection: dict[str, Any],
    *,
    manifest_path: Path,
    document_id: str,
    source_sha256: str,
    page_count: int,
) -> tuple[int, str, dict[str, Any], dict[str, Any], dict[str, str]]:
    if selection.get("selected_for_retrieval_prototype") is not True:
        raise EvidenceSelectionError(
            "Every recovery selection must explicitly set selected_for_retrieval_prototype=true."
        )
    page_number = selection.get("physical_page_number")
    if not isinstance(page_number, int) or not 1 <= page_number <= page_count:
        raise EvidenceSelectionError(f"Invalid recovery selection page number: {page_number!r}.")

    corrected_path = _resolve_manifest_path(
        manifest_path, selection.get("corrected_text_path"), "corrected_text_path"
    )
    review_path = _resolve_manifest_path(
        manifest_path, selection.get("review_record_path"), "review_record_path"
    )
    candidate_path = _resolve_manifest_path(
        manifest_path, selection.get("original_candidate_path"), "original_candidate_path"
    )
    recovery_path = _resolve_manifest_path(
        manifest_path, selection.get("recovery_metadata_path"), "recovery_metadata_path"
    )

    corrected_bytes = _read_bytes(corrected_path, "corrected recovery text")
    candidate_bytes = _read_bytes(candidate_path, "original recovery candidate")
    recovery_bytes = _read_bytes(recovery_path, "recovery metadata")
    review_bytes = _read_bytes(review_path, "review record")
    review = _read_json(review_path, "review record")
    recovery = _read_json(recovery_path, "recovery metadata")

    _require_equal(review.get("schema_version"), REVIEW_SCHEMA_VERSION, "Review schema version")
    _require_equal(review.get("document_id"), document_id, "Review document ID")
    _require_equal(review.get("source_sha256"), source_sha256, "Review source hash")
    _require_equal(review.get("physical_page_number"), page_number, "Review page number")
    _require_equal(recovery.get("document_id"), document_id, "Recovery document ID")
    _require_equal(recovery.get("source_sha256"), source_sha256, "Recovery source hash")
    _require_equal(recovery.get("physical_page_number"), page_number, "Recovery page number")
    _require_equal(recovery.get("source_page_count"), page_count, "Recovery source page count")
    _require_equal(recovery.get("engine"), "tesseract", "Recovery engine")
    _require_equal(review.get("recovery_engine"), "tesseract", "Review recovery engine")
    if review.get("permission_for_retrieval_prototype") is not True:
        raise EvidenceSelectionError(
            f"Review for physical page {page_number} lacks prototype-use permission."
        )
    if not isinstance(review.get("review_provenance"), dict):
        raise EvidenceSelectionError("Review record 'review_provenance' must be an object.")

    artifacts = review.get("artifacts")
    if not isinstance(artifacts, dict):
        raise EvidenceSelectionError("Review record 'artifacts' must be an object.")
    actual_hashes = {
        "original_candidate_sha256": _sha256_bytes(candidate_bytes),
        "recovery_metadata_sha256": _sha256_bytes(recovery_bytes),
        "corrected_text_sha256": _sha256_bytes(corrected_bytes),
        "review_record_sha256": _sha256_bytes(review_bytes),
    }
    for field, actual_hash in actual_hashes.items():
        _require_equal(actual_hash, selection.get(field), f"Manifest {field}")
        if field != "review_record_sha256":
            _require_equal(actual_hash, artifacts.get(field), f"Review {field}")

    try:
        candidate_text = candidate_bytes.decode("utf-8")
        corrected_text = corrected_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceSelectionError(f"Recovery text is not valid UTF-8: {exc}.") from None
    _require_equal(recovery.get("text"), candidate_text, "Recovery candidate text")

    correction = review.get("correction")
    if not isinstance(correction, dict):
        raise EvidenceSelectionError("Review record 'correction' must be an object.")
    expected_phrase = correction.get("expected_phrase")
    corrected_phrase = correction.get("corrected_phrase")
    if not isinstance(expected_phrase, str) or not isinstance(corrected_phrase, str):
        raise EvidenceSelectionError("Review correction phrases must be strings.")
    occurrence_count = candidate_text.count(expected_phrase)
    _require_equal(occurrence_count, 1, "Correction occurrence count")
    _require_equal(correction.get("occurrence_count"), 1, "Recorded correction occurrence count")
    _require_equal(
        corrected_text,
        candidate_text.replace(expected_phrase, corrected_phrase, 1),
        "Corrected recovery text",
    )

    return page_number, corrected_text, review, recovery, actual_hashes


def select_page_evidence(
    canonical_directory: Path, review_manifest_path: Path | None = None
) -> EvidenceSelection:
    """Select one explicit, provenance-bound representation for each usable physical page."""
    canonical_directory = Path(canonical_directory).resolve()
    document_path = canonical_directory / "document.json"
    pages_path = canonical_directory / "pages.jsonl"
    inspection_path = canonical_directory / "inspection.md"
    document = _read_json(document_path, "canonical document metadata")
    pages = _load_canonical_pages(pages_path)

    document_id = document.get("document_id")
    source_sha256 = document.get("sha256")
    page_count = document.get("page_count")
    if not isinstance(document_id, str) or not isinstance(source_sha256, str):
        raise EvidenceSelectionError("Canonical document identity or source hash is missing.")
    _require_equal(document_id, f"sha256-{source_sha256}", "Canonical document ID")
    if not isinstance(page_count, int) or page_count != len(pages):
        raise EvidenceSelectionError("Canonical page count does not match pages.jsonl.")

    seen_pages: set[int] = set()
    for expected_page, page in enumerate(pages, start=1):
        page_number = page.get("physical_page_number")
        _require_equal(page_number, expected_page, "Canonical physical page order")
        _require_equal(page.get("document_id"), document_id, "Canonical page document ID")
        if page_number in seen_pages:
            raise EvidenceSelectionError(f"Duplicate canonical physical page {page_number}.")
        seen_pages.add(page_number)

    manifest: dict[str, Any] = {
        "schema_version": REVIEW_MANIFEST_SCHEMA_VERSION,
        "document_id": document_id,
        "source_sha256": source_sha256,
        "reviewed_recoveries": [],
        "page_limitations": [],
    }
    manifest_path: Path | None = None
    input_hashes = {
        "canonical_document_json": _sha256_bytes(
            _read_bytes(document_path, "canonical document metadata")
        ),
        "canonical_pages_jsonl": _sha256_bytes(_read_bytes(pages_path, "canonical pages")),
    }
    if inspection_path.is_file():
        input_hashes["canonical_inspection_md"] = _sha256_bytes(
            _read_bytes(inspection_path, "canonical inspection")
        )
    if review_manifest_path is not None:
        manifest_path = Path(review_manifest_path).resolve()
        manifest_bytes = _read_bytes(manifest_path, "review manifest")
        manifest = _read_json(manifest_path, "review manifest")
        input_hashes["review_manifest"] = _sha256_bytes(manifest_bytes)
        _require_equal(
            manifest.get("schema_version"),
            REVIEW_MANIFEST_SCHEMA_VERSION,
            "Review manifest schema version",
        )
        _require_equal(manifest.get("document_id"), document_id, "Manifest document ID")
        _require_equal(manifest.get("source_sha256"), source_sha256, "Manifest source hash")

    page_limitations = _load_page_limitations(manifest, page_count=page_count)
    recovery_records = manifest.get("reviewed_recoveries", [])
    if not isinstance(recovery_records, list):
        raise EvidenceSelectionError("Review manifest 'reviewed_recoveries' must be a list.")
    recoveries: dict[int, tuple[str, dict[str, Any], dict[str, Any], dict[str, str]]] = {}
    for item in recovery_records:
        if not isinstance(item, dict) or manifest_path is None:
            raise EvidenceSelectionError("Each reviewed recovery selection must be an object.")
        page_number, text, review, recovery, review_hashes = _validate_review_selection(
            item,
            manifest_path=manifest_path,
            document_id=document_id,
            source_sha256=source_sha256,
            page_count=page_count,
        )
        if page_number in recoveries:
            raise EvidenceSelectionError(
                f"Duplicate or conflicting recovery selections for physical page {page_number}."
            )
        recoveries[page_number] = (text, review, recovery, review_hashes)
        for name, digest in review_hashes.items():
            input_hashes[f"reviewed_page_{page_number}_{name}"] = digest

    selected: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    canonical_count = 0
    recovery_count = 0
    limited_pages: list[int] = []
    for page in pages:
        page_number = int(page["physical_page_number"])
        canonical_quality = page.get("text_quality")
        if not isinstance(canonical_quality, dict):
            raise EvidenceSelectionError(
                f"Canonical physical page {page_number} lacks screening metadata."
            )
        limitations = list(page_limitations.get(page_number, []))
        if page_number in recoveries:
            text, review, recovery, review_hashes = recoveries[page_number]
            review_limitations = review.get("limitations", [])
            if not isinstance(review_limitations, list) or not all(
                isinstance(item, str) for item in review_limitations
            ):
                raise EvidenceSelectionError(
                    f"Review limitations for physical page {page_number} are invalid."
                )
            limitations.extend(item for item in review_limitations if item not in limitations)
            record = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "document_id": document_id,
                "company": document.get("company"),
                "report_year": document.get("report_year"),
                "physical_page_number": page_number,
                "text": text,
                "text_sha256": _sha256_bytes(text.encode("utf-8")),
                "method": "reviewed_ocr_recovery",
                "provenance": {
                    "source_sha256": source_sha256,
                    "recovery_engine": recovery.get("engine"),
                    "extractor": recovery.get("extractor"),
                    "extraction_settings": recovery.get("extraction_settings"),
                    "original_candidate_sha256": review["artifacts"]["original_candidate_sha256"],
                    "recovery_metadata_sha256": review["artifacts"]["recovery_metadata_sha256"],
                    "review_record_sha256": review_hashes["review_record_sha256"],
                    "corrected_text_sha256": review_hashes["corrected_text_sha256"],
                },
                "screening_findings": {
                    "canonical": canonical_quality,
                    "selected_recovery": recovery.get("text_quality"),
                    "screening_is_not_verification": True,
                },
                "review_provenance": review.get("review_provenance"),
                "known_limitations": limitations,
            }
            recovery_count += 1
        elif canonical_quality.get("automatic_indexing_eligible") is True:
            text = page.get("text")
            if not isinstance(text, str):
                raise EvidenceSelectionError(
                    f"Canonical physical page {page_number} text must be a string."
                )
            record = {
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "document_id": document_id,
                "company": document.get("company"),
                "report_year": document.get("report_year"),
                "physical_page_number": page_number,
                "text": text,
                "text_sha256": _sha256_bytes(text.encode("utf-8")),
                "method": "canonical_extraction",
                "provenance": {
                    "source_sha256": source_sha256,
                    "canonical_schema_version": page.get("schema_version"),
                    "extractor": document.get("extractor"),
                    "extraction_settings": document.get("extraction_settings"),
                },
                "screening_findings": {
                    "canonical": canonical_quality,
                    "screening_is_not_verification": True,
                },
                "review_provenance": None,
                "known_limitations": limitations,
            }
            canonical_count += 1
        else:
            excluded.append(
                {
                    "physical_page_number": page_number,
                    "reason": (
                        "canonical_screening_disallows_automatic_indexing_and_"
                        "no_reviewed_recovery_was_selected"
                    ),
                    "screening_findings": canonical_quality,
                }
            )
            continue
        if limitations:
            limited_pages.append(page_number)
        selected.append(record)

    return EvidenceSelection(
        document_id=document_id,
        source_sha256=source_sha256,
        pages=tuple(selected),
        excluded_pages=tuple(excluded),
        canonical_count=canonical_count,
        reviewed_recovery_count=recovery_count,
        pages_with_limitations=tuple(limited_pages),
        input_hashes=input_hashes,
    )


def prepare_evidence(
    canonical_directory: Path,
    *,
    review_manifest_path: Path | None = None,
    output_root: Path = Path("artifacts/evidence"),
    overwrite: bool = False,
) -> EvidenceExportResult:
    """Atomically publish selected page evidence and its provenance manifest."""
    selection = select_page_evidence(canonical_directory, review_manifest_path)
    output_directory = Path(output_root) / selection.document_id
    if output_directory.exists() and not overwrite:
        raise EvidenceSelectionError(
            f"Evidence output already exists at '{output_directory}'. "
            "Use --overwrite to replace it."
        )
    try:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temp_directory = Path(
            tempfile.mkdtemp(prefix=f".{selection.document_id}.", dir=output_directory.parent)
        )
    except OSError as exc:
        raise EvidenceSelectionError(f"Could not prepare evidence output: {exc}") from None
    try:
        with (temp_directory / "evidence.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for page in selection.pages:
                stream.write(json.dumps(page, ensure_ascii=False) + "\n")
        export_manifest = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "document_id": selection.document_id,
            "source_sha256": selection.source_sha256,
            "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "input_hashes": selection.input_hashes,
            "counts": {
                "selected_pages": len(selection.pages),
                "canonical_pages": selection.canonical_count,
                "reviewed_recoveries": selection.reviewed_recovery_count,
                "excluded_pages": len(selection.excluded_pages),
                "pages_with_limitations": len(selection.pages_with_limitations),
            },
            "selection_decisions": [
                {
                    "physical_page_number": page["physical_page_number"],
                    "method": page["method"],
                    "selected_text_sha256": page["text_sha256"],
                    "known_limitations": page["known_limitations"],
                }
                for page in selection.pages
            ],
            "excluded_pages": list(selection.excluded_pages),
            "pages_with_limitations": list(selection.pages_with_limitations),
            "limitations": [
                "Automatic screening is not verification.",
                "Selected evidence is for a page-level retrieval prototype, not unattended "
                "numerical interpretation.",
                "Sampling and selection do not establish that every source page is accurate.",
            ],
        }
        (temp_directory / "manifest.json").write_text(
            json.dumps(export_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _publish(temp_directory, output_directory, overwrite)
    except OSError as exc:
        raise EvidenceSelectionError(f"Could not publish evidence output: {exc}") from None
    finally:
        if temp_directory.exists():
            shutil.rmtree(temp_directory, ignore_errors=True)

    return EvidenceExportResult(
        document_id=selection.document_id,
        canonical_count=selection.canonical_count,
        reviewed_recovery_count=selection.reviewed_recovery_count,
        excluded_pages=tuple(
            int(record["physical_page_number"]) for record in selection.excluded_pages
        ),
        pages_with_limitations=selection.pages_with_limitations,
        output_directory=output_directory.resolve(),
    )
