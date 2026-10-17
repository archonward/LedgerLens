"""Extract one page with an optional alternative engine for manual review."""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ledgerlens.ingestion.pdf import IngestionError, _publish, compute_sha256
from ledgerlens.ingestion.quality import TextQuality, assess_text_quality

RECOVERY_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class RecoveryResult:
    """Summary of an alternative extraction candidate."""

    document_id: str
    physical_page_number: int
    page_count: int
    character_count: int
    quality: TextQuality
    output_directory: Path


def _load_pypdf() -> tuple[Any, type[Exception], str]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:
        raise IngestionError(
            "The optional recovery dependency is not installed. "
            "Run 'uv sync --extra recovery --group dev' and try again."
        ) from None
    return PdfReader, PdfReadError, version("pypdf")


def recover_page(
    pdf_path: Path,
    *,
    physical_page_number: int,
    output_root: Path = Path("artifacts/recovery"),
    overwrite: bool = False,
) -> RecoveryResult:
    """Extract one physical page with pypdf as an unreviewed candidate."""
    path = Path(pdf_path)
    if not path.exists():
        raise IngestionError(f"PDF does not exist: '{path}'.")
    if not path.is_file():
        raise IngestionError(f"PDF path is not a file: '{path}'.")
    if physical_page_number < 1:
        raise IngestionError("Physical page number must be at least 1.")

    full_hash = compute_sha256(path)
    document_id = f"sha256-{full_hash}"
    output_root = Path(output_root)
    output_directory = output_root / document_id / f"page-{physical_page_number}"
    if output_directory.exists() and not overwrite:
        raise IngestionError(
            f"Recovery output already exists at '{output_directory}'. "
            "Use --overwrite to replace it."
        )

    PdfReader, PdfReadError, extractor_version = _load_pypdf()
    try:
        reader = PdfReader(path)
        page_count = len(reader.pages)
        if physical_page_number > page_count:
            raise IngestionError(
                f"Physical page {physical_page_number} is out of range; "
                f"the PDF has {page_count} pages."
            )
        text = reader.pages[physical_page_number - 1].extract_text() or ""
    except IngestionError:
        raise
    except PdfReadError as exc:
        raise IngestionError(f"Could not parse PDF '{path}' with pypdf: {exc}") from None
    except OSError as exc:
        raise IngestionError(f"Could not read PDF '{path}': {exc}") from None
    except Exception as exc:
        raise IngestionError(
            f"Could not recover physical page {physical_page_number} with pypdf: {exc}"
        ) from None

    quality = assess_text_quality(text)
    candidate_warning = (
        "Alternative extraction candidate requires visual review and must not automatically "
        "replace canonical ingestion text or be treated as verified evidence."
    )
    record: dict[str, object] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "document_id": document_id,
        "source_sha256": full_hash,
        "original_filename": path.name,
        "physical_page_number": physical_page_number,
        "source_page_count": page_count,
        "extracted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "extractor": {"name": "pypdf", "version": extractor_version},
        "extraction_settings": {
            "method": "PageObject.extract_text",
            "arguments": {},
            "physical_page_number_basis": 1,
        },
        "candidate_status": "requires_review",
        "automatic_indexing_eligible": False,
        "approved_for_evidence": False,
        "approved_for_canonical_replacement": False,
        "canonical_ingestion_modified": False,
        "text": text,
        "character_count": len(text),
        "text_quality": quality.as_dict(),
        "warnings": [candidate_warning, *quality.warnings],
    }

    try:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temp_directory = Path(
            tempfile.mkdtemp(prefix=f".page-{physical_page_number}.", dir=output_directory.parent)
        )
    except OSError as exc:
        raise IngestionError(f"Cannot prepare recovery output: {exc}") from None

    try:
        (temp_directory / "recovery.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temp_directory / "candidate.txt").write_text(text, encoding="utf-8")
        try:
            _publish(temp_directory, output_directory, overwrite)
        except OSError as exc:
            raise IngestionError(f"Could not publish recovery output: {exc}") from None
    except IngestionError:
        raise
    except OSError as exc:
        raise IngestionError(f"Could not write recovery output: {exc}") from None
    finally:
        if temp_directory.exists():
            shutil.rmtree(temp_directory, ignore_errors=True)

    return RecoveryResult(
        document_id=document_id,
        physical_page_number=physical_page_number,
        page_count=page_count,
        character_count=len(text),
        quality=quality,
        output_directory=output_directory.resolve(),
    )
