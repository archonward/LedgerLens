"""Extract page-linked text from a local PDF without interpreting its contents."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pdfplumber
from pdfminer.pdfdocument import PDFPasswordIncorrect
from pdfminer.pdfexceptions import PDFException
from pdfminer.pdfparser import PDFSyntaxError
from pdfplumber.utils.exceptions import PdfminerException

from ledgerlens.ingestion.quality import assess_text_quality

SCHEMA_VERSION = "1.2"
EXTRACTOR_NAME = "pdfplumber"
TEXT_SETTINGS = {
    "method": "Page.extract_text",
    "layout": False,
    "x_tolerance": 3,
    "y_tolerance": 3,
    "ocr": False,
    "structured_tables": False,
}


class IngestionError(Exception):
    """An expected, user-actionable PDF ingestion failure."""


@dataclass(frozen=True)
class ExtractionResult:
    """Summary of a successfully published extraction."""

    document_id: str
    page_count: int
    pages_without_text: tuple[int, ...]
    pages_requiring_review: tuple[int, ...]
    unresolved_character_code_findings: tuple[tuple[int, int], ...]
    replacement_character_findings: tuple[tuple[int, int], ...]
    unexpected_control_character_findings: tuple[tuple[int, int], ...]
    output_directory: Path
    warnings: tuple[str, ...]


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IngestionError(f"Cannot read PDF '{path}': {exc}") from None
    return digest.hexdigest()


def _page_preview(text: str, limit: int = 500) -> str:
    if not text.strip():
        return "    (no extracted text)"

    preview = text[:limit]
    if len(text) > limit:
        preview += "\n[preview truncated]"
    return "\n".join(f"    {line}" for line in preview.splitlines())


def _format_counted_pages(findings: object, noun: str) -> str:
    assert isinstance(findings, list)
    return (
        ", ".join(
            f"{finding['physical_page_number']} ({finding['count']} {noun}"
            f"{'s' if finding['count'] != 1 else ''})"
            for finding in findings
        )
        or "none"
    )


def _inspection_markdown(metadata: dict[str, object], pages: list[dict[str, object]]) -> str:
    warnings = metadata["warnings"]
    quality_summary = metadata["quality_summary"]
    assert isinstance(warnings, list)
    assert isinstance(quality_summary, dict)
    unresolved = quality_summary["unresolved_character_codes"]
    assert isinstance(unresolved, dict)
    unresolved_pages = unresolved["pages"]
    assert isinstance(unresolved_pages, list)
    replacements = quality_summary["replacement_characters"]
    controls = quality_summary["unexpected_control_characters"]
    assert isinstance(replacements, dict)
    assert isinstance(controls, dict)

    affected_pages = (
        ", ".join(
            f"{finding['physical_page_number']} ({finding['count']} tokens)"
            for finding in unresolved_pages
        )
        or "none"
    )
    review_pages = quality_summary["pages_requiring_review"]
    assert isinstance(review_pages, list)

    lines = [
        "# Extraction inspection",
        "",
        f"- Document ID: `{metadata['document_id']}`",
        f"- Original filename: `{metadata['original_filename']}`",
        f"- Company: {metadata['company']}",
        f"- Report year: {metadata['report_year']}",
        f"- Physical PDF pages: {metadata['page_count']}",
        f"- Extracted at (UTC): `{metadata['extracted_at_utc']}`",
    ]
    if metadata["source_url"]:
        lines.append(f"- Source URL (metadata only): {metadata['source_url']}")

    lines.extend(
        [
            "",
            "## Text quality summary",
            "",
            f"- Status: `{quality_summary['status']}`",
            "- Pages requiring review: "
            + (", ".join(str(page) for page in review_pages) or "none"),
            f"- Pages with unresolved character codes: {affected_pages}",
            f"- Total unresolved character-code tokens: {unresolved['total_count']}",
            "- Pages with Unicode replacement characters: "
            + _format_counted_pages(replacements["pages"], "character"),
            f"- Total Unicode replacement characters: {replacements['total_count']}",
            "- Pages with unexpected control characters: "
            + _format_counted_pages(controls["pages"], "character"),
            f"- Total unexpected control characters: {controls['total_count']}",
            "- A status of `no_detected_issues` does not mean the text is verified.",
        ]
    )

    lines.extend(
        [
            "",
            "## Warnings",
            "",
        ]
    )
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No page-level extraction warnings.")

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- Page numbers below are physical PDF page numbers and may differ from printed page "
            "labels.",
            "- Report year is document-level metadata; individual pages may contain figures for "
            "multiple years.",
            "- Currency and units have not yet been interpreted.",
            "- Extracted text is not validated financial data.",
            "- OCR and structured table extraction were not performed. Table structure and reading "
            "order require comparison with the source PDF.",
            "",
            "## Page previews",
        ]
    )

    for page in pages:
        page_warnings = page["warnings"]
        assert isinstance(page_warnings, list)
        lines.extend(
            [
                "",
                f"### Physical page {page['physical_page_number']}",
                "",
                f"Characters extracted: {page['character_count']}",
                f"Text quality status: `{page['text_quality']['status']}`",
                "Unresolved character-code tokens: "
                f"{page['text_quality']['unresolved_character_code_count']}",
                "Unicode replacement characters: "
                f"{page['text_quality']['replacement_character_count']}",
                "Unexpected Unicode control characters: "
                f"{page['text_quality']['unexpected_control_character_count']}",
                f"Requires review: {'yes' if page['text_quality']['requires_review'] else 'no'}",
                "Eligible for automatic indexing: "
                f"{'yes' if page['text_quality']['automatic_indexing_eligible'] else 'no'}",
                "",
            ]
        )
        if page_warnings:
            lines.extend(f"Warning: {warning}" for warning in page_warnings)
            lines.append("")
        lines.append(_page_preview(str(page["text"])))

    return "\n".join(lines) + "\n"


def _publish(temp_directory: Path, output_directory: Path, overwrite: bool) -> None:
    if not output_directory.exists():
        temp_directory.replace(output_directory)
        return

    if not overwrite:
        raise IngestionError(
            f"Output already exists at '{output_directory}'. Use --overwrite to replace it."
        )

    backup = output_directory.with_name(f".{output_directory.name}.backup-{uuid.uuid4().hex}")
    output_directory.replace(backup)
    try:
        temp_directory.replace(output_directory)
    except OSError:
        backup.replace(output_directory)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def ingest_pdf(
    pdf_path: Path,
    *,
    company: str,
    report_year: int,
    source_url: str | None = None,
    output_root: Path = Path("data/processed"),
    overwrite: bool = False,
) -> ExtractionResult:
    """Extract a local PDF and publish page-linked JSON plus an inspection report."""
    path = Path(pdf_path)
    if not path.exists():
        raise IngestionError(f"PDF does not exist: '{path}'.")
    if not path.is_file():
        raise IngestionError(f"PDF path is not a file: '{path}'.")
    if not company.strip():
        raise IngestionError("Company must not be empty.")
    if report_year < 1:
        raise IngestionError("Report year must be a positive integer.")

    full_hash = compute_sha256(path)
    document_id = f"sha256-{full_hash}"
    output_root = Path(output_root)
    output_directory = output_root / document_id

    if output_directory.exists() and not overwrite:
        raise IngestionError(
            f"Output already exists at '{output_directory}'. Use --overwrite to replace it."
        )

    try:
        output_root.mkdir(parents=True, exist_ok=True)
        temp_directory = Path(tempfile.mkdtemp(prefix=f".{document_id}.", dir=output_root))
    except OSError as exc:
        raise IngestionError(f"Cannot prepare output directory '{output_root}': {exc}") from None

    pages: list[dict[str, object]] = []
    pages_without_text: list[int] = []
    pages_requiring_review: list[int] = []
    unresolved_character_code_findings: list[tuple[int, int]] = []
    replacement_character_findings: list[tuple[int, int]] = []
    unexpected_control_character_findings: list[tuple[int, int]] = []
    warnings: list[str] = []

    try:
        try:
            with pdfplumber.open(path) as pdf:
                if not pdf.pages:
                    raise IngestionError(f"PDF contains no pages: '{path}'.")

                pages_path = temp_directory / "pages.jsonl"
                with pages_path.open("w", encoding="utf-8", newline="\n") as pages_file:
                    for physical_page_number, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text(
                            layout=False,
                            x_tolerance=3,
                            y_tolerance=3,
                        )
                        extracted_text = text if text is not None else ""
                        quality = assess_text_quality(extracted_text)
                        page_warnings = list(quality.warnings)
                        if not extracted_text.strip():
                            pages_without_text.append(physical_page_number)
                        if quality.requires_review:
                            pages_requiring_review.append(physical_page_number)
                        if quality.unresolved_character_code_count:
                            unresolved_character_code_findings.append(
                                (physical_page_number, quality.unresolved_character_code_count)
                            )
                        if quality.replacement_character_count:
                            replacement_character_findings.append(
                                (physical_page_number, quality.replacement_character_count)
                            )
                        if quality.unexpected_control_character_count:
                            unexpected_control_character_findings.append(
                                (physical_page_number, quality.unexpected_control_character_count)
                            )
                        warnings.extend(
                            f"Physical page {physical_page_number}: {warning}"
                            for warning in page_warnings
                        )

                        page_record: dict[str, object] = {
                            "schema_version": SCHEMA_VERSION,
                            "document_id": document_id,
                            "physical_page_number": physical_page_number,
                            "text": extracted_text,
                            "character_count": len(extracted_text),
                            "warnings": page_warnings,
                            "text_quality": quality.as_dict(),
                        }
                        pages.append(page_record)
                        pages_file.write(json.dumps(page_record, ensure_ascii=False) + "\n")
        except PDFPasswordIncorrect:
            raise IngestionError(
                f"PDF is password-protected and cannot be opened without a password: '{path}'."
            ) from None
        except PdfminerException as exc:
            underlying_error = exc.args[0] if exc.args else None
            if isinstance(underlying_error, PDFPasswordIncorrect):
                raise IngestionError(
                    f"PDF is password-protected and cannot be opened without a password: '{path}'."
                ) from None
            raise IngestionError(
                f"Could not parse PDF '{path}': {underlying_error or exc}."
            ) from None
        except PDFSyntaxError as exc:
            raise IngestionError(f"PDF is malformed or invalid: '{path}' ({exc}).") from None
        except PDFException as exc:
            raise IngestionError(f"Could not parse PDF '{path}': {exc}.") from None
        except OSError as exc:
            raise IngestionError(f"Could not read PDF '{path}': {exc}.") from None
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(f"Could not extract text from PDF '{path}': {exc}.") from None

        metadata: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "document_id": document_id,
            "sha256": full_hash,
            "original_filename": path.name,
            "company": company.strip(),
            "report_year": report_year,
            "source_url": source_url,
            "page_count": len(pages),
            "extracted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "extractor": {
                "name": EXTRACTOR_NAME,
                "version": version(EXTRACTOR_NAME),
            },
            "extraction_settings": TEXT_SETTINGS,
            "warnings": warnings,
            "quality_summary": {
                "status": ("issues_detected" if pages_requiring_review else "no_detected_issues"),
                "pages_requiring_review": pages_requiring_review,
                "pages_unsuitable_for_automatic_indexing": pages_requiring_review,
                "pages_with_no_extracted_text": pages_without_text,
                "unresolved_character_codes": {
                    "total_count": sum(count for _, count in unresolved_character_code_findings),
                    "pages": [
                        {
                            "physical_page_number": page_number,
                            "count": count,
                        }
                        for page_number, count in unresolved_character_code_findings
                    ],
                },
                "replacement_characters": {
                    "total_count": sum(count for _, count in replacement_character_findings),
                    "pages": [
                        {"physical_page_number": page_number, "count": count}
                        for page_number, count in replacement_character_findings
                    ],
                },
                "unexpected_control_characters": {
                    "total_count": sum(count for _, count in unexpected_control_character_findings),
                    "pages": [
                        {"physical_page_number": page_number, "count": count}
                        for page_number, count in unexpected_control_character_findings
                    ],
                },
            },
        }

        (temp_directory / "document.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temp_directory / "inspection.md").write_text(
            _inspection_markdown(metadata, pages),
            encoding="utf-8",
        )

        try:
            _publish(temp_directory, output_directory, overwrite)
        except OSError as exc:
            raise IngestionError(f"Could not publish output '{output_directory}': {exc}") from None
    except IngestionError:
        raise
    except OSError as exc:
        raise IngestionError(f"Could not write extraction output: {exc}") from None
    finally:
        if temp_directory.exists():
            shutil.rmtree(temp_directory, ignore_errors=True)

    return ExtractionResult(
        document_id=document_id,
        page_count=len(pages),
        pages_without_text=tuple(pages_without_text),
        pages_requiring_review=tuple(pages_requiring_review),
        unresolved_character_code_findings=tuple(unresolved_character_code_findings),
        replacement_character_findings=tuple(replacement_character_findings),
        unexpected_control_character_findings=tuple(unexpected_control_character_findings),
        output_directory=output_directory.resolve(),
        warnings=tuple(warnings),
    )
