"""Command-line interface for LedgerLens development utilities."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

from ledgerlens import __version__
from ledgerlens.ingestion import IngestionError, ingest_pdf, recover_page


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerlens",
        description="Evidence-based analysis tools for financial reports.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report local environment information without running GPU workloads.",
    )
    doctor_parser.set_defaults(handler=_run_doctor)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Extract page-linked text from a local PDF.",
    )
    ingest_parser.add_argument("pdf", type=Path, help="Path to a local PDF file.")
    ingest_parser.add_argument("--company", required=True, help="Company named by the report.")
    ingest_parser.add_argument(
        "--report-year",
        required=True,
        type=int,
        help="Document-level report year metadata.",
    )
    ingest_parser.add_argument(
        "--source-url",
        help="Optional provenance URL to record without fetching it.",
    )
    ingest_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Output parent directory (default: data/processed).",
    )
    ingest_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing extraction for identical PDF content.",
    )
    ingest_parser.set_defaults(handler=_run_ingest)

    recovery_parser = subparsers.add_parser(
        "recover-page",
        help="Extract one page as an unapproved recovery candidate.",
    )
    recovery_parser.add_argument("pdf", type=Path, help="Path to a local PDF file.")
    recovery_parser.add_argument(
        "--page",
        required=True,
        type=int,
        help="Physical PDF page number, starting from 1.",
    )
    recovery_parser.add_argument(
        "--engine",
        choices=("pypdf", "tesseract"),
        default="pypdf",
        help="Recovery engine (default: pypdf).",
    )
    recovery_parser.add_argument(
        "--tesseract-path",
        type=Path,
        help="Explicit path to the Tesseract executable (OCR engine only).",
    )
    recovery_parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="PDF rendering resolution for OCR (default: 300).",
    )
    recovery_parser.add_argument(
        "--language",
        default="eng",
        help="Tesseract language code (default: eng).",
    )
    recovery_parser.add_argument(
        "--psm",
        type=int,
        default=3,
        help="Tesseract page segmentation mode (default: 3).",
    )
    recovery_parser.add_argument(
        "--oem",
        type=int,
        default=1,
        help="Tesseract OCR engine mode (default: 1).",
    )
    recovery_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=120,
        help="Timeout for each Tesseract invocation (default: 120).",
    )
    recovery_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/recovery"),
        help="Recovery output parent directory (default: artifacts/recovery).",
    )
    recovery_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing recovery candidate for this page.",
    )
    recovery_parser.set_defaults(handler=_run_recover_page)
    return parser


def _run_doctor(_: argparse.Namespace) -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    print(f"Python: {platform.python_version()}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"LedgerLens: {__version__}")
    print(f"nvidia-smi on PATH: {'yes' if nvidia_smi else 'no'}")
    print(
        "GPU availability: unverified; finding nvidia-smi alone does not establish "
        "a usable GPU allocation."
    )
    return 0


def _run_ingest(args: argparse.Namespace) -> int:
    try:
        result = ingest_pdf(
            args.pdf,
            company=args.company,
            report_year=args.report_year,
            source_url=args.source_url,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    pages_without_text = (
        ", ".join(str(page) for page in result.pages_without_text)
        if result.pages_without_text
        else "none"
    )
    print(f"Document ID: {result.document_id}")
    print(f"Pages: {result.page_count}")
    print(f"Pages with no extracted text: {pages_without_text}")
    unresolved_findings = (
        ", ".join(
            f"{page} ({count:,} tokens)"
            for page, count in result.unresolved_character_code_findings
        )
        if result.unresolved_character_code_findings
        else "none"
    )
    pages_requiring_review = (
        ", ".join(str(page) for page in result.pages_requiring_review)
        if result.pages_requiring_review
        else "none"
    )
    print(f"Pages with unresolved character codes: {unresolved_findings}")
    replacement_findings = (
        ", ".join(
            f"{page} ({count:,} characters)"
            for page, count in result.replacement_character_findings
        )
        if result.replacement_character_findings
        else "none"
    )
    control_findings = (
        ", ".join(
            f"{page} ({count:,} characters)"
            for page, count in result.unexpected_control_character_findings
        )
        if result.unexpected_control_character_findings
        else "none"
    )
    print(f"Pages with Unicode replacement characters: {replacement_findings}")
    print(f"Pages with unexpected control characters: {control_findings}")
    print(f"Pages requiring review: {pages_requiring_review}")
    print(f"Output: {result.output_directory}")
    if result.warnings:
        print("Extraction warnings:")
        for warning in result.warnings:
            print(f"- {warning}")
    else:
        print("Extraction warnings: none")
    return 0


def _run_recover_page(args: argparse.Namespace) -> int:
    try:
        result = recover_page(
            args.pdf,
            physical_page_number=args.page,
            engine=args.engine,
            output_root=args.output_root,
            overwrite=args.overwrite,
            tesseract_path=args.tesseract_path,
            dpi=args.dpi,
            language=args.language,
            psm=args.psm,
            oem=args.oem,
            timeout_seconds=args.timeout_seconds,
        )
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Document ID: {result.document_id}")
    print(f"Physical page: {result.physical_page_number} of {result.page_count}")
    print(f"Recovery engine: {result.engine}")
    print(f"Characters extracted: {result.character_count}")
    print(f"Unresolved character-code tokens: {result.quality.unresolved_character_code_count:,}")
    print(f"Unicode replacement characters: {result.quality.replacement_character_count:,}")
    print(
        "Unexpected Unicode control characters: "
        f"{result.quality.unexpected_control_character_count:,}"
    )
    if result.ocr_word_count is not None:
        print(f"OCR words: {result.ocr_word_count:,}")
        confidence = (
            f"{result.mean_word_confidence:.2f}"
            if result.mean_word_confidence is not None
            else "unavailable"
        )
        print(f"Mean OCR word confidence: {confidence} (not proof of correctness)")
    print(f"Text quality status: {result.quality.status}")
    print("Candidate status: requires review; canonical ingestion unchanged")
    print(f"Output: {result.output_directory}")
    if result.quality.warnings:
        print("Recovery warnings:")
        for warning in result.quality.warnings:
            print(f"- {warning}")
    else:
        print("Recovery warnings: no detected text-quality issues; candidate is not verified")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LedgerLens command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
