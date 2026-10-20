"""Command-line interface for LedgerLens development utilities."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

from ledgerlens import __version__
from ledgerlens.evaluation import (
    EvaluationError,
    evaluate_bm25,
    load_evaluation_set,
    write_evaluation_report,
)
from ledgerlens.evidence import prepare_evidence
from ledgerlens.ingestion import IngestionError, ingest_pdf, recover_page
from ledgerlens.retrieval import BM25Retriever, RetrievalError, load_evidence_export


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

    evidence_parser = subparsers.add_parser(
        "prepare-evidence",
        help="Select canonical pages and explicitly reviewed recoveries for a prototype.",
    )
    evidence_parser.add_argument(
        "canonical_extraction_directory",
        type=Path,
        help="Directory containing canonical document.json and pages.jsonl.",
    )
    evidence_parser.add_argument(
        "--review-manifest",
        type=Path,
        help="Optional explicit reviewed-recovery selection manifest.",
    )
    evidence_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/evidence"),
        help="Evidence output parent directory (default: artifacts/evidence).",
    )
    evidence_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evidence export for this document.",
    )
    evidence_parser.set_defaults(handler=_run_prepare_evidence)

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Rank selected evidence pages with the BM25 baseline.",
    )
    retrieve_parser.add_argument(
        "evidence_directory",
        type=Path,
        help="Directory containing selected evidence.jsonl and manifest.json.",
    )
    retrieve_parser.add_argument("--query", required=True, help="Natural-language search query.")
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of positive-score pages to return (default: 5).",
    )
    retrieve_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable view.",
    )
    retrieve_parser.set_defaults(handler=_run_retrieve)

    evaluation_parser = subparsers.add_parser(
        "evaluate-retrieval",
        help="Evaluate BM25 against independently supplied page judgments.",
    )
    evaluation_parser.add_argument(
        "evidence_directory",
        type=Path,
        help="Directory containing selected evidence.jsonl and manifest.json.",
    )
    evaluation_parser.add_argument(
        "evaluation_set",
        type=Path,
        help="JSON file containing questions and page-level relevance judgments.",
    )
    evaluation_parser.add_argument(
        "--split",
        required=True,
        choices=("development", "held_out"),
        help="Evaluate exactly one predefined dataset split.",
    )
    evaluation_parser.add_argument(
        "--k",
        dest="cutoffs",
        type=int,
        nargs="+",
        default=(1, 3, 5, 10),
        help="Positive rank cutoffs for Hit@K and Recall@K (default: 1 3 5 10).",
    )
    evaluation_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path for the structured JSON evaluation report.",
    )
    evaluation_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluation report explicitly.",
    )
    evaluation_parser.set_defaults(handler=_run_evaluate_retrieval)
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


def _run_prepare_evidence(args: argparse.Namespace) -> int:
    try:
        result = prepare_evidence(
            args.canonical_extraction_directory,
            review_manifest_path=args.review_manifest,
            output_root=args.output_root,
            overwrite=args.overwrite,
        )
    except IngestionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    excluded = ", ".join(str(page) for page in result.excluded_pages) or "none"
    limited = ", ".join(str(page) for page in result.pages_with_limitations) or "none"
    print(f"Document ID: {result.document_id}")
    print(f"Canonical pages selected: {result.canonical_count}")
    print(f"Reviewed recoveries selected: {result.reviewed_recovery_count}")
    print(f"Excluded physical pages: {excluded}")
    print(f"Pages carrying limitations: {limited}")
    print(f"Output: {result.output_directory}")
    return 0


def _run_retrieve(args: argparse.Namespace) -> int:
    try:
        corpus = load_evidence_export(args.evidence_directory)
        retriever = BM25Retriever(corpus)
        results = retriever.search(args.query, top_k=args.top_k)
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(
            json.dumps(
                {
                    "query": args.query,
                    "document_id": corpus.document_id,
                    "bm25": {"k1": retriever.k1, "b": retriever.b},
                    "result_count": len(results),
                    "no_match": not results,
                    "results": [result.to_dict() for result in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"Document ID: {corpus.document_id}")
    print(f"Query: {args.query}")
    print(f"BM25: k1={retriever.k1}, b={retriever.b}")
    print(f"Positive matches returned: {len(results)}")
    print("Scores rank lexical relevance; they are not confidence or accuracy measures.")
    if not results:
        print("No positive BM25 matches found.")
        return 0

    for result in results:
        page = result.page
        print()
        print(
            f"{result.rank}. Physical PDF page {page.physical_page_number} "
            f"| score {result.score:.6f} | {page.method}"
        )
        print(f"   Selected-text SHA-256: {page.text_sha256}")
        print(f"   Source SHA-256: {page.provenance.get('source_sha256', 'unavailable')}")
        print("   Provenance: " + json.dumps(page.provenance, ensure_ascii=False, sort_keys=True))
        if page.review_provenance is not None:
            print(
                "   Review provenance: "
                + json.dumps(page.review_provenance, ensure_ascii=False, sort_keys=True)
            )
        warnings = page.warnings
        print(f"   Warnings: {'; '.join(warnings) if warnings else 'none recorded'}")
        limitations = "; ".join(page.known_limitations) or "none recorded"
        print(f"   Limitations: {limitations}")
        print("   Excerpt:")
        for line in result.excerpt.splitlines() or [""]:
            print(textwrap.indent(line, "     "))
    return 0


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _run_evaluate_retrieval(args: argparse.Namespace) -> int:
    try:
        corpus = load_evidence_export(args.evidence_directory)
        evaluation_set = load_evaluation_set(args.evaluation_set)
        report = evaluate_bm25(
            corpus,
            evaluation_set,
            split=args.split,
            cutoffs=tuple(args.cutoffs),
        )
        protected_paths = {
            args.evaluation_set.resolve(),
            (args.evidence_directory / "evidence.jsonl").resolve(),
            (args.evidence_directory / "manifest.json").resolve(),
        }
        if args.output.resolve() in protected_paths:
            raise EvaluationError(
                "Evaluation output must be separate from judgments and selected-evidence inputs"
            )
        output_path = write_evaluation_report(
            report,
            args.output,
            overwrite=args.overwrite,
        )
    except (RetrievalError, EvaluationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Dataset: {evaluation_set.dataset_id}")
    print(f"Split: {args.split}")
    print(f"Questions in split: {report.payload['counts']['questions_in_split']}")
    print(f"Verified questions evaluated: {report.verified_question_count}")
    print(
        f"Provisional questions excluded from official metrics: {report.provisional_question_count}"
    )
    aggregate = report.payload["aggregate_metrics"]
    if aggregate is None:
        print("Official metrics: unavailable; this split has no verified questions")
    else:
        for cutoff in report.payload["cutoffs"]:
            key = str(cutoff)
            print(f"Hit@{cutoff}: {_format_metric(aggregate['hit_at_k'][key])}")
            print(f"Recall@{cutoff}: {_format_metric(aggregate['recall_at_k'][key])}")
        print(f"MRR: {_format_metric(aggregate['mrr'])}")
    print(f"Report: {output_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LedgerLens command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
