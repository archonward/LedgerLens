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
    evaluate_chunk_semantic,
    evaluate_hybrid,
    evaluate_semantic,
    load_evaluation_set,
    write_evaluation_report,
)
from ledgerlens.evidence import prepare_evidence
from ledgerlens.ingestion import IngestionError, ingest_pdf, recover_page
from ledgerlens.retrieval import (
    BM25Retriever,
    ChunkingConfig,
    ChunkSemanticRetriever,
    HybridRetriever,
    RetrievalError,
    SemanticRetriever,
    build_chunk_embedding_index,
    build_embedding_index,
    create_fastembed_backend,
    load_chunk_embedding_index,
    load_embedding_index,
    load_evidence_export,
)


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

    embed_parser = subparsers.add_parser(
        "embed-evidence",
        help="Build a reusable CPU semantic-embedding cache for selected evidence.",
    )
    embed_parser.add_argument(
        "evidence_directory",
        type=Path,
        help="Directory containing selected evidence.jsonl and manifest.json.",
    )
    embed_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/embeddings"),
        help="Embedding-index parent directory (default: artifacts/embeddings).",
    )
    embed_parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("artifacts/models"),
        help="Ignored model-download cache (default: artifacts/models).",
    )
    embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of pages embedded per inference batch (default: 32).",
    )
    embed_parser.add_argument(
        "--threads",
        type=int,
        help="Optional ONNX Runtime CPU thread count.",
    )
    embed_parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the pinned model to be present in the local model cache.",
    )
    embed_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cache for the same evidence and model.",
    )
    embed_parser.set_defaults(handler=_run_embed_evidence)

    semantic_parser = subparsers.add_parser(
        "semantic-retrieve",
        help="Rank selected evidence pages using a validated embedding cache.",
    )
    semantic_parser.add_argument(
        "evidence_directory",
        type=Path,
        help="Directory containing selected evidence.jsonl and manifest.json.",
    )
    semantic_parser.add_argument(
        "embedding_index",
        type=Path,
        help="Directory produced by embed-evidence.",
    )
    semantic_parser.add_argument("--query", required=True, help="Natural-language search query.")
    semantic_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of pages to return (default: 5).",
    )
    semantic_parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("artifacts/models"),
        help="Existing ignored model cache (default: artifacts/models).",
    )
    semantic_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable view.",
    )
    semantic_parser.set_defaults(handler=_run_semantic_retrieve)

    hybrid_parser = subparsers.add_parser(
        "hybrid-retrieve",
        help="Fuse BM25 and page-semantic rankings with reciprocal rank fusion.",
    )
    hybrid_parser.add_argument(
        "evidence_directory",
        type=Path,
        help="Directory containing selected evidence.jsonl and manifest.json.",
    )
    hybrid_parser.add_argument(
        "embedding_index",
        type=Path,
        help="Existing page-embedding index produced by embed-evidence.",
    )
    hybrid_parser.add_argument("--query", required=True, help="Natural-language search query.")
    hybrid_parser.add_argument(
        "--top-k", type=int, default=5, help="Maximum physical pages to return (default: 5)."
    )
    hybrid_parser.add_argument(
        "--rrf-k", type=int, default=60, help="RRF rank constant (default: 60)."
    )
    hybrid_parser.add_argument(
        "--bm25-weight", type=float, default=1.0, help="Positive BM25 rank weight (default: 1)."
    )
    hybrid_parser.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
        help="Positive page-semantic rank weight (default: 1).",
    )
    hybrid_parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("artifacts/models"),
        help="Existing ignored model cache (default: artifacts/models).",
    )
    hybrid_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the human-readable view.",
    )
    hybrid_parser.set_defaults(handler=_run_hybrid_retrieve)

    chunk_embed_parser = subparsers.add_parser(
        "embed-evidence-chunks",
        help="Build token-aware chunk embeddings within selected physical pages.",
    )
    chunk_embed_parser.add_argument(
        "evidence_directory",
        type=Path,
        help="Directory containing selected evidence.jsonl and manifest.json.",
    )
    chunk_embed_parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/chunk-embeddings"),
        help="Chunk-index parent directory (default: artifacts/chunk-embeddings).",
    )
    chunk_embed_parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("artifacts/models"),
        help="Ignored model-download cache (default: artifacts/models).",
    )
    chunk_embed_parser.add_argument(
        "--max-tokens",
        type=int,
        default=384,
        help="Maximum model tokens per chunk (default: 384).",
    )
    chunk_embed_parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=64,
        help="Token overlap between consecutive chunks (default: 64).",
    )
    chunk_embed_parser.add_argument(
        "--boundary-window-tokens",
        type=int,
        default=64,
        help="Trailing token window searched for paragraph or line breaks (default: 64).",
    )
    chunk_embed_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks embedded per inference batch (default: 32).",
    )
    chunk_embed_parser.add_argument("--threads", type=int, help="Optional ONNX CPU thread count.")
    chunk_embed_parser.add_argument(
        "--offline",
        action="store_true",
        help="Require the pinned model to be present in the local model cache.",
    )
    chunk_embed_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a cache with identical evidence, model, and chunk settings.",
    )
    chunk_embed_parser.set_defaults(handler=_run_embed_evidence_chunks)

    chunk_retrieve_parser = subparsers.add_parser(
        "chunk-semantic-retrieve",
        help="Rank physical pages by aggregated semantic chunk scores.",
    )
    chunk_retrieve_parser.add_argument(
        "evidence_directory",
        type=Path,
        help="Directory containing selected evidence.jsonl and manifest.json.",
    )
    chunk_retrieve_parser.add_argument(
        "embedding_index", type=Path, help="Existing chunk-embedding index directory."
    )
    chunk_retrieve_parser.add_argument("--query", required=True, help="Natural-language query.")
    chunk_retrieve_parser.add_argument(
        "--top-k", type=int, default=5, help="Maximum physical pages to return (default: 5)."
    )
    chunk_retrieve_parser.add_argument(
        "--aggregation", choices=("max", "mean_top_n"), default="max"
    )
    chunk_retrieve_parser.add_argument("--aggregation-top-n", type=int, default=2)
    chunk_retrieve_parser.add_argument("--model-cache", type=Path, default=Path("artifacts/models"))
    chunk_retrieve_parser.add_argument("--json", action="store_true")
    chunk_retrieve_parser.set_defaults(handler=_run_chunk_semantic_retrieve)

    evaluation_parser = subparsers.add_parser(
        "evaluate-retrieval",
        help="Evaluate BM25, semantic, or hybrid retrieval against independent judgments.",
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
        "--retriever",
        choices=("bm25", "semantic", "chunk-semantic", "hybrid"),
        default="bm25",
        help="Retrieval method to evaluate (default: bm25).",
    )
    evaluation_parser.add_argument(
        "--embedding-index",
        type=Path,
        help="Required page/chunk index for semantic, chunk-semantic, or hybrid retrieval.",
    )
    evaluation_parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path("artifacts/models"),
        help="Existing ignored model cache for semantic evaluation.",
    )
    evaluation_parser.add_argument(
        "--chunk-aggregation",
        choices=("max", "mean_top_n"),
        default="max",
        help="Page aggregation for chunk-semantic evaluation (default: max).",
    )
    evaluation_parser.add_argument(
        "--aggregation-top-n",
        type=int,
        default=2,
        help="Number of chunk scores averaged by mean_top_n (default: 2).",
    )
    evaluation_parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF rank constant for hybrid evaluation (default: 60).",
    )
    evaluation_parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="Positive BM25 rank weight for hybrid evaluation (default: 1).",
    )
    evaluation_parser.add_argument(
        "--semantic-weight",
        type=float,
        default=1.0,
        help="Positive page-semantic rank weight for hybrid evaluation (default: 1).",
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


def _run_embed_evidence(args: argparse.Namespace) -> int:
    try:
        if args.threads is not None and args.threads < 1:
            raise RetrievalError("ONNX Runtime thread count must be a positive integer")
        corpus = load_evidence_export(args.evidence_directory)
        backend = create_fastembed_backend(
            args.model_cache,
            allow_download=not args.offline,
            threads=args.threads,
        )
        result = build_embedding_index(
            corpus,
            backend,
            output_root=args.output_root,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Document ID: {result.document_id}")
    print(f"Model: {result.model.model_name}")
    print(f"Model revision: {result.model.model_revision}")
    print(f"Backend: {result.model.backend} {result.model.backend_version}")
    print(f"Physical pages embedded: {result.page_count}")
    print(f"Embedding dimension: {result.dimension}")
    print(f"Output: {result.output_directory}")
    print("Page text beyond the model input limit is truncated during embedding.")
    return 0


def _run_semantic_retrieve(args: argparse.Namespace) -> int:
    try:
        corpus = load_evidence_export(args.evidence_directory)
        index = load_embedding_index(args.embedding_index, corpus)
        backend = create_fastembed_backend(args.model_cache, allow_download=False)
        retriever = SemanticRetriever(corpus, index, backend)
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
                    "retriever": "page_level_semantic",
                    "model": index.manifest["model"],
                    "embedding_index": {
                        "manifest_sha256": index.manifest_sha256,
                        "vectors_jsonl_sha256": index.vectors_jsonl_sha256,
                    },
                    "result_count": len(results),
                    "results": [result.to_dict() for result in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"Document ID: {corpus.document_id}")
    print(f"Query: {args.query}")
    print(f"Model: {index.model.model_name} @ {index.model.model_revision}")
    print(f"Cosine-ranked pages returned: {len(results)}")
    print("Scores rank semantic similarity; they are not confidence or accuracy measures.")
    for result in results:
        page = result.page
        print()
        print(
            f"{result.rank}. Physical PDF page {page.physical_page_number} "
            f"| score {result.score:.6f} | {page.method}"
        )
        print(f"   Selected-text SHA-256: {page.text_sha256}")
        print(f"   Source SHA-256: {page.provenance.get('source_sha256', 'unavailable')}")
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


def _run_hybrid_retrieve(args: argparse.Namespace) -> int:
    try:
        corpus = load_evidence_export(args.evidence_directory)
        index = load_embedding_index(args.embedding_index, corpus)
        backend = create_fastembed_backend(args.model_cache, allow_download=False)
        retriever = HybridRetriever(
            corpus,
            index,
            backend,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            semantic_weight=args.semantic_weight,
        )
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
                    "retriever": "hybrid_rrf",
                    "fusion": {
                        "method": "weighted_reciprocal_rank_fusion",
                        "rrf_k": retriever.rrf_k,
                        "bm25_weight": retriever.bm25_weight,
                        "semantic_weight": retriever.semantic_weight,
                        "tie_breaker": "ascending_physical_page_number",
                    },
                    "embedding_index": {
                        "model": index.manifest["model"],
                        "manifest_sha256": index.manifest_sha256,
                        "vectors_jsonl_sha256": index.vectors_jsonl_sha256,
                    },
                    "result_count": len(results),
                    "results": [result.to_dict() for result in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"Document ID: {corpus.document_id}")
    print(f"Query: {args.query}")
    print(f"Model: {index.model.model_name} @ {index.model.model_revision}")
    print(
        f"Fusion: RRF k={retriever.rrf_k}, BM25 weight={retriever.bm25_weight:g}, "
        f"semantic weight={retriever.semantic_weight:g}"
    )
    print("RRF scores combine ranks; they are not confidence or accuracy measures.")
    for result in results:
        page = result.page
        bm25_rank = result.bm25.rank if result.bm25.rank is not None else "not ranked"
        semantic_rank = result.semantic.rank if result.semantic.rank is not None else "not ranked"
        print()
        print(
            f"{result.rank}. Physical PDF page {page.physical_page_number} "
            f"| fused score {result.score:.8f} | {page.method}"
        )
        print(
            f"   BM25: rank {bm25_rank}, raw score {result.bm25.raw_score}, "
            f"RRF contribution {result.bm25.contribution:.8f}"
        )
        print(
            f"   Page semantic: rank {semantic_rank}, raw score {result.semantic.raw_score}, "
            f"RRF contribution {result.semantic.contribution:.8f}"
        )
        print(f"   Excerpt source: {result.excerpt_source}")
        print(f"   Selected-text SHA-256: {page.text_sha256}")
        print(f"   Source SHA-256: {page.provenance.get('source_sha256', 'unavailable')}")
        if page.review_provenance is not None:
            print(
                "   Review provenance: "
                + json.dumps(page.review_provenance, ensure_ascii=False, sort_keys=True)
            )
        warnings = page.warnings
        print(f"   Warnings: {'; '.join(warnings) if warnings else 'none recorded'}")
        limitations = "; ".join(page.known_limitations) or "none recorded"
        print(f"   Limitations: {limitations}")
        print("   Exact source excerpt:")
        for line in result.excerpt.splitlines() or [""]:
            print(textwrap.indent(line, "     "))
    return 0


def _run_embed_evidence_chunks(args: argparse.Namespace) -> int:
    try:
        if args.threads is not None and args.threads < 1:
            raise RetrievalError("ONNX Runtime thread count must be a positive integer")
        corpus = load_evidence_export(args.evidence_directory)
        backend = create_fastembed_backend(
            args.model_cache, allow_download=not args.offline, threads=args.threads
        )
        config = ChunkingConfig(
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            boundary_window_tokens=args.boundary_window_tokens,
        )
        result = build_chunk_embedding_index(
            corpus,
            backend,
            output_root=args.output_root,
            config=config,
            batch_size=args.batch_size,
            overwrite=args.overwrite,
        )
    except RetrievalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Document ID: {result.document_id}")
    print(f"Model: {result.model.model_name} @ {result.model.model_revision}")
    print(f"Physical pages represented: {result.page_count}")
    print(f"Chunks embedded: {result.chunk_count}")
    print(
        f"Chunking: max_tokens={result.chunking.max_tokens}, "
        f"overlap_tokens={result.chunking.overlap_tokens}, "
        f"boundary_window_tokens={result.chunking.boundary_window_tokens}"
    )
    print(f"Output: {result.output_directory}")
    return 0


def _run_chunk_semantic_retrieve(args: argparse.Namespace) -> int:
    try:
        corpus = load_evidence_export(args.evidence_directory)
        index = load_chunk_embedding_index(args.embedding_index, corpus)
        backend = create_fastembed_backend(args.model_cache, allow_download=False)
        retriever = ChunkSemanticRetriever(
            corpus,
            index,
            backend,
            aggregation=args.aggregation,
            aggregation_top_n=args.aggregation_top_n,
        )
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
                    "retriever": "chunk_aware_page_semantic",
                    "model": index.manifest["model"],
                    "chunking": index.manifest["chunking"],
                    "aggregation": args.aggregation,
                    "aggregation_top_n": args.aggregation_top_n,
                    "embedding_index": {
                        "manifest_sha256": index.manifest_sha256,
                        "chunks_jsonl_sha256": index.chunks_jsonl_sha256,
                    },
                    "result_count": len(results),
                    "results": [result.to_dict() for result in results],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    print(f"Document ID: {corpus.document_id}")
    print(f"Query: {args.query}")
    print(f"Model: {index.model.model_name} @ {index.model.model_revision}")
    print(
        f"Chunking: max_tokens={index.chunking.max_tokens}, "
        f"overlap_tokens={index.chunking.overlap_tokens}"
    )
    print(f"Page aggregation: {args.aggregation}")
    print("Scores rank semantic similarity; they are not confidence or accuracy measures.")
    for result in results:
        page = result.page
        chunk = result.best_chunk
        print()
        print(
            f"{result.rank}. Physical PDF page {page.physical_page_number} "
            f"| page score {result.score:.6f} | {page.method}"
        )
        print(
            f"   Best chunk: {chunk.chunk_id} | score {result.best_chunk_score:.6f} "
            f"| chars [{chunk.char_start}, {chunk.char_end}) "
            f"| tokens [{chunk.token_start}, {chunk.token_end})"
        )
        print(f"   Chunk-text SHA-256: {chunk.text_sha256}")
        print(f"   Selected-text SHA-256: {page.text_sha256}")
        if page.review_provenance is not None:
            print(
                "   Review provenance: "
                + json.dumps(page.review_provenance, ensure_ascii=False, sort_keys=True)
            )
        warnings = page.warnings
        print(f"   Warnings: {'; '.join(warnings) if warnings else 'none recorded'}")
        limitations = "; ".join(page.known_limitations) or "none recorded"
        print(f"   Limitations: {limitations}")
        print("   Exact source excerpt:")
        for line in result.excerpt.splitlines() or [""]:
            print(textwrap.indent(line, "     "))
    return 0


def _format_metric(value: float) -> str:
    return f"{value:.4f}"


def _run_evaluate_retrieval(args: argparse.Namespace) -> int:
    try:
        corpus = load_evidence_export(args.evidence_directory)
        evaluation_set = load_evaluation_set(args.evaluation_set)
        if args.retriever == "hybrid":
            if args.embedding_index is None:
                raise EvaluationError(
                    "--embedding-index is required when --retriever hybrid is selected"
                )
            index = load_embedding_index(args.embedding_index, corpus)
            backend = create_fastembed_backend(args.model_cache, allow_download=False)
            report = evaluate_hybrid(
                corpus,
                index,
                backend,
                evaluation_set,
                split=args.split,
                cutoffs=tuple(args.cutoffs),
                rrf_k=args.rrf_k,
                bm25_weight=args.bm25_weight,
                semantic_weight=args.semantic_weight,
            )
        elif args.retriever == "chunk-semantic":
            if args.embedding_index is None:
                raise EvaluationError(
                    "--embedding-index is required when --retriever chunk-semantic is selected"
                )
            index = load_chunk_embedding_index(args.embedding_index, corpus)
            backend = create_fastembed_backend(args.model_cache, allow_download=False)
            report = evaluate_chunk_semantic(
                corpus,
                index,
                backend,
                evaluation_set,
                split=args.split,
                cutoffs=tuple(args.cutoffs),
                aggregation=args.chunk_aggregation,
                aggregation_top_n=args.aggregation_top_n,
            )
        elif args.retriever == "semantic":
            if args.embedding_index is None:
                raise EvaluationError(
                    "--embedding-index is required when --retriever semantic is selected"
                )
            index = load_embedding_index(args.embedding_index, corpus)
            backend = create_fastembed_backend(args.model_cache, allow_download=False)
            report = evaluate_semantic(
                corpus,
                index,
                backend,
                evaluation_set,
                split=args.split,
                cutoffs=tuple(args.cutoffs),
            )
        else:
            if args.embedding_index is not None:
                raise EvaluationError(
                    "--embedding-index is only valid with --retriever semantic, "
                    "chunk-semantic, or hybrid"
                )
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
        if args.embedding_index is not None:
            protected_paths.update(
                {
                    (args.embedding_index / "manifest.json").resolve(),
                    (args.embedding_index / "vectors.jsonl").resolve(),
                    (args.embedding_index / "chunks.jsonl").resolve(),
                }
            )
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
    print(f"Retriever: {args.retriever}")
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
