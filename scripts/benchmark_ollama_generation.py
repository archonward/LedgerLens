"""Run the opt-in qwen3 Ollama benchmark over reviewed NVIDIA FY2026 inputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from demonstrate_nvidia_fy2026 import (
    DOCUMENT_ID,
    DemonstrationError,
    build_demonstration,
)

from ledgerlens.calculations import FactOperand, profit_margin, year_over_year_growth
from ledgerlens.financial_facts import Verification, VerificationStatus, load_financial_facts
from ledgerlens.generation import (
    AnswerGenerator,
    AnswerRequest,
    AnswerTask,
    InferenceConfig,
    RejectedModelOutput,
)
from ledgerlens.generation.benchmark import (
    AcceptanceCase,
    BenchmarkError,
    CaseKind,
    run_acceptance_benchmark,
    write_benchmark_artifacts,
    write_rejected_output_debug_file,
)
from ledgerlens.generation.ollama import OllamaBackend, OllamaBackendError
from ledgerlens.retrieval.bm25 import RetrievalError, load_evidence_export

DEFAULT_OUTPUT = Path("artifacts/evaluation/ollama-generation-qwen3-4b-instruct")


def _build_parser() -> argparse.ArgumentParser:
    artifact_root = Path("artifacts")
    evidence = artifact_root / "evidence" / DOCUMENT_ID
    facts = artifact_root / "financial-facts" / DOCUMENT_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="required acknowledgement that this command will execute the installed local model",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--source-pdf", type=Path, default=Path("data/raw/nvidia-2026-annual-report.pdf")
    )
    parser.add_argument("--evidence-directory", type=Path, default=evidence)
    parser.add_argument("--facts", type=Path, default=facts / "facts.jsonl")
    parser.add_argument("--review-record", type=Path, default=facts / "review.json")
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--debug-rejected-output",
        type=Path,
        help=(
            "explicitly retain raw rejected model text in this separate local file; the path "
            "must be beneath ignored artifacts/"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _cases(corpus, facts_by_id, repeats: int) -> tuple[AcceptanceCase, ...]:
    revenue = facts_by_id["nvidia-fy2026-revenue"]
    prior_revenue = facts_by_id["nvidia-fy2025-revenue"]
    net_income = facts_by_id["nvidia-fy2026-net-income"]
    growth = year_over_year_growth(FactOperand(revenue, corpus), FactOperand(prior_revenue, corpus))
    margin = profit_margin(FactOperand(net_income, corpus), FactOperand(revenue, corpus))
    growth_request = AnswerRequest(
        "How much did NVIDIA revenue grow in FY2026?",
        AnswerTask.REVENUE_GROWTH,
        (revenue, prior_revenue),
        growth,
        corpus,
    )
    margin_request = AnswerRequest(
        "What was NVIDIA's FY2026 net profit margin?",
        AnswerTask.NET_PROFIT_MARGIN,
        (net_income, revenue),
        margin,
        corpus,
    )
    unsupported = replace(growth_request, question="Why did operating expenses change?")
    mismatched = replace(
        growth_request,
        question="What was NVIDIA's net profit margin?",
        task=AnswerTask.NET_PROFIT_MARGIN,
    )
    unverified_revenue = replace(
        revenue,
        verification=Verification(
            VerificationStatus.UNVERIFIED,
            note="Controlled benchmark negative: verification intentionally removed.",
        ),
    )
    unverified = replace(growth_request, facts=(unverified_revenue, prior_revenue))
    return (
        AcceptanceCase("revenue-growth", CaseKind.POSITIVE, growth_request, repeats),
        AcceptanceCase("net-profit-margin", CaseKind.POSITIVE, margin_request, repeats),
        AcceptanceCase(
            "unsupported-question",
            CaseKind.NEGATIVE,
            unsupported,
            expected_abstention_reason=("question_is_not_supported_by_revenue_growth_evidence"),
        ),
        AcceptanceCase(
            "task-calculation-mismatch",
            CaseKind.NEGATIVE,
            mismatched,
            expected_abstention_reason="calculation_type_does_not_match_answer_task",
        ),
        AcceptanceCase(
            "unverified-input",
            CaseKind.NEGATIVE,
            unverified,
            expected_abstention_reason="financial_facts_are_not_verified",
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate local inputs, run the fixed benchmark, and publish ignored artifacts."""
    args = _build_parser().parse_args(argv)
    if not args.run_live:
        print("error: --run-live is required to execute the installed local model", file=sys.stderr)
        return 2
    try:
        if args.output_directory.exists() and not args.overwrite:
            raise BenchmarkError(
                f"benchmark output already exists: {args.output_directory}; "
                "choose a separate directory or use --overwrite"
            )
        debug_path = None
        if args.debug_rejected_output is not None:
            debug_path = args.debug_rejected_output.resolve()
            try:
                debug_path.relative_to(Path("artifacts").resolve())
            except ValueError as exc:
                raise BenchmarkError(
                    "--debug-rejected-output must be located beneath ignored artifacts/"
                ) from exc
            if debug_path.exists() and not args.overwrite:
                raise BenchmarkError(
                    f"debug output already exists: {debug_path}; use --overwrite to replace it"
                )
        demonstration = build_demonstration(
            args.source_pdf,
            args.evidence_directory,
            args.facts,
            args.review_record,
        )
        corpus = load_evidence_export(args.evidence_directory)
        facts = load_financial_facts(args.facts, corpus)
        facts_by_id = {fact.fact_id: fact for fact in facts}
        config = InferenceConfig(max_new_tokens=96, temperature=0.0)
        backend = OllamaBackend(timeout_seconds=args.timeout_seconds)
        model = backend.verify_installed_model()
        rejected_outputs: list[RejectedModelOutput] = []
        generator = AnswerGenerator(
            backend,
            inference_config=config,
            rejected_output_handler=(
                rejected_outputs.append if args.debug_rejected_output is not None else None
            ),
        )
        source = {
            "document_id": corpus.document_id,
            "source_sha256": corpus.source_sha256,
            "evidence_jsonl_sha256": corpus.evidence_jsonl_sha256,
            "evidence_manifest_sha256": corpus.manifest_sha256,
            "fact_ids": sorted(facts_by_id),
            "physical_pdf_pages": sorted({fact.evidence.physical_page_number for fact in facts}),
            "review": demonstration["review"],
        }
        payload = run_acceptance_benchmark(
            generator,
            _cases(corpus, facts_by_id, args.repeats),
            model=model,
            inference=config,
            source=source,
        )
        destination = write_benchmark_artifacts(
            payload, args.output_directory, overwrite=args.overwrite
        )
        debug_destination = None
        if debug_path is not None:
            debug_destination = write_rejected_output_debug_file(
                tuple(rejected_outputs),
                debug_path,
                model=model,
                inference=config,
                allow_raw_output=True,
                overwrite=args.overwrite,
            )
        summary = payload["summary"]
        print(
            json.dumps(
                {
                    "output_directory": str(destination),
                    "configuration_id": payload["configuration_id"],
                    "model": payload["model"],
                    "summary": summary,
                    "debug_rejected_output": (
                        str(debug_destination) if debug_destination is not None else None
                    ),
                },
                indent=2,
            )
        )
    except (
        BenchmarkError,
        DemonstrationError,
        KeyError,
        OllamaBackendError,
        OSError,
        RetrievalError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
