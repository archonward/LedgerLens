"""User-facing CLI adapter for deterministic financial answer orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date

from ledgerlens.financial_facts import FinancialFactValidationError, load_financial_facts
from ledgerlens.generation import AnswerGenerator, InferenceConfig, LocalModelBackend
from ledgerlens.generation.ollama import OllamaBackend
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationStatus,
)
from ledgerlens.retrieval import RetrievalError, load_evidence_export

ANSWER_ABSTAINED_EXIT_CODE = 3
_INFERENCE_CONFIG = InferenceConfig(max_new_tokens=96, temperature=0.0)


def add_answer_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the narrow answer command on the existing root parser."""
    parser = subparsers.add_parser(
        "answer",
        help="Answer one supported financial question from reviewed local facts.",
    )
    parser.add_argument("question", help="Explicit financial question to answer.")
    parser.add_argument("--company", required=True, help="Exact company name in the facts.")
    parser.add_argument(
        "--target-period-end",
        required=True,
        type=_iso_date,
        metavar="YYYY-MM-DD",
        help="Exact reporting-period end date; fiscal periods are never inferred.",
    )
    parser.add_argument(
        "--evidence-directory",
        required=True,
        help="Local selected-evidence directory containing manifest.json and evidence.jsonl.",
    )
    parser.add_argument(
        "--financial-facts",
        required=True,
        help="Local reviewed financial-facts JSONL path.",
    )
    parser.add_argument(
        "--ollama",
        required=True,
        action="store_true",
        help="Explicitly allow local qwen3:4b-instruct inference through Ollama.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the complete machine-readable orchestration result.",
    )
    parser.set_defaults(handler=run_answer)


def _iso_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be an ISO date in YYYY-MM-DD form") from None
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("must be an ISO date in YYYY-MM-DD form")
    return parsed


def _build_backend() -> LocalModelBackend:
    """Construct the only supported live backend after explicit CLI opt-in."""
    return OllamaBackend()


def run_answer(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[[], LocalModelBackend] | None = None,
) -> int:
    """Load validated local inputs and execute the existing orchestrator."""
    try:
        corpus = load_evidence_export(args.evidence_directory)
        facts = load_financial_facts(args.financial_facts, corpus)
    except (RetrievalError, FinancialFactValidationError) as exc:
        _render_input_error(str(exc), as_json=args.json)
        return 2

    backend = (backend_factory or _build_backend)()
    result = FinancialAnswerOrchestrator(
        AnswerGenerator(backend, inference_config=_INFERENCE_CONFIG)
    ).answer(
        OrchestrationRequest(
            question=args.question,
            company=args.company,
            target_period_end=args.target_period_end,
            facts=facts,
            corpus=corpus,
        )
    )
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _render_human(result)
    return 0 if result.status is OrchestrationStatus.ANSWERED else ANSWER_ABSTAINED_EXIT_CODE


def _render_input_error(detail: str, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {"status": "error", "error": {"code": "invalid_local_input", "detail": detail}},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"error: {detail}", file=sys.stderr)


def _render_human(result: OrchestrationResult) -> None:
    print(f"Status: {result.status.value}")
    print(f"Question: {result.question}")
    print(f"Task: {result.task.value if result.task else 'unresolved'}")
    if result.status is OrchestrationStatus.ABSTAINED:
        reason = result.abstention_reason.value if result.abstention_reason else "unspecified"
        print(f"Abstention reason: {reason}")
        if result.detail:
            print(f"Detail: {result.detail}")
    if result.selected_fact_ids:
        print(f"Selected facts: {', '.join(result.selected_fact_ids)}")

    print("Selection audit:")
    if not result.selection_audit.decisions:
        print("- No fact selection was attempted.")
    for decision in result.selection_audit.decisions:
        candidates = ", ".join(decision.candidate_fact_ids) or "none"
        print(
            f"- {decision.role}: {decision.outcome}; metric={decision.metric}; "
            f"candidates={candidates}; rule={decision.period_rule}"
        )

    if result.answer is None:
        return
    if result.status is OrchestrationStatus.ANSWERED:
        print("Answer:")
        print(result.answer.answer_text)
        print("Evidence citations:")
        for citation in result.answer.citations:
            print(
                f"- {citation.marker} document={citation.document_id}; "
                f"source_sha256={citation.source_sha256}"
            )
            print(f"  Supporting evidence: {citation.supporting_text}")
            print(
                "  Evidence hashes: "
                f"selected_text={citation.selected_text_sha256}; "
                f"facts_export={citation.evidence_jsonl_sha256}; "
                f"manifest={citation.evidence_manifest_sha256}"
            )
            if citation.review_provenance is not None:
                print(
                    "  Review provenance: "
                    + json.dumps(citation.review_provenance, ensure_ascii=False, sort_keys=True)
                )
            _print_items("  Warnings", citation.screening_warnings)
            _print_items("  Evidence limitations", citation.known_limitations)
        fact_limitations = tuple(
            dict.fromkeys(
                limitation
                for value in result.answer.reported_values
                for limitation in value.limitations
            )
        )
        _print_items("Fact limitations", fact_limitations)
    print(
        "Generation provenance: "
        + json.dumps(result.answer.provenance.to_dict(), ensure_ascii=False, sort_keys=True)
    )


def _print_items(label: str, values: tuple[str, ...]) -> None:
    if values:
        print(f"{label}: {'; '.join(values)}")
    else:
        print(f"{label}: none")


__all__ = ["ANSWER_ABSTAINED_EXIT_CODE", "add_answer_parser", "run_answer"]
