"""CLI adapter for explicit multi-document financial answering."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from ledgerlens.generation import LocalModelBackend
from ledgerlens.generation.ollama import OllamaBackend
from ledgerlens.multi_document_answer_generation import MultiDocumentAnswerStatus
from ledgerlens.multi_document_evaluation import (
    MultiDocumentEvaluationError,
    load_multi_document_evaluation_set,
    run_multi_document_evaluation,
    write_multi_document_evaluation,
)
from ledgerlens.multi_document_workflow import (
    MultiDocumentWorkflowError,
    MultiDocumentWorkflowResult,
    load_workflow_request,
    run_multi_document_workflow,
)

MULTI_ANSWER_ABSTAINED_EXIT_CODE = 3


def add_answer_multi_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the explicit request-file based multi-document answer command."""
    parser = subparsers.add_parser(
        "answer-multi",
        help="Answer from explicitly selected facts in separate reviewed annual reports.",
    )
    parser.add_argument(
        "request",
        type=Path,
        help="Versioned JSON request with exact sources, reconciliation IDs, facts, and roles.",
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
        help="Emit the complete machine-readable workflow result.",
    )
    parser.set_defaults(handler=run_answer_multi)


def add_evaluate_answer_multi_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the explicit reference-case multi-document evaluation command."""
    parser = subparsers.add_parser(
        "evaluate-answer-multi",
        help="Evaluate explicit multi-document cases against reviewed reference labels.",
    )
    parser.add_argument("evaluation_set", type=Path, help="Versioned JSON reference-case set.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/evaluation"),
        help="Protected parent for fingerprinted outputs (default: artifacts/evaluation).",
    )
    parser.add_argument(
        "--ollama",
        required=True,
        action="store_true",
        help="Explicitly allow local qwen3:4b-instruct inference through Ollama.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace this fingerprint."
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the complete evaluation payload."
    )
    parser.set_defaults(handler=run_evaluate_answer_multi)


def _build_backend() -> LocalModelBackend:
    return OllamaBackend()


def run_answer_multi(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[[], LocalModelBackend] | None = None,
) -> int:
    """Load the explicit request and invoke the trusted multi-document pipeline."""
    try:
        request = load_workflow_request(args.request)
        result = run_multi_document_workflow(request, (backend_factory or _build_backend)())
    except MultiDocumentWorkflowError as exc:
        _render_input_error(str(exc), as_json=args.json)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        _render_human(result)
    if result.answer and result.answer.status is MultiDocumentAnswerStatus.ANSWERED:
        return 0
    return MULTI_ANSWER_ABSTAINED_EXIT_CODE


def run_evaluate_answer_multi(
    args: argparse.Namespace,
    *,
    backend_factory: Callable[[], LocalModelBackend] | None = None,
) -> int:
    """Run and publish explicit reference cases through the same trusted workflow."""
    try:
        evaluation_set = load_multi_document_evaluation_set(args.evaluation_set)
        payload = run_multi_document_evaluation(
            evaluation_set, (backend_factory or _build_backend)()
        )
        destination = write_multi_document_evaluation(
            payload, args.output_root, overwrite=args.overwrite
        )
    except (MultiDocumentEvaluationError, MultiDocumentWorkflowError) as exc:
        _render_input_error(str(exc), as_json=args.json)
        return 2
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Evaluation: {payload['evaluation']}")
        print(f"Configuration: {payload['configuration_id']}")
        print(f"Output: {destination}")
        print("Summary: " + json.dumps(payload["summary"], sort_keys=True))
    summary = payload["summary"]
    assert isinstance(summary, dict)
    return 0 if summary.get("reference_checks_passed") is True else MULTI_ANSWER_ABSTAINED_EXIT_CODE


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


def _print_items(label: str, values: tuple[str, ...]) -> None:
    print(f"{label}: {'; '.join(values) if values else 'none'}")


def _render_human(result: MultiDocumentWorkflowResult) -> None:
    print(f"Status: {result.status}")
    print(f"Question: {result.request.question}")
    print(f"Task: {result.request.task.value}")
    print(f"Company: {result.request.company}")
    print("Reconciliation sources:")
    for source in result.reconciliation.sources:
        print(
            f"- document={source.document_id}; facts={source.path}; "
            f"facts_sha256={source.facts_sha256}; source_sha256={source.source_sha256}"
        )
        print(
            f"  Evidence hashes: jsonl={source.evidence_jsonl_sha256}; "
            f"manifest={source.evidence_manifest_sha256}"
        )
    print("Source-selection audit:")
    if result.orchestration.selection is None:
        print("- Selection was not completed.")
    else:
        for decision in result.orchestration.selection.decisions:
            selected = decision.operand.fact.fact_id if decision.operand else "none"
            reason = decision.abstention_reason.value if decision.abstention_reason else "none"
            print(
                f"- {decision.request.role}: status={decision.status.value}; "
                f"reconciliation_result={decision.request.reconciliation_result_id}; "
                f"selected_fact={selected}; abstention={reason}"
            )
            if decision.detail:
                print(f"  Detail: {decision.detail}")
    if result.answer is None:
        reason = result.orchestration.abstention_reason
        print(f"Abstention reason: {reason.value if reason else 'unspecified'}")
        if result.orchestration.detail:
            print(f"Detail: {result.orchestration.detail}")
        return
    answer = result.answer
    if answer.abstention_reason:
        print(f"Abstention reason: {answer.abstention_reason}")
    if answer.reported_values:
        print("Source-reported values:")
        for value in answer.reported_values:
            print(f"- {value.render()}")
    if answer.computed_value and result.presentation:
        print("Computed result:")
        print(
            f"- {result.presentation.display_value} "
            f"(exact Decimal: {result.presentation.exact_value})"
        )
        print(f"- Formula: {answer.computed_value.formula}")
        print(f"- Calculation ID: {answer.computed_value.calculation_id}")
    if answer.model_generated_explanation is not None:
        print("Model-generated explanation:")
        print(f"- {answer.model_generated_explanation}")
    elif answer.status is MultiDocumentAnswerStatus.MODEL_OUTPUT_REJECTED:
        print("Model-generated explanation: rejected by grounding policy 1.1")
    elif answer.status is MultiDocumentAnswerStatus.INFERENCE_FAILED:
        print("Model-generated explanation: unavailable because inference failed")
    print("Annual-report citations:")
    for citation in answer.citations:
        report = citation.annual_report
        print(
            f"- {citation.marker} role={citation.role}; fact={citation.fact_id}; "
            f"source_sha256={report.source_sha256}"
        )
        print(
            "  Evidence hashes: "
            f"support={citation.supporting_text_sha256}; "
            f"selected_text={citation.selected_text_sha256}; "
            f"jsonl={citation.evidence_jsonl_sha256}; "
            f"manifest={citation.evidence_manifest_sha256}"
        )
        _print_items("  Screening warnings", citation.screening_warnings)
        _print_items("  Evidence limitations", citation.evidence_limitations)
        _print_items("  Fact warnings", citation.fact_warnings)
        _print_items("  Fact limitations", citation.fact_limitations)
    print(
        "Generation provenance: "
        + json.dumps(answer.provenance.to_dict(), ensure_ascii=False, sort_keys=True)
    )


__all__ = [
    "MULTI_ANSWER_ABSTAINED_EXIT_CODE",
    "add_answer_multi_parser",
    "add_evaluate_answer_multi_parser",
    "run_answer_multi",
    "run_evaluate_answer_multi",
]
