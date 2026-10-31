"""User-facing CLI adapter for deterministic financial answer orchestration."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

from ledgerlens.financial_facts import FinancialFactValidationError, load_financial_facts
from ledgerlens.generation import AnswerGenerator, InferenceConfig, LocalModelBackend
from ledgerlens.generation.ollama import OllamaBackend
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationStatus,
)
from ledgerlens.retrieval import (
    HybridRetriever,
    RetrievalError,
    create_fastembed_backend,
    load_embedding_index,
    load_evidence_export,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus
from ledgerlens.retrieval_facts import (
    RetrievalFinancialAnswerOrchestrator,
    RetrievalFinancialAnswerResult,
)

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
        "--hybrid-embedding-index",
        type=Path,
        help=(
            "Optional existing page-embedding index; when supplied, only reviewed facts "
            "associated with hybrid-retrieved evidence are eligible."
        ),
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=_positive_integer,
        default=5,
        help="Maximum hybrid-retrieved pages eligible for fact association (default: 5).",
    )
    parser.add_argument(
        "--retrieval-model-cache",
        type=Path,
        default=Path("artifacts/models"),
        help="Existing offline semantic model cache (default: artifacts/models).",
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


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _build_backend() -> LocalModelBackend:
    """Construct the only supported live backend after explicit CLI opt-in."""
    return OllamaBackend()


def _build_hybrid_retriever(args: argparse.Namespace, corpus: EvidenceCorpus) -> HybridRetriever:
    backend = create_fastembed_backend(
        args.retrieval_model_cache,
        allow_download=False,
    )
    index = load_embedding_index(args.hybrid_embedding_index, corpus)
    return HybridRetriever(corpus, index, backend)


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
    request = OrchestrationRequest(
        question=args.question,
        company=args.company,
        target_period_end=args.target_period_end,
        facts=facts,
        corpus=corpus,
    )
    orchestrator = FinancialAnswerOrchestrator(
        AnswerGenerator(backend, inference_config=_INFERENCE_CONFIG)
    )
    if args.hybrid_embedding_index is not None:
        try:
            retriever = _build_hybrid_retriever(args, corpus)
        except RetrievalError as exc:
            _render_input_error(str(exc), as_json=args.json)
            return 2
        result = RetrievalFinancialAnswerOrchestrator(orchestrator, retriever).answer(
            request, top_k=args.retrieval_top_k
        )
    else:
        result = orchestrator.answer(request)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    elif isinstance(result, RetrievalFinancialAnswerResult):
        _render_retrieval_human(result)
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


def _render_retrieval_human(result: RetrievalFinancialAnswerResult) -> None:
    print("Retrieval-to-fact association:")
    print(f"- Retrieved pages: {len(result.association.retrieved_pages)}")
    for page in result.association.retrieved_pages:
        facts = ", ".join(page.associated_fact_ids) or "none"
        print(
            f"- Rank {page.rank}: physical PDF page {page.physical_page_number}; "
            f"associated facts={facts}; selected_text_sha256={page.selected_text_sha256}"
        )
    print("Fact association decisions:")
    for decision in result.association.fact_decisions:
        rank = str(decision.retrieval_rank) if decision.retrieval_rank is not None else "none"
        print(
            f"- {decision.fact_id}: {decision.outcome.value}; "
            f"verification={decision.verification_status.value}; retrieval_rank={rank}"
        )
        if decision.detail:
            print(f"  Detail: {decision.detail}")
    if result.abstention_reason is not None:
        print(f"Retrieval abstention reason: {result.abstention_reason.value}")
        if result.detail:
            print(f"Retrieval detail: {result.detail}")
    if result.orchestration is not None:
        _render_human(result.orchestration)
    else:
        print(f"Status: {result.status.value}")
        print(f"Question: {result.question}")


def _print_items(label: str, values: tuple[str, ...]) -> None:
    if values:
        print(f"{label}: {'; '.join(values)}")
    else:
        print(f"{label}: none")


__all__ = ["ANSWER_ABSTAINED_EXIT_CODE", "add_answer_parser", "run_answer"]
