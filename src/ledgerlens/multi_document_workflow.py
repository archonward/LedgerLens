"""Reusable end-to-end multi-document workflow and deterministic presentation policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path

from ledgerlens.financial_facts import FinancialFactValidationError
from ledgerlens.generation import AnswerTask, LocalModelBackend
from ledgerlens.multi_document_answer_generation import (
    MultiDocumentAnswerGenerator,
    MultiDocumentGeneratedAnswer,
)
from ledgerlens.multi_document_generation import (
    MultiDocumentGenerationRequest,
    SourceFactBinding,
)
from ledgerlens.multi_document_orchestration import (
    MultiDocumentFinancialAnswerOrchestrator,
    MultiDocumentOrchestrationRequest,
    MultiDocumentOrchestrationResult,
    MultiDocumentOrchestrationStatus,
)
from ledgerlens.reconciliation import (
    ReconciliationError,
    ReconciliationReport,
    VerifiedFactSource,
    reconcile_financial_facts,
)
from ledgerlens.reconciliation_selection import (
    FactSelectionRequest,
    ReconciliationSelectionPolicy,
    SingleSourcePolicy,
)
from ledgerlens.retrieval import RetrievalError, load_evidence_export
from ledgerlens.retrieval.bm25 import EvidenceCorpus

MULTI_DOCUMENT_REQUEST_SCHEMA_VERSION = "1.0"
PERCENTAGE_PRESENTATION_POLICY_VERSION = "1.0"
PERCENTAGE_DECIMAL_PLACES = 2
_PERCENTAGE_QUANTUM = Decimal("0.01")


class MultiDocumentWorkflowError(ValueError):
    """A request definition or local source artifact is unsafe or invalid."""


@dataclass(frozen=True)
class MultiDocumentSourceInput:
    """One exact evidence/fact source declared by the caller."""

    document_id: str
    source_sha256: str
    evidence_directory: Path
    financial_facts: tuple[Path, ...]

    def to_dict(self) -> dict[str, str]:
        return {
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "evidence_directory": self.evidence_directory.as_posix(),
            "financial_facts": [path.as_posix() for path in self.financial_facts],
        }


@dataclass(frozen=True)
class MultiDocumentWorkflowRequest:
    """Versioned explicit request used by the CLI and evaluation runner."""

    question: str
    task: AnswerTask
    company: str
    sources: tuple[MultiDocumentSourceInput, ...]
    selections: tuple[FactSelectionRequest, ...]
    selection_policy: ReconciliationSelectionPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MULTI_DOCUMENT_REQUEST_SCHEMA_VERSION,
            "question": self.question,
            "task": self.task.value,
            "company": self.company,
            "sources": [source.to_dict() for source in self.sources],
            "selections": [selection.to_dict() for selection in self.selections],
            "selection_policy": {"single_source": self.selection_policy.single_source.value},
        }


@dataclass(frozen=True)
class PercentagePresentation:
    """Human display derived from, but never replacing, an exact Decimal result."""

    exact_value: str
    display_value: str
    policy_version: str = PERCENTAGE_PRESENTATION_POLICY_VERSION
    decimal_places: int = PERCENTAGE_DECIMAL_PLACES
    rounding: str = ROUND_HALF_EVEN

    def to_dict(self) -> dict[str, object]:
        return {
            "exact_value": self.exact_value,
            "display_value": self.display_value,
            "policy_version": self.policy_version,
            "decimal_places": self.decimal_places,
            "rounding": self.rounding,
            "source_of_truth": "calculation.computed_value",
        }


@dataclass(frozen=True)
class MultiDocumentWorkflowResult:
    """Complete reconciliation, selection, calculation, validation, and generation result."""

    request: MultiDocumentWorkflowRequest
    reconciliation: ReconciliationReport
    orchestration: MultiDocumentOrchestrationResult
    answer: MultiDocumentGeneratedAnswer | None

    @property
    def status(self) -> str:
        return self.answer.status.value if self.answer else "validation_abstained"

    @property
    def presentation(self) -> PercentagePresentation | None:
        if self.answer is None or self.answer.computed_value is None:
            return None
        calculation = self.answer.validation.reproduced_calculation
        if calculation is None:
            return None
        return present_percentage(calculation.computed_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "status": self.status,
            "request": self.request.to_dict(),
            "reconciliation": self.reconciliation.to_dict(),
            "orchestration": self.orchestration.to_dict(),
            "answer": self.answer.to_dict() if self.answer else None,
            "presentation": self.presentation.to_dict() if self.presentation else None,
        }


def present_percentage(value: Decimal) -> PercentagePresentation:
    """Round only the human display to two places using Decimal half-even."""
    if not isinstance(value, Decimal) or not value.is_finite():
        raise MultiDocumentWorkflowError("percentage presentation requires a finite Decimal")
    try:
        rounded = value.quantize(_PERCENTAGE_QUANTUM, rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise MultiDocumentWorkflowError("percentage could not be presented safely") from exc
    if rounded == 0:
        rounded = abs(rounded)
    return PercentagePresentation(str(value), f"{rounded:.2f}%")


def _require_object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MultiDocumentWorkflowError(f"{context} must be an object")
    return value


def _require_text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultiDocumentWorkflowError(f"{context} must be a non-empty string")
    return value


def _require_sha256(value: object, context: str) -> str:
    digest = _require_text(value, context)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise MultiDocumentWorkflowError(f"{context} must be a lowercase SHA-256 digest")
    return digest


def _resolve_path(value: object, context: str, base_directory: Path) -> Path:
    raw = _require_text(value, context)
    path = Path(raw)
    return path if path.is_absolute() else base_directory / path


def _resolve_fact_paths(value: object, context: str, base_directory: Path) -> tuple[Path, ...]:
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not values:
        raise MultiDocumentWorkflowError(f"{context} must be a path or non-empty path array")
    paths = tuple(
        _resolve_path(item, f"{context}[{index}]", base_directory)
        for index, item in enumerate(values)
    )
    if len({path.resolve() for path in paths}) != len(paths):
        raise MultiDocumentWorkflowError(f"{context} contains duplicate paths")
    return paths


def workflow_request_from_dict(
    value: object,
    *,
    base_directory: str | Path = ".",
) -> MultiDocumentWorkflowRequest:
    """Parse one strict JSON-compatible workflow request without inferring selections."""
    record = _require_object(value, "multi-document request")
    if record.get("schema_version") != MULTI_DOCUMENT_REQUEST_SCHEMA_VERSION:
        raise MultiDocumentWorkflowError("unsupported multi-document request schema_version")
    try:
        task = AnswerTask(_require_text(record.get("task"), "task"))
    except ValueError as exc:
        raise MultiDocumentWorkflowError("task is not supported") from exc
    source_values = record.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise MultiDocumentWorkflowError("sources must be a non-empty array")
    base = Path(base_directory)
    sources: list[MultiDocumentSourceInput] = []
    for index, source_value in enumerate(source_values):
        source = _require_object(source_value, f"sources[{index}]")
        sources.append(
            MultiDocumentSourceInput(
                _require_text(source.get("document_id"), f"sources[{index}].document_id"),
                _require_sha256(source.get("source_sha256"), f"sources[{index}].source_sha256"),
                _resolve_path(
                    source.get("evidence_directory"),
                    f"sources[{index}].evidence_directory",
                    base,
                ),
                _resolve_fact_paths(
                    source.get("financial_facts"),
                    f"sources[{index}].financial_facts",
                    base,
                ),
            )
        )
    if len({source.document_id for source in sources}) != len(sources):
        raise MultiDocumentWorkflowError("source document IDs must be unique")
    if len({source.evidence_directory.resolve() for source in sources}) != len(sources):
        raise MultiDocumentWorkflowError("evidence directories must be unique")
    fact_paths = [path.resolve() for source in sources for path in source.financial_facts]
    if len(set(fact_paths)) != len(fact_paths):
        raise MultiDocumentWorkflowError("financial fact paths must be unique")

    selection_values = record.get("selections")
    if not isinstance(selection_values, list) or not selection_values:
        raise MultiDocumentWorkflowError("selections must be a non-empty array")
    selections: list[FactSelectionRequest] = []
    for index, selection_value in enumerate(selection_values):
        item = _require_object(selection_value, f"selections[{index}]")
        selections.append(
            FactSelectionRequest(
                _require_text(item.get("role"), f"selections[{index}].role"),
                _require_text(
                    item.get("reconciliation_result_id"),
                    f"selections[{index}].reconciliation_result_id",
                ),
                _require_text(
                    item.get("selected_fact_id"), f"selections[{index}].selected_fact_id"
                ),
            )
        )
    policy_value = _require_object(record.get("selection_policy", {}), "selection_policy")
    try:
        policy = ReconciliationSelectionPolicy(
            SingleSourcePolicy(policy_value.get("single_source", "abstain"))
        )
    except ValueError as exc:
        raise MultiDocumentWorkflowError("selection_policy.single_source is unsupported") from exc
    return MultiDocumentWorkflowRequest(
        _require_text(record.get("question"), "question"),
        task,
        _require_text(record.get("company"), "company"),
        tuple(sources),
        tuple(selections),
        policy,
    )


def load_workflow_request(path: str | Path) -> MultiDocumentWorkflowRequest:
    """Load a request whose relative artifact paths are anchored to the JSON file."""
    request_path = Path(path)
    try:
        value = json.loads(request_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MultiDocumentWorkflowError(
            f"could not read request file {request_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise MultiDocumentWorkflowError(f"request file is not valid JSON: {exc}") from exc
    return workflow_request_from_dict(value, base_directory=request_path.parent)


def _load_sources(
    request: MultiDocumentWorkflowRequest,
) -> tuple[ReconciliationReport, dict[str, EvidenceCorpus]]:
    corpora: dict[str, EvidenceCorpus] = {}
    verified_sources: list[VerifiedFactSource] = []
    try:
        for source in request.sources:
            corpus = load_evidence_export(source.evidence_directory)
            if (
                corpus.document_id != source.document_id
                or corpus.source_sha256 != source.source_sha256
            ):
                raise MultiDocumentWorkflowError(
                    f"source {source.document_id!r} does not match its declared evidence identity"
                )
            if corpus.document_id in corpora:
                raise MultiDocumentWorkflowError(
                    f"more than one corpus maps to document {corpus.document_id!r}"
                )
            corpora[corpus.document_id] = corpus
            verified_sources.extend(
                VerifiedFactSource(path, corpus) for path in source.financial_facts
            )
        report = reconcile_financial_facts(verified_sources)
    except (RetrievalError, FinancialFactValidationError, ReconciliationError) as exc:
        raise MultiDocumentWorkflowError(str(exc)) from exc
    return report, corpora


def _generation_request(
    result: MultiDocumentOrchestrationResult,
) -> MultiDocumentGenerationRequest:
    selection = result.selection
    calculation = result.calculation
    if selection is None or selection.operands is None or calculation is None:
        raise MultiDocumentWorkflowError("orchestration did not produce generation inputs")
    bindings = tuple(
        SourceFactBinding(
            decision.request.role,
            decision.operand.fact,
            decision.operand.corpus.document_id,
            decision.operand.corpus.source_sha256,
        )
        for decision in selection.decisions
        if decision.operand is not None
    )
    return MultiDocumentGenerationRequest(
        result.request.question,
        result.request.task,
        result.request.company,
        bindings,
        calculation,
        selection,
    )


def run_multi_document_workflow(
    request: MultiDocumentWorkflowRequest,
    backend: LocalModelBackend,
) -> MultiDocumentWorkflowResult:
    """Run the established reconciliation-to-grounded-generation pipeline."""
    report, corpora = _load_sources(request)
    orchestration_request = MultiDocumentOrchestrationRequest(
        request.question,
        request.task,
        request.company,
        request.selections,
        request.selection_policy,
    )
    orchestration = MultiDocumentFinancialAnswerOrchestrator().orchestrate(
        orchestration_request, report, corpora
    )
    if orchestration.status is MultiDocumentOrchestrationStatus.ABSTAINED:
        return MultiDocumentWorkflowResult(request, report, orchestration, None)
    answer = MultiDocumentAnswerGenerator(backend).generate(
        _generation_request(orchestration), corpora
    )
    return MultiDocumentWorkflowResult(request, report, orchestration, answer)


__all__ = [
    "MULTI_DOCUMENT_REQUEST_SCHEMA_VERSION",
    "PERCENTAGE_DECIMAL_PLACES",
    "PERCENTAGE_PRESENTATION_POLICY_VERSION",
    "MultiDocumentSourceInput",
    "MultiDocumentWorkflowError",
    "MultiDocumentWorkflowRequest",
    "MultiDocumentWorkflowResult",
    "PercentagePresentation",
    "load_workflow_request",
    "present_percentage",
    "run_multi_document_workflow",
    "workflow_request_from_dict",
]
