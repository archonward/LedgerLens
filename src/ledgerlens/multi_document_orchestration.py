"""Conservative multi-document orchestration over explicit reconciled facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from ledgerlens.calculations import (
    CalculationError,
    CalculationResult,
    FactOperand,
    profit_margin,
    year_over_year_growth,
)
from ledgerlens.financial_facts import FinancialFact, ResolutionStatus
from ledgerlens.generation import GROUNDING_POLICY_VERSION, AnswerTask
from ledgerlens.reconciliation import ReconciliationReport
from ledgerlens.reconciliation_selection import (
    CalculationFactSelection,
    FactSelectionRequest,
    ReconciliationSelectionPolicy,
    SelectionStatus,
    prepare_calculation_facts,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus

MULTI_DOCUMENT_ORCHESTRATION_SCHEMA_VERSION = "1.0"


class MultiDocumentOrchestrationError(ValueError):
    """A multi-document orchestration request is structurally invalid."""


class MultiDocumentOrchestrationStatus(StrEnum):
    """Whether deterministic calculation completed without answer generation."""

    CALCULATED = "calculated"
    ABSTAINED = "abstained"


class MultiDocumentAbstentionReason(StrEnum):
    """Closed orchestration-level reasons for deterministic abstention."""

    INVALID_ROLE_SET = "invalid_role_set"
    FACT_SELECTION_ABSTAINED = "fact_selection_abstained"
    COMPANY_MISMATCH = "company_mismatch"
    METRIC_MISMATCH = "metric_mismatch"
    UNRESOLVED_REPORTING_PERIOD = "unresolved_reporting_period"
    INCOMPATIBLE_FINANCIAL_FACTS = "incompatible_financial_facts"


class GenerationBoundaryReason(StrEnum):
    """Why the existing single-corpus generator was not invoked."""

    ORCHESTRATION_ABSTAINED = "orchestration_abstained"
    SINGLE_CORPUS_INTERFACE = "existing_generator_requires_one_validating_corpus"


@dataclass(frozen=True)
class MultiDocumentOrchestrationRequest:
    """Explicit task, company, selection requests, and single-source policy."""

    question: str
    task: AnswerTask
    company: str
    fact_selections: tuple[FactSelectionRequest, ...]
    selection_policy: ReconciliationSelectionPolicy = ReconciliationSelectionPolicy()

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise MultiDocumentOrchestrationError("question must be a non-empty string")
        if not isinstance(self.task, AnswerTask):
            raise MultiDocumentOrchestrationError("task must be a supported AnswerTask")
        if not isinstance(self.company, str) or not self.company.strip():
            raise MultiDocumentOrchestrationError("company must be a non-empty string")
        if not isinstance(self.fact_selections, tuple) or not self.fact_selections:
            raise MultiDocumentOrchestrationError("fact_selections must be a non-empty tuple")
        if any(
            not isinstance(selection, FactSelectionRequest) for selection in self.fact_selections
        ):
            raise MultiDocumentOrchestrationError(
                "fact_selections must contain FactSelectionRequest values"
            )
        if not isinstance(self.selection_policy, ReconciliationSelectionPolicy):
            raise MultiDocumentOrchestrationError(
                "selection_policy must be a ReconciliationSelectionPolicy"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "task": self.task.value,
            "company": self.company,
            "fact_selections": [item.to_dict() for item in self.fact_selections],
            "selection_policy": self.selection_policy.to_dict(),
        }


@dataclass(frozen=True)
class MultiDocumentOrchestrationResult:
    """Structured deterministic calculation or abstention; never a generated answer."""

    status: MultiDocumentOrchestrationStatus
    request: MultiDocumentOrchestrationRequest
    selection: CalculationFactSelection | None
    selected_facts: tuple[FinancialFact, ...]
    calculation: CalculationResult | None
    generation_boundary_reason: GenerationBoundaryReason
    abstention_reason: MultiDocumentAbstentionReason | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status is MultiDocumentOrchestrationStatus.CALCULATED:
            if (
                self.selection is None
                or self.selection.operands is None
                or self.calculation is None
                or self.abstention_reason is not None
                or self.generation_boundary_reason
                is not GenerationBoundaryReason.SINGLE_CORPUS_INTERFACE
            ):
                raise MultiDocumentOrchestrationError(
                    "calculated results require selection, calculation, and generation boundary"
                )
            operands = self.selection.operands
            assert operands is not None
            operand_facts = tuple(operand.fact for operand in operands)
            calculation_facts = tuple(item.fact for item in self.calculation.inputs)
            if self.selected_facts != operand_facts or self.selected_facts != calculation_facts:
                raise MultiDocumentOrchestrationError(
                    "selected facts must match prepared operands and calculation inputs"
                )
        else:
            if (
                self.calculation is not None
                or self.abstention_reason is None
                or self.generation_boundary_reason
                is not GenerationBoundaryReason.ORCHESTRATION_ABSTAINED
            ):
                raise MultiDocumentOrchestrationError(
                    "abstained results require a reason, no calculation, and no generation"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MULTI_DOCUMENT_ORCHESTRATION_SCHEMA_VERSION,
            "status": self.status.value,
            "request": self.request.to_dict(),
            "selection": self.selection.to_dict() if self.selection else None,
            "selected_fact_ids": [fact.fact_id for fact in self.selected_facts],
            "selected_facts": [fact.to_dict() for fact in self.selected_facts],
            "calculation": self.calculation.to_dict() if self.calculation else None,
            "generation": {
                "status": "not_attempted",
                "reason": self.generation_boundary_reason.value,
                "model_invoked": False,
                "answer_generator_invoked": False,
                "grounding_policy_version": GROUNDING_POLICY_VERSION,
                "boundary_detail": (
                    "The existing AnswerRequest accepts one EvidenceCorpus and revalidates all "
                    "facts against it; cross-document facts cannot be supplied safely."
                    if self.generation_boundary_reason
                    is GenerationBoundaryReason.SINGLE_CORPUS_INTERFACE
                    else "Orchestration abstained before the generation boundary."
                ),
            },
            "abstention_reason": (self.abstention_reason.value if self.abstention_reason else None),
            "detail": self.detail,
        }


def _expected_roles(task: AnswerTask) -> tuple[str, str]:
    if task is AnswerTask.REVENUE_GROWTH:
        return "current", "prior"
    return "profit", "revenue"


def _ordered_requests(
    request: MultiDocumentOrchestrationRequest,
) -> tuple[FactSelectionRequest, FactSelectionRequest] | None:
    expected = _expected_roles(request.task)
    by_role = {selection.role: selection for selection in request.fact_selections}
    if len(by_role) != len(request.fact_selections) or set(by_role) != set(expected):
        return None
    return by_role[expected[0]], by_role[expected[1]]


def _selection_detail(selection: CalculationFactSelection) -> str:
    abstentions = [
        f"{decision.request.role}: {decision.abstention_reason.value}"
        for decision in selection.decisions
        if decision.status is SelectionStatus.ABSTAINED and decision.abstention_reason is not None
    ]
    return "; ".join(abstentions) or "one or more fact selections abstained"


def _selected_facts(operands: tuple[FactOperand, ...]) -> tuple[FinancialFact, ...]:
    return tuple(operand.fact for operand in operands)


def _precalculation_error(
    request: MultiDocumentOrchestrationRequest,
    facts: tuple[FinancialFact, FinancialFact],
) -> tuple[MultiDocumentAbstentionReason, str] | None:
    if any(fact.company != request.company for fact in facts):
        return (
            MultiDocumentAbstentionReason.COMPANY_MISMATCH,
            "selected fact companies must exactly match the requested company",
        )
    expected_metrics = (
        ("revenue", "revenue")
        if request.task is AnswerTask.REVENUE_GROWTH
        else ("net_income", "revenue")
    )
    actual_metrics = tuple(fact.metric for fact in facts)
    if actual_metrics != expected_metrics:
        return (
            MultiDocumentAbstentionReason.METRIC_MISMATCH,
            f"selected metrics {actual_metrics!r} do not match {expected_metrics!r}",
        )
    if any(fact.reporting_period.status is not ResolutionStatus.KNOWN for fact in facts):
        return (
            MultiDocumentAbstentionReason.UNRESOLVED_REPORTING_PERIOD,
            "selected facts require known reporting periods",
        )
    return None


def _calculate(
    task: AnswerTask,
    operands: tuple[FactOperand, FactOperand],
) -> CalculationResult:
    if task is AnswerTask.REVENUE_GROWTH:
        return year_over_year_growth(operands[0], operands[1])
    return profit_margin(operands[0], operands[1])


def _abstain(
    request: MultiDocumentOrchestrationRequest,
    reason: MultiDocumentAbstentionReason,
    detail: str,
    *,
    selection: CalculationFactSelection | None = None,
    selected_facts: tuple[FinancialFact, ...] = (),
) -> MultiDocumentOrchestrationResult:
    return MultiDocumentOrchestrationResult(
        status=MultiDocumentOrchestrationStatus.ABSTAINED,
        request=request,
        selection=selection,
        selected_facts=selected_facts,
        calculation=None,
        generation_boundary_reason=GenerationBoundaryReason.ORCHESTRATION_ABSTAINED,
        abstention_reason=reason,
        detail=detail,
    )


class MultiDocumentFinancialAnswerOrchestrator:
    """Prepare explicit reconciled facts and calculate without unsafe generation."""

    def orchestrate(
        self,
        request: MultiDocumentOrchestrationRequest,
        report: ReconciliationReport,
        corpora: Mapping[str, EvidenceCorpus],
    ) -> MultiDocumentOrchestrationResult:
        ordered = _ordered_requests(request)
        if ordered is None:
            expected = ", ".join(_expected_roles(request.task))
            return _abstain(
                request,
                MultiDocumentAbstentionReason.INVALID_ROLE_SET,
                f"task requires exactly these roles: {expected}",
            )
        selection = prepare_calculation_facts(
            report,
            ordered,
            corpora,
            policy=request.selection_policy,
        )
        operands = selection.operands
        if operands is None:
            return _abstain(
                request,
                MultiDocumentAbstentionReason.FACT_SELECTION_ABSTAINED,
                _selection_detail(selection),
                selection=selection,
            )
        if len(operands) != 2:
            raise MultiDocumentOrchestrationError(
                "selection adapter returned an unexpected operand count"
            )
        typed_operands = (operands[0], operands[1])
        facts = _selected_facts(typed_operands)
        compatibility_error = _precalculation_error(request, facts)
        if compatibility_error:
            reason, detail = compatibility_error
            return _abstain(
                request,
                reason,
                detail,
                selection=selection,
                selected_facts=facts,
            )
        try:
            calculation = _calculate(request.task, typed_operands)
        except CalculationError as exc:
            return _abstain(
                request,
                MultiDocumentAbstentionReason.INCOMPATIBLE_FINANCIAL_FACTS,
                str(exc),
                selection=selection,
                selected_facts=facts,
            )
        return MultiDocumentOrchestrationResult(
            status=MultiDocumentOrchestrationStatus.CALCULATED,
            request=request,
            selection=selection,
            selected_facts=facts,
            calculation=calculation,
            generation_boundary_reason=GenerationBoundaryReason.SINGLE_CORPUS_INTERFACE,
            detail=(
                "deterministic calculation completed; answer generation was not attempted "
                "because the existing generator has a single-corpus validation contract"
            ),
        )


__all__ = [
    "MULTI_DOCUMENT_ORCHESTRATION_SCHEMA_VERSION",
    "GenerationBoundaryReason",
    "MultiDocumentAbstentionReason",
    "MultiDocumentFinancialAnswerOrchestrator",
    "MultiDocumentOrchestrationError",
    "MultiDocumentOrchestrationRequest",
    "MultiDocumentOrchestrationResult",
    "MultiDocumentOrchestrationStatus",
]
