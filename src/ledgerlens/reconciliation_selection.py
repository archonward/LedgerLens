"""Auditable selection of reconciled source facts for deterministic calculations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ledgerlens.calculations import CalculationError, FactOperand
from ledgerlens.reconciliation import (
    ReconciliationObservation,
    ReconciliationReport,
    ReconciliationResult,
    ReconciliationStatus,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus


class ReconciliationSelectionError(ValueError):
    """A selection request or policy is structurally invalid."""


class SingleSourcePolicy(StrEnum):
    """Whether an explicitly named single-source fact may feed a calculation."""

    ABSTAIN = "abstain"
    ALLOW_EXPLICIT = "allow_explicit"


class SelectionStatus(StrEnum):
    """Whether a source-specific fact was prepared for calculation."""

    SELECTED = "selected"
    ABSTAINED = "abstained"


class SelectionAbstentionReason(StrEnum):
    """Closed reasons that prevent reconciliation-to-calculation selection."""

    RESULT_NOT_FOUND = "result_not_found"
    AMBIGUOUS_RESULT_ID = "ambiguous_result_id"
    CONFLICT = "conflict"
    INCOMPATIBLE = "incompatible"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    DUPLICATE_AMBIGUITY = "duplicate_ambiguity"
    EXPLICIT_SOURCE_REQUIRED = "explicit_source_required"
    SELECTED_FACT_NOT_IN_RESULT = "selected_fact_not_in_result"
    INVALID_RECONCILIATION_RESULT = "invalid_reconciliation_result"
    INVALID_OR_STALE_EVIDENCE = "invalid_or_stale_evidence"


@dataclass(frozen=True)
class ReconciliationSelectionPolicy:
    """Explicit policy applied uniformly to every selection request."""

    single_source: SingleSourcePolicy = SingleSourcePolicy.ABSTAIN

    def __post_init__(self) -> None:
        if not isinstance(self.single_source, SingleSourcePolicy):
            raise ReconciliationSelectionError("single_source must be a SingleSourcePolicy value")

    def to_dict(self) -> dict[str, object]:
        return {
            "agreement_source_rule": "explicit_fact_id",
            "single_source": self.single_source.value,
            "conflicts_allowed": False,
            "incompatible_observations_allowed": False,
            "duplicate_ambiguity_allowed": False,
            "evidence_revalidation": "all_candidate_observations",
        }


@dataclass(frozen=True)
class FactSelectionRequest:
    """One calculation role and the exact reconciled source fact requested for it."""

    role: str
    reconciliation_result_id: str
    selected_fact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise ReconciliationSelectionError("selection role must be a non-empty string")
        if (
            not isinstance(self.reconciliation_result_id, str)
            or not self.reconciliation_result_id.strip()
        ):
            raise ReconciliationSelectionError(
                "reconciliation_result_id must be a non-empty string"
            )
        if self.selected_fact_id is not None and (
            not isinstance(self.selected_fact_id, str) or not self.selected_fact_id.strip()
        ):
            raise ReconciliationSelectionError(
                "selected_fact_id must be a non-empty string or null"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "reconciliation_result_id": self.reconciliation_result_id,
            "requested_fact_id": self.selected_fact_id,
        }


@dataclass(frozen=True)
class FactSelectionDecision:
    """One selection or abstention with all candidate provenance retained."""

    request: FactSelectionRequest
    status: SelectionStatus
    reconciliation_status: ReconciliationStatus | None
    candidate_observations: tuple[ReconciliationObservation, ...]
    operand: FactOperand | None = None
    abstention_reason: SelectionAbstentionReason | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status is SelectionStatus.SELECTED:
            if self.operand is None or self.abstention_reason is not None:
                raise ReconciliationSelectionError(
                    "selected decisions require an operand and no abstention reason"
                )
        elif self.operand is not None or self.abstention_reason is None:
            raise ReconciliationSelectionError(
                "abstained decisions require a reason and must not contain an operand"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.request.to_dict(),
            "status": self.status.value,
            "reconciliation_status": (
                self.reconciliation_status.value if self.reconciliation_status else None
            ),
            "candidate_observations": [
                observation.to_dict() for observation in self.candidate_observations
            ],
            "selected_fact": self.operand.fact.to_dict() if self.operand else None,
            "abstention_reason": (self.abstention_reason.value if self.abstention_reason else None),
            "detail": self.detail,
        }


@dataclass(frozen=True)
class CalculationFactSelection:
    """Ordered decisions and an all-or-nothing operand set for a calculation call."""

    policy: ReconciliationSelectionPolicy
    decisions: tuple[FactSelectionDecision, ...]

    @property
    def operands(self) -> tuple[FactOperand, ...] | None:
        """Return operands in request order only when every role was selected."""
        if any(decision.status is SelectionStatus.ABSTAINED for decision in self.decisions):
            return None
        return tuple(
            decision.operand for decision in self.decisions if decision.operand is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": (
                SelectionStatus.SELECTED.value
                if self.operands is not None
                else SelectionStatus.ABSTAINED.value
            ),
            "policy": self.policy.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def _abstain(
    request: FactSelectionRequest,
    reason: SelectionAbstentionReason,
    detail: str,
    *,
    result: ReconciliationResult | None = None,
) -> FactSelectionDecision:
    return FactSelectionDecision(
        request=request,
        status=SelectionStatus.ABSTAINED,
        reconciliation_status=result.status if result else None,
        candidate_observations=result.observations if result else (),
        abstention_reason=reason,
        detail=detail,
    )


def _duplicate_detail(report: ReconciliationReport, result: ReconciliationResult) -> str | None:
    fact_id_counts = Counter(observation.fact.fact_id for observation in result.observations)
    duplicate_ids = sorted(
        fact_id
        for fact_id, count in fact_id_counts.items()
        if count > 1 or fact_id in report.duplicate_fact_ids
    )
    document_counts = Counter(
        observation.fact.evidence.document_id for observation in result.observations
    )
    duplicate_documents = sorted(
        document_id for document_id, count in document_counts.items() if count > 1
    )
    if not duplicate_ids and not duplicate_documents:
        return None
    parts: list[str] = []
    if duplicate_ids:
        parts.append(f"duplicate fact IDs: {', '.join(duplicate_ids)}")
    if duplicate_documents:
        parts.append(
            "multiple candidate records from source documents: " + ", ".join(duplicate_documents)
        )
    return "; ".join(parts)


def _validate_result_shape(result: ReconciliationResult) -> str | None:
    if result.comparison_key is None:
        return "eligible reconciliation results require a comparison key"
    if not result.observations:
        return "eligible reconciliation results require at least one observation"
    if any(
        observation.fact.company != result.company or observation.fact.metric != result.metric
        for observation in result.observations
    ):
        return "candidate facts do not match the reconciliation company and metric"
    values = {observation.base_value for observation in result.observations}
    documents = {observation.fact.evidence.document_id for observation in result.observations}
    if result.status is ReconciliationStatus.AGREEMENT:
        if len(values) != 1 or len(documents) < 2:
            return "agreement does not contain one value from distinct source documents"
    elif result.status is ReconciliationStatus.INSUFFICIENT_EVIDENCE:
        if len(values) != 1 or len(documents) != 1:
            return "insufficient-evidence result has inconsistent values or source documents"
    else:
        return f"unsupported reconciliation status: {result.status!r}"
    return None


def _revalidate_candidates(
    result: ReconciliationResult,
    corpora: Mapping[str, EvidenceCorpus],
) -> tuple[dict[str, FactOperand] | None, str | None]:
    operands: dict[str, FactOperand] = {}
    for observation in result.observations:
        document_id = observation.fact.evidence.document_id
        corpus = corpora.get(document_id)
        if corpus is None:
            return None, f"no evidence corpus was supplied for document {document_id!r}"
        try:
            operand = FactOperand(observation.fact, corpus)
        except CalculationError as exc:
            return None, str(exc)
        operands[observation.fact.fact_id] = operand
    return operands, None


def _select_one(
    report: ReconciliationReport,
    request: FactSelectionRequest,
    corpora: Mapping[str, EvidenceCorpus],
    policy: ReconciliationSelectionPolicy,
) -> FactSelectionDecision:
    matches = [
        result for result in report.results if result.result_id == request.reconciliation_result_id
    ]
    if not matches:
        return _abstain(
            request,
            SelectionAbstentionReason.RESULT_NOT_FOUND,
            "the requested reconciliation result is absent",
        )
    if len(matches) > 1:
        return _abstain(
            request,
            SelectionAbstentionReason.AMBIGUOUS_RESULT_ID,
            "the reconciliation report contains the requested result ID more than once",
        )
    result = matches[0]
    if result.status is ReconciliationStatus.CONFLICT:
        return _abstain(
            request,
            SelectionAbstentionReason.CONFLICT,
            "conflicting values are unresolved and cannot feed a calculation",
            result=result,
        )
    if result.status is ReconciliationStatus.INCOMPATIBLE:
        return _abstain(
            request,
            SelectionAbstentionReason.INCOMPATIBLE,
            "incompatible observations cannot feed a calculation",
            result=result,
        )

    shape_error = _validate_result_shape(result)
    if shape_error:
        return _abstain(
            request,
            SelectionAbstentionReason.INVALID_RECONCILIATION_RESULT,
            shape_error,
            result=result,
        )
    duplicate_detail = _duplicate_detail(report, result)
    if duplicate_detail:
        return _abstain(
            request,
            SelectionAbstentionReason.DUPLICATE_AMBIGUITY,
            duplicate_detail,
            result=result,
        )
    if (
        result.status is ReconciliationStatus.INSUFFICIENT_EVIDENCE
        and policy.single_source is SingleSourcePolicy.ABSTAIN
    ):
        return _abstain(
            request,
            SelectionAbstentionReason.INSUFFICIENT_EVIDENCE,
            "single-source observations are disabled by policy",
            result=result,
        )
    if request.selected_fact_id is None:
        return _abstain(
            request,
            SelectionAbstentionReason.EXPLICIT_SOURCE_REQUIRED,
            "the caller must name one exact source-specific fact ID",
            result=result,
        )
    candidate_ids = {observation.fact.fact_id for observation in result.observations}
    if request.selected_fact_id not in candidate_ids:
        return _abstain(
            request,
            SelectionAbstentionReason.SELECTED_FACT_NOT_IN_RESULT,
            f"fact {request.selected_fact_id!r} is not a candidate in this result",
            result=result,
        )

    operands, validation_error = _revalidate_candidates(result, corpora)
    if operands is None:
        return _abstain(
            request,
            SelectionAbstentionReason.INVALID_OR_STALE_EVIDENCE,
            validation_error or "candidate evidence validation failed",
            result=result,
        )
    return FactSelectionDecision(
        request=request,
        status=SelectionStatus.SELECTED,
        reconciliation_status=result.status,
        candidate_observations=result.observations,
        operand=operands[request.selected_fact_id],
        detail="selected only because the caller explicitly named this candidate fact ID",
    )


def prepare_calculation_facts(
    report: ReconciliationReport,
    requests: Iterable[FactSelectionRequest],
    corpora: Mapping[str, EvidenceCorpus],
    *,
    policy: ReconciliationSelectionPolicy = ReconciliationSelectionPolicy(),
) -> CalculationFactSelection:
    """Prepare source-specific ``FactOperand`` values from reconciliation results.

    The returned operands preserve request order and are available only when every request
    succeeds. Selection is never inferred from observation order or report recency.
    """
    requested = tuple(requests)
    if not requested:
        raise ReconciliationSelectionError("at least one fact selection request is required")
    roles = [request.role for request in requested]
    if len(roles) != len(set(roles)):
        raise ReconciliationSelectionError("fact selection request roles must be unique")
    decisions = tuple(_select_one(report, request, corpora, policy) for request in requested)
    return CalculationFactSelection(policy, decisions)


__all__ = [
    "CalculationFactSelection",
    "FactSelectionDecision",
    "FactSelectionRequest",
    "ReconciliationSelectionError",
    "ReconciliationSelectionPolicy",
    "SelectionAbstentionReason",
    "SelectionStatus",
    "SingleSourcePolicy",
    "prepare_calculation_facts",
]
