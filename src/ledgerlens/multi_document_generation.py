"""Validation boundary for future generation over multiple annual reports.

This module deliberately performs no model inference and renders no natural-language
answer.  It validates an explicit, source-specific request into a structured result that
Phase 3B can require before presenting any material to a local model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from ledgerlens.calculations import (
    CalculationError,
    CalculationResult,
    CalculationType,
    FactOperand,
    profit_margin,
    year_over_year_growth,
)
from ledgerlens.financial_facts import FinancialFact
from ledgerlens.generation import GROUNDING_POLICY_VERSION, AnswerTask
from ledgerlens.reconciliation import ReconciliationStatus
from ledgerlens.reconciliation_selection import (
    CalculationFactSelection,
    FactSelectionDecision,
    ReconciliationSelectionPolicy,
    SelectionStatus,
    SingleSourcePolicy,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus

MULTI_DOCUMENT_GENERATION_SCHEMA_VERSION = "1.0"


class MultiDocumentGenerationError(ValueError):
    """A multi-document generation request is structurally invalid."""


class MultiDocumentGenerationValidationStatus(StrEnum):
    """Whether a request is safe to pass to a future generation component."""

    VALIDATED = "validated"
    ABSTAINED = "abstained"


class MultiDocumentGenerationAbstentionReason(StrEnum):
    """Closed validation failures at the multi-document generation boundary."""

    INVALID_ROLE_SET = "invalid_role_set"
    INVALID_FACT_BINDINGS = "invalid_fact_bindings"
    MISSING_EVIDENCE_CORPUS = "missing_evidence_corpus"
    AMBIGUOUS_SOURCE_MAPPING = "ambiguous_source_mapping"
    INVALID_OR_STALE_EVIDENCE = "invalid_or_stale_evidence"
    UNSUPPORTED_SELECTION_POLICY = "unsupported_selection_policy"
    SELECTION_ABSTAINED = "selection_abstained"
    INVALID_SELECTION = "invalid_selection"
    CALCULATION_INPUT_MISMATCH = "calculation_input_mismatch"
    CALCULATION_COULD_NOT_BE_REPRODUCED = "calculation_could_not_be_reproduced"
    CALCULATION_RESULT_MISMATCH = "calculation_result_mismatch"


@dataclass(frozen=True)
class SourceFactBinding:
    """One calculation role explicitly bound to an exact annual-report identity."""

    role: str
    fact: FinancialFact
    corpus_document_id: str
    corpus_source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise MultiDocumentGenerationError("binding role must be a non-empty string")
        if not isinstance(self.fact, FinancialFact):
            raise MultiDocumentGenerationError("binding fact must be a FinancialFact")
        if not isinstance(self.corpus_document_id, str) or not self.corpus_document_id.strip():
            raise MultiDocumentGenerationError(
                "binding corpus_document_id must be a non-empty string"
            )
        if (
            not isinstance(self.corpus_source_sha256, str)
            or len(self.corpus_source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.corpus_source_sha256)
        ):
            raise MultiDocumentGenerationError(
                "binding corpus_source_sha256 must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "fact": self.fact.to_dict(),
            "corpus_document_id": self.corpus_document_id,
            "corpus_source_sha256": self.corpus_source_sha256,
        }


@dataclass(frozen=True)
class MultiDocumentGenerationRequest:
    """Explicit facts, source bindings, selection audit, and supplied calculation."""

    question: str
    task: AnswerTask
    company: str
    bindings: tuple[SourceFactBinding, ...]
    calculation: CalculationResult
    selection: CalculationFactSelection

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise MultiDocumentGenerationError("question must be a non-empty string")
        if not isinstance(self.task, AnswerTask):
            raise MultiDocumentGenerationError("task must be a supported AnswerTask")
        if not isinstance(self.company, str) or not self.company.strip():
            raise MultiDocumentGenerationError("company must be a non-empty string")
        if not isinstance(self.bindings, tuple) or not self.bindings:
            raise MultiDocumentGenerationError("bindings must be a non-empty tuple")
        if any(not isinstance(binding, SourceFactBinding) for binding in self.bindings):
            raise MultiDocumentGenerationError("bindings must contain SourceFactBinding values")
        if not isinstance(self.calculation, CalculationResult):
            raise MultiDocumentGenerationError("calculation must be a CalculationResult")
        if not isinstance(self.selection, CalculationFactSelection):
            raise MultiDocumentGenerationError("selection must be a CalculationFactSelection")

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "task": self.task.value,
            "company": self.company,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "calculation": self.calculation.to_dict(),
            "selection": self.selection.to_dict(),
        }


@dataclass(frozen=True)
class AnnualReportIdentity:
    """Exact source-document identity shown on every source-specific citation."""

    document_id: str
    company: str
    report_year: int
    source_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "company": self.company,
            "report_year": self.report_year,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class SourceSpecificCitation:
    """Citation for one financial value and its validating annual report."""

    role: str
    fact_id: str
    annual_report: AnnualReportIdentity
    physical_page_number: int
    supporting_text: str
    supporting_text_sha256: str
    selected_text_sha256: str
    evidence_jsonl_sha256: str
    evidence_manifest_sha256: str
    evidence_method: str
    screening_warnings: tuple[str, ...]
    evidence_limitations: tuple[str, ...]
    fact_warnings: tuple[str, ...]
    fact_limitations: tuple[str, ...]
    evidence_review_provenance: dict[str, object] | None
    fact_verification_provenance: dict[str, object]

    @property
    def marker(self) -> str:
        report = self.annual_report
        return (
            f"[{report.company} {report.report_year} annual report "
            f"{report.document_id}, PDF p. {self.physical_page_number}]"
        )

    @classmethod
    def from_binding(cls, binding: SourceFactBinding) -> SourceSpecificCitation:
        fact = binding.fact
        evidence = fact.evidence
        return cls(
            role=binding.role,
            fact_id=fact.fact_id,
            annual_report=AnnualReportIdentity(
                evidence.document_id,
                evidence.source_company,
                evidence.source_report_year,
                evidence.source_sha256,
            ),
            physical_page_number=evidence.physical_page_number,
            supporting_text=evidence.supporting_text,
            supporting_text_sha256=evidence.supporting_text_sha256,
            selected_text_sha256=evidence.selected_text_sha256,
            evidence_jsonl_sha256=evidence.evidence_jsonl_sha256,
            evidence_manifest_sha256=evidence.evidence_manifest_sha256,
            evidence_method=evidence.evidence_method,
            screening_warnings=evidence.screening_warnings,
            evidence_limitations=evidence.known_limitations,
            fact_warnings=fact.warnings,
            fact_limitations=fact.limitations,
            evidence_review_provenance=evidence.review_provenance,
            fact_verification_provenance=fact.verification.to_dict(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "role": self.role,
            "fact_id": self.fact_id,
            "annual_report": self.annual_report.to_dict(),
            "physical_page_number": self.physical_page_number,
            "supporting_text": self.supporting_text,
            "supporting_text_sha256": self.supporting_text_sha256,
            "selected_text_sha256": self.selected_text_sha256,
            "evidence_jsonl_sha256": self.evidence_jsonl_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "evidence_method": self.evidence_method,
            "screening_warnings": list(self.screening_warnings),
            "evidence_limitations": list(self.evidence_limitations),
            "fact_warnings": list(self.fact_warnings),
            "fact_limitations": list(self.fact_limitations),
            "evidence_review_provenance": self.evidence_review_provenance,
            "fact_verification_provenance": self.fact_verification_provenance,
        }


@dataclass(frozen=True)
class MultiDocumentGenerationValidationResult:
    """Validated generation inputs or a structured conservative abstention."""

    status: MultiDocumentGenerationValidationStatus
    request: MultiDocumentGenerationRequest
    reproduced_calculation: CalculationResult | None
    citations: tuple[SourceSpecificCitation, ...]
    abstention_reason: MultiDocumentGenerationAbstentionReason | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.status is MultiDocumentGenerationValidationStatus.VALIDATED:
            if self.reproduced_calculation is None or self.abstention_reason is not None:
                raise MultiDocumentGenerationError(
                    "validated results require a reproduced calculation and no abstention"
                )
            if len(self.citations) != len(self.request.bindings):
                raise MultiDocumentGenerationError(
                    "validated results require one source-specific citation per binding"
                )
        elif (
            self.reproduced_calculation is not None
            or self.citations
            or self.abstention_reason is None
        ):
            raise MultiDocumentGenerationError(
                "abstained results require a reason and no validated calculation or citations"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MULTI_DOCUMENT_GENERATION_SCHEMA_VERSION,
            "status": self.status.value,
            "request": self.request.to_dict(),
            "reproduced_calculation": (
                self.reproduced_calculation.to_dict() if self.reproduced_calculation else None
            ),
            "citations": [citation.to_dict() for citation in self.citations],
            "generation": {
                "status": "not_attempted",
                "model_invoked": False,
                "grounding_policy_version": GROUNDING_POLICY_VERSION,
            },
            "abstention_reason": (self.abstention_reason.value if self.abstention_reason else None),
            "detail": self.detail,
        }


def _expected_roles(task: AnswerTask) -> tuple[str, str]:
    if task is AnswerTask.REVENUE_GROWTH:
        return "current", "prior"
    return "profit", "revenue"


def _abstain(
    request: MultiDocumentGenerationRequest,
    reason: MultiDocumentGenerationAbstentionReason,
    detail: str,
) -> MultiDocumentGenerationValidationResult:
    return MultiDocumentGenerationValidationResult(
        MultiDocumentGenerationValidationStatus.ABSTAINED,
        request,
        None,
        (),
        reason,
        detail,
    )


def _validate_selection(
    request: MultiDocumentGenerationRequest,
    by_role: dict[str, SourceFactBinding],
) -> tuple[MultiDocumentGenerationAbstentionReason, str] | None:
    selection = request.selection
    if type(selection.policy) is not ReconciliationSelectionPolicy or not isinstance(
        selection.policy.single_source, SingleSourcePolicy
    ):
        return (
            MultiDocumentGenerationAbstentionReason.UNSUPPORTED_SELECTION_POLICY,
            "selection uses an unsupported reconciliation policy",
        )
    decisions = selection.decisions
    if (
        not isinstance(decisions, tuple)
        or not decisions
        or any(not isinstance(decision, FactSelectionDecision) for decision in decisions)
    ):
        return (
            MultiDocumentGenerationAbstentionReason.INVALID_SELECTION,
            "selection must contain preserved FactSelectionDecision values",
        )
    if any(decision.status is SelectionStatus.ABSTAINED for decision in decisions):
        return (
            MultiDocumentGenerationAbstentionReason.SELECTION_ABSTAINED,
            "one or more preserved reconciliation decisions abstained",
        )
    if len(decisions) != len(by_role) or {item.request.role for item in decisions} != set(by_role):
        return (
            MultiDocumentGenerationAbstentionReason.INVALID_SELECTION,
            "selection roles do not exactly match source bindings",
        )
    for decision in decisions:
        binding = by_role[decision.request.role]
        if (
            decision.operand is None
            or decision.operand.fact != binding.fact
            or decision.request.selected_fact_id != binding.fact.fact_id
            or decision.reconciliation_status
            not in {ReconciliationStatus.AGREEMENT, ReconciliationStatus.INSUFFICIENT_EVIDENCE}
        ):
            return (
                MultiDocumentGenerationAbstentionReason.INVALID_SELECTION,
                f"selection decision for role {binding.role!r} does not preserve the binding",
            )
        if (
            decision.reconciliation_status is ReconciliationStatus.INSUFFICIENT_EVIDENCE
            and selection.policy.single_source is not SingleSourcePolicy.ALLOW_EXPLICIT
        ):
            return (
                MultiDocumentGenerationAbstentionReason.UNSUPPORTED_SELECTION_POLICY,
                "single-source selection was not explicitly allowed",
            )
    return None


def _reproduce(
    request: MultiDocumentGenerationRequest,
    operands: dict[str, FactOperand],
) -> CalculationResult:
    if request.task is AnswerTask.REVENUE_GROWTH:
        return year_over_year_growth(operands["current"], operands["prior"])
    return profit_margin(operands["profit"], operands["revenue"])


class MultiDocumentGenerationValidator:
    """Revalidate source bindings, selection, and Decimal calculation without inference."""

    def validate(
        self,
        request: MultiDocumentGenerationRequest,
        corpora: Mapping[str, EvidenceCorpus],
    ) -> MultiDocumentGenerationValidationResult:
        if not isinstance(corpora, Mapping):
            return _abstain(
                request,
                MultiDocumentGenerationAbstentionReason.AMBIGUOUS_SOURCE_MAPPING,
                "corpora must map exact document IDs to EvidenceCorpus values",
            )
        expected_roles = _expected_roles(request.task)
        roles = [binding.role for binding in request.bindings]
        if len(roles) != len(set(roles)) or set(roles) != set(expected_roles):
            return _abstain(
                request,
                MultiDocumentGenerationAbstentionReason.INVALID_ROLE_SET,
                f"task requires exactly these unique roles: {', '.join(expected_roles)}",
            )
        if len({binding.fact.fact_id for binding in request.bindings}) != len(request.bindings):
            return _abstain(
                request,
                MultiDocumentGenerationAbstentionReason.INVALID_FACT_BINDINGS,
                "source bindings must contain unique fact IDs",
            )
        by_role = {binding.role: binding for binding in request.bindings}
        selection_error = _validate_selection(request, by_role)
        if selection_error:
            return _abstain(request, *selection_error)

        operands: dict[str, FactOperand] = {}
        expected_metrics = (
            {"current": "revenue", "prior": "revenue"}
            if request.task is AnswerTask.REVENUE_GROWTH
            else {"profit": "net_income", "revenue": "revenue"}
        )
        for role in expected_roles:
            binding = by_role[role]
            fact = binding.fact
            if fact.company != request.company or fact.metric.casefold() != expected_metrics[role]:
                return _abstain(
                    request,
                    MultiDocumentGenerationAbstentionReason.INVALID_FACT_BINDINGS,
                    f"fact for role {role!r} does not match the requested company and metric",
                )
            if (
                fact.evidence.document_id != binding.corpus_document_id
                or fact.evidence.source_sha256 != binding.corpus_source_sha256
            ):
                return _abstain(
                    request,
                    MultiDocumentGenerationAbstentionReason.AMBIGUOUS_SOURCE_MAPPING,
                    f"fact for role {role!r} does not match its declared annual report",
                )
            corpus = corpora.get(binding.corpus_document_id)
            if corpus is None:
                return _abstain(
                    request,
                    MultiDocumentGenerationAbstentionReason.MISSING_EVIDENCE_CORPUS,
                    f"no corpus was supplied for document {binding.corpus_document_id!r}",
                )
            if (
                not isinstance(corpus, EvidenceCorpus)
                or corpus.document_id != binding.corpus_document_id
                or corpus.source_sha256 != binding.corpus_source_sha256
            ):
                return _abstain(
                    request,
                    MultiDocumentGenerationAbstentionReason.AMBIGUOUS_SOURCE_MAPPING,
                    f"corpus mapping for role {role!r} has a different document identity",
                )
            try:
                operands[role] = FactOperand(fact, corpus)
            except CalculationError as exc:
                return _abstain(
                    request,
                    MultiDocumentGenerationAbstentionReason.INVALID_OR_STALE_EVIDENCE,
                    str(exc),
                )

        calculation_roles = {item.role: item.fact for item in request.calculation.inputs}
        bound_roles = {role: by_role[role].fact for role in expected_roles}
        if calculation_roles != bound_roles:
            return _abstain(
                request,
                MultiDocumentGenerationAbstentionReason.CALCULATION_INPUT_MISMATCH,
                "calculation inputs do not exactly match the source-specific bindings",
            )
        expected_type = (
            CalculationType.YEAR_OVER_YEAR_GROWTH
            if request.task is AnswerTask.REVENUE_GROWTH
            else CalculationType.PROFIT_MARGIN
        )
        if request.calculation.calculation_type is not expected_type:
            return _abstain(
                request,
                MultiDocumentGenerationAbstentionReason.CALCULATION_INPUT_MISMATCH,
                "calculation type does not match the requested answer task",
            )
        try:
            reproduced = _reproduce(request, operands)
        except CalculationError as exc:
            return _abstain(
                request,
                MultiDocumentGenerationAbstentionReason.CALCULATION_COULD_NOT_BE_REPRODUCED,
                str(exc),
            )
        if reproduced.to_dict() != request.calculation.to_dict():
            return _abstain(
                request,
                MultiDocumentGenerationAbstentionReason.CALCULATION_RESULT_MISMATCH,
                "supplied calculation does not match independent Decimal reproduction",
            )
        citations = tuple(
            SourceSpecificCitation.from_binding(by_role[role]) for role in expected_roles
        )
        return MultiDocumentGenerationValidationResult(
            MultiDocumentGenerationValidationStatus.VALIDATED,
            request,
            reproduced,
            citations,
        )


__all__ = [
    "MULTI_DOCUMENT_GENERATION_SCHEMA_VERSION",
    "AnnualReportIdentity",
    "MultiDocumentGenerationAbstentionReason",
    "MultiDocumentGenerationError",
    "MultiDocumentGenerationRequest",
    "MultiDocumentGenerationValidationResult",
    "MultiDocumentGenerationValidationStatus",
    "MultiDocumentGenerationValidator",
    "SourceFactBinding",
    "SourceSpecificCitation",
]
