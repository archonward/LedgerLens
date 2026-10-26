"""Narrow deterministic orchestration from reviewed facts to grounded answers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

from ledgerlens.calculations import (
    CalculationError,
    CalculationResult,
    FactOperand,
    profit_margin,
    year_over_year_growth,
)
from ledgerlens.financial_facts import (
    FinancialFact,
    FinancialFactValidationError,
    ResolutionStatus,
    VerificationStatus,
)
from ledgerlens.generation import (
    AnswerGenerator,
    AnswerRequest,
    AnswerStatus,
    AnswerTask,
    GeneratedAnswer,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus

ORCHESTRATION_SCHEMA_VERSION = "1.0"
_WORDS = re.compile(r"[A-Za-z]+")


class OrchestrationStatus(StrEnum):
    ANSWERED = "answered"
    ABSTAINED = "abstained"


class OrchestrationAbstentionReason(StrEnum):
    UNSUPPORTED_TASK = "unsupported_task"
    AMBIGUOUS_TASK = "ambiguous_task"
    DUPLICATE_FACT_ID = "duplicate_fact_id"
    MISSING_FACT = "missing_fact"
    AMBIGUOUS_FACT = "ambiguous_fact"
    UNRESOLVED_FACT_CONTEXT = "unresolved_fact_context"
    UNVERIFIED_FACT = "unverified_fact"
    INVALID_OR_STALE_EVIDENCE = "invalid_or_stale_evidence"
    INCOMPATIBLE_FINANCIAL_FACTS = "incompatible_financial_facts"
    ANSWER_GENERATION_ABSTAINED = "answer_generation_abstained"


@dataclass(frozen=True)
class OrchestrationRequest:
    """Explicit scope and reviewed fact collection for one question."""

    question: str
    company: str
    target_period_end: date
    facts: tuple[FinancialFact, ...]
    corpus: EvidenceCorpus

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")
        if not isinstance(self.company, str) or not self.company.strip():
            raise ValueError("company must be a non-empty string")
        if not isinstance(self.target_period_end, date):
            raise ValueError("target_period_end must be a date")


@dataclass(frozen=True)
class FactSelectionDecision:
    """Auditable candidates and outcome for one required calculation role."""

    role: str
    metric: str
    period_rule: str
    candidate_fact_ids: tuple[str, ...]
    selected_fact_id: str | None
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "metric": self.metric,
            "period_rule": self.period_rule,
            "candidate_fact_ids": list(self.candidate_fact_ids),
            "selected_fact_id": self.selected_fact_id,
            "outcome": self.outcome,
        }


@dataclass(frozen=True)
class SelectionAudit:
    """Task matching and fact-selection trace retained independently of model output."""

    question_tokens: tuple[str, ...]
    matched_tasks: tuple[AnswerTask, ...]
    company: str
    target_period_end: date
    available_fact_ids: tuple[str, ...]
    decisions: tuple[FactSelectionDecision, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "question_tokens": list(self.question_tokens),
            "matched_tasks": [task.value for task in self.matched_tasks],
            "company": self.company,
            "target_period_end": self.target_period_end.isoformat(),
            "available_fact_ids": list(self.available_fact_ids),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


@dataclass(frozen=True)
class OrchestrationResult:
    """A grounded generated answer or structured orchestration abstention."""

    status: OrchestrationStatus
    question: str
    task: AnswerTask | None
    selected_fact_ids: tuple[str, ...]
    selection_audit: SelectionAudit
    calculation: CalculationResult | None
    answer: GeneratedAnswer | None
    abstention_reason: OrchestrationAbstentionReason | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": ORCHESTRATION_SCHEMA_VERSION,
            "status": self.status.value,
            "question": self.question,
            "task": self.task.value if self.task else None,
            "selected_fact_ids": list(self.selected_fact_ids),
            "selection_audit": self.selection_audit.to_dict(),
            "calculation": self.calculation.to_dict() if self.calculation else None,
            "answer": self.answer.to_dict() if self.answer else None,
            "abstention_reason": self.abstention_reason.value if self.abstention_reason else None,
            "detail": self.detail,
        }


class FinancialAnswerOrchestrator:
    """Select reviewed facts and invoke existing calculation and generation boundaries."""

    def __init__(self, generator: AnswerGenerator) -> None:
        self._generator = generator

    def answer(self, request: OrchestrationRequest) -> OrchestrationResult:
        tokens, matched = _identify_tasks(request.question)
        audit = SelectionAudit(
            tokens,
            matched,
            request.company,
            request.target_period_end,
            tuple(sorted(fact.fact_id for fact in request.facts)),
            (),
        )
        if not matched:
            return _abstain(request, audit, None, OrchestrationAbstentionReason.UNSUPPORTED_TASK)
        if len(matched) != 1:
            return _abstain(request, audit, None, OrchestrationAbstentionReason.AMBIGUOUS_TASK)
        task = matched[0]
        if len({fact.fact_id for fact in request.facts}) != len(request.facts):
            return _abstain(request, audit, task, OrchestrationAbstentionReason.DUPLICATE_FACT_ID)

        selected, decisions, error = self._select(request, task)
        audit = SelectionAudit(
            tokens,
            matched,
            request.company,
            request.target_period_end,
            tuple(sorted(fact.fact_id for fact in request.facts)),
            decisions,
        )
        if error is not None:
            return _abstain(request, audit, task, error)
        assert selected is not None
        for fact in selected:
            if fact.verification.status is not VerificationStatus.VERIFIED:
                return _abstain(
                    request,
                    audit,
                    task,
                    OrchestrationAbstentionReason.UNVERIFIED_FACT,
                    selected,
                )
            try:
                fact.validate_against(request.corpus)
            except FinancialFactValidationError as exc:
                return _abstain(
                    request,
                    audit,
                    task,
                    OrchestrationAbstentionReason.INVALID_OR_STALE_EVIDENCE,
                    selected,
                    str(exc),
                )
        try:
            calculation = _calculate(task, selected, request.corpus)
        except CalculationError as exc:
            return _abstain(
                request,
                audit,
                task,
                OrchestrationAbstentionReason.INCOMPATIBLE_FINANCIAL_FACTS,
                selected,
                str(exc),
            )
        answer = self._generator.generate(
            AnswerRequest(
                request.question,
                task,
                selected,
                calculation,
                request.corpus,
            )
        )
        if answer.status is AnswerStatus.ABSTAINED:
            return OrchestrationResult(
                OrchestrationStatus.ABSTAINED,
                request.question,
                task,
                tuple(fact.fact_id for fact in selected),
                audit,
                calculation,
                answer,
                OrchestrationAbstentionReason.ANSWER_GENERATION_ABSTAINED,
                answer.abstention_reason,
            )
        return OrchestrationResult(
            OrchestrationStatus.ANSWERED,
            request.question,
            task,
            tuple(fact.fact_id for fact in selected),
            audit,
            calculation,
            answer,
        )

    def _select(
        self, request: OrchestrationRequest, task: AnswerTask
    ) -> tuple[
        tuple[FinancialFact, FinancialFact] | None,
        tuple[FactSelectionDecision, ...],
        OrchestrationAbstentionReason | None,
    ]:
        company_facts = tuple(
            fact for fact in request.facts if fact.company.casefold() == request.company.casefold()
        )
        current, current_decision, error = _select_exact_period(
            company_facts,
            "current" if task is AnswerTask.REVENUE_GROWTH else "revenue",
            "revenue",
            request.target_period_end,
        )
        if error:
            return None, (current_decision,), error
        assert current is not None
        if task is AnswerTask.REVENUE_GROWTH:
            if current.reporting_period.start_date is None:
                return (
                    None,
                    (current_decision,),
                    OrchestrationAbstentionReason.UNRESOLVED_FACT_CONTEXT,
                )
            prior_end = current.reporting_period.start_date - timedelta(days=1)
            second, second_decision, error = _select_exact_period(
                company_facts, "prior", "revenue", prior_end
            )
            selected = (current, second) if second else None
        else:
            second, second_decision, error = _select_exact_period(
                company_facts, "profit", "net_income", request.target_period_end
            )
            selected = (second, current) if second else None
        decisions = (current_decision, second_decision)
        return selected, decisions, error


def _identify_tasks(question: str) -> tuple[tuple[str, ...], tuple[AnswerTask, ...]]:
    tokens = tuple(word.casefold() for word in _WORDS.findall(question))
    words = set(tokens)
    matched: list[AnswerTask] = []
    if "revenue" in words and words & {
        "change",
        "changed",
        "grew",
        "grow",
        "growth",
        "increase",
        "increased",
    }:
        matched.append(AnswerTask.REVENUE_GROWTH)
    if "margin" in words and words & {"income", "net", "profit"}:
        matched.append(AnswerTask.NET_PROFIT_MARGIN)
    return tokens, tuple(matched)


def _select_exact_period(
    facts: tuple[FinancialFact, ...], role: str, metric: str, period_end: date
) -> tuple[FinancialFact | None, FactSelectionDecision, OrchestrationAbstentionReason | None]:
    broad = tuple(fact for fact in facts if fact.metric.casefold() == metric)
    candidates = tuple(
        fact
        for fact in broad
        if fact.reporting_period.status is ResolutionStatus.KNOWN
        and fact.reporting_period.end_date == period_end
    )
    rule = f"known reporting period ending {period_end.isoformat()}"
    if len(candidates) > 1:
        decision = FactSelectionDecision(
            role, metric, rule, tuple(sorted(f.fact_id for f in candidates)), None, "ambiguous"
        )
        return None, decision, OrchestrationAbstentionReason.AMBIGUOUS_FACT
    if not candidates:
        unresolved = tuple(
            fact for fact in broad if fact.reporting_period.status is not ResolutionStatus.KNOWN
        )
        reason = (
            OrchestrationAbstentionReason.UNRESOLVED_FACT_CONTEXT
            if unresolved
            else OrchestrationAbstentionReason.MISSING_FACT
        )
        decision = FactSelectionDecision(
            role,
            metric,
            rule,
            tuple(sorted(f.fact_id for f in unresolved)),
            None,
            reason.value,
        )
        return None, decision, reason
    fact = candidates[0]
    decision = FactSelectionDecision(role, metric, rule, (fact.fact_id,), fact.fact_id, "selected")
    return fact, decision, None


def _calculate(
    task: AnswerTask,
    facts: tuple[FinancialFact, FinancialFact],
    corpus: EvidenceCorpus,
) -> CalculationResult:
    if task is AnswerTask.REVENUE_GROWTH:
        return year_over_year_growth(FactOperand(facts[0], corpus), FactOperand(facts[1], corpus))
    return profit_margin(FactOperand(facts[0], corpus), FactOperand(facts[1], corpus))


def _abstain(
    request: OrchestrationRequest,
    audit: SelectionAudit,
    task: AnswerTask | None,
    reason: OrchestrationAbstentionReason,
    selected: tuple[FinancialFact, ...] = (),
    detail: str | None = None,
) -> OrchestrationResult:
    return OrchestrationResult(
        OrchestrationStatus.ABSTAINED,
        request.question,
        task,
        tuple(fact.fact_id for fact in selected),
        audit,
        None,
        None,
        reason,
        detail,
    )


__all__ = [
    "ORCHESTRATION_SCHEMA_VERSION",
    "FactSelectionDecision",
    "FinancialAnswerOrchestrator",
    "OrchestrationAbstentionReason",
    "OrchestrationRequest",
    "OrchestrationResult",
    "OrchestrationStatus",
    "SelectionAudit",
]
