from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal

from ledgerlens.financial_facts import (
    Currency,
    FinancialFact,
    PeriodType,
    ReportingPeriod,
    ResolutionStatus,
    SupportingEvidence,
    TableColumn,
    UnitKind,
    Units,
    Verification,
    VerificationStatus,
)
from ledgerlens.generation import AnswerGenerator, AnswerStatus, InferenceConfig, ModelIdentity
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationAbstentionReason,
    OrchestrationRequest,
    OrchestrationStatus,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _period(start: date, end: date, label: str) -> ReportingPeriod:
    return ReportingPeriod(ResolutionStatus.KNOWN, PeriodType.DURATION, start, end, label)


def _fixture() -> tuple[EvidenceCorpus, dict[str, FinancialFact]]:
    rows = (("Revenue", "$150"), ("Revenue", "$100"), ("Net income", "$30"))
    pages = []
    for number, (metric, value) in enumerate(rows, 1):
        text = f"Controlled statement\n{metric}: {value}"
        pages.append(
            EvidencePage(
                document_id="sha256-" + "a" * 64,
                company="Synthetic Company",
                report_year=2026,
                physical_page_number=number,
                text=text,
                text_sha256=_sha256(text),
                method="canonical_extraction",
                provenance={"source_sha256": "a" * 64},
                screening_findings={"canonical": {"status": "pass", "warnings": []}},
                review_provenance=None,
                known_limitations=("Controlled fixture only.",),
            )
        )
    corpus = EvidenceCorpus(
        "sha256-" + "a" * 64,
        "a" * 64,
        tuple(pages),
        (),
        {},
        _sha256("evidence"),
        _sha256("manifest"),
    )
    current = _period(date(2025, 1, 1), date(2025, 12, 31), "FY2025")
    prior = _period(date(2024, 1, 1), date(2024, 12, 31), "FY2024")

    def fact(
        page: int, fact_id: str, metric: str, value: str, normalized: str, period: ReportingPeriod
    ) -> FinancialFact:
        evidence_page = corpus.pages[page - 1]
        return FinancialFact(
            fact_id,
            metric,
            value,
            Decimal(normalized),
            evidence_page.company,
            period,
            Currency(ResolutionStatus.KNOWN, "USD"),
            Units(
                ResolutionStatus.KNOWN,
                UnitKind.MONETARY,
                Decimal("1000000"),
                "USD millions",
            ),
            TableColumn(ResolutionStatus.KNOWN, period.label),
            SupportingEvidence.from_evidence(corpus, evidence_page, evidence_page.text),
            Verification(
                VerificationStatus.VERIFIED,
                "fixture reviewer",
                "2026-10-26T00:00:00Z",
                "controlled offline fixture",
            ),
            limitations=("Controlled fixture only.",),
        )

    return corpus, {
        "current_revenue": fact(1, "revenue-2025", "revenue", "$150", "150", current),
        "prior_revenue": fact(2, "revenue-2024", "revenue", "$100", "100", prior),
        "net_income": fact(3, "net-income-2025", "net_income", "$30", "30", current),
    }


@dataclass
class MockBackend:
    response: str
    calls: int = 0
    identity: ModelIdentity = ModelIdentity("mock", "controlled")

    def generate(self, _prompt: str, _config: InferenceConfig) -> str:
        self.calls += 1
        return self.response


def _request(
    corpus: EvidenceCorpus,
    facts: dict[str, FinancialFact],
    question: str,
    selected: tuple[FinancialFact, ...] | None = None,
) -> OrchestrationRequest:
    return OrchestrationRequest(
        question,
        "Synthetic Company",
        date(2025, 12, 31),
        selected or tuple(facts.values()),
        corpus,
    )


def test_orchestrates_revenue_growth_with_auditable_selection() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("Revenue increased from the prior fiscal year.")
    orchestrator = FinancialAnswerOrchestrator(AnswerGenerator(backend))

    result = orchestrator.answer(_request(corpus, facts, "How much did revenue grow?"))

    assert result.status is OrchestrationStatus.ANSWERED
    assert result.selected_fact_ids == ("revenue-2025", "revenue-2024")
    assert result.calculation is not None
    assert result.calculation.computed_value == Decimal("50.0")
    assert result.answer is not None and result.answer.status is AnswerStatus.ANSWERED
    assert backend.calls == 1
    decisions = result.to_dict()["selection_audit"]["decisions"]  # type: ignore[index]
    assert decisions[0]["role"] == "current"  # type: ignore[index]
    assert decisions[0]["candidate_fact_ids"] == ["revenue-2025"]  # type: ignore[index]
    assert decisions[1]["role"] == "prior"  # type: ignore[index]


def test_orchestrates_net_profit_margin_for_exact_period() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("Net income represented a portion of revenue for the fiscal year.")

    result = FinancialAnswerOrchestrator(AnswerGenerator(backend)).answer(
        _request(corpus, facts, "What was the net profit margin?")
    )

    assert result.status is OrchestrationStatus.ANSWERED
    assert result.selected_fact_ids == ("net-income-2025", "revenue-2025")
    assert result.calculation is not None
    assert result.calculation.computed_value == Decimal("20.0")


def test_unsupported_and_ambiguous_tasks_abstain_before_generation() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("unused")
    orchestrator = FinancialAnswerOrchestrator(AnswerGenerator(backend))

    unsupported = orchestrator.answer(_request(corpus, facts, "What were operating expenses?"))
    ambiguous = orchestrator.answer(
        _request(corpus, facts, "Compare revenue growth and net profit margin.")
    )

    assert unsupported.abstention_reason is OrchestrationAbstentionReason.UNSUPPORTED_TASK
    assert ambiguous.abstention_reason is OrchestrationAbstentionReason.AMBIGUOUS_TASK
    assert len(ambiguous.selection_audit.matched_tasks) == 2
    assert backend.calls == 0


def test_missing_ambiguous_and_unverified_facts_abstain_auditably() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("unused")
    orchestrator = FinancialAnswerOrchestrator(AnswerGenerator(backend))
    growth_question = "How much did revenue grow?"
    missing = orchestrator.answer(
        _request(corpus, facts, growth_question, (facts["current_revenue"],))
    )
    duplicate_current = replace(facts["current_revenue"], fact_id="revenue-2025-copy")
    ambiguous = orchestrator.answer(
        _request(corpus, facts, growth_question, (*facts.values(), duplicate_current))
    )
    unverified_current = replace(
        facts["current_revenue"], verification=Verification(VerificationStatus.UNVERIFIED)
    )
    unverified = orchestrator.answer(
        _request(
            corpus,
            facts,
            growth_question,
            (unverified_current, facts["prior_revenue"]),
        )
    )

    assert missing.abstention_reason is OrchestrationAbstentionReason.MISSING_FACT
    assert ambiguous.abstention_reason is OrchestrationAbstentionReason.AMBIGUOUS_FACT
    assert ambiguous.selection_audit.decisions[0].candidate_fact_ids == (
        "revenue-2025",
        "revenue-2025-copy",
    )
    assert unverified.abstention_reason is OrchestrationAbstentionReason.UNVERIFIED_FACT
    assert backend.calls == 0


def test_unresolved_period_and_duplicate_fact_ids_abstain() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("unused")
    orchestrator = FinancialAnswerOrchestrator(AnswerGenerator(backend))
    unresolved = replace(
        facts["current_revenue"],
        reporting_period=ReportingPeriod(
            ResolutionStatus.AMBIGUOUS,
            note="Controlled fixture has no selected reporting-period interpretation.",
        ),
        verification=Verification(VerificationStatus.UNVERIFIED),
        warnings=("Reporting period is ambiguous in this controlled negative.",),
    )
    unresolved_result = orchestrator.answer(
        _request(
            corpus,
            facts,
            "How much did revenue grow?",
            (unresolved, facts["prior_revenue"]),
        )
    )
    duplicate_result = orchestrator.answer(
        _request(
            corpus,
            facts,
            "How much did revenue grow?",
            (facts["current_revenue"], facts["current_revenue"]),
        )
    )

    assert unresolved_result.abstention_reason is (
        OrchestrationAbstentionReason.UNRESOLVED_FACT_CONTEXT
    )
    assert unresolved_result.selection_audit.decisions[0].candidate_fact_ids == ("revenue-2025",)
    assert duplicate_result.abstention_reason is OrchestrationAbstentionReason.DUPLICATE_FACT_ID
    assert backend.calls == 0


def test_stale_or_incompatible_selected_facts_abstain_before_model() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("unused")
    orchestrator = FinancialAnswerOrchestrator(AnswerGenerator(backend))
    request = _request(corpus, facts, "What was the net profit margin?")
    stale = orchestrator.answer(replace(request, corpus=replace(corpus, manifest_sha256="b" * 64)))
    incompatible_revenue = replace(
        facts["current_revenue"], currency=Currency(ResolutionStatus.KNOWN, "EUR")
    )
    incompatible = orchestrator.answer(
        replace(request, facts=(facts["net_income"], incompatible_revenue))
    )

    assert stale.abstention_reason is OrchestrationAbstentionReason.INVALID_OR_STALE_EVIDENCE
    assert incompatible.abstention_reason is (
        OrchestrationAbstentionReason.INCOMPATIBLE_FINANCIAL_FACTS
    )
    assert backend.calls == 0


def test_generation_grounding_abstention_is_preserved() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("Revenue increased because of demand.")

    result = FinancialAnswerOrchestrator(AnswerGenerator(backend)).answer(
        _request(corpus, facts, "How much did revenue grow?")
    )

    assert result.status is OrchestrationStatus.ABSTAINED
    assert result.abstention_reason is OrchestrationAbstentionReason.ANSWER_GENERATION_ABSTAINED
    assert result.detail == "model_output_failed_grounding"
    assert result.answer is not None
    assert result.answer.provenance.grounding_diagnostic is not None
    assert result.calculation is not None
    assert backend.calls == 1
