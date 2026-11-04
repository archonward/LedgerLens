from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ledgerlens.calculations import CalculationType
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
from ledgerlens.generation import AnswerTask
from ledgerlens.multi_document_orchestration import (
    GenerationBoundaryReason,
    MultiDocumentAbstentionReason,
    MultiDocumentFinancialAnswerOrchestrator,
    MultiDocumentOrchestrationRequest,
    MultiDocumentOrchestrationStatus,
)
from ledgerlens.reconciliation import (
    ReconciliationReport,
    ReconciliationResult,
    ReconciliationStatus,
    VerifiedFactSource,
    reconcile_financial_facts,
)
from ledgerlens.reconciliation_selection import (
    FactSelectionRequest,
    ReconciliationSelectionPolicy,
    SelectionAbstentionReason,
    SingleSourcePolicy,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage, load_evidence_export

CURRENT_PERIOD = ReportingPeriod(
    ResolutionStatus.KNOWN,
    PeriodType.DURATION,
    date(2025, 1, 1),
    date(2025, 12, 31),
    "FY2025",
)
PRIOR_PERIOD = ReportingPeriod(
    ResolutionStatus.KNOWN,
    PeriodType.DURATION,
    date(2024, 1, 1),
    date(2024, 12, 31),
    "FY2024",
)


def _corpus(suffix: str, values: tuple[str, str, str]) -> EvidenceCorpus:
    document_id = f"sha256-{suffix * 64}"
    pages = []
    rows = (
        ("Revenue", values[0]),
        ("Revenue", values[1]),
        ("Net income", values[2]),
    )
    for number, (metric, value) in enumerate(rows, 1):
        text = f"Controlled statement\n{metric}: {value}"
        pages.append(
            EvidencePage(
                document_id=document_id,
                company="Synthetic Company",
                report_year=2026,
                physical_page_number=number,
                text=text,
                text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                method="canonical_extraction",
                provenance={"source_sha256": suffix * 64},
                screening_findings={},
                review_provenance=None,
                known_limitations=("Controlled synthetic fixture only.",),
            )
        )
    return EvidenceCorpus(
        document_id=document_id,
        source_sha256=suffix * 64,
        pages=tuple(pages),
        excluded_pages=(),
        manifest={},
        evidence_jsonl_sha256=("a" if suffix != "a" else "c") * 64,
        manifest_sha256=("b" if suffix != "b" else "d") * 64,
    )


def _fact(
    corpus: EvidenceCorpus,
    page_number: int,
    fact_id: str,
    metric: str,
    reported_value: str,
    normalized_value: str,
    period: ReportingPeriod,
) -> FinancialFact:
    page = corpus.pages[page_number - 1]
    return FinancialFact(
        fact_id=fact_id,
        metric=metric,
        reported_value=reported_value,
        normalized_value=Decimal(normalized_value),
        company="Synthetic Company",
        reporting_period=period,
        currency=Currency(ResolutionStatus.KNOWN, "USD"),
        units=Units(
            ResolutionStatus.KNOWN,
            UnitKind.MONETARY,
            Decimal("1000000"),
            "USD millions",
        ),
        table_column=TableColumn(ResolutionStatus.KNOWN, period.label),
        evidence=SupportingEvidence.from_evidence(corpus, page, page.text),
        verification=Verification(
            VerificationStatus.VERIFIED,
            reviewed_by="Synthetic reviewer",
            reviewed_at_utc="2026-11-04T00:00:00Z",
            method="Controlled offline fixture review",
            note="Synthetic fixture only.",
        ),
        limitations=("Controlled synthetic fixture only.",),
    )


def _facts(
    corpus: EvidenceCorpus,
    prefix: str,
    values: tuple[str, str, str] = ("$150", "$100", "$30"),
    normalized: tuple[str, str, str] = ("150", "100", "30"),
) -> tuple[FinancialFact, FinancialFact, FinancialFact]:
    return (
        _fact(
            corpus,
            1,
            f"{prefix}-current-revenue",
            "revenue",
            values[0],
            normalized[0],
            CURRENT_PERIOD,
        ),
        _fact(
            corpus,
            2,
            f"{prefix}-prior-revenue",
            "revenue",
            values[1],
            normalized[1],
            PRIOR_PERIOD,
        ),
        _fact(
            corpus,
            3,
            f"{prefix}-net-income",
            "net_income",
            values[2],
            normalized[2],
            CURRENT_PERIOD,
        ),
    )


def _report(
    tmp_path: Path,
    sources: tuple[tuple[EvidenceCorpus, tuple[FinancialFact, ...]], ...],
) -> tuple[ReconciliationReport, dict[str, EvidenceCorpus]]:
    fact_sources = []
    corpora = {}
    for index, (corpus, facts) in enumerate(sources):
        path = tmp_path / f"source-{index}.jsonl"
        path.write_text(
            "".join(json.dumps(fact.to_dict()) + "\n" for fact in facts),
            encoding="utf-8",
        )
        fact_sources.append(VerifiedFactSource(path, corpus))
        corpora[corpus.document_id] = corpus
    return reconcile_financial_facts(fact_sources), corpora


def _agreement(
    report: ReconciliationReport,
    metric: str,
    period_end: date,
) -> ReconciliationResult:
    matches = [
        result
        for result in report.results
        if result.status is ReconciliationStatus.AGREEMENT
        and result.metric == metric
        and result.comparison_key is not None
        and result.comparison_key.end_date == period_end.isoformat()
    ]
    assert len(matches) == 1
    return matches[0]


def _request(
    task: AnswerTask,
    selections: tuple[FactSelectionRequest, FactSelectionRequest],
    *,
    policy: ReconciliationSelectionPolicy = ReconciliationSelectionPolicy(),
) -> MultiDocumentOrchestrationRequest:
    return MultiDocumentOrchestrationRequest(
        question=(
            "How much did revenue grow?"
            if task is AnswerTask.REVENUE_GROWTH
            else "What was the net profit margin?"
        ),
        task=task,
        company="Synthetic Company",
        fact_selections=selections,
        selection_policy=policy,
    )


def _two_source_fixture(
    tmp_path: Path,
) -> tuple[ReconciliationReport, dict[str, EvidenceCorpus]]:
    left = _corpus("1", ("$150", "$100", "$30"))
    right = _corpus("2", ("$150", "$100", "$30"))
    return _report(tmp_path, ((left, _facts(left, "left")), (right, _facts(right, "right"))))


def test_cross_document_growth_and_margin_stop_at_structured_calculation(
    tmp_path: Path,
) -> None:
    report, corpora = _two_source_fixture(tmp_path)
    current = _agreement(report, "revenue", CURRENT_PERIOD.end_date)
    prior = _agreement(report, "revenue", PRIOR_PERIOD.end_date)
    profit = _agreement(report, "net_income", CURRENT_PERIOD.end_date)
    orchestrator = MultiDocumentFinancialAnswerOrchestrator()

    growth = orchestrator.orchestrate(
        _request(
            AnswerTask.REVENUE_GROWTH,
            (
                FactSelectionRequest("prior", prior.result_id, "right-prior-revenue"),
                FactSelectionRequest("current", current.result_id, "left-current-revenue"),
            ),
        ),
        report,
        corpora,
    )
    margin = orchestrator.orchestrate(
        _request(
            AnswerTask.NET_PROFIT_MARGIN,
            (
                FactSelectionRequest("profit", profit.result_id, "left-net-income"),
                FactSelectionRequest("revenue", current.result_id, "right-current-revenue"),
            ),
        ),
        report,
        corpora,
    )

    assert growth.status is MultiDocumentOrchestrationStatus.CALCULATED
    assert growth.calculation is not None
    assert growth.calculation.calculation_type is CalculationType.YEAR_OVER_YEAR_GROWTH
    assert growth.calculation.computed_value == Decimal("50.0")
    assert margin.calculation is not None
    assert margin.calculation.calculation_type is CalculationType.PROFIT_MARGIN
    assert margin.calculation.computed_value == Decimal("20.0")
    assert growth.generation_boundary_reason is GenerationBoundaryReason.SINGLE_CORPUS_INTERFACE
    payload = growth.to_dict()
    assert payload["generation"]["model_invoked"] is False  # type: ignore[index]
    assert payload["selected_facts"][0]["evidence"]["physical_page_number"] == 1  # type: ignore[index]
    assert payload["selected_facts"][0]["verification"]["method"]  # type: ignore[index]
    assert payload["selected_facts"][0]["limitations"] == [  # type: ignore[index]
        "Controlled synthetic fixture only."
    ]
    selected_documents = {fact.evidence.document_id for fact in growth.selected_facts}
    assert len(selected_documents) == 2


def test_conflict_selection_abstention_is_preserved(tmp_path: Path) -> None:
    left = _corpus("3", ("$150", "$100", "$30"))
    right = _corpus("4", ("$151", "$100", "$30"))
    report, corpora = _report(
        tmp_path,
        (
            (left, _facts(left, "left")),
            (right, _facts(right, "right", ("$151", "$100", "$30"), ("151", "100", "30"))),
        ),
    )
    conflict = next(
        result
        for result in report.results
        if result.status is ReconciliationStatus.CONFLICT and result.metric == "revenue"
    )
    prior = _agreement(report, "revenue", PRIOR_PERIOD.end_date)

    result = MultiDocumentFinancialAnswerOrchestrator().orchestrate(
        _request(
            AnswerTask.REVENUE_GROWTH,
            (
                FactSelectionRequest("current", conflict.result_id, "left-current-revenue"),
                FactSelectionRequest("prior", prior.result_id, "right-prior-revenue"),
            ),
        ),
        report,
        corpora,
    )

    assert result.abstention_reason is MultiDocumentAbstentionReason.FACT_SELECTION_ABSTAINED
    assert result.selection is not None
    assert result.selection.decisions[0].abstention_reason is SelectionAbstentionReason.CONFLICT
    assert result.calculation is None


def test_duplicate_fact_id_ambiguity_is_preserved(tmp_path: Path) -> None:
    left = _corpus("5", ("$150", "$100", "$30"))
    right = _corpus("6", ("$150", "$100", "$30"))
    left_facts = _facts(left, "shared")
    right_facts = _facts(right, "shared")
    report, corpora = _report(tmp_path, ((left, left_facts), (right, right_facts)))
    current = _agreement(report, "revenue", CURRENT_PERIOD.end_date)
    prior = _agreement(report, "revenue", PRIOR_PERIOD.end_date)

    result = MultiDocumentFinancialAnswerOrchestrator().orchestrate(
        _request(
            AnswerTask.REVENUE_GROWTH,
            (
                FactSelectionRequest("current", current.result_id, "shared-current-revenue"),
                FactSelectionRequest("prior", prior.result_id, "shared-prior-revenue"),
            ),
        ),
        report,
        corpora,
    )

    assert result.selection is not None
    assert all(
        decision.abstention_reason is SelectionAbstentionReason.DUPLICATE_AMBIGUITY
        for decision in result.selection.decisions
    )
    assert result.calculation is None


def test_single_source_requires_explicit_allow_policy(tmp_path: Path) -> None:
    corpus = _corpus("7", ("$150", "$100", "$30"))
    facts = _facts(corpus, "single")
    report, corpora = _report(tmp_path, ((corpus, facts),))
    current = next(
        result
        for result in report.results
        if result.status is ReconciliationStatus.INSUFFICIENT_EVIDENCE
        and result.metric == "revenue"
        and result.comparison_key is not None
        and result.comparison_key.end_date == CURRENT_PERIOD.end_date.isoformat()
    )
    prior = next(
        result
        for result in report.results
        if result.status is ReconciliationStatus.INSUFFICIENT_EVIDENCE
        and result.metric == "revenue"
        and result.comparison_key is not None
        and result.comparison_key.end_date == PRIOR_PERIOD.end_date.isoformat()
    )
    selections = (
        FactSelectionRequest("current", current.result_id, "single-current-revenue"),
        FactSelectionRequest("prior", prior.result_id, "single-prior-revenue"),
    )
    orchestrator = MultiDocumentFinancialAnswerOrchestrator()

    default = orchestrator.orchestrate(
        _request(AnswerTask.REVENUE_GROWTH, selections), report, corpora
    )
    allowed = orchestrator.orchestrate(
        _request(
            AnswerTask.REVENUE_GROWTH,
            selections,
            policy=ReconciliationSelectionPolicy(SingleSourcePolicy.ALLOW_EXPLICIT),
        ),
        report,
        corpora,
    )

    assert default.abstention_reason is MultiDocumentAbstentionReason.FACT_SELECTION_ABSTAINED
    assert allowed.status is MultiDocumentOrchestrationStatus.CALCULATED
    assert allowed.selection is not None
    assert all(
        decision.reconciliation_status is ReconciliationStatus.INSUFFICIENT_EVIDENCE
        for decision in allowed.selection.decisions
    )


def test_incompatible_periods_and_stale_corroboration_abstain(tmp_path: Path) -> None:
    report, corpora = _two_source_fixture(tmp_path)
    current = _agreement(report, "revenue", CURRENT_PERIOD.end_date)
    same_period_request = _request(
        AnswerTask.REVENUE_GROWTH,
        (
            FactSelectionRequest("current", current.result_id, "left-current-revenue"),
            FactSelectionRequest("prior", current.result_id, "right-current-revenue"),
        ),
    )
    orchestrator = MultiDocumentFinancialAnswerOrchestrator()

    incompatible = orchestrator.orchestrate(same_period_request, report, corpora)
    stale_corpora = dict(corpora)
    right_document = next(key for key in corpora if key.endswith("2" * 64))
    stale_corpora[right_document] = replace(corpora[right_document], manifest_sha256="0" * 64)
    stale = orchestrator.orchestrate(same_period_request, report, stale_corpora)

    assert incompatible.abstention_reason is (
        MultiDocumentAbstentionReason.INCOMPATIBLE_FINANCIAL_FACTS
    )
    assert "consecutive fiscal years" in (incompatible.detail or "")
    assert stale.abstention_reason is MultiDocumentAbstentionReason.FACT_SELECTION_ABSTAINED
    assert stale.selection is not None
    assert all(
        decision.abstention_reason is SelectionAbstentionReason.INVALID_OR_STALE_EVIDENCE
        for decision in stale.selection.decisions
    )


def test_ambiguous_result_company_and_metric_mismatches_abstain(tmp_path: Path) -> None:
    report, corpora = _two_source_fixture(tmp_path)
    current = _agreement(report, "revenue", CURRENT_PERIOD.end_date)
    prior = _agreement(report, "revenue", PRIOR_PERIOD.end_date)
    profit = _agreement(report, "net_income", CURRENT_PERIOD.end_date)
    selections = (
        FactSelectionRequest("current", current.result_id, "left-current-revenue"),
        FactSelectionRequest("prior", prior.result_id, "right-prior-revenue"),
    )
    orchestrator = MultiDocumentFinancialAnswerOrchestrator()

    ambiguous_report = replace(report, results=(*report.results, current))
    ambiguous = orchestrator.orchestrate(
        _request(AnswerTask.REVENUE_GROWTH, selections),
        ambiguous_report,
        corpora,
    )
    wrong_company = orchestrator.orchestrate(
        replace(
            _request(AnswerTask.REVENUE_GROWTH, selections),
            company="Other Company",
        ),
        report,
        corpora,
    )
    wrong_metric = orchestrator.orchestrate(
        _request(
            AnswerTask.REVENUE_GROWTH,
            (
                FactSelectionRequest("current", profit.result_id, "left-net-income"),
                FactSelectionRequest("prior", prior.result_id, "right-prior-revenue"),
            ),
        ),
        report,
        corpora,
    )

    assert ambiguous.selection is not None
    assert ambiguous.selection.decisions[0].abstention_reason is (
        SelectionAbstentionReason.AMBIGUOUS_RESULT_ID
    )
    assert wrong_company.abstention_reason is MultiDocumentAbstentionReason.COMPANY_MISMATCH
    assert wrong_metric.abstention_reason is MultiDocumentAbstentionReason.METRIC_MISMATCH


def test_real_nvidia_multi_report_orchestration_when_artifacts_exist() -> None:
    root = Path(__file__).parents[1]
    document_ids = (
        "sha256-b5751f4bb240eeb63d119211d6cc495e1d623897f1f2af1887e387bad1404c84",
        "sha256-757b344ab62ba282f6e44e0702e8ce5f190d88fb23b5e0a2beaf4cb277a395cc",
        "sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
    )
    evidence_paths = [root / "artifacts" / "evidence" / item for item in document_ids]
    fact_paths = [
        root / "artifacts" / "financial-facts" / item / "milestone-8b-verified-facts.jsonl"
        for item in document_ids
    ]
    fact_paths.append(root / "artifacts" / "financial-facts" / document_ids[2] / "facts.jsonl")
    required = [
        *(path / "manifest.json" for path in evidence_paths),
        *(path / "evidence.jsonl" for path in evidence_paths),
        *fact_paths,
    ]
    if not all(path.is_file() for path in required):
        pytest.skip("Git-ignored Milestone 8B artifacts are not present")
    corpora_values = [load_evidence_export(path) for path in evidence_paths]
    report = reconcile_financial_facts(
        (
            VerifiedFactSource(fact_paths[0], corpora_values[0]),
            VerifiedFactSource(fact_paths[1], corpora_values[1]),
            VerifiedFactSource(fact_paths[2], corpora_values[2]),
            VerifiedFactSource(fact_paths[3], corpora_values[2]),
        )
    )
    corpora = {corpus.document_id: corpus for corpus in corpora_values}
    fy2025 = _agreement(report, "revenue", date(2025, 1, 26))
    fy2024_revenue = _agreement(report, "revenue", date(2024, 1, 28))
    fy2024_profit = _agreement(report, "net_income", date(2024, 1, 28))
    orchestrator = MultiDocumentFinancialAnswerOrchestrator()

    growth = orchestrator.orchestrate(
        MultiDocumentOrchestrationRequest(
            "How much did NVIDIA revenue grow?",
            AnswerTask.REVENUE_GROWTH,
            "NVIDIA",
            (
                FactSelectionRequest(
                    "current", fy2025.result_id, "nvidia-fy2025-report-fy2025-revenue"
                ),
                FactSelectionRequest(
                    "prior", fy2024_revenue.result_id, "nvidia-fy2024-report-fy2024-revenue"
                ),
            ),
        ),
        report,
        corpora,
    )
    margin = orchestrator.orchestrate(
        MultiDocumentOrchestrationRequest(
            "What was NVIDIA's FY2024 net profit margin?",
            AnswerTask.NET_PROFIT_MARGIN,
            "NVIDIA",
            (
                FactSelectionRequest(
                    "profit", fy2024_profit.result_id, "nvidia-fy2024-report-fy2024-net-income"
                ),
                FactSelectionRequest(
                    "revenue", fy2024_revenue.result_id, "nvidia-fy2026-report-fy2024-revenue"
                ),
            ),
        ),
        report,
        corpora,
    )

    assert growth.status is MultiDocumentOrchestrationStatus.CALCULATED
    assert margin.status is MultiDocumentOrchestrationStatus.CALCULATED
    assert growth.calculation is not None and margin.calculation is not None
    assert len({fact.evidence.document_id for fact in growth.selected_facts}) == 2
    assert len({fact.evidence.document_id for fact in margin.selected_facts}) == 2
    assert growth.to_dict()["generation"]["model_invoked"] is False  # type: ignore[index]
