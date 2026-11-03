from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ledgerlens.calculations import financial_ratio
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
    SelectionStatus,
    SingleSourcePolicy,
    prepare_calculation_facts,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage


def _corpus(suffix: str, reported_value: str) -> EvidenceCorpus:
    document_id = f"sha256-{suffix * 64}"
    source_sha256 = suffix * 64
    text = f"Controlled statement\nRevenue: {reported_value}"
    page = EvidencePage(
        document_id=document_id,
        company="Synthetic Company",
        report_year=2025,
        physical_page_number=7,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        method="canonical_extraction",
        provenance={"source_sha256": source_sha256},
        screening_findings={},
        review_provenance=None,
        known_limitations=("Controlled synthetic fixture only.",),
    )
    return EvidenceCorpus(
        document_id=document_id,
        source_sha256=source_sha256,
        pages=(page,),
        excluded_pages=(),
        manifest={},
        evidence_jsonl_sha256=("a" if suffix != "a" else "c") * 64,
        manifest_sha256=("b" if suffix != "b" else "d") * 64,
    )


def _period(year: int = 2025) -> ReportingPeriod:
    return ReportingPeriod(
        ResolutionStatus.KNOWN,
        PeriodType.DURATION,
        date(year - 1, 1, 1),
        date(year - 1, 12, 31),
        f"Year ended December 31, {year - 1}",
    )


def _fact(
    corpus: EvidenceCorpus,
    fact_id: str,
    reported_value: str,
    normalized_value: str,
    *,
    period: ReportingPeriod | None = None,
) -> FinancialFact:
    return FinancialFact(
        fact_id=fact_id,
        metric="revenue",
        reported_value=reported_value,
        normalized_value=Decimal(normalized_value),
        company="Synthetic Company",
        reporting_period=period or _period(),
        currency=Currency(ResolutionStatus.KNOWN, "USD"),
        units=Units(
            ResolutionStatus.KNOWN,
            UnitKind.MONETARY,
            Decimal("1000000"),
            "USD millions",
        ),
        table_column=TableColumn(ResolutionStatus.KNOWN, "2024"),
        evidence=SupportingEvidence.from_evidence(
            corpus,
            corpus.pages[0],
            corpus.pages[0].text,
        ),
        verification=Verification(
            VerificationStatus.VERIFIED,
            reviewed_by="Synthetic reviewer",
            reviewed_at_utc="2026-11-03T00:00:00Z",
            method="Controlled offline fixture review",
        ),
        limitations=("Controlled synthetic fixture only.",),
    )


def _report(
    tmp_path: Path,
    records: tuple[tuple[EvidenceCorpus, FinancialFact], ...],
) -> tuple[ReconciliationReport, dict[str, EvidenceCorpus]]:
    sources = []
    corpora = {}
    for index, (corpus, fact) in enumerate(records):
        path = tmp_path / f"facts-{index}.jsonl"
        path.write_text(json.dumps(fact.to_dict()) + "\n", encoding="utf-8")
        sources.append(VerifiedFactSource(path, corpus))
        corpora[corpus.document_id] = corpus
    return reconcile_financial_facts(sources), corpora


def _result(report: ReconciliationReport, status: ReconciliationStatus) -> ReconciliationResult:
    matches = [result for result in report.results if result.status is status]
    assert len(matches) == 1
    return matches[0]


def test_agreement_requires_explicit_fact_ids_and_prepares_existing_operands(
    tmp_path: Path,
) -> None:
    left_corpus = _corpus("1", "$100")
    right_corpus = _corpus("2", "$100")
    report, corpora = _report(
        tmp_path,
        (
            (left_corpus, _fact(left_corpus, "left-revenue", "$100", "100")),
            (right_corpus, _fact(right_corpus, "right-revenue", "$100", "100")),
        ),
    )
    agreement = _result(report, ReconciliationStatus.AGREEMENT)

    missing_choice = prepare_calculation_facts(
        report,
        (FactSelectionRequest("numerator", agreement.result_id),),
        corpora,
    )
    selection = prepare_calculation_facts(
        report,
        (
            FactSelectionRequest("numerator", agreement.result_id, "left-revenue"),
            FactSelectionRequest("denominator", agreement.result_id, "right-revenue"),
        ),
        corpora,
    )

    assert missing_choice.decisions[0].abstention_reason is (
        SelectionAbstentionReason.EXPLICIT_SOURCE_REQUIRED
    )
    assert selection.operands is not None
    assert [operand.fact.fact_id for operand in selection.operands] == [
        "left-revenue",
        "right-revenue",
    ]
    assert financial_ratio(*selection.operands, ratio_name="source comparison").computed_value == 1
    payload = selection.to_dict()
    assert payload["policy"]["agreement_source_rule"] == "explicit_fact_id"  # type: ignore[index]
    assert len(payload["decisions"][0]["candidate_observations"]) == 2  # type: ignore[index]


def test_conflict_abstains_even_with_an_explicit_source(tmp_path: Path) -> None:
    left_corpus = _corpus("3", "$100")
    right_corpus = _corpus("4", "$101")
    report, corpora = _report(
        tmp_path,
        (
            (left_corpus, _fact(left_corpus, "left-revenue", "$100", "100")),
            (right_corpus, _fact(right_corpus, "right-revenue", "$101", "101")),
        ),
    )
    conflict = _result(report, ReconciliationStatus.CONFLICT)

    selection = prepare_calculation_facts(
        report,
        (FactSelectionRequest("revenue", conflict.result_id, "left-revenue"),),
        corpora,
    )

    assert selection.operands is None
    assert selection.decisions[0].abstention_reason is SelectionAbstentionReason.CONFLICT


def test_single_source_defaults_to_abstention_and_can_be_explicitly_allowed(
    tmp_path: Path,
) -> None:
    corpus = _corpus("5", "$100")
    report, corpora = _report(
        tmp_path,
        ((corpus, _fact(corpus, "single-revenue", "$100", "100")),),
    )
    insufficient = _result(report, ReconciliationStatus.INSUFFICIENT_EVIDENCE)
    request = FactSelectionRequest("revenue", insufficient.result_id, "single-revenue")

    default = prepare_calculation_facts(report, (request,), corpora)
    allowed = prepare_calculation_facts(
        report,
        (request,),
        corpora,
        policy=ReconciliationSelectionPolicy(SingleSourcePolicy.ALLOW_EXPLICIT),
    )

    assert default.decisions[0].abstention_reason is (
        SelectionAbstentionReason.INSUFFICIENT_EVIDENCE
    )
    assert allowed.operands is not None
    assert allowed.operands[0].fact.fact_id == "single-revenue"
    assert allowed.decisions[0].status is SelectionStatus.SELECTED
    assert allowed.decisions[0].reconciliation_status is (
        ReconciliationStatus.INSUFFICIENT_EVIDENCE
    )


def test_duplicate_fact_id_ambiguity_is_rejected(tmp_path: Path) -> None:
    left_corpus = _corpus("6", "$100")
    right_corpus = _corpus("7", "$100")
    report, corpora = _report(
        tmp_path,
        (
            (left_corpus, _fact(left_corpus, "duplicate-revenue", "$100", "100")),
            (right_corpus, _fact(right_corpus, "duplicate-revenue", "$100", "100")),
        ),
    )
    agreement = _result(report, ReconciliationStatus.AGREEMENT)

    selection = prepare_calculation_facts(
        report,
        (FactSelectionRequest("revenue", agreement.result_id, "duplicate-revenue"),),
        corpora,
    )

    assert selection.operands is None
    assert selection.decisions[0].abstention_reason is (
        SelectionAbstentionReason.DUPLICATE_AMBIGUITY
    )
    assert "duplicate fact IDs" in (selection.decisions[0].detail or "")


def test_incompatible_period_result_is_rejected(tmp_path: Path) -> None:
    current_corpus = _corpus("8", "$100")
    prior_corpus = _corpus("9", "$100")
    report, corpora = _report(
        tmp_path,
        (
            (
                current_corpus,
                _fact(current_corpus, "current-revenue", "$100", "100", period=_period(2025)),
            ),
            (
                prior_corpus,
                _fact(prior_corpus, "prior-revenue", "$100", "100", period=_period(2024)),
            ),
        ),
    )
    incompatible = _result(report, ReconciliationStatus.INCOMPATIBLE)

    selection = prepare_calculation_facts(
        report,
        (FactSelectionRequest("revenue", incompatible.result_id, "current-revenue"),),
        corpora,
    )

    assert selection.decisions[0].abstention_reason is SelectionAbstentionReason.INCOMPATIBLE
    assert selection.decisions[0].candidate_observations == incompatible.observations


def test_every_agreement_candidate_is_revalidated_for_stale_evidence(tmp_path: Path) -> None:
    left_corpus = _corpus("e", "$100")
    right_corpus = _corpus("f", "$100")
    report, corpora = _report(
        tmp_path,
        (
            (left_corpus, _fact(left_corpus, "left-revenue", "$100", "100")),
            (right_corpus, _fact(right_corpus, "right-revenue", "$100", "100")),
        ),
    )
    agreement = _result(report, ReconciliationStatus.AGREEMENT)
    corpora[right_corpus.document_id] = replace(
        right_corpus,
        manifest_sha256="0" * 64,
    )

    selection = prepare_calculation_facts(
        report,
        (FactSelectionRequest("revenue", agreement.result_id, "left-revenue"),),
        corpora,
    )

    assert selection.operands is None
    assert selection.decisions[0].abstention_reason is (
        SelectionAbstentionReason.INVALID_OR_STALE_EVIDENCE
    )
    assert "manifest hash is stale" in (selection.decisions[0].detail or "")


def test_invalid_single_source_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="SingleSourcePolicy"):
        ReconciliationSelectionPolicy("allow")  # type: ignore[arg-type]
