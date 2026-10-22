from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from ledgerlens.financial_facts import (
    Currency,
    FinancialFact,
    FinancialFactValidationError,
    PeriodType,
    ReportingPeriod,
    ResolutionStatus,
    SupportingEvidence,
    TableColumn,
    UnitKind,
    Units,
    Verification,
    VerificationStatus,
    financial_fact_from_dict,
    load_financial_facts,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def corpus() -> EvidenceCorpus:
    text = "In millions, except per share data\nYear Ended January 28, 2024\nRevenue $60,922"
    page = EvidencePage(
        document_id=f"sha256-{'a' * 64}",
        company="NVIDIA Corporation",
        report_year=2024,
        physical_page_number=91,
        text=text,
        text_sha256=_sha256(text),
        method="canonical_extraction",
        provenance={"source_sha256": "a" * 64},
        screening_findings={"canonical": {"status": "pass", "warnings": ["Tables are flattened."]}},
        review_provenance=None,
        known_limitations=("Column association was manually reviewed.",),
    )
    return EvidenceCorpus(
        document_id=page.document_id,
        source_sha256="a" * 64,
        pages=(page,),
        excluded_pages=(),
        manifest={},
        evidence_jsonl_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )


def _verified_fact(corpus: EvidenceCorpus) -> FinancialFact:
    evidence = SupportingEvidence.from_evidence(
        corpus,
        corpus.pages[0],
        "Year Ended January 28, 2024\nRevenue $60,922",
    )
    return FinancialFact(
        fact_id="nvda-2024-revenue",
        metric="revenue",
        reported_value="$60,922",
        normalized_value=Decimal("60922"),
        company="NVIDIA Corporation",
        reporting_period=ReportingPeriod(
            status=ResolutionStatus.KNOWN,
            period_type=PeriodType.DURATION,
            start_date=date(2023, 1, 30),
            end_date=date(2024, 1, 28),
            label="Year Ended January 28, 2024",
        ),
        currency=Currency(status=ResolutionStatus.KNOWN, code="USD"),
        units=Units(
            status=ResolutionStatus.KNOWN,
            kind=UnitKind.MONETARY,
            scale=Decimal("1000000"),
            label="USD in millions",
        ),
        table_column=TableColumn(
            status=ResolutionStatus.KNOWN,
            label="Year Ended January 28, 2024",
        ),
        evidence=evidence,
        verification=Verification(
            status=VerificationStatus.VERIFIED,
            reviewed_by="controlled-test-reviewer",
            reviewed_at_utc="2026-10-22T02:00:00Z",
            method="manual comparison with the rendered source page",
        ),
    )


def test_verified_fact_round_trips_and_preserves_provenance(
    corpus: EvidenceCorpus, tmp_path: Path
) -> None:
    fact = _verified_fact(corpus)
    fact.validate_against(corpus)

    payload = fact.to_dict()
    restored = financial_fact_from_dict(payload)
    assert restored == fact
    assert payload["reported_value"] == "$60,922"
    assert payload["normalized_value"] == "60922"
    assert payload["units"]["scale"] == "1000000"  # type: ignore[index]
    assert payload["evidence"]["physical_page_number"] == 91  # type: ignore[index]
    assert payload["evidence"]["screening_warnings"] == [  # type: ignore[index]
        "Tables are flattened."
    ]

    path = tmp_path / "facts.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert load_financial_facts(path, corpus) == (fact,)


def test_supporting_text_must_be_exact_selected_evidence(corpus: EvidenceCorpus) -> None:
    with pytest.raises(FinancialFactValidationError, match="exact substring"):
        SupportingEvidence.from_evidence(
            corpus,
            corpus.pages[0],
            "Revenue was 60,922 million dollars",
        )


def test_reported_value_must_appear_in_supporting_text(corpus: EvidenceCorpus) -> None:
    fact = _verified_fact(corpus)
    with pytest.raises(FinancialFactValidationError, match="reported_value"):
        replace(fact, reported_value="$99,999")


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (
            "reporting_period",
            ReportingPeriod(
                status=ResolutionStatus.AMBIGUOUS,
                note="Two flattened year columns are plausible.",
            ),
        ),
        (
            "currency",
            Currency(
                status=ResolutionStatus.MISSING,
                note="No currency declaration is present in selected evidence.",
            ),
        ),
        (
            "units",
            Units(
                status=ResolutionStatus.AMBIGUOUS,
                note="The scale heading cannot be associated with this value.",
            ),
        ),
        (
            "table_column",
            TableColumn(
                status=ResolutionStatus.AMBIGUOUS,
                note="Flattening destroyed the column boundary.",
            ),
        ),
    ],
)
def test_verified_fact_rejects_unresolved_critical_context(
    corpus: EvidenceCorpus, field: str, replacement: object
) -> None:
    fact = _verified_fact(corpus)
    with pytest.raises(FinancialFactValidationError, match="critical context"):
        replace(fact, **{field: replacement}, warnings=("Critical context is unresolved.",))


def test_ambiguous_fact_requires_an_explicit_warning(corpus: EvidenceCorpus) -> None:
    fact = _verified_fact(corpus)
    period = ReportingPeriod(
        status=ResolutionStatus.AMBIGUOUS,
        note="The flattened table contains two possible fiscal-year headings.",
    )
    verification = Verification(
        status=VerificationStatus.AMBIGUOUS,
        reviewed_by="controlled-test-reviewer",
        reviewed_at_utc="2026-10-22T10:00:00Z",
        method="manual source review",
        note="The value is legible but its year association is not.",
    )

    with pytest.raises(FinancialFactValidationError, match="warning or limitation"):
        replace(fact, reporting_period=period, verification=verification)

    ambiguous = replace(
        fact,
        reporting_period=period,
        verification=verification,
        warnings=("Do not use this value until its reporting period is resolved.",),
    )
    assert ambiguous.verification.status is VerificationStatus.AMBIGUOUS
    assert ambiguous.reporting_period.status is ResolutionStatus.AMBIGUOUS


def test_stale_evidence_hash_is_rejected(corpus: EvidenceCorpus) -> None:
    fact = _verified_fact(corpus)
    stale = replace(
        fact,
        evidence=replace(fact.evidence, evidence_manifest_sha256="d" * 64),
    )
    with pytest.raises(FinancialFactValidationError, match="manifest hash is stale"):
        stale.validate_against(corpus)


def test_changed_inherited_evidence_metadata_is_rejected(corpus: EvidenceCorpus) -> None:
    fact = _verified_fact(corpus)
    changed = replace(
        fact,
        evidence=replace(fact.evidence, screening_warnings=("Different warning.",)),
    )
    with pytest.raises(FinancialFactValidationError, match="warnings do not match"):
        changed.validate_against(corpus)


def test_verified_monetary_fact_requires_currency(corpus: EvidenceCorpus) -> None:
    fact = _verified_fact(corpus)
    with pytest.raises(FinancialFactValidationError, match="require a known currency"):
        replace(
            fact,
            currency=Currency(
                status=ResolutionStatus.NOT_APPLICABLE,
                note="Currency was incorrectly treated as inapplicable.",
            ),
        )


def test_duplicate_fact_ids_are_rejected(corpus: EvidenceCorpus, tmp_path: Path) -> None:
    payload = json.dumps(_verified_fact(corpus).to_dict())
    path = tmp_path / "duplicate.jsonl"
    path.write_text(f"{payload}\n{payload}\n", encoding="utf-8")

    with pytest.raises(FinancialFactValidationError, match="duplicate financial fact_id"):
        load_financial_facts(path, corpus)
