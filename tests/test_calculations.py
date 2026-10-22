from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import date
from decimal import Decimal, localcontext

import pytest

from ledgerlens.calculations import (
    CalculationError,
    CalculationType,
    FactOperand,
    ResultUnitKind,
    financial_ratio,
    profit_margin,
    year_over_year_growth,
)
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
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _period(year: int, *, label: str | None = None) -> ReportingPeriod:
    return ReportingPeriod(
        status=ResolutionStatus.KNOWN,
        period_type=PeriodType.DURATION,
        start_date=date(year - 1, 1, 1),
        end_date=date(year - 1, 12, 31),
        label=label or f"Year Ended December 31, {year - 1}",
    )


def _corpus(
    source_character: str,
    report_year: int,
    rows: tuple[tuple[str, str], ...],
    *,
    company: str = "Synthetic Company",
) -> EvidenceCorpus:
    source_hash = source_character * 64
    pages: list[EvidencePage] = []
    for page_number, (metric, reported_value) in enumerate(rows, start=1):
        text = f"Controlled fixture\n{metric}: {reported_value}"
        pages.append(
            EvidencePage(
                document_id=f"sha256-{source_hash}",
                company=company,
                report_year=report_year,
                physical_page_number=page_number,
                text=text,
                text_sha256=_sha256(text),
                method="canonical_extraction",
                provenance={"source_sha256": source_hash},
                screening_findings={"canonical": {"status": "pass", "warnings": []}},
                review_provenance=None,
                known_limitations=("Controlled synthetic fixture only.",),
            )
        )
    return EvidenceCorpus(
        document_id=f"sha256-{source_hash}",
        source_sha256=source_hash,
        pages=tuple(pages),
        excluded_pages=(),
        manifest={},
        evidence_jsonl_sha256=_sha256(f"evidence-{source_character}"),
        manifest_sha256=_sha256(f"manifest-{source_character}"),
    )


def _operand(
    corpus: EvidenceCorpus,
    page_number: int,
    *,
    fact_id: str,
    metric: str,
    reported_value: str,
    normalized_value: str,
    period: ReportingPeriod,
    scale: str = "1000000",
    unit_kind: UnitKind = UnitKind.MONETARY,
    currency_code: str | None = "USD",
) -> FactOperand:
    page = corpus.pages[page_number - 1]
    currency = (
        Currency(status=ResolutionStatus.KNOWN, code=currency_code)
        if currency_code is not None
        else Currency(
            status=ResolutionStatus.NOT_APPLICABLE,
            note="The controlled non-monetary fixture has no currency.",
        )
    )
    fact = FinancialFact(
        fact_id=fact_id,
        metric=metric,
        reported_value=reported_value,
        normalized_value=Decimal(normalized_value),
        company=page.company,
        reporting_period=period,
        currency=currency,
        units=Units(
            status=ResolutionStatus.KNOWN,
            kind=unit_kind,
            scale=Decimal(scale),
            label=f"controlled {unit_kind.value} units",
        ),
        table_column=TableColumn(
            status=ResolutionStatus.KNOWN,
            label=period.label,
        ),
        evidence=SupportingEvidence.from_evidence(corpus, page, page.text),
        verification=Verification(
            status=VerificationStatus.VERIFIED,
            reviewed_by="controlled-fixture-reviewer",
            reviewed_at_utc="2026-10-23T00:00:00Z",
            method="controlled fixture construction",
        ),
        limitations=("Controlled synthetic fixture only.",),
    )
    return FactOperand(fact, corpus)


def test_year_over_year_growth_converts_scales_and_records_provenance() -> None:
    current_corpus = _corpus("a", 2025, (("Revenue", "$1.5"),))
    prior_corpus = _corpus("b", 2024, (("Revenue", "$1,000"),))
    current = _operand(
        current_corpus,
        1,
        fact_id="synthetic-revenue-2024",
        metric="revenue",
        reported_value="$1.5",
        normalized_value="1.5",
        period=_period(2025),
        scale="1000000",
    )
    prior = _operand(
        prior_corpus,
        1,
        fact_id="synthetic-revenue-2023",
        metric="revenue",
        reported_value="$1,000",
        normalized_value="1000",
        period=_period(2024),
        scale="1000",
    )

    result = year_over_year_growth(current, prior)
    payload = result.to_dict()

    assert result.computed_value == Decimal("50.0")
    assert result.calculation_type is CalculationType.YEAR_OVER_YEAR_GROWTH
    assert result.unit.kind is ResultUnitKind.PERCENTAGE
    assert payload["calculation_id"].startswith("sha256-")  # type: ignore[union-attr]
    assert payload["arithmetic"] == {  # type: ignore[index]
        "number_type": "decimal",
        "precision": 34,
        "rounding": "ROUND_HALF_EVEN",
        "emin": -999999,
        "emax": 999999,
        "clamp": 0,
    }
    inputs = payload["inputs"]
    assert inputs[0]["role"] == "current"  # type: ignore[index]
    assert inputs[1]["fact"]["fact_id"] == "synthetic-revenue-2023"  # type: ignore[index]
    assert inputs[1]["fact"]["evidence"]["physical_page_number"] == 1  # type: ignore[index]
    assert result.calculation_id == year_over_year_growth(current, prior).calculation_id


def test_profit_margin_uses_same_period_monetary_facts() -> None:
    corpus = _corpus("c", 2025, (("Net income", "$25"), ("Revenue", "$100")))
    period = _period(2025)
    profit = _operand(
        corpus,
        1,
        fact_id="synthetic-net-income",
        metric="net_income",
        reported_value="$25",
        normalized_value="25",
        period=period,
    )
    revenue = _operand(
        corpus,
        2,
        fact_id="synthetic-revenue",
        metric="revenue",
        reported_value="$100",
        normalized_value="100",
        period=period,
    )

    result = profit_margin(profit, revenue)

    assert result.computed_value == Decimal("25.00")
    assert result.formula.endswith("* 100")
    assert [item.role for item in result.inputs] == ["profit", "revenue"]


def test_profit_margin_rejects_instant_periods() -> None:
    corpus = _corpus("9", 2025, (("Profit", "$25"), ("Revenue", "$100")))
    period = ReportingPeriod(
        status=ResolutionStatus.KNOWN,
        period_type=PeriodType.INSTANT,
        end_date=date(2024, 12, 31),
        label="As of December 31, 2024",
    )
    profit = _operand(
        corpus,
        1,
        fact_id="synthetic-profit-instant",
        metric="profit",
        reported_value="$25",
        normalized_value="25",
        period=period,
    )
    revenue = _operand(
        corpus,
        2,
        fact_id="synthetic-revenue-instant",
        metric="revenue",
        reported_value="$100",
        normalized_value="100",
        period=period,
    )

    with pytest.raises(CalculationError, match="duration reporting period"):
        profit_margin(profit, revenue)


def test_financial_ratio_converts_scales_and_uses_caller_label() -> None:
    corpus = _corpus("d", 2025, (("Current assets", "$2"), ("Current liabilities", "$500")))
    period = _period(2025)
    assets = _operand(
        corpus,
        1,
        fact_id="synthetic-current-assets",
        metric="current_assets",
        reported_value="$2",
        normalized_value="2",
        period=period,
        scale="1000000000",
    )
    liabilities = _operand(
        corpus,
        2,
        fact_id="synthetic-current-liabilities",
        metric="current_liabilities",
        reported_value="$500",
        normalized_value="500",
        period=period,
        scale="1000000",
    )

    result = financial_ratio(assets, liabilities, ratio_name="current ratio")

    assert result.computed_value == Decimal("4")
    assert result.unit.kind is ResultUnitKind.RATIO
    assert result.unit.label == "current ratio"


def test_calculation_is_independent_of_process_decimal_context() -> None:
    corpus = _corpus("e", 2025, (("Profit", "$1"), ("Revenue", "$3")))
    period = _period(2025)
    profit = _operand(
        corpus,
        1,
        fact_id="synthetic-profit",
        metric="profit",
        reported_value="$1",
        normalized_value="1",
        period=period,
    )
    revenue = _operand(
        corpus,
        2,
        fact_id="synthetic-revenue",
        metric="revenue",
        reported_value="$3",
        normalized_value="3",
        period=period,
    )

    with localcontext() as context:
        context.prec = 3
        value = profit_margin(profit, revenue).computed_value

    assert value == Decimal("33.33333333333333333333333333333333")


def test_unverified_and_ambiguous_facts_are_rejected() -> None:
    corpus = _corpus("f", 2025, (("Revenue", "$100"),))
    verified = _operand(
        corpus,
        1,
        fact_id="synthetic-revenue",
        metric="revenue",
        reported_value="$100",
        normalized_value="100",
        period=_period(2025),
    )
    unverified = replace(
        verified.fact,
        verification=Verification(status=VerificationStatus.UNVERIFIED),
    )
    ambiguous = replace(
        verified.fact,
        verification=Verification(
            status=VerificationStatus.AMBIGUOUS,
            note="The controlled value is intentionally ambiguous.",
        ),
    )

    with pytest.raises(CalculationError, match="not verified"):
        FactOperand(unverified, corpus)
    with pytest.raises(CalculationError, match="not verified"):
        FactOperand(ambiguous, corpus)


def test_stale_fact_is_rejected_before_calculation() -> None:
    corpus = _corpus("1", 2025, (("Revenue", "$100"),))
    operand = _operand(
        corpus,
        1,
        fact_id="synthetic-revenue",
        metric="revenue",
        reported_value="$100",
        normalized_value="100",
        period=_period(2025),
    )
    stale = replace(
        operand.fact,
        evidence=replace(operand.fact.evidence, evidence_manifest_sha256="0" * 64),
    )

    with pytest.raises(CalculationError, match="manifest hash is stale"):
        FactOperand(stale, corpus)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("company", "same company"),
        ("currency", "compatible currencies"),
        ("unit", "same unit type"),
        ("period", "same reporting period"),
    ],
)
def test_ratio_rejects_incompatible_inputs(change: str, message: str) -> None:
    corpus = _corpus("2", 2025, (("Assets", "$100"), ("Liabilities", "$50")))
    period = _period(2025)
    numerator = _operand(
        corpus,
        1,
        fact_id="synthetic-assets",
        metric="assets",
        reported_value="$100",
        normalized_value="100",
        period=period,
    )
    denominator = _operand(
        corpus,
        2,
        fact_id="synthetic-liabilities",
        metric="liabilities",
        reported_value="$50",
        normalized_value="50",
        period=period,
    )
    if change == "company":
        other_corpus = _corpus(
            "3",
            2025,
            (("Liabilities", "$50"),),
            company="Other Synthetic Company",
        )
        denominator = _operand(
            other_corpus,
            1,
            fact_id="other-liabilities",
            metric="liabilities",
            reported_value="$50",
            normalized_value="50",
            period=period,
        )
    elif change == "currency":
        denominator = FactOperand(
            replace(denominator.fact, currency=Currency(ResolutionStatus.KNOWN, "EUR")),
            corpus,
        )
    elif change == "unit":
        denominator = FactOperand(
            replace(
                denominator.fact,
                currency=Currency(
                    ResolutionStatus.NOT_APPLICABLE,
                    note="Count has no currency.",
                ),
                units=Units(
                    ResolutionStatus.KNOWN,
                    UnitKind.COUNT,
                    Decimal("1"),
                    "items",
                ),
            ),
            corpus,
        )
    else:
        denominator = FactOperand(
            replace(
                denominator.fact,
                reporting_period=_period(2024),
                table_column=TableColumn(
                    ResolutionStatus.KNOWN,
                    _period(2024).label,
                ),
            ),
            corpus,
        )

    with pytest.raises(CalculationError, match=message):
        financial_ratio(numerator, denominator, ratio_name="test ratio")


@pytest.mark.parametrize(
    ("calculation", "zero_role"),
    [
        ("growth", "prior value"),
        ("margin", "revenue"),
        ("ratio", "denominator"),
    ],
)
def test_division_by_zero_is_rejected(calculation: str, zero_role: str) -> None:
    if calculation == "growth":
        left_corpus = _corpus("4", 2025, (("Revenue", "$100"),))
        right_corpus = _corpus("5", 2024, (("Revenue", "$0"),))
        left_period = _period(2025)
        right_period = _period(2024)
    else:
        left_corpus = _corpus("4", 2025, (("Numerator", "$100"),))
        right_corpus = _corpus("5", 2025, (("Denominator", "$0"),))
        left_period = right_period = _period(2025)
    left = _operand(
        left_corpus,
        1,
        fact_id="synthetic-numerator",
        metric="revenue" if calculation == "growth" else "numerator",
        reported_value="$100",
        normalized_value="100",
        period=left_period,
    )
    right = _operand(
        right_corpus,
        1,
        fact_id="synthetic-zero",
        metric="revenue" if calculation == "growth" else "denominator",
        reported_value="$0",
        normalized_value="0",
        period=right_period,
    )

    with pytest.raises(CalculationError, match=zero_role):
        if calculation == "growth":
            year_over_year_growth(left, right)
        elif calculation == "margin":
            profit_margin(left, right)
        else:
            financial_ratio(left, right, ratio_name="zero ratio")


def test_year_over_year_growth_rejects_nonconsecutive_or_different_metrics() -> None:
    current_corpus = _corpus("6", 2025, (("Revenue", "$100"),))
    prior_corpus = _corpus("7", 2023, (("Revenue", "$80"),))
    current = _operand(
        current_corpus,
        1,
        fact_id="synthetic-current",
        metric="revenue",
        reported_value="$100",
        normalized_value="100",
        period=_period(2025),
    )
    prior = _operand(
        prior_corpus,
        1,
        fact_id="synthetic-prior",
        metric="revenue",
        reported_value="$80",
        normalized_value="80",
        period=_period(2023),
    )

    with pytest.raises(CalculationError, match="consecutive fiscal years"):
        year_over_year_growth(current, prior)
    with pytest.raises(CalculationError, match="same metric"):
        changed_prior = FactOperand(replace(prior.fact, metric="profit"), prior.corpus)
        year_over_year_growth(current, changed_prior)


def test_year_over_year_growth_rejects_a_gap_between_annual_periods() -> None:
    current_corpus = _corpus("a", 2025, (("Revenue", "$100"),))
    prior_corpus = _corpus("b", 2024, (("Revenue", "$80"),))
    current_period = ReportingPeriod(
        status=ResolutionStatus.KNOWN,
        period_type=PeriodType.DURATION,
        start_date=date(2024, 1, 8),
        end_date=date(2024, 12, 31),
        label="Synthetic shifted fiscal year",
    )
    current = _operand(
        current_corpus,
        1,
        fact_id="synthetic-current-gap",
        metric="revenue",
        reported_value="$100",
        normalized_value="100",
        period=current_period,
    )
    prior = _operand(
        prior_corpus,
        1,
        fact_id="synthetic-prior-gap",
        metric="revenue",
        reported_value="$80",
        normalized_value="80",
        period=_period(2024),
    )

    with pytest.raises(CalculationError, match="consecutive fiscal years"):
        year_over_year_growth(current, prior)


def test_ratio_name_is_required() -> None:
    corpus = _corpus("8", 2025, (("Assets", "$100"), ("Liabilities", "$50")))
    period = _period(2025)
    numerator = _operand(
        corpus,
        1,
        fact_id="synthetic-assets",
        metric="assets",
        reported_value="$100",
        normalized_value="100",
        period=period,
    )
    denominator = _operand(
        corpus,
        2,
        fact_id="synthetic-liabilities",
        metric="liabilities",
        reported_value="$50",
        normalized_value="50",
        period=period,
    )

    with pytest.raises(CalculationError, match="ratio_name"):
        financial_ratio(numerator, denominator, ratio_name=" ")
