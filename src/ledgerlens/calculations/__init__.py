"""Deterministic financial calculations over verified, provenance-bound facts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum

from ledgerlens.financial_facts import (
    FinancialFact,
    FinancialFactValidationError,
    PeriodType,
    ResolutionStatus,
    UnitKind,
    VerificationStatus,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus

CALCULATION_SCHEMA_VERSION = "1.0"
DECIMAL_PRECISION = 34
DECIMAL_ROUNDING = ROUND_HALF_EVEN
DECIMAL_EMIN = -999999
DECIMAL_EMAX = 999999
_ANNUAL_PERIOD_MIN_DAYS = 350
_ANNUAL_PERIOD_MAX_DAYS = 380


class CalculationError(ValueError):
    """A calculation is undefined or its financial facts are incompatible."""


class CalculationType(StrEnum):
    """Supported deterministic calculation types."""

    YEAR_OVER_YEAR_GROWTH = "year_over_year_growth"
    PROFIT_MARGIN = "profit_margin"
    FINANCIAL_RATIO = "financial_ratio"


class ResultUnitKind(StrEnum):
    """Units emitted by the initial dimensionless calculations."""

    PERCENTAGE = "percentage"
    RATIO = "ratio"


@dataclass(frozen=True)
class FactOperand:
    """A financial fact paired with the evidence corpus that validates it."""

    fact: FinancialFact
    corpus: EvidenceCorpus

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Revalidate verification state and provenance at calculation time."""
        if self.fact.verification.status is not VerificationStatus.VERIFIED:
            raise CalculationError(
                f"fact {self.fact.fact_id!r} is not verified and cannot be calculated"
            )
        if self.fact.normalized_value is None:
            raise CalculationError(f"fact {self.fact.fact_id!r} has no normalized value")
        try:
            self.fact.validate_against(self.corpus)
        except FinancialFactValidationError as exc:
            raise CalculationError(
                f"fact {self.fact.fact_id!r} failed evidence validation: {exc}"
            ) from exc


@dataclass(frozen=True)
class CalculationInputReference:
    """Role and complete immutable fact snapshot used by a calculation."""

    role: str
    fact: FinancialFact

    def __post_init__(self) -> None:
        if not isinstance(self.role, str) or not self.role.strip():
            raise CalculationError("calculation input role must be a non-empty string")

    def to_dict(self) -> dict[str, object]:
        return {"role": self.role, "fact": self.fact.to_dict()}


@dataclass(frozen=True)
class ResultUnit:
    """Explicit unit for a dimensionless calculation result."""

    kind: ResultUnitKind
    label: str
    scale: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise CalculationError("result unit label must be a non-empty string")
        if not self.scale.is_finite() or self.scale <= 0:
            raise CalculationError("result unit scale must be finite and positive")

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "label": self.label, "scale": str(self.scale)}


@dataclass(frozen=True)
class CalculationResult:
    """A reproducible Decimal result with formulas, inputs, units, and provenance."""

    calculation_type: CalculationType
    formula: str
    inputs: tuple[CalculationInputReference, ...]
    computed_value: Decimal
    unit: ResultUnit
    precision: int = DECIMAL_PRECISION
    rounding: str = DECIMAL_ROUNDING

    def __post_init__(self) -> None:
        if not isinstance(self.formula, str) or not self.formula.strip():
            raise CalculationError("calculation formula must be a non-empty string")
        if len(self.inputs) != 2:
            raise CalculationError("supported calculations require exactly two inputs")
        if len({item.role for item in self.inputs}) != len(self.inputs):
            raise CalculationError("calculation input roles must be unique")
        if not isinstance(self.computed_value, Decimal) or not self.computed_value.is_finite():
            raise CalculationError("computed value must be a finite Decimal")
        if self.precision != DECIMAL_PRECISION or self.rounding != DECIMAL_ROUNDING:
            raise CalculationError("unsupported Decimal arithmetic context")

    @property
    def calculation_id(self) -> str:
        """Return a deterministic content identifier for this exact calculation."""
        canonical = json.dumps(
            self._payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"sha256-{hashlib.sha256(canonical).hexdigest()}"

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": CALCULATION_SCHEMA_VERSION,
            "calculation_type": self.calculation_type.value,
            "formula": self.formula,
            "inputs": [item.to_dict() for item in self.inputs],
            "computed_value": str(self.computed_value),
            "unit": self.unit.to_dict(),
            "arithmetic": {
                "number_type": "decimal",
                "precision": self.precision,
                "rounding": self.rounding,
                "emin": DECIMAL_EMIN,
                "emax": DECIMAL_EMAX,
                "clamp": 0,
            },
        }

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON representation with the content-derived identifier."""
        return {"calculation_id": self.calculation_id, **self._payload()}


def _facts(left: FactOperand, right: FactOperand) -> tuple[FinancialFact, FinancialFact]:
    left.validate()
    right.validate()
    return left.fact, right.fact


def _require_same_company(left: FinancialFact, right: FinancialFact) -> None:
    if left.company != right.company:
        raise CalculationError(
            f"facts must have the same company: {left.company!r} != {right.company!r}"
        )


def _require_same_currency(left: FinancialFact, right: FinancialFact) -> None:
    if (
        left.currency.status is not right.currency.status
        or left.currency.code != right.currency.code
    ):
        raise CalculationError("facts must have compatible currencies")


def _require_same_unit_kind(left: FinancialFact, right: FinancialFact) -> None:
    if left.units.kind is not right.units.kind:
        raise CalculationError("facts must have the same unit type")


def _require_same_period(left: FinancialFact, right: FinancialFact) -> None:
    left_period = left.reporting_period
    right_period = right.reporting_period
    if (
        left_period.status is not right_period.status
        or left_period.period_type is not right_period.period_type
        or left_period.start_date != right_period.start_date
        or left_period.end_date != right_period.end_date
    ):
        raise CalculationError("facts must have the same reporting period")


def _base_value(fact: FinancialFact) -> Decimal:
    if fact.normalized_value is None or fact.units.scale is None:
        raise CalculationError(f"fact {fact.fact_id!r} lacks a value or unit scale")
    return fact.normalized_value * fact.units.scale


def _decimal_context() -> Context:
    return Context(
        prec=DECIMAL_PRECISION,
        rounding=DECIMAL_ROUNDING,
        Emin=DECIMAL_EMIN,
        Emax=DECIMAL_EMAX,
        capitals=1,
        clamp=0,
    )


def _divide(numerator: Decimal, denominator: Decimal, denominator_role: str) -> Decimal:
    if denominator == 0:
        raise CalculationError(f"{denominator_role} must not be zero")
    with localcontext(_decimal_context()):
        return numerator / denominator


def _result(
    calculation_type: CalculationType,
    formula: str,
    roles: tuple[tuple[str, FinancialFact], tuple[str, FinancialFact]],
    value: Decimal,
    unit: ResultUnit,
) -> CalculationResult:
    if not value.is_finite():
        raise CalculationError("calculation produced a non-finite value")
    return CalculationResult(
        calculation_type=calculation_type,
        formula=formula,
        inputs=tuple(CalculationInputReference(role=role, fact=fact) for role, fact in roles),
        computed_value=value,
        unit=unit,
    )


def year_over_year_growth(current: FactOperand, prior: FactOperand) -> CalculationResult:
    """Calculate percentage change between comparable consecutive annual facts."""
    current_fact, prior_fact = _facts(current, prior)
    _require_same_company(current_fact, prior_fact)
    _require_same_unit_kind(current_fact, prior_fact)
    _require_same_currency(current_fact, prior_fact)
    if current_fact.metric != prior_fact.metric:
        raise CalculationError("year-over-year facts must have the same metric")
    if current_fact.units.kind not in {
        UnitKind.MONETARY,
        UnitKind.PER_SHARE,
        UnitKind.COUNT,
        UnitKind.PERCENTAGE,
    }:
        raise CalculationError("year-over-year growth does not support this unit type")
    current_period = current_fact.reporting_period
    prior_period = prior_fact.reporting_period
    if (
        current_period.status is not ResolutionStatus.KNOWN
        or prior_period.status is not ResolutionStatus.KNOWN
        or current_period.period_type is not PeriodType.DURATION
        or prior_period.period_type is not PeriodType.DURATION
        or current_period.start_date is None
        or current_period.end_date is None
        or prior_period.start_date is None
        or prior_period.end_date is None
    ):
        raise CalculationError("year-over-year growth requires known duration periods")
    current_days = (current_period.end_date - current_period.start_date).days + 1
    prior_days = (prior_period.end_date - prior_period.start_date).days + 1
    if not (
        _ANNUAL_PERIOD_MIN_DAYS <= current_days <= _ANNUAL_PERIOD_MAX_DAYS
        and _ANNUAL_PERIOD_MIN_DAYS <= prior_days <= _ANNUAL_PERIOD_MAX_DAYS
    ):
        raise CalculationError("year-over-year growth requires annual reporting periods")
    if (
        current_period.end_date <= prior_period.end_date
        or current_period.start_date != prior_period.end_date + timedelta(days=1)
        or current_period.end_date.year != prior_period.end_date.year + 1
    ):
        raise CalculationError("year-over-year periods must be ordered consecutive fiscal years")

    with localcontext(_decimal_context()):
        current_value = _base_value(current_fact)
        prior_value = _base_value(prior_fact)
        growth = _divide(current_value - prior_value, prior_value, "prior value") * Decimal("100")
    return _result(
        CalculationType.YEAR_OVER_YEAR_GROWTH,
        "((current.normalized_value * current.units.scale) - "
        "(prior.normalized_value * prior.units.scale)) / "
        "(prior.normalized_value * prior.units.scale) * 100",
        (("current", current_fact), ("prior", prior_fact)),
        growth,
        ResultUnit(ResultUnitKind.PERCENTAGE, "percent"),
    )


def profit_margin(profit: FactOperand, revenue: FactOperand) -> CalculationResult:
    """Calculate profit as a percentage of revenue for one reporting period."""
    profit_fact, revenue_fact = _facts(profit, revenue)
    _require_same_company(profit_fact, revenue_fact)
    _require_same_period(profit_fact, revenue_fact)
    if profit_fact.reporting_period.period_type is not PeriodType.DURATION:
        raise CalculationError("profit margin requires a duration reporting period")
    if (
        profit_fact.units.kind is not UnitKind.MONETARY
        or revenue_fact.units.kind is not UnitKind.MONETARY
    ):
        raise CalculationError("profit margin requires two monetary facts")
    _require_same_currency(profit_fact, revenue_fact)

    with localcontext(_decimal_context()):
        profit_value = _base_value(profit_fact)
        revenue_value = _base_value(revenue_fact)
        margin = _divide(profit_value, revenue_value, "revenue") * Decimal("100")
    return _result(
        CalculationType.PROFIT_MARGIN,
        "(profit.normalized_value * profit.units.scale) / "
        "(revenue.normalized_value * revenue.units.scale) * 100",
        (("profit", profit_fact), ("revenue", revenue_fact)),
        margin,
        ResultUnit(ResultUnitKind.PERCENTAGE, "percent"),
    )


def financial_ratio(
    numerator: FactOperand, denominator: FactOperand, *, ratio_name: str
) -> CalculationResult:
    """Calculate a named dimensionless ratio between compatible same-period facts."""
    if not isinstance(ratio_name, str) or not ratio_name.strip():
        raise CalculationError("ratio_name must be a non-empty string")
    numerator_fact, denominator_fact = _facts(numerator, denominator)
    _require_same_company(numerator_fact, denominator_fact)
    _require_same_unit_kind(numerator_fact, denominator_fact)
    _require_same_currency(numerator_fact, denominator_fact)
    _require_same_period(numerator_fact, denominator_fact)
    if numerator_fact.units.kind not in {
        UnitKind.MONETARY,
        UnitKind.PER_SHARE,
        UnitKind.PERCENTAGE,
        UnitKind.COUNT,
    }:
        raise CalculationError("financial ratio does not support this unit type")

    with localcontext(_decimal_context()):
        ratio = _divide(
            _base_value(numerator_fact),
            _base_value(denominator_fact),
            "denominator",
        )
    return _result(
        CalculationType.FINANCIAL_RATIO,
        "(numerator.normalized_value * numerator.units.scale) / "
        "(denominator.normalized_value * denominator.units.scale)",
        (("numerator", numerator_fact), ("denominator", denominator_fact)),
        ratio,
        ResultUnit(ResultUnitKind.RATIO, ratio_name.strip()),
    )


__all__ = [
    "CALCULATION_SCHEMA_VERSION",
    "DECIMAL_PRECISION",
    "DECIMAL_ROUNDING",
    "DECIMAL_EMIN",
    "DECIMAL_EMAX",
    "CalculationError",
    "CalculationInputReference",
    "CalculationResult",
    "CalculationType",
    "FactOperand",
    "ResultUnit",
    "ResultUnitKind",
    "financial_ratio",
    "profit_margin",
    "year_over_year_growth",
]
