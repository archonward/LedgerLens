"""Validated, provenance-bound representations of reported financial facts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage

FINANCIAL_FACT_SCHEMA_VERSION = "1.0"
_SHA256_LENGTH = 64


class FinancialFactValidationError(ValueError):
    """A financial fact is structurally invalid or is not bound to its evidence."""


class ResolutionStatus(StrEnum):
    """How completely a critical piece of financial context is resolved."""

    KNOWN = "known"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class VerificationStatus(StrEnum):
    """Review state of a financial fact, distinct from extraction quality."""

    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    UNVERIFIED = "unverified"


class PeriodType(StrEnum):
    """Whether a fact applies over a duration or at one instant."""

    DURATION = "duration"
    INSTANT = "instant"


class UnitKind(StrEnum):
    """Broad semantic kind of a reported unit."""

    MONETARY = "monetary"
    PER_SHARE = "per_share"
    PERCENTAGE = "percentage"
    COUNT = "count"
    OTHER = "other"


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FinancialFactValidationError(f"{field} must be a non-empty string")


def _validate_sha256(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FinancialFactValidationError(f"{field} must be a lowercase SHA-256 digest")


def _parse_date(value: object, field: str) -> date | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FinancialFactValidationError(f"{field} must be an ISO date or null")
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise FinancialFactValidationError(f"{field} must be an ISO date") from None


def _parse_decimal(value: object, field: str) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FinancialFactValidationError(f"{field} must be a decimal string or null")
    try:
        result = Decimal(value)
    except InvalidOperation:
        raise FinancialFactValidationError(f"{field} must be a decimal string") from None
    if not result.is_finite():
        raise FinancialFactValidationError(f"{field} must be finite")
    return result


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise FinancialFactValidationError(f"{field} must be a list of non-empty strings")
    return tuple(value)


@dataclass(frozen=True)
class ReportingPeriod:
    """Explicit reporting-period interpretation, including unresolved cases."""

    status: ResolutionStatus
    period_type: PeriodType | None = None
    start_date: date | None = None
    end_date: date | None = None
    label: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.status is ResolutionStatus.KNOWN:
            if self.period_type is None or self.end_date is None or not self.label:
                raise FinancialFactValidationError(
                    "known reporting periods require period_type, end_date, and label"
                )
            if self.period_type is PeriodType.DURATION and self.start_date is None:
                raise FinancialFactValidationError(
                    "known duration reporting periods require start_date"
                )
            if self.period_type is PeriodType.INSTANT and self.start_date is not None:
                raise FinancialFactValidationError(
                    "instant reporting periods must not have start_date"
                )
            if self.start_date is not None and self.start_date > self.end_date:
                raise FinancialFactValidationError("reporting period starts after it ends")
        else:
            if self.status is ResolutionStatus.NOT_APPLICABLE:
                raise FinancialFactValidationError(
                    "reporting period cannot be marked not_applicable"
                )
            if any(
                item is not None
                for item in (self.period_type, self.start_date, self.end_date, self.label)
            ):
                raise FinancialFactValidationError(
                    "unresolved reporting periods must not contain a selected interpretation"
                )
            _require_text(self.note or "", "unresolved reporting period note")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "period_type": self.period_type.value if self.period_type else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "label": self.label,
            "note": self.note,
        }


@dataclass(frozen=True)
class Currency:
    """Reported currency, or an explicit record that it is unresolved/inapplicable."""

    status: ResolutionStatus
    code: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.status is ResolutionStatus.KNOWN:
            if (
                not isinstance(self.code, str)
                or len(self.code) != 3
                or not self.code.isascii()
                or not self.code.isalpha()
                or not self.code.isupper()
            ):
                raise FinancialFactValidationError(
                    "known currency code must be a three-letter uppercase code"
                )
            if self.note is not None:
                _require_text(self.note, "currency note")
        else:
            if self.code is not None:
                raise FinancialFactValidationError(
                    "unresolved or inapplicable currency must not contain a code"
                )
            _require_text(self.note or "", "currency note")

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "code": self.code, "note": self.note}


@dataclass(frozen=True)
class Units:
    """Reported unit kind and multiplier; scale is not applied to normalized_value."""

    status: ResolutionStatus
    kind: UnitKind | None = None
    scale: Decimal | None = None
    label: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.status is ResolutionStatus.NOT_APPLICABLE:
            raise FinancialFactValidationError("units cannot be marked not_applicable")
        if self.status is ResolutionStatus.KNOWN:
            if self.kind is None or self.scale is None or not self.label:
                raise FinancialFactValidationError("known units require kind, scale, and label")
            if not self.scale.is_finite() or self.scale <= 0:
                raise FinancialFactValidationError("unit scale must be a finite positive decimal")
        else:
            if any(item is not None for item in (self.kind, self.scale, self.label)):
                raise FinancialFactValidationError(
                    "unresolved units must not contain a selected interpretation"
                )
            _require_text(self.note or "", "unresolved units note")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "kind": self.kind.value if self.kind else None,
            "scale": str(self.scale) if self.scale is not None else None,
            "label": self.label,
            "note": self.note,
        }


@dataclass(frozen=True)
class TableColumn:
    """Association between a value and a table column, when applicable."""

    status: ResolutionStatus
    label: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if self.status is ResolutionStatus.KNOWN:
            _require_text(self.label or "", "known table column label")
        else:
            if self.label is not None:
                raise FinancialFactValidationError(
                    "unresolved or inapplicable table column must not contain a label"
                )
            _require_text(self.note or "", "table column note")

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "label": self.label, "note": self.note}


@dataclass(frozen=True)
class Verification:
    """Human verification state and its audit metadata."""

    status: VerificationStatus
    reviewed_by: str | None = None
    reviewed_at_utc: str | None = None
    method: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        supplied = (self.reviewed_by, self.reviewed_at_utc, self.method)
        if self.status is VerificationStatus.VERIFIED and not all(supplied):
            raise FinancialFactValidationError(
                "verified facts require reviewed_by, reviewed_at_utc, and review method"
            )
        if any(supplied) and not all(supplied):
            raise FinancialFactValidationError("review metadata must be supplied as a complete set")
        if self.status is VerificationStatus.AMBIGUOUS:
            _require_text(self.note or "", "ambiguous verification note")
        if self.reviewed_at_utc is not None:
            try:
                parsed = datetime.fromisoformat(self.reviewed_at_utc.replace("Z", "+00:00"))
            except ValueError:
                raise FinancialFactValidationError(
                    "reviewed_at_utc must be an ISO-8601 timestamp"
                ) from None
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise FinancialFactValidationError("reviewed_at_utc must include a timezone")
            if parsed.utcoffset().total_seconds() != 0:
                raise FinancialFactValidationError("reviewed_at_utc must use UTC")
        if self.note is not None:
            _require_text(self.note, "verification note")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reviewed_by": self.reviewed_by,
            "reviewed_at_utc": self.reviewed_at_utc,
            "method": self.method,
            "note": self.note,
        }


@dataclass(frozen=True)
class SupportingEvidence:
    """Exact supporting text and integrity metadata from one selected evidence page."""

    document_id: str
    source_company: str
    source_report_year: int
    source_sha256: str
    physical_page_number: int
    supporting_text: str
    supporting_text_sha256: str
    selected_text_sha256: str
    evidence_method: str
    evidence_jsonl_sha256: str
    evidence_manifest_sha256: str
    screening_warnings: tuple[str, ...]
    known_limitations: tuple[str, ...]
    review_provenance: dict[str, Any] | None

    def __post_init__(self) -> None:
        _require_text(self.document_id, "evidence document_id")
        _require_text(self.source_company, "evidence source_company")
        if not isinstance(self.source_report_year, int) or isinstance(
            self.source_report_year, bool
        ):
            raise FinancialFactValidationError("evidence source_report_year must be an integer")
        if (
            not isinstance(self.physical_page_number, int)
            or isinstance(self.physical_page_number, bool)
            or self.physical_page_number < 1
        ):
            raise FinancialFactValidationError(
                "evidence physical_page_number must be a positive integer"
            )
        _require_text(self.supporting_text, "supporting_text")
        _require_text(self.evidence_method, "evidence_method")
        for field, value in (
            ("source_sha256", self.source_sha256),
            ("supporting_text_sha256", self.supporting_text_sha256),
            ("selected_text_sha256", self.selected_text_sha256),
            ("evidence_jsonl_sha256", self.evidence_jsonl_sha256),
            ("evidence_manifest_sha256", self.evidence_manifest_sha256),
        ):
            _validate_sha256(value, field)
        actual = hashlib.sha256(self.supporting_text.encode("utf-8")).hexdigest()
        if actual != self.supporting_text_sha256:
            raise FinancialFactValidationError(
                "supporting_text_sha256 does not match supporting_text"
            )
        for field, values in (
            ("screening_warnings", self.screening_warnings),
            ("known_limitations", self.known_limitations),
        ):
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise FinancialFactValidationError(f"{field} must contain non-empty strings")
        if self.review_provenance is not None and not isinstance(self.review_provenance, dict):
            raise FinancialFactValidationError("review_provenance must be an object or null")

    @classmethod
    def from_evidence(
        cls, corpus: EvidenceCorpus, page: EvidencePage, supporting_text: str
    ) -> SupportingEvidence:
        """Bind an exact substring to a page in a validated evidence corpus."""
        if page not in corpus.pages:
            raise FinancialFactValidationError("evidence page is not part of the supplied corpus")
        if not supporting_text or supporting_text not in page.text:
            raise FinancialFactValidationError(
                "supporting_text must be a non-empty exact substring of selected page text"
            )
        return cls(
            document_id=page.document_id,
            source_company=page.company,
            source_report_year=page.report_year,
            source_sha256=corpus.source_sha256,
            physical_page_number=page.physical_page_number,
            supporting_text=supporting_text,
            supporting_text_sha256=hashlib.sha256(supporting_text.encode("utf-8")).hexdigest(),
            selected_text_sha256=page.text_sha256,
            evidence_method=page.method,
            evidence_jsonl_sha256=corpus.evidence_jsonl_sha256,
            evidence_manifest_sha256=corpus.manifest_sha256,
            screening_warnings=page.warnings,
            known_limitations=page.known_limitations,
            review_provenance=page.review_provenance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "source_company": self.source_company,
            "source_report_year": self.source_report_year,
            "source_sha256": self.source_sha256,
            "physical_page_number": self.physical_page_number,
            "supporting_text": self.supporting_text,
            "supporting_text_sha256": self.supporting_text_sha256,
            "selected_text_sha256": self.selected_text_sha256,
            "evidence_method": self.evidence_method,
            "evidence_jsonl_sha256": self.evidence_jsonl_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "screening_warnings": list(self.screening_warnings),
            "known_limitations": list(self.known_limitations),
            "review_provenance": self.review_provenance,
        }


@dataclass(frozen=True)
class FinancialFact:
    """One reported financial fact with explicit context, evidence, and review state."""

    fact_id: str
    metric: str
    reported_value: str
    normalized_value: Decimal | None
    company: str
    reporting_period: ReportingPeriod
    currency: Currency
    units: Units
    table_column: TableColumn
    evidence: SupportingEvidence
    verification: Verification
    warnings: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field, value in (
            ("fact_id", self.fact_id),
            ("metric", self.metric),
            ("reported_value", self.reported_value),
            ("company", self.company),
        ):
            _require_text(value, field)
        if self.normalized_value is not None:
            if (
                not isinstance(self.normalized_value, Decimal)
                or not self.normalized_value.is_finite()
            ):
                raise FinancialFactValidationError(
                    "normalized_value must be a finite Decimal or null"
                )
        for field, values in (("warnings", self.warnings), ("limitations", self.limitations)):
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise FinancialFactValidationError(f"{field} must contain non-empty strings")
        if self.reported_value not in self.evidence.supporting_text:
            raise FinancialFactValidationError(
                "reported_value must occur verbatim in supporting_text"
            )

        unresolved = {
            "reporting_period": self.reporting_period.status,
            "currency": self.currency.status,
            "units": self.units.status,
            "table_column": self.table_column.status,
        }
        resolved_statuses = {ResolutionStatus.KNOWN, ResolutionStatus.NOT_APPLICABLE}
        if self.verification.status is VerificationStatus.VERIFIED:
            if self.normalized_value is None:
                raise FinancialFactValidationError("verified facts require normalized_value")
            if any(status not in resolved_statuses for status in unresolved.values()):
                raise FinancialFactValidationError(
                    "verified facts cannot contain missing or ambiguous critical context"
                )
            if (
                self.units.kind in {UnitKind.MONETARY, UnitKind.PER_SHARE}
                and self.currency.status is not ResolutionStatus.KNOWN
            ):
                raise FinancialFactValidationError(
                    "verified monetary and per-share facts require a known currency"
                )
            if (
                self.units.kind in {UnitKind.PERCENTAGE, UnitKind.COUNT}
                and self.currency.status is not ResolutionStatus.NOT_APPLICABLE
            ):
                raise FinancialFactValidationError(
                    "verified percentage and count facts require currency to be not_applicable"
                )
        if any(
            status in {ResolutionStatus.MISSING, ResolutionStatus.AMBIGUOUS}
            for status in unresolved.values()
        ) and not (self.warnings or self.limitations):
            raise FinancialFactValidationError(
                "facts with unresolved critical context require a warning or limitation"
            )

    def validate_against(self, corpus: EvidenceCorpus) -> None:
        """Confirm this fact still points to the same selected evidence artifact."""
        if self.evidence.document_id != corpus.document_id:
            raise FinancialFactValidationError("fact document_id does not match evidence corpus")
        if self.evidence.source_sha256 != corpus.source_sha256:
            raise FinancialFactValidationError("fact source_sha256 does not match evidence corpus")
        if self.evidence.evidence_jsonl_sha256 != corpus.evidence_jsonl_sha256:
            raise FinancialFactValidationError("fact evidence.jsonl hash is stale")
        if self.evidence.evidence_manifest_sha256 != corpus.manifest_sha256:
            raise FinancialFactValidationError("fact evidence manifest hash is stale")
        page = next(
            (
                item
                for item in corpus.pages
                if item.physical_page_number == self.evidence.physical_page_number
            ),
            None,
        )
        if page is None:
            raise FinancialFactValidationError("fact physical page is absent from evidence corpus")
        if page.document_id != self.evidence.document_id:
            raise FinancialFactValidationError("fact page document identity does not match")
        if page.company != self.evidence.source_company or page.company != self.company:
            raise FinancialFactValidationError("fact company does not match selected evidence")
        if page.report_year != self.evidence.source_report_year:
            raise FinancialFactValidationError(
                "fact source report year does not match selected evidence"
            )
        if page.text_sha256 != self.evidence.selected_text_sha256:
            raise FinancialFactValidationError("fact selected-text hash is stale")
        if page.method != self.evidence.evidence_method:
            raise FinancialFactValidationError("fact evidence method does not match")
        if self.evidence.supporting_text not in page.text:
            raise FinancialFactValidationError(
                "fact supporting text is not present verbatim on the selected page"
            )
        if self.evidence.screening_warnings != page.warnings:
            raise FinancialFactValidationError("fact screening warnings do not match evidence page")
        if self.evidence.known_limitations != page.known_limitations:
            raise FinancialFactValidationError("fact limitations do not match evidence page")
        if self.evidence.review_provenance != page.review_provenance:
            raise FinancialFactValidationError(
                "fact recovery review provenance does not match evidence page"
            )

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON representation without converting numbers to floats."""
        return {
            "schema_version": FINANCIAL_FACT_SCHEMA_VERSION,
            "fact_id": self.fact_id,
            "metric": self.metric,
            "reported_value": self.reported_value,
            "normalized_value": (
                str(self.normalized_value) if self.normalized_value is not None else None
            ),
            "company": self.company,
            "reporting_period": self.reporting_period.to_dict(),
            "currency": self.currency.to_dict(),
            "units": self.units.to_dict(),
            "table_column": self.table_column.to_dict(),
            "evidence": self.evidence.to_dict(),
            "verification": self.verification.to_dict(),
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
        }


def financial_fact_from_dict(value: object) -> FinancialFact:
    """Parse and validate one JSON-compatible financial fact record."""
    if not isinstance(value, dict):
        raise FinancialFactValidationError("financial fact record must be an object")
    if value.get("schema_version") != FINANCIAL_FACT_SCHEMA_VERSION:
        raise FinancialFactValidationError("unsupported financial fact schema_version")
    try:
        period_value = value["reporting_period"]
        currency_value = value["currency"]
        units_value = value["units"]
        column_value = value["table_column"]
        evidence_value = value["evidence"]
        verification_value = value["verification"]
        if not all(
            isinstance(item, dict)
            for item in (
                period_value,
                currency_value,
                units_value,
                column_value,
                evidence_value,
                verification_value,
            )
        ):
            raise FinancialFactValidationError("nested financial fact fields must be objects")
        period_type_value = period_value.get("period_type")
        unit_kind_value = units_value.get("kind")
        fact = FinancialFact(
            fact_id=value["fact_id"],
            metric=value["metric"],
            reported_value=value["reported_value"],
            normalized_value=_parse_decimal(value.get("normalized_value"), "normalized_value"),
            company=value["company"],
            reporting_period=ReportingPeriod(
                status=ResolutionStatus(period_value["status"]),
                period_type=PeriodType(period_type_value) if period_type_value else None,
                start_date=_parse_date(period_value.get("start_date"), "start_date"),
                end_date=_parse_date(period_value.get("end_date"), "end_date"),
                label=period_value.get("label"),
                note=period_value.get("note"),
            ),
            currency=Currency(
                status=ResolutionStatus(currency_value["status"]),
                code=currency_value.get("code"),
                note=currency_value.get("note"),
            ),
            units=Units(
                status=ResolutionStatus(units_value["status"]),
                kind=UnitKind(unit_kind_value) if unit_kind_value else None,
                scale=_parse_decimal(units_value.get("scale"), "units.scale"),
                label=units_value.get("label"),
                note=units_value.get("note"),
            ),
            table_column=TableColumn(
                status=ResolutionStatus(column_value["status"]),
                label=column_value.get("label"),
                note=column_value.get("note"),
            ),
            evidence=SupportingEvidence(
                document_id=evidence_value["document_id"],
                source_company=evidence_value["source_company"],
                source_report_year=evidence_value["source_report_year"],
                source_sha256=evidence_value["source_sha256"],
                physical_page_number=evidence_value["physical_page_number"],
                supporting_text=evidence_value["supporting_text"],
                supporting_text_sha256=evidence_value["supporting_text_sha256"],
                selected_text_sha256=evidence_value["selected_text_sha256"],
                evidence_method=evidence_value["evidence_method"],
                evidence_jsonl_sha256=evidence_value["evidence_jsonl_sha256"],
                evidence_manifest_sha256=evidence_value["evidence_manifest_sha256"],
                screening_warnings=_string_tuple(
                    evidence_value["screening_warnings"], "screening_warnings"
                ),
                known_limitations=_string_tuple(
                    evidence_value["known_limitations"], "known_limitations"
                ),
                review_provenance=evidence_value.get("review_provenance"),
            ),
            verification=Verification(
                status=VerificationStatus(verification_value["status"]),
                reviewed_by=verification_value.get("reviewed_by"),
                reviewed_at_utc=verification_value.get("reviewed_at_utc"),
                method=verification_value.get("method"),
                note=verification_value.get("note"),
            ),
            warnings=_string_tuple(value["warnings"], "warnings"),
            limitations=_string_tuple(value["limitations"], "limitations"),
        )
    except KeyError as exc:
        raise FinancialFactValidationError(
            f"missing required financial fact field: {exc}"
        ) from None
    except (TypeError, ValueError) as exc:
        if isinstance(exc, FinancialFactValidationError):
            raise
        raise FinancialFactValidationError(f"invalid financial fact enum value: {exc}") from None
    return fact


def load_financial_facts(path: str | Path, corpus: EvidenceCorpus) -> tuple[FinancialFact, ...]:
    """Load JSONL facts, reject duplicate IDs, and validate every evidence binding."""
    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise FinancialFactValidationError(
            f"could not read financial facts at {source}: {exc}"
        ) from exc
    facts: list[FinancialFact] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise FinancialFactValidationError(f"financial facts line {line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FinancialFactValidationError(
                f"malformed financial facts line {line_number}: {exc}"
            ) from exc
        try:
            fact = financial_fact_from_dict(value)
            fact.validate_against(corpus)
        except FinancialFactValidationError as exc:
            raise FinancialFactValidationError(
                f"financial facts line {line_number}: {exc}"
            ) from exc
        if fact.fact_id in seen_ids:
            raise FinancialFactValidationError(f"duplicate financial fact_id: {fact.fact_id!r}")
        seen_ids.add(fact.fact_id)
        facts.append(fact)
    return tuple(facts)
