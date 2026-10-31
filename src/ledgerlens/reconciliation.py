"""Deterministic reconciliation of verified, provenance-bound financial facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import MAX_EMAX, MIN_EMIN, Decimal, localcontext
from enum import StrEnum
from pathlib import Path

from ledgerlens.financial_facts import (
    FinancialFact,
    FinancialFactValidationError,
    PeriodType,
    ResolutionStatus,
    UnitKind,
    VerificationStatus,
    load_financial_facts,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus

RECONCILIATION_SCHEMA_VERSION = "1.0"


class ReconciliationError(ValueError):
    """A reconciliation input is invalid or cannot be compared safely."""


class ReconciliationStatus(StrEnum):
    """Auditable outcomes without source selection or conflict resolution."""

    AGREEMENT = "agreement"
    CONFLICT = "conflict"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True)
class VerifiedFactSource:
    """One source-specific fact file and the evidence corpus that validates it."""

    path: Path
    corpus: EvidenceCorpus

    def __init__(self, path: str | Path, corpus: EvidenceCorpus) -> None:
        object.__setattr__(self, "path", Path(path))
        object.__setattr__(self, "corpus", corpus)


@dataclass(frozen=True)
class SourceSummary:
    """Identity and integrity metadata for one loaded fact source."""

    path: str
    facts_sha256: str
    fact_count: int
    document_id: str
    source_sha256: str
    evidence_jsonl_sha256: str
    evidence_manifest_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "facts_sha256": self.facts_sha256,
            "fact_count": self.fact_count,
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "evidence_jsonl_sha256": self.evidence_jsonl_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
        }


@dataclass(frozen=True)
class ReconciliationObservation:
    """A complete fact snapshot plus its source-file and normalized-value context."""

    source_path: str
    fact: FinancialFact
    base_value: Decimal

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "base_value": str(self.base_value),
            "fact": self.fact.to_dict(),
        }


@dataclass(frozen=True)
class ComparisonKey:
    """The exact dimensions that make observations eligible for value comparison."""

    company: str
    metric: str
    period_type: PeriodType
    start_date: str | None
    end_date: str
    unit_kind: UnitKind
    currency_status: ResolutionStatus
    currency_code: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "company": self.company,
            "metric": self.metric,
            "reporting_period": {
                "period_type": self.period_type.value,
                "start_date": self.start_date,
                "end_date": self.end_date,
            },
            "unit_semantics": {
                "kind": self.unit_kind.value,
                "currency_status": self.currency_status.value,
                "currency_code": self.currency_code,
            },
        }


@dataclass(frozen=True)
class ReconciliationResult:
    """One comparison or incompatibility result retaining every input observation."""

    status: ReconciliationStatus
    company: str
    metric: str
    observations: tuple[ReconciliationObservation, ...]
    reasons: tuple[str, ...]
    comparison_key: ComparisonKey | None = None

    @property
    def result_id(self) -> str:
        """Return a deterministic content identifier for this exact result."""
        identity_payload = {
            **self._payload(),
            "observations": [
                {
                    "base_value": str(observation.base_value),
                    "fact": observation.fact.to_dict(),
                }
                for observation in self.observations
            ],
        }
        canonical = json.dumps(
            identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"sha256-{hashlib.sha256(canonical).hexdigest()}"

    def _payload(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "company": self.company,
            "metric": self.metric,
            "comparison_key": self.comparison_key.to_dict() if self.comparison_key else None,
            "reasons": list(self.reasons),
            "observations": [observation.to_dict() for observation in self.observations],
        }

    def to_dict(self) -> dict[str, object]:
        return {"result_id": self.result_id, **self._payload()}


@dataclass(frozen=True)
class ReconciliationReport:
    """Structured reconciliation output with source integrity and full provenance."""

    sources: tuple[SourceSummary, ...]
    results: tuple[ReconciliationResult, ...]
    duplicate_fact_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        counts = {status.value: 0 for status in ReconciliationStatus}
        for result in self.results:
            counts[result.status.value] += 1
        return {
            "schema_version": RECONCILIATION_SCHEMA_VERSION,
            "sources": [source.to_dict() for source in self.sources],
            "duplicate_fact_ids": list(self.duplicate_fact_ids),
            "counts": {
                "sources": len(self.sources),
                "observations": sum(source.fact_count for source in self.sources),
                "results": len(self.results),
                **counts,
            },
            "results": [result.to_dict() for result in self.results],
            "policy": {
                "number_type": "decimal",
                "source_preference": None,
                "conflict_resolution": None,
                "restatement_inference": False,
                "records_merged": False,
            },
        }


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ReconciliationError(f"could not read financial fact source at {path}: {exc}") from exc


def _base_value(fact: FinancialFact) -> Decimal:
    if fact.normalized_value is None or fact.units.scale is None:
        raise ReconciliationError(f"verified fact {fact.fact_id!r} lacks a value or unit scale")
    # Decimal multiplication obeys the active context. Supply enough coefficient precision so
    # normalization remains exact and independent of process-global Decimal settings.
    precision = len(fact.normalized_value.as_tuple().digits) + len(
        fact.units.scale.as_tuple().digits
    )
    with localcontext() as context:
        context.prec = max(1, precision)
        context.Emax = MAX_EMAX
        context.Emin = MIN_EMIN
        return fact.normalized_value * fact.units.scale


def _comparison_key(fact: FinancialFact) -> ComparisonKey:
    period = fact.reporting_period
    if (
        period.status is not ResolutionStatus.KNOWN
        or period.period_type is None
        or period.end_date is None
        or fact.units.status is not ResolutionStatus.KNOWN
        or fact.units.kind is None
    ):
        raise ReconciliationError(
            f"verified fact {fact.fact_id!r} has unresolved comparison context"
        )
    return ComparisonKey(
        company=fact.company,
        metric=fact.metric,
        period_type=period.period_type,
        start_date=period.start_date.isoformat() if period.start_date else None,
        end_date=period.end_date.isoformat(),
        unit_kind=fact.units.kind,
        currency_status=fact.currency.status,
        currency_code=fact.currency.code,
    )


def _observation_sort_key(observation: ReconciliationObservation) -> tuple[str, ...]:
    fact = observation.fact
    return (
        fact.evidence.document_id,
        fact.fact_id,
        observation.source_path,
        str(observation.base_value),
    )


def _key_sort_key(key: ComparisonKey) -> tuple[str, ...]:
    return (
        key.company,
        key.metric,
        key.end_date,
        key.start_date or "",
        key.period_type.value,
        key.unit_kind.value,
        key.currency_status.value,
        key.currency_code or "",
    )


def _incompatibility_reasons(left: ComparisonKey, right: ComparisonKey) -> tuple[str, ...]:
    reasons: list[str] = []
    if (
        left.period_type is not right.period_type
        or left.start_date != right.start_date
        or left.end_date != right.end_date
    ):
        reasons.append("reporting periods are not exactly equal")
    if left.unit_kind is not right.unit_kind:
        reasons.append("unit kinds are incompatible")
    if (
        left.currency_status is not right.currency_status
        or left.currency_code != right.currency_code
    ):
        reasons.append("currency semantics are incompatible")
    return tuple(reasons)


def _load_sources(
    sources: Iterable[VerifiedFactSource],
) -> tuple[tuple[SourceSummary, ...], tuple[ReconciliationObservation, ...]]:
    summaries: list[SourceSummary] = []
    observations: list[ReconciliationObservation] = []
    for source in sources:
        facts_sha256 = _file_sha256(source.path)
        try:
            facts = load_financial_facts(source.path, source.corpus)
        except FinancialFactValidationError as exc:
            raise ReconciliationError(
                f"invalid financial fact source {source.path}: {exc}"
            ) from exc
        unverified = [
            fact.fact_id
            for fact in facts
            if fact.verification.status is not VerificationStatus.VERIFIED
        ]
        if unverified:
            raise ReconciliationError(
                f"source {source.path} contains non-verified facts: {', '.join(unverified)}"
            )
        if _file_sha256(source.path) != facts_sha256:
            raise ReconciliationError(f"financial fact source changed while loading: {source.path}")
        source_path = source.path.as_posix()
        summaries.append(
            SourceSummary(
                path=source_path,
                facts_sha256=facts_sha256,
                fact_count=len(facts),
                document_id=source.corpus.document_id,
                source_sha256=source.corpus.source_sha256,
                evidence_jsonl_sha256=source.corpus.evidence_jsonl_sha256,
                evidence_manifest_sha256=source.corpus.manifest_sha256,
            )
        )
        observations.extend(
            ReconciliationObservation(source_path, fact, _base_value(fact)) for fact in facts
        )
    summaries.sort(
        key=lambda summary: (
            summary.document_id,
            summary.path,
            summary.facts_sha256,
        )
    )
    return tuple(summaries), tuple(observations)


def reconcile_financial_facts(sources: Iterable[VerifiedFactSource]) -> ReconciliationReport:
    """Load, validate, and compare source-specific verified facts deterministically.

    Equivalent observations remain separate records. The result never chooses a preferred source,
    infers a restatement, resolves a conflict, or changes a fact's verification state.
    """
    summaries, observations = _load_sources(sources)
    if not summaries:
        raise ReconciliationError("at least one verified financial fact source is required")

    fact_id_counts: dict[str, int] = {}
    grouped: dict[ComparisonKey, list[ReconciliationObservation]] = {}
    for observation in observations:
        fact_id_counts[observation.fact.fact_id] = (
            fact_id_counts.get(observation.fact.fact_id, 0) + 1
        )
        grouped.setdefault(_comparison_key(observation.fact), []).append(observation)

    results: list[ReconciliationResult] = []
    ordered_keys = sorted(grouped, key=_key_sort_key)
    for key in ordered_keys:
        group = tuple(sorted(grouped[key], key=_observation_sort_key))
        values = {observation.base_value for observation in group}
        document_ids = {observation.fact.evidence.document_id for observation in group}
        if len(values) > 1:
            status = ReconciliationStatus.CONFLICT
            reasons = ("normalized Decimal values differ; no value was selected",)
        elif len(document_ids) > 1:
            status = ReconciliationStatus.AGREEMENT
            reasons = ("normalized Decimal values agree across distinct source documents",)
        else:
            status = ReconciliationStatus.INSUFFICIENT_EVIDENCE
            reasons = ("fewer than two distinct source documents support this exact observation",)
        results.append(ReconciliationResult(status, key.company, key.metric, group, reasons, key))

    # Emit explicit records for non-equivalent candidate observations. Pairing comparison groups,
    # rather than individual records, keeps duplicate source records visible without multiplying
    # identical incompatibility findings.
    for index, left_key in enumerate(ordered_keys):
        for right_key in ordered_keys[index + 1 :]:
            if left_key.company != right_key.company or left_key.metric != right_key.metric:
                continue
            reasons = _incompatibility_reasons(left_key, right_key)
            if not reasons:
                continue
            incompatible = tuple(
                sorted((*grouped[left_key], *grouped[right_key]), key=_observation_sort_key)
            )
            results.append(
                ReconciliationResult(
                    ReconciliationStatus.INCOMPATIBLE,
                    left_key.company,
                    left_key.metric,
                    incompatible,
                    reasons,
                )
            )

    status_order = {status: index for index, status in enumerate(ReconciliationStatus)}
    results.sort(
        key=lambda result: (
            result.company,
            result.metric,
            status_order[result.status],
            result.result_id,
        )
    )
    duplicates = tuple(sorted(fact_id for fact_id, count in fact_id_counts.items() if count > 1))
    return ReconciliationReport(summaries, tuple(results), duplicates)


__all__ = [
    "RECONCILIATION_SCHEMA_VERSION",
    "ComparisonKey",
    "ReconciliationError",
    "ReconciliationObservation",
    "ReconciliationReport",
    "ReconciliationResult",
    "ReconciliationStatus",
    "SourceSummary",
    "VerifiedFactSource",
    "reconcile_financial_facts",
]
