from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

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
    ReconciliationError,
    ReconciliationReport,
    ReconciliationResult,
    ReconciliationStatus,
    VerifiedFactSource,
    reconcile_financial_facts,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage, load_evidence_export


def _corpus(document_suffix: str, text: str) -> EvidenceCorpus:
    document_id = f"sha256-{document_suffix * 64}"
    source_sha256 = document_suffix * 64
    text_sha256 = hashlib.sha256(text.encode()).hexdigest()
    page = EvidencePage(
        document_id=document_id,
        company="Synthetic Company",
        report_year=2025,
        physical_page_number=7,
        text=text,
        text_sha256=text_sha256,
        method="canonical_extraction",
        provenance={"source_sha256": source_sha256},
        screening_findings={},
        review_provenance=None,
        known_limitations=("Synthetic table association requires review.",),
    )
    return EvidenceCorpus(
        document_id=document_id,
        source_sha256=source_sha256,
        pages=(page,),
        excluded_pages=(),
        manifest={},
        evidence_jsonl_sha256=("a" if document_suffix != "a" else "c") * 64,
        manifest_sha256=("b" if document_suffix != "b" else "d") * 64,
    )


def _period(year: int = 2025) -> ReportingPeriod:
    return ReportingPeriod(
        status=ResolutionStatus.KNOWN,
        period_type=PeriodType.DURATION,
        start_date=date(year - 1, 1, 1),
        end_date=date(year - 1, 12, 31),
        label=f"Year ended December 31, {year - 1}",
    )


def _fact(
    corpus: EvidenceCorpus,
    *,
    fact_id: str,
    reported_value: str,
    normalized_value: str,
    scale: str = "1000000",
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
            Decimal(scale),
            "USD reported units",
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
            reviewed_at_utc="2026-10-31T00:00:00Z",
            method="Offline synthetic fixture review",
            note="Synthetic fixture; not a real financial claim.",
        ),
        limitations=("Synthetic fixture; not a real financial claim.",),
    )


def _source(
    tmp_path: Path,
    name: str,
    corpus: EvidenceCorpus,
    fact: FinancialFact,
) -> VerifiedFactSource:
    path = tmp_path / f"{name}.jsonl"
    path.write_text(json.dumps(fact.to_dict()) + "\n", encoding="utf-8")
    return VerifiedFactSource(path, corpus)


def _results(
    report: ReconciliationReport, status: ReconciliationStatus
) -> list[ReconciliationResult]:
    return [result for result in report.results if result.status is status]


def test_matching_values_agree_and_preserve_complete_provenance(tmp_path: Path) -> None:
    left_corpus = _corpus("1", "Revenue $100")
    right_corpus = _corpus("2", "Revenue $100")
    left = _fact(
        left_corpus,
        fact_id="left-revenue",
        reported_value="$100",
        normalized_value="100",
    )
    right = _fact(
        right_corpus,
        fact_id="right-revenue",
        reported_value="$100",
        normalized_value="100",
    )

    sources = (
        _source(tmp_path, "left", left_corpus, left),
        _source(tmp_path, "right", right_corpus, right),
    )
    report = reconcile_financial_facts(sources)

    agreement = _results(report, ReconciliationStatus.AGREEMENT)
    assert len(agreement) == 1
    assert [item.fact.fact_id for item in agreement[0].observations] == [
        "left-revenue",
        "right-revenue",
    ]
    serialized = agreement[0].to_dict()
    assert serialized["observations"][0]["fact"]["evidence"]["physical_page_number"] == 7
    assert serialized["observations"][0]["fact"]["verification"]["method"]
    assert serialized["observations"][0]["fact"]["limitations"] == [
        "Synthetic fixture; not a real financial claim."
    ]
    assert report.to_dict()["policy"]["source_preference"] is None
    assert report.to_dict()["policy"]["records_merged"] is False
    assert reconcile_financial_facts(reversed(sources)).to_dict() == report.to_dict()


def test_conflicting_values_are_reported_without_resolution(tmp_path: Path) -> None:
    left_corpus = _corpus("3", "Revenue $100")
    right_corpus = _corpus("4", "Revenue $101")
    sources = (
        _source(
            tmp_path,
            "left",
            left_corpus,
            _fact(
                left_corpus,
                fact_id="left-revenue",
                reported_value="$100",
                normalized_value="100",
            ),
        ),
        _source(
            tmp_path,
            "right",
            right_corpus,
            _fact(
                right_corpus,
                fact_id="right-revenue",
                reported_value="$101",
                normalized_value="101",
            ),
        ),
    )

    report = reconcile_financial_facts(sources)

    conflict = _results(report, ReconciliationStatus.CONFLICT)
    assert len(conflict) == 1
    assert {item.base_value for item in conflict[0].observations} == {
        Decimal("100000000"),
        Decimal("101000000"),
    }
    assert "no value was selected" in conflict[0].reasons[0]


def test_incompatible_periods_are_explicit_not_compared(tmp_path: Path) -> None:
    left_corpus = _corpus("5", "Revenue $100")
    right_corpus = _corpus("6", "Revenue $100")
    sources = (
        _source(
            tmp_path,
            "left",
            left_corpus,
            _fact(
                left_corpus,
                fact_id="left-revenue",
                reported_value="$100",
                normalized_value="100",
                period=_period(2025),
            ),
        ),
        _source(
            tmp_path,
            "right",
            right_corpus,
            _fact(
                right_corpus,
                fact_id="right-revenue",
                reported_value="$100",
                normalized_value="100",
                period=_period(2024),
            ),
        ),
    )

    report = reconcile_financial_facts(sources)

    assert len(_results(report, ReconciliationStatus.INSUFFICIENT_EVIDENCE)) == 2
    incompatible = _results(report, ReconciliationStatus.INCOMPATIBLE)
    assert len(incompatible) == 1
    assert incompatible[0].reasons == ("reporting periods are not exactly equal",)


def test_different_unit_scales_compare_as_exact_decimals(tmp_path: Path) -> None:
    millions = _corpus("7", "Revenue $1")
    thousands = _corpus("8", "Revenue $1,000")
    sources = (
        _source(
            tmp_path,
            "millions",
            millions,
            _fact(
                millions,
                fact_id="millions-revenue",
                reported_value="$1",
                normalized_value="1",
                scale="1000000",
            ),
        ),
        _source(
            tmp_path,
            "thousands",
            thousands,
            _fact(
                thousands,
                fact_id="thousands-revenue",
                reported_value="$1,000",
                normalized_value="1000",
                scale="1000",
            ),
        ),
    )

    with localcontext() as context:
        context.prec = 2
        report = reconcile_financial_facts(sources)

    agreement = _results(report, ReconciliationStatus.AGREEMENT)
    assert len(agreement) == 1
    assert {item.base_value for item in agreement[0].observations} == {Decimal("1000000")}


def test_different_currencies_are_incompatible_not_conflicting(tmp_path: Path) -> None:
    usd_corpus = _corpus("a", "Revenue $100")
    eur_corpus = _corpus("b", "Revenue EUR100")
    usd = _fact(
        usd_corpus,
        fact_id="usd-revenue",
        reported_value="$100",
        normalized_value="100",
    )
    eur = replace(
        _fact(
            eur_corpus,
            fact_id="eur-revenue",
            reported_value="EUR100",
            normalized_value="100",
        ),
        currency=Currency(ResolutionStatus.KNOWN, "EUR"),
    )

    report = reconcile_financial_facts(
        (
            _source(tmp_path, "usd", usd_corpus, usd),
            _source(tmp_path, "eur", eur_corpus, eur),
        )
    )

    assert not _results(report, ReconciliationStatus.CONFLICT)
    incompatible = _results(report, ReconciliationStatus.INCOMPATIBLE)
    assert len(incompatible) == 1
    assert incompatible[0].reasons == ("currency semantics are incompatible",)


def test_duplicate_observations_in_one_document_do_not_create_agreement(
    tmp_path: Path,
) -> None:
    corpus = _corpus("d", "Revenue $100")
    first = _fact(
        corpus,
        fact_id="first-source-record",
        reported_value="$100",
        normalized_value="100",
    )
    second = replace(first, fact_id="second-source-record")
    path = tmp_path / "same-document.jsonl"
    path.write_text(
        "\n".join(json.dumps(fact.to_dict()) for fact in (first, second)) + "\n",
        encoding="utf-8",
    )

    report = reconcile_financial_facts((VerifiedFactSource(path, corpus),))

    insufficient = _results(report, ReconciliationStatus.INSUFFICIENT_EVIDENCE)
    assert len(insufficient) == 1
    assert len(insufficient[0].observations) == 2
    assert not _results(report, ReconciliationStatus.AGREEMENT)
    assert report.duplicate_fact_ids == ()


def test_duplicate_fact_ids_across_source_entries_remain_visible(
    tmp_path: Path,
) -> None:
    corpus = _corpus("9", "Revenue $100")
    source = _source(
        tmp_path,
        "duplicate",
        corpus,
        _fact(
            corpus,
            fact_id="duplicate-revenue",
            reported_value="$100",
            normalized_value="100",
        ),
    )

    report = reconcile_financial_facts((source, source))

    insufficient = _results(report, ReconciliationStatus.INSUFFICIENT_EVIDENCE)
    assert len(insufficient) == 1
    assert len(insufficient[0].observations) == 2
    assert report.duplicate_fact_ids == ("duplicate-revenue",)
    assert report.to_dict()["counts"]["observations"] == 2


def test_fact_file_must_validate_against_its_corresponding_corpus(tmp_path: Path) -> None:
    fact_corpus = _corpus("e", "Revenue $100")
    wrong_corpus = _corpus("f", "Revenue $100")
    source = _source(
        tmp_path,
        "mismatched",
        wrong_corpus,
        _fact(
            fact_corpus,
            fact_id="mismatched-revenue",
            reported_value="$100",
            normalized_value="100",
        ),
    )

    with pytest.raises(ReconciliationError, match="document_id does not match"):
        reconcile_financial_facts((source,))


def test_non_verified_fact_is_rejected(tmp_path: Path) -> None:
    corpus = _corpus("0", "Revenue $100")
    verified = _fact(
        corpus,
        fact_id="unverified-revenue",
        reported_value="$100",
        normalized_value="100",
    )
    unverified = replace(
        verified,
        verification=Verification(VerificationStatus.UNVERIFIED),
    )
    source = _source(tmp_path, "unverified", corpus, unverified)

    with pytest.raises(ReconciliationError, match="contains non-verified facts"):
        reconcile_financial_facts((source,))


def test_local_eighteen_fact_corpus_reconciles_when_artifacts_exist() -> None:
    root = Path(__file__).parents[1]
    document_ids = (
        "sha256-b5751f4bb240eeb63d119211d6cc495e1d623897f1f2af1887e387bad1404c84",
        "sha256-757b344ab62ba282f6e44e0702e8ce5f190d88fb23b5e0a2beaf4cb277a395cc",
        "sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c",
    )
    evidence_paths = [root / "artifacts" / "evidence" / item for item in document_ids]
    fact_paths = [
        root / "artifacts" / "financial-facts" / document_id / "milestone-8b-verified-facts.jsonl"
        for document_id in document_ids
    ]
    fact_paths.append(root / "artifacts" / "financial-facts" / document_ids[2] / "facts.jsonl")
    if not all(
        path.is_file()
        for path in [
            *(path / "manifest.json" for path in evidence_paths),
            *(path / "evidence.jsonl" for path in evidence_paths),
            *fact_paths,
        ]
    ):
        pytest.skip("Git-ignored Milestone 8B artifacts are not present")

    corpora = [load_evidence_export(path) for path in evidence_paths]
    report = reconcile_financial_facts(
        (
            VerifiedFactSource(fact_paths[0], corpora[0]),
            VerifiedFactSource(fact_paths[1], corpora[1]),
            VerifiedFactSource(fact_paths[2], corpora[2]),
            VerifiedFactSource(fact_paths[3], corpora[2]),
        )
    )

    assert report.to_dict()["counts"]["observations"] == 18
    assert len(_results(report, ReconciliationStatus.AGREEMENT)) == 6
    assert len(_results(report, ReconciliationStatus.CONFLICT)) == 0
    assert len(_results(report, ReconciliationStatus.INSUFFICIENT_EVIDENCE)) == 4
    assert len(_results(report, ReconciliationStatus.INCOMPATIBLE)) == 20
    assert report.duplicate_fact_ids == ()
