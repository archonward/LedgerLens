"""Offline tests for the Phase 3C CLI, presentation, and evaluation workflow."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

import ledgerlens.multi_document_cli as multi_cli
from ledgerlens.cli import main
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
from ledgerlens.generation import InferenceConfig, ModelIdentity
from ledgerlens.multi_document_evaluation import (
    MultiDocumentEvaluationError,
    load_multi_document_evaluation_set,
    run_multi_document_evaluation,
    write_multi_document_evaluation,
)
from ledgerlens.multi_document_workflow import present_percentage
from ledgerlens.reconciliation import (
    ReconciliationStatus,
    VerifiedFactSource,
    reconcile_financial_facts,
)
from ledgerlens.retrieval import load_evidence_export
from ledgerlens.retrieval.bm25 import EvidenceCorpus


@dataclass
class MockBackend:
    output: object = "Revenue increased from the prior fiscal year."
    calls: int = 0
    identity: ModelIdentity = ModelIdentity("mock", "multi-cli", "fixture-v1")

    def generate(self, _prompt: str, config: InferenceConfig) -> str:
        assert config == InferenceConfig(max_new_tokens=96, temperature=0.0)
        self.calls += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output  # type: ignore[return-value]


def _write_source(
    root: Path,
    suffix: str,
    prefix: str,
) -> tuple[Path, Path, EvidenceCorpus]:
    source_root = root / prefix
    evidence_directory = source_root / "evidence"
    evidence_directory.mkdir(parents=True)
    document_id = "sha256-" + suffix * 64
    source_sha256 = suffix * 64
    rows = (("Revenue", "$150"), ("Revenue", "$100"), ("Net income", "$30"))
    records: list[dict[str, object]] = []
    for page_number, (metric, reported) in enumerate(rows, 1):
        text = f"Controlled {prefix} statement\n{metric}: {reported}"
        records.append(
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "company": "Synthetic Company",
                "report_year": 2025 if prefix == "left" else 2024,
                "physical_page_number": page_number,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "method": "canonical_extraction",
                "provenance": {"source_sha256": source_sha256},
                "screening_findings": {"canonical": {"status": "pass", "warnings": []}},
                "review_provenance": None,
                "known_limitations": [f"Controlled {prefix} fixture only."],
            }
        )
    (evidence_directory / "evidence.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in records), encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "document_id": document_id,
        "source_sha256": source_sha256,
        "selection_decisions": [
            {
                "physical_page_number": item["physical_page_number"],
                "method": item["method"],
                "selected_text_sha256": item["text_sha256"],
                "known_limitations": item["known_limitations"],
            }
            for item in records
        ],
        "excluded_pages": [],
        "counts": {
            "selected_pages": 3,
            "canonical_pages": 3,
            "reviewed_recoveries": 0,
            "excluded_pages": 0,
            "pages_with_limitations": 3,
        },
        "pages_with_limitations": [1, 2, 3],
    }
    (evidence_directory / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    corpus = load_evidence_export(evidence_directory)
    current = ReportingPeriod(
        ResolutionStatus.KNOWN,
        PeriodType.DURATION,
        date(2025, 1, 1),
        date(2025, 12, 31),
        "FY2025",
    )
    prior = ReportingPeriod(
        ResolutionStatus.KNOWN,
        PeriodType.DURATION,
        date(2024, 1, 1),
        date(2024, 12, 31),
        "FY2024",
    )

    def fact(
        page_number: int,
        name: str,
        metric: str,
        reported: str,
        normalized: str,
        period: ReportingPeriod,
    ) -> FinancialFact:
        page = corpus.pages[page_number - 1]
        return FinancialFact(
            f"{prefix}-{name}",
            metric,
            reported,
            Decimal(normalized),
            "Synthetic Company",
            period,
            Currency(ResolutionStatus.KNOWN, "USD"),
            Units(ResolutionStatus.KNOWN, UnitKind.MONETARY, Decimal("1000000"), "USD millions"),
            TableColumn(ResolutionStatus.KNOWN, period.label),
            SupportingEvidence.from_evidence(corpus, page, page.text),
            Verification(
                VerificationStatus.VERIFIED,
                "fixture reviewer",
                "2026-11-13T00:00:00Z",
                "controlled offline fixture",
            ),
            limitations=(f"Controlled {prefix} fixture only.",),
        )

    facts = (
        fact(1, "current-revenue", "revenue", "$150", "150", current),
        fact(2, "prior-revenue", "revenue", "$100", "100", prior),
        fact(3, "net-income", "net_income", "$30", "30", current),
    )
    facts_path = source_root / "facts.jsonl"
    facts_path.write_text(
        "".join(json.dumps(item.to_dict()) + "\n" for item in facts), encoding="utf-8"
    )
    return evidence_directory, facts_path, corpus


def _agreement(report: object, metric: str, end: str) -> object:
    matches = [
        item
        for item in report.results  # type: ignore[attr-defined]
        if item.status is ReconciliationStatus.AGREEMENT
        and item.metric == metric
        and item.comparison_key is not None
        and item.comparison_key.end_date == end
    ]
    assert len(matches) == 1
    return matches[0]


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    left_evidence, left_facts, left = _write_source(tmp_path, "1", "left")
    right_evidence, right_facts, right = _write_source(tmp_path, "2", "right")
    report = reconcile_financial_facts(
        (VerifiedFactSource(left_facts, left), VerifiedFactSource(right_facts, right))
    )
    current = _agreement(report, "revenue", "2025-12-31")
    prior = _agreement(report, "revenue", "2024-12-31")
    request = {
        "schema_version": "1.0",
        "question": "How much did revenue grow?",
        "task": "revenue_growth",
        "company": "Synthetic Company",
        "sources": [
            {
                "document_id": left.document_id,
                "source_sha256": left.source_sha256,
                "evidence_directory": left_evidence.relative_to(tmp_path).as_posix(),
                "financial_facts": left_facts.relative_to(tmp_path).as_posix(),
            },
            {
                "document_id": right.document_id,
                "source_sha256": right.source_sha256,
                "evidence_directory": right_evidence.relative_to(tmp_path).as_posix(),
                "financial_facts": right_facts.relative_to(tmp_path).as_posix(),
            },
        ],
        "selections": [
            {
                "role": "current",
                "reconciliation_result_id": current.result_id,  # type: ignore[attr-defined]
                "selected_fact_id": "left-current-revenue",
            },
            {
                "role": "prior",
                "reconciliation_result_id": prior.result_id,  # type: ignore[attr-defined]
                "selected_fact_id": "right-prior-revenue",
            },
        ],
        "selection_policy": {"single_source": "abstain"},
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
    return request_path, request


def _arguments(request: Path, *extra: str) -> list[str]:
    return ["answer-multi", str(request), "--ollama", *extra]


def test_answer_multi_json_runs_trusted_pipeline_and_preserves_full_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, _ = _fixture(tmp_path)
    backend = MockBackend()
    monkeypatch.setattr(multi_cli, "_build_backend", lambda: backend)

    assert main(_arguments(request_path, "--json")) == 0
    payload = json.loads(capsys.readouterr().out)

    assert backend.calls == 1
    assert payload["status"] == "answered"
    assert payload["answer"]["computed_value"]["computed_value"] == "50.0"
    assert payload["presentation"]["exact_value"] == "50.0"
    assert payload["presentation"]["display_value"] == "50.00%"
    assert (
        len({item["annual_report"]["document_id"] for item in payload["answer"]["citations"]}) == 2
    )
    assert payload["reconciliation"]["policy"]["source_preference"] is None
    assert payload["answer"]["provenance"]["explanation_validation"] == "passed"
    selected = payload["orchestration"]["selection"]["decisions"]
    assert [item["selected_fact"]["fact_id"] for item in selected] == [
        "left-current-revenue",
        "right-prior-revenue",
    ]


def test_answer_multi_human_output_uses_deterministic_rounding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    request_path, _ = _fixture(tmp_path)
    monkeypatch.setattr(multi_cli, "_build_backend", lambda: MockBackend())

    assert main(_arguments(request_path)) == 0
    output = capsys.readouterr().out

    assert "Computed result:" in output
    assert "50.00% (exact Decimal: 50.0)" in output
    assert "Source-selection audit:" in output
    assert "Annual-report citations:" in output
    assert "explanation_validation" in output


@pytest.mark.parametrize("failure", ["missing", "duplicate", "mismatch", "stale"])
def test_answer_multi_rejects_unsafe_source_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    request_path, request = _fixture(tmp_path)
    if failure == "missing":
        request["sources"][0]["evidence_directory"] = "missing"  # type: ignore[index]
    elif failure == "duplicate":
        request["sources"].append(dict(request["sources"][0]))  # type: ignore[index]
    elif failure == "mismatch":
        request["sources"][0]["document_id"] = "sha256-" + "9" * 64  # type: ignore[index]
    else:
        manifest = tmp_path / request["sources"][0]["evidence_directory"] / "manifest.json"  # type: ignore[index,operator]
        manifest.write_text(manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
    request_path.write_text(json.dumps(request) + "\n", encoding="utf-8")
    backend = MockBackend()
    monkeypatch.setattr(multi_cli, "_build_backend", lambda: backend)

    assert main(_arguments(request_path, "--json")) == 2
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == "error"
    assert payload["error"]["code"] == "invalid_local_input"
    assert backend.calls == 0


@pytest.mark.parametrize(
    ("output", "status"),
    [
        ("Revenue increased because demand grew.", "model_output_rejected"),
        (RuntimeError("offline failure"), "inference_failed"),
    ],
)
def test_answer_multi_distinguishes_grounding_rejection_and_inference_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output: object,
    status: str,
) -> None:
    request_path, _ = _fixture(tmp_path)
    monkeypatch.setattr(multi_cli, "_build_backend", lambda: MockBackend(output))

    assert main(_arguments(request_path, "--json")) == 3
    payload = json.loads(capsys.readouterr().out)

    assert payload["status"] == status
    assert payload["answer"]["computed_value"]["computed_value"] == "50.0"
    assert len(payload["answer"]["citations"]) == 2
    assert payload["answer"]["model_generated_explanation"] is None


def test_percentage_presentation_is_half_even_and_never_changes_exact_value() -> None:
    assert present_percentage(Decimal("1.005")).to_dict()["display_value"] == "1.00%"
    presentation = present_percentage(Decimal("1.015"))
    assert presentation.display_value == "1.02%"
    assert presentation.exact_value == "1.015"


def _evaluation_file(tmp_path: Path, request: dict[str, object], *, wrong: bool = False) -> Path:
    case = {
        "case_id": "synthetic-growth",
        "question": request["question"],
        "task": request["task"],
        "company": request["company"],
        "selections": request["selections"],
        "selection_policy": request["selection_policy"],
        "reference": {
            "expected_computed_value": "51" if wrong else "50.0",
            "expected_citations": [
                {
                    "role": "current",
                    "document_id": (
                        "sha256-" + "9" * 64 if wrong else request["sources"][0]["document_id"]  # type: ignore[index]
                    ),
                },
                {
                    "role": "prior",
                    "document_id": request["sources"][1]["document_id"],  # type: ignore[index]
                },
            ],
            "provenance": {
                "reviewed_by": "fixture reviewer",
                "reviewed_at_utc": "2026-11-13T00:00:00Z",
                "method": "controlled explicit reference",
                "note": "Synthetic test only.",
            },
        },
    }
    value = {
        "schema_version": "1.0",
        "dataset_id": "synthetic-multi-document-v1",
        "sources": request["sources"],
        "cases": [case],
    }
    path = tmp_path / ("evaluation-wrong.json" if wrong else "evaluation.json")
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def test_evaluation_scores_references_and_publishes_fingerprinted_output(tmp_path: Path) -> None:
    _, request = _fixture(tmp_path)
    evaluation_set = load_multi_document_evaluation_set(_evaluation_file(tmp_path, request))

    payload = run_multi_document_evaluation(
        evaluation_set,
        MockBackend(),
        created_at_utc="2026-11-13T00:00:00Z",
    )

    assert payload["summary"] == {
        "cases": 1,
        "validated_calculations": 1,
        "numerically_correct": 1,
        "citation_correct": 1,
        "model_invocation_attempted": 1,
        "model_invocation_completed": 1,
        "grounding_accepted": 1,
        "grounding_rejected": 0,
        "validation_abstained": 0,
        "inference_failed": 0,
        "reference_checks_passed": True,
    }
    destination = write_multi_document_evaluation(payload, tmp_path / "artifacts/evaluation")
    written = destination / "evaluation.json"
    assert json.loads(written.read_text(encoding="utf-8")) == payload
    with pytest.raises(MultiDocumentEvaluationError, match="already exists"):
        write_multi_document_evaluation(payload, tmp_path / "artifacts/evaluation")


def test_evaluation_reports_incorrect_numeric_and_citation_references(tmp_path: Path) -> None:
    _, request = _fixture(tmp_path)
    evaluation_set = load_multi_document_evaluation_set(
        _evaluation_file(tmp_path, request, wrong=True)
    )

    payload = run_multi_document_evaluation(
        evaluation_set,
        MockBackend(),
        created_at_utc="2026-11-13T00:00:00Z",
    )
    case = payload["cases"][0]  # type: ignore[index]

    assert case["checks"] == {"numerical_correct": False, "citation_correct": False}
    assert payload["summary"]["reference_checks_passed"] is False  # type: ignore[index]


def test_evaluate_answer_multi_cli_writes_reproducible_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, request = _fixture(tmp_path)
    evaluation_path = _evaluation_file(tmp_path, request)
    output_root = tmp_path / "artifacts/evaluation"
    monkeypatch.setattr(multi_cli, "_build_backend", lambda: MockBackend())

    exit_code = main(
        [
            "evaluate-answer-multi",
            str(evaluation_path),
            "--output-root",
            str(output_root),
            "--ollama",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["summary"]["reference_checks_passed"] is True
    destination = output_root / payload["configuration_id"] / "evaluation.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
