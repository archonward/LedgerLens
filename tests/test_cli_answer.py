"""Offline behavior tests for the narrow financial answer CLI."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

import ledgerlens.answer_cli as answer_cli
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
from ledgerlens.retrieval import FusionComponent, HybridSearchResult, load_evidence_export
from ledgerlens.retrieval.bm25 import EvidenceCorpus


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass
class MockBackend:
    response: str
    calls: int = 0
    identity: ModelIdentity = ModelIdentity("mock", "controlled-cli")

    def generate(self, _prompt: str, config: InferenceConfig) -> str:
        assert config == InferenceConfig(max_new_tokens=96, temperature=0.0)
        self.calls += 1
        return self.response


@dataclass
class FixedHybridRetriever:
    corpus: EvidenceCorpus
    physical_pages: tuple[int, ...]

    def search(
        self, _query: str, *, top_k: int = 5, excerpt_characters: int = 500
    ) -> tuple[HybridSearchResult, ...]:
        assert excerpt_characters > 0
        selected = self.physical_pages[:top_k]
        return tuple(
            HybridSearchResult(
                rank,
                1.0 / rank,
                self.corpus.pages[page_number - 1],
                self.corpus.pages[page_number - 1].text,
                self.corpus.evidence_jsonl_sha256,
                self.corpus.manifest_sha256,
                60,
                FusionComponent(rank, 1.0, 1.0, 1.0 / (60 + rank)),
                FusionComponent(rank, 1.0, 1.0, 1.0 / (60 + rank)),
                "bm25_matched_passage",
            )
            for rank, page_number in enumerate(selected, start=1)
        )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    evidence_directory = tmp_path / "evidence"
    evidence_directory.mkdir()
    document_id = "sha256-" + "a" * 64
    source_sha256 = "a" * 64
    rows = (("Revenue", "$150"), ("Revenue", "$100"), ("Net income", "$30"))
    evidence_records = []
    for page_number, (metric, reported) in enumerate(rows, start=1):
        text = f"Controlled statement\n{metric}: {reported}"
        evidence_records.append(
            {
                "schema_version": "1.0",
                "document_id": document_id,
                "company": "Synthetic Company",
                "report_year": 2025,
                "physical_page_number": page_number,
                "text": text,
                "text_sha256": _sha256_bytes(text.encode()),
                "method": "canonical_extraction",
                "provenance": {"source_sha256": source_sha256},
                "screening_findings": {"canonical": {"status": "pass", "warnings": []}},
                "review_provenance": None,
                "known_limitations": ["Controlled fixture only."],
            }
        )
    evidence_raw = "".join(json.dumps(row) + "\n" for row in evidence_records).encode()
    (evidence_directory / "evidence.jsonl").write_bytes(evidence_raw)
    manifest = {
        "schema_version": "1.0",
        "document_id": document_id,
        "source_sha256": source_sha256,
        "selection_decisions": [
            {
                "physical_page_number": row["physical_page_number"],
                "method": row["method"],
                "selected_text_sha256": row["text_sha256"],
                "known_limitations": row["known_limitations"],
            }
            for row in evidence_records
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
        fact_id: str,
        metric: str,
        reported: str,
        normalized: str,
        period: ReportingPeriod,
    ) -> FinancialFact:
        page = corpus.pages[page_number - 1]
        return FinancialFact(
            fact_id,
            metric,
            reported,
            Decimal(normalized),
            page.company,
            period,
            Currency(ResolutionStatus.KNOWN, "USD"),
            Units(ResolutionStatus.KNOWN, UnitKind.MONETARY, Decimal("1000000"), "USD millions"),
            TableColumn(ResolutionStatus.KNOWN, period.label),
            SupportingEvidence.from_evidence(corpus, page, page.text),
            Verification(
                VerificationStatus.VERIFIED,
                "fixture reviewer",
                "2026-10-29T00:00:00Z",
                "controlled offline fixture",
            ),
            limitations=("Controlled fixture only.",),
        )

    facts = (
        fact(1, "revenue-2025", "revenue", "$150", "150", current),
        fact(2, "revenue-2024", "revenue", "$100", "100", prior),
        fact(3, "net-income-2025", "net_income", "$30", "30", current),
    )
    facts_path = tmp_path / "facts.jsonl"
    facts_path.write_text(
        "".join(json.dumps(fact.to_dict()) + "\n" for fact in facts), encoding="utf-8"
    )
    return evidence_directory, facts_path


def _arguments(evidence: Path, facts: Path, question: str, *extra: str) -> list[str]:
    return [
        "answer",
        question,
        "--company",
        "Synthetic Company",
        "--target-period-end",
        "2025-12-31",
        "--evidence-directory",
        str(evidence),
        "--financial-facts",
        str(facts),
        "--ollama",
        *extra,
    ]


def test_answer_json_preserves_calculation_citations_and_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence, facts = _write_inputs(tmp_path)
    backend = MockBackend("Revenue increased from the prior fiscal year.")
    monkeypatch.setattr(answer_cli, "_build_backend", lambda: backend)

    exit_code = main(_arguments(evidence, facts, "How much did revenue grow?", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert backend.calls == 1
    assert payload["status"] == "answered"
    assert payload["selected_fact_ids"] == ["revenue-2025", "revenue-2024"]
    assert payload["calculation"]["computed_value"] == "50.0"
    assert payload["answer"]["citations"][0]["marker"] == "[PDF p. 1]"
    assert payload["answer"]["citations"][0]["source_sha256"] == "a" * 64
    assert payload["answer"]["citations"][0]["known_limitations"] == ["Controlled fixture only."]
    assert payload["answer"]["provenance"]["model"]["model"] == "controlled-cli"
    assert payload["answer"]["provenance"]["explanation_validation"] == "passed"


def test_answer_human_output_is_readable_and_includes_evidence_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence, facts = _write_inputs(tmp_path)
    monkeypatch.setattr(
        answer_cli,
        "_build_backend",
        lambda: MockBackend("Revenue increased from the prior fiscal year."),
    )

    assert main(_arguments(evidence, facts, "How much did revenue grow?")) == 0
    output = capsys.readouterr().out

    assert "Status: answered" in output
    assert "Deterministic year over year growth: 50.0 percent" in output
    assert "[PDF p. 1]" in output
    assert "source_sha256=" + "a" * 64 in output
    assert "Evidence limitations: Controlled fixture only." in output
    assert '"explanation_validation": "passed"' in output


def test_answer_json_supports_net_profit_margin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence, facts = _write_inputs(tmp_path)
    backend = MockBackend("Net income represented a portion of revenue for the fiscal year.")
    monkeypatch.setattr(answer_cli, "_build_backend", lambda: backend)

    exit_code = main(_arguments(evidence, facts, "What was the net profit margin?", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert backend.calls == 1
    assert payload["task"] == "net_profit_margin"
    assert payload["selected_fact_ids"] == ["net-income-2025", "revenue-2025"]
    assert payload["calculation"]["computed_value"] == "20.0"
    assert {item["marker"] for item in payload["answer"]["citations"]} == {
        "[PDF p. 1]",
        "[PDF p. 3]",
    }


def test_answer_cli_optional_hybrid_path_associates_only_retrieved_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence, facts = _write_inputs(tmp_path)
    corpus = load_evidence_export(evidence)
    backend = MockBackend("Revenue increased from the prior fiscal year.")
    monkeypatch.setattr(answer_cli, "_build_backend", lambda: backend)
    monkeypatch.setattr(
        answer_cli,
        "_build_hybrid_retriever",
        lambda _args, _corpus: FixedHybridRetriever(corpus, (1, 2)),
    )

    exit_code = main(
        _arguments(
            evidence,
            facts,
            "How much did revenue grow?",
            "--hybrid-embedding-index",
            "controlled-index",
            "--retrieval-top-k",
            "2",
            "--json",
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert backend.calls == 1
    assert payload["status"] == "answered"
    assert payload["retrieval_association"]["associated_fact_ids"] == [
        "revenue-2025",
        "revenue-2024",
    ]
    assert payload["orchestration"]["selected_fact_ids"] == [
        "revenue-2025",
        "revenue-2024",
    ]
    assert (
        payload["retrieval_association"]["retrieved_pages"][0]["selected_text_sha256"]
        == corpus.pages[0].text_sha256
    )


def test_answer_cli_hybrid_failure_abstains_without_all_fact_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence, facts = _write_inputs(tmp_path)
    corpus = load_evidence_export(evidence)
    backend = MockBackend("unused")
    monkeypatch.setattr(answer_cli, "_build_backend", lambda: backend)
    monkeypatch.setattr(
        answer_cli,
        "_build_hybrid_retriever",
        lambda _args, _corpus: FixedHybridRetriever(corpus, (3,)),
    )

    exit_code = main(
        _arguments(
            evidence,
            facts,
            "How much did revenue grow?",
            "--hybrid-embedding-index",
            "controlled-index",
            "--retrieval-top-k",
            "1",
            "--json",
        )
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == answer_cli.ANSWER_ABSTAINED_EXIT_CODE
    assert backend.calls == 0
    assert payload["abstention_reason"] == "insufficient_associated_facts"
    assert payload["retrieval_association"]["associated_fact_ids"] == ["net-income-2025"]
    assert payload["orchestration"]["abstention_reason"] == "missing_fact"


def test_unsupported_question_returns_structured_abstention_without_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence, facts = _write_inputs(tmp_path)
    backend = MockBackend("unused")
    monkeypatch.setattr(answer_cli, "_build_backend", lambda: backend)

    exit_code = main(_arguments(evidence, facts, "What were operating expenses?", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == answer_cli.ANSWER_ABSTAINED_EXIT_CODE
    assert backend.calls == 0
    assert payload["status"] == "abstained"
    assert payload["abstention_reason"] == "unsupported_task"
    assert payload["calculation"] is None
    assert payload["answer"] is None


def test_grounding_failure_preserves_downstream_structured_abstention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence, facts = _write_inputs(tmp_path)
    monkeypatch.setattr(
        answer_cli,
        "_build_backend",
        lambda: MockBackend("Revenue increased significantly from the prior fiscal year."),
    )

    exit_code = main(_arguments(evidence, facts, "How much did revenue grow?", "--json"))
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == answer_cli.ANSWER_ABSTAINED_EXIT_CODE
    assert payload["abstention_reason"] == "answer_generation_abstained"
    assert payload["detail"] == "model_output_failed_grounding"
    assert payload["calculation"]["computed_value"] == "50.0"
    diagnostic = payload["answer"]["provenance"]["grounding_diagnostic"]
    assert diagnostic["rejection_codes"] == ["disallowed_vocabulary"]
    assert diagnostic["disallowed_words"] == ["significantly"]


def test_invalid_local_input_is_a_structured_cli_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(
        [
            "answer",
            "How much did revenue grow?",
            "--company",
            "Synthetic Company",
            "--target-period-end",
            "2025-12-31",
            "--evidence-directory",
            str(tmp_path / "missing"),
            "--financial-facts",
            str(tmp_path / "missing.jsonl"),
            "--ollama",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["status"] == "error"
    assert payload["error"]["code"] == "invalid_local_input"


@pytest.mark.parametrize("invalid_date", ["2025-2-03", "FY2025", "2025-02-30"])
def test_target_period_end_requires_an_exact_iso_date(invalid_date: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "answer",
                "How much did revenue grow?",
                "--company",
                "Synthetic Company",
                "--target-period-end",
                invalid_date,
                "--evidence-directory",
                "evidence",
                "--financial-facts",
                "facts.jsonl",
                "--ollama",
            ]
        )
    assert exc_info.value.code == 2


def test_ollama_opt_in_flag_is_required() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "answer",
                "How much did revenue grow?",
                "--company",
                "Synthetic Company",
                "--target-period-end",
                "2025-12-31",
                "--evidence-directory",
                "evidence",
                "--financial-facts",
                "facts.jsonl",
            ]
        )
    assert exc_info.value.code == 2
