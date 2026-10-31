from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ledgerlens.calculations import FactOperand, profit_margin, year_over_year_growth
from ledgerlens.financial_facts import load_financial_facts
from ledgerlens.generation import (
    AnswerGenerator,
    AnswerRequest,
    AnswerStatus,
    AnswerTask,
    CallableLocalModelBackend,
    ModelIdentity,
)
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationRequest,
    OrchestrationStatus,
)
from ledgerlens.retrieval import (
    HybridRetriever,
    RetrievalError,
    create_fastembed_backend,
    load_embedding_index,
)
from ledgerlens.retrieval.bm25 import load_evidence_export
from ledgerlens.retrieval_facts import (
    RetrievalAnswerAbstentionReason,
    RetrievalFinancialAnswerOrchestrator,
)

DOCUMENT_ID = "sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c"
SOURCE_SHA256 = DOCUMENT_ID.removeprefix("sha256-")
SOURCE_PDF = Path("data/raw/nvidia-2026-annual-report.pdf")
EVIDENCE_DIRECTORY = Path("artifacts/evidence") / DOCUMENT_ID
FACTS_DIRECTORY = Path("artifacts/financial-facts") / DOCUMENT_ID
EMBEDDING_INDEX = Path("artifacts/embeddings") / DOCUMENT_ID / "baai-bge-small-en-v1-5-c32e6154d1bb"
MODEL_CACHE = Path("artifacts/models")
SCRIPT = Path("scripts/demonstrate_nvidia_fy2026.py")


def _require_local_artifacts() -> None:
    required = (
        SOURCE_PDF,
        EVIDENCE_DIRECTORY / "evidence.jsonl",
        EVIDENCE_DIRECTORY / "manifest.json",
        FACTS_DIRECTORY / "facts.jsonl",
        FACTS_DIRECTORY / "review.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip("ignored NVIDIA review artifacts are unavailable: " + ", ".join(missing))


def _run_demo(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-pdf",
            str(SOURCE_PDF),
            "--evidence-directory",
            str(EVIDENCE_DIRECTORY),
            "--facts",
            str(FACTS_DIRECTORY / "facts.jsonl"),
            "--review-record",
            str(FACTS_DIRECTORY / "review.json"),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _real_hybrid_retriever(corpus) -> HybridRetriever:
    if os.environ.get("LEDGERLENS_RUN_REAL_HYBRID_FACTS") != "1":
        pytest.skip("set LEDGERLENS_RUN_REAL_HYBRID_FACTS=1 to execute the cached semantic model")
    required = (
        EMBEDDING_INDEX / "manifest.json",
        EMBEDDING_INDEX / "vectors.jsonl",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.skip("ignored NVIDIA embedding artifacts are unavailable: " + ", ".join(missing))
    try:
        backend = create_fastembed_backend(MODEL_CACHE, allow_download=False)
    except RetrievalError as exc:
        if "dependencies are unavailable" in str(exc):
            pytest.skip(f"optional semantic dependencies are unavailable: {exc}")
        raise
    index = load_embedding_index(EMBEDDING_INDEX, corpus)
    return HybridRetriever(corpus, index, backend)


def test_reviewed_nvidia_facts_reproduce_expected_calculations() -> None:
    _require_local_artifacts()

    completed = _run_demo()

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["source"]["document_id"] == DOCUMENT_ID
    assert payload["source"]["source_sha256"] == SOURCE_SHA256
    assert payload["review"] == {
        "review_record_sha256": "79ace3d1e3282991aa11d2989830cfafc5e1408db2f513821f49ee32c0cc99e5",
        "reviewer_identity": "OpenAI Codex automated coding agent",
        "method": (
            "Codex-assisted visual inspection of original PDF renderings, followed by exact "
            "comparison with the selected page-139 evidence text; no independent human review"
        ),
        "independent_human_verification_claimed": False,
        "physical_pages_reviewed": [22, 69, 139, 145],
    }

    facts = {fact["fact_id"]: fact for fact in payload["facts"]}
    assert {fact_id: fact["reported_value"] for fact_id, fact in facts.items()} == {
        "nvidia-fy2026-revenue": "$ 215,938",
        "nvidia-fy2025-revenue": "$ 130,497",
        "nvidia-fy2026-net-income": "$ 120,067",
        "nvidia-fy2025-net-income": "$ 72,880",
    }
    for fact in facts.values():
        assert fact["verification"]["status"] == "verified"
        assert fact["verification"]["reviewed_by"] == "OpenAI Codex automated coding agent"
        assert "not independent human verification" in fact["verification"]["method"]
        assert fact["currency"] == {"status": "known", "code": "USD", "note": None}
        assert fact["units"]["scale"] == "1000000"
        assert fact["evidence"]["physical_page_number"] == 139
        assert fact["evidence"]["source_sha256"] == SOURCE_SHA256

    calculations = payload["calculations"]
    assert calculations["revenue_growth_fy2026_vs_fy2025"]["computed_value"] == (
        "65.47353579009479145114447075411695"
    )
    assert calculations["net_profit_margin_fy2026"]["computed_value"] == (
        "55.60253406070260908223656790375015"
    )
    assert calculations["net_profit_margin_fy2025"]["computed_value"] == (
        "55.84802715771243783381993455788256"
    )
    for calculation in calculations.values():
        assert calculation["calculation_id"].startswith("sha256-")
        assert len(calculation["inputs"]) == 2
        assert all("evidence" in item["fact"] for item in calculation["inputs"])


def test_demonstration_rejects_a_changed_review_record(tmp_path: Path) -> None:
    _require_local_artifacts()
    review = json.loads((FACTS_DIRECTORY / "review.json").read_text(encoding="utf-8"))
    review["review"]["method"] += " Changed after fact creation."
    changed_review = tmp_path / "changed-review.json"
    changed_review.write_text(json.dumps(review), encoding="utf-8")

    completed = _run_demo("--review-record", str(changed_review))

    assert completed.returncode == 2
    assert "not bound to the supplied review record" in completed.stderr


def test_reviewed_nvidia_facts_ground_revenue_growth_and_margin_answers() -> None:
    _require_local_artifacts()
    completed = _run_demo()
    assert completed.returncode == 0, completed.stderr
    corpus = load_evidence_export(EVIDENCE_DIRECTORY)
    facts = {
        fact.fact_id: fact for fact in load_financial_facts(FACTS_DIRECTORY / "facts.jsonl", corpus)
    }
    revenue_current = facts["nvidia-fy2026-revenue"]
    revenue_prior = facts["nvidia-fy2025-revenue"]
    net_income = facts["nvidia-fy2026-net-income"]
    growth = year_over_year_growth(
        FactOperand(revenue_current, corpus), FactOperand(revenue_prior, corpus)
    )
    margin = profit_margin(FactOperand(net_income, corpus), FactOperand(revenue_current, corpus))
    identity = ModelIdentity("mock-local", "controlled-open-model", "integration-test")

    growth_answer = AnswerGenerator(
        CallableLocalModelBackend(
            identity,
            lambda _prompt, _config: "Revenue increased from the prior fiscal year.",
        )
    ).generate(
        AnswerRequest(
            "How much did NVIDIA revenue grow in FY2026?",
            AnswerTask.REVENUE_GROWTH,
            (revenue_current, revenue_prior),
            growth,
            corpus,
        )
    )
    margin_answer = AnswerGenerator(
        CallableLocalModelBackend(
            identity,
            lambda _prompt, _config: (
                "Net income represented a portion of revenue for the fiscal year."
            ),
        )
    ).generate(
        AnswerRequest(
            "What was NVIDIA's FY2026 net profit margin?",
            AnswerTask.NET_PROFIT_MARGIN,
            (net_income, revenue_current),
            margin,
            corpus,
        )
    )

    assert growth_answer.status is AnswerStatus.ANSWERED
    assert growth_answer.computed_value is not None
    assert growth_answer.computed_value.computed_value == ("65.47353579009479145114447075411695")
    assert margin_answer.status is AnswerStatus.ANSWERED
    assert margin_answer.computed_value is not None
    assert margin_answer.computed_value.computed_value == ("55.60253406070260908223656790375015")
    for answer in (growth_answer, margin_answer):
        assert {citation.marker for citation in answer.citations} == {"[PDF p. 139]"}
        assert all(citation.source_sha256 == SOURCE_SHA256 for citation in answer.citations)
        assert "Source-reported values:" in answer.answer_text
        assert "Computed result:" in answer.answer_text
        assert "Model-generated explanation:" in answer.answer_text


def test_reviewed_nvidia_facts_are_selected_by_deterministic_orchestration() -> None:
    _require_local_artifacts()
    corpus = load_evidence_export(EVIDENCE_DIRECTORY)
    facts = load_financial_facts(FACTS_DIRECTORY / "facts.jsonl", corpus)
    by_id = {fact.fact_id: fact for fact in facts}
    current_revenue = by_id["nvidia-fy2026-revenue"]
    target_end = current_revenue.reporting_period.end_date
    assert target_end is not None
    responses = iter(
        (
            "Revenue increased from the prior fiscal year.",
            "Net income represented a portion of revenue for the fiscal year.",
        )
    )
    backend = CallableLocalModelBackend(
        ModelIdentity("mock-local", "controlled-open-model", "orchestration-test"),
        lambda _prompt, _config: next(responses),
    )
    orchestrator = FinancialAnswerOrchestrator(AnswerGenerator(backend))

    growth = orchestrator.answer(
        OrchestrationRequest(
            "How much did NVIDIA revenue grow in FY2026?",
            current_revenue.company,
            target_end,
            facts,
            corpus,
        )
    )
    margin = orchestrator.answer(
        OrchestrationRequest(
            "What was NVIDIA's FY2026 net profit margin?",
            current_revenue.company,
            target_end,
            facts,
            corpus,
        )
    )

    assert growth.status is OrchestrationStatus.ANSWERED
    assert growth.selected_fact_ids == (
        "nvidia-fy2026-revenue",
        "nvidia-fy2025-revenue",
    )
    assert growth.calculation is not None
    assert str(growth.calculation.computed_value) == "65.47353579009479145114447075411695"
    assert margin.status is OrchestrationStatus.ANSWERED
    assert margin.selected_fact_ids == (
        "nvidia-fy2026-net-income",
        "nvidia-fy2026-revenue",
    )
    assert margin.calculation is not None
    assert str(margin.calculation.computed_value) == "55.60253406070260908223656790375015"


def test_real_hybrid_retrieval_associates_nvidia_facts_for_growth_and_margin() -> None:
    _require_local_artifacts()
    corpus = load_evidence_export(EVIDENCE_DIRECTORY)
    facts = load_financial_facts(FACTS_DIRECTORY / "facts.jsonl", corpus)
    by_id = {fact.fact_id: fact for fact in facts}
    target_end = by_id["nvidia-fy2026-revenue"].reporting_period.end_date
    assert target_end is not None
    retriever = _real_hybrid_retriever(corpus)

    def explain(prompt, _config):
        if "net profit margin" in prompt.casefold():
            return "Net income represented a portion of revenue for the fiscal year."
        return "Revenue increased from the prior fiscal year."

    backend = CallableLocalModelBackend(
        ModelIdentity("mock-local", "controlled-open-model", "retrieval-integration-test"),
        explain,
    )
    integration = RetrievalFinancialAnswerOrchestrator(
        FinancialAnswerOrchestrator(AnswerGenerator(backend)),
        retriever,
    )
    growth = integration.answer(
        OrchestrationRequest(
            "How much did NVIDIA revenue grow in FY2026?",
            "NVIDIA",
            target_end,
            facts,
            corpus,
        ),
        top_k=5,
    )
    margin = integration.answer(
        OrchestrationRequest(
            "What was NVIDIA's FY2026 net profit margin?",
            "NVIDIA",
            target_end,
            facts,
            corpus,
        ),
        top_k=5,
    )

    assert growth.status is OrchestrationStatus.ANSWERED
    assert growth.orchestration is not None
    assert growth.orchestration.selected_fact_ids == (
        "nvidia-fy2026-revenue",
        "nvidia-fy2025-revenue",
    )
    assert growth.orchestration.calculation is not None
    assert str(growth.orchestration.calculation.computed_value) == (
        "65.47353579009479145114447075411695"
    )
    assert margin.status is OrchestrationStatus.ANSWERED
    assert margin.orchestration is not None
    assert margin.orchestration.selected_fact_ids == (
        "nvidia-fy2026-net-income",
        "nvidia-fy2026-revenue",
    )
    assert margin.orchestration.calculation is not None
    assert str(margin.orchestration.calculation.computed_value) == (
        "55.60253406070260908223656790375015"
    )
    for result in (growth, margin):
        statement = next(
            page for page in result.association.retrieved_pages if page.physical_page_number == 139
        )
        assert statement.associated_fact_ids == tuple(sorted(by_id))
        assert result.orchestration is not None
        assert result.orchestration.answer is not None
        assert {item.marker for item in result.orchestration.answer.citations} == {"[PDF p. 139]"}


def test_real_hybrid_retrieval_cutoff_abstains_without_page_139_fallback() -> None:
    _require_local_artifacts()
    corpus = load_evidence_export(EVIDENCE_DIRECTORY)
    facts = load_financial_facts(FACTS_DIRECTORY / "facts.jsonl", corpus)
    current = next(fact for fact in facts if fact.fact_id == "nvidia-fy2026-revenue")
    target_end = current.reporting_period.end_date
    assert target_end is not None
    retriever = _real_hybrid_retriever(corpus)
    calls: list[str] = []
    backend = CallableLocalModelBackend(
        ModelIdentity("mock-local", "controlled-open-model", "retrieval-failure-test"),
        lambda prompt, _config: calls.append(prompt) or "Revenue increased from the prior year.",
    )
    integration = RetrievalFinancialAnswerOrchestrator(
        FinancialAnswerOrchestrator(AnswerGenerator(backend)),
        retriever,
    )

    result = integration.answer(
        OrchestrationRequest(
            "How much did NVIDIA revenue grow in FY2026?",
            "NVIDIA",
            target_end,
            facts,
            corpus,
        ),
        top_k=4,
    )

    assert [page.physical_page_number for page in result.association.retrieved_pages] == [
        166,
        23,
        129,
        124,
    ]
    assert result.status is OrchestrationStatus.ABSTAINED
    assert result.abstention_reason is (
        RetrievalAnswerAbstentionReason.NO_ASSOCIATED_REVIEWED_FACTS
    )
    assert result.orchestration is None
    assert calls == []
