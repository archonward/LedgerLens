"""Opt-in live orchestration checks over ignored reviewed NVIDIA artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ledgerlens.financial_facts import load_financial_facts
from ledgerlens.generation import AnswerGenerator, InferenceConfig
from ledgerlens.generation.ollama import OllamaBackend
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationAbstentionReason,
    OrchestrationRequest,
    OrchestrationStatus,
)
from ledgerlens.retrieval.bm25 import load_evidence_export

DOCUMENT_ID = "sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c"
EVIDENCE_DIRECTORY = Path("artifacts/evidence") / DOCUMENT_ID
FACTS_PATH = Path("artifacts/financial-facts") / DOCUMENT_ID / "facts.jsonl"


def _require_opt_in_and_artifacts() -> None:
    if os.environ.get("LEDGERLENS_RUN_ORCHESTRATION_OLLAMA") != "1":
        pytest.skip("set LEDGERLENS_RUN_ORCHESTRATION_OLLAMA=1 to execute live orchestration")
    missing = [
        str(path)
        for path in (
            EVIDENCE_DIRECTORY / "evidence.jsonl",
            EVIDENCE_DIRECTORY / "manifest.json",
            FACTS_PATH,
        )
        if not path.is_file()
    ]
    if missing:
        pytest.skip("ignored NVIDIA evidence artifacts are unavailable: " + ", ".join(missing))


def test_live_ollama_orchestrates_reviewed_nvidia_tasks() -> None:
    _require_opt_in_and_artifacts()
    corpus = load_evidence_export(EVIDENCE_DIRECTORY)
    facts = load_financial_facts(FACTS_PATH, corpus)
    by_id = {fact.fact_id: fact for fact in facts}
    current_revenue = by_id["nvidia-fy2026-revenue"]
    target_end = current_revenue.reporting_period.end_date
    assert target_end is not None
    backend = OllamaBackend(timeout_seconds=300)
    orchestrator = FinancialAnswerOrchestrator(
        AnswerGenerator(
            backend,
            inference_config=InferenceConfig(max_new_tokens=96, temperature=0.0),
        )
    )

    results = (
        orchestrator.answer(
            OrchestrationRequest(
                "How much did NVIDIA revenue grow in FY2026?",
                current_revenue.company,
                target_end,
                facts,
                corpus,
            )
        ),
        orchestrator.answer(
            OrchestrationRequest(
                "What was NVIDIA's FY2026 net profit margin?",
                current_revenue.company,
                target_end,
                facts,
                corpus,
            )
        ),
    )

    assert results[0].selected_fact_ids == (
        "nvidia-fy2026-revenue",
        "nvidia-fy2025-revenue",
    )
    assert results[1].selected_fact_ids == (
        "nvidia-fy2026-net-income",
        "nvidia-fy2026-revenue",
    )
    for result in results:
        assert result.calculation is not None
        assert result.answer is not None
        assert result.status in {OrchestrationStatus.ANSWERED, OrchestrationStatus.ABSTAINED}
        if result.status is OrchestrationStatus.ANSWERED:
            assert {citation.marker for citation in result.answer.citations} == {"[PDF p. 139]"}
        else:
            assert result.abstention_reason is (
                OrchestrationAbstentionReason.ANSWER_GENERATION_ABSTAINED
            )
            assert result.detail == "model_output_failed_grounding"
