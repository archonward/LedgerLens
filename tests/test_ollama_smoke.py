"""Opt-in smoke test for the exact local Ollama model and ignored NVIDIA artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ledgerlens.calculations import FactOperand, profit_margin, year_over_year_growth
from ledgerlens.financial_facts import load_financial_facts
from ledgerlens.generation import (
    AnswerGenerator,
    AnswerRequest,
    AnswerStatus,
    AnswerTask,
    ExplanationValidationStatus,
    InferenceConfig,
)
from ledgerlens.generation.ollama import OLLAMA_MODEL, OllamaBackend
from ledgerlens.retrieval.bm25 import load_evidence_export

DOCUMENT_ID = "sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c"
EVIDENCE_DIRECTORY = Path("artifacts/evidence") / DOCUMENT_ID
FACTS_PATH = Path("artifacts/financial-facts") / DOCUMENT_ID / "facts.jsonl"


def _require_opt_in_and_artifacts() -> None:
    if os.environ.get("LEDGERLENS_RUN_OLLAMA_SMOKE") != "1":
        pytest.skip("set LEDGERLENS_RUN_OLLAMA_SMOKE=1 to execute the local model")
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


def test_qwen3_ollama_answers_reviewed_nvidia_examples() -> None:
    """Run only when explicitly requested; strict grounding failure remains a valid outcome."""
    _require_opt_in_and_artifacts()
    corpus = load_evidence_export(EVIDENCE_DIRECTORY)
    facts = {fact.fact_id: fact for fact in load_financial_facts(FACTS_PATH, corpus)}
    revenue = facts["nvidia-fy2026-revenue"]
    prior_revenue = facts["nvidia-fy2025-revenue"]
    net_income = facts["nvidia-fy2026-net-income"]
    requests = (
        AnswerRequest(
            "How much did NVIDIA revenue grow in FY2026?",
            AnswerTask.REVENUE_GROWTH,
            (revenue, prior_revenue),
            year_over_year_growth(FactOperand(revenue, corpus), FactOperand(prior_revenue, corpus)),
            corpus,
        ),
        AnswerRequest(
            "What was NVIDIA's FY2026 net profit margin?",
            AnswerTask.NET_PROFIT_MARGIN,
            (net_income, revenue),
            profit_margin(FactOperand(net_income, corpus), FactOperand(revenue, corpus)),
            corpus,
        ),
    )
    config = InferenceConfig(max_new_tokens=96, temperature=0.0)
    backend = OllamaBackend(timeout_seconds=300)
    generator = AnswerGenerator(backend, inference_config=config)
    records: list[dict[str, object]] = []

    for request in requests:
        answer = generator.generate(request)
        validation = answer.provenance.explanation_validation
        assert answer.abstention_reason != "model_inference_failed"
        assert validation in {
            ExplanationValidationStatus.PASSED,
            ExplanationValidationStatus.FAILED,
        }
        assert answer.provenance.model.model == OLLAMA_MODEL
        assert answer.provenance.model.revision is not None
        assert answer.provenance.response_latency_ms is not None
        if validation is ExplanationValidationStatus.PASSED:
            assert answer.status is AnswerStatus.ANSWERED
            assert {citation.marker for citation in answer.citations} == {"[PDF p. 139]"}
        else:
            assert answer.status is AnswerStatus.ABSTAINED
            assert answer.abstention_reason == "model_output_failed_grounding"
        records.append(
            {
                "task": request.task.value,
                "answer_status": answer.status.value,
                "explanation_validation": validation.value,
                "response_latency_ms": answer.provenance.response_latency_ms,
                "model": answer.provenance.model.to_dict(),
                "inference": answer.provenance.inference.to_dict(),
                "abstention_reason": answer.abstention_reason,
            }
        )

    print(json.dumps(records, indent=2))
