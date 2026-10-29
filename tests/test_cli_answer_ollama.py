"""Explicitly opt-in live Ollama coverage for the user-facing answer CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ledgerlens.cli import main

DOCUMENT_ID = "sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c"
EVIDENCE_DIRECTORY = Path("artifacts/evidence") / DOCUMENT_ID
FACTS_PATH = Path("artifacts/financial-facts") / DOCUMENT_ID / "facts.jsonl"


def _require_opt_in_and_artifacts() -> None:
    if os.environ.get("LEDGERLENS_RUN_ANSWER_CLI_OLLAMA") != "1":
        pytest.skip("set LEDGERLENS_RUN_ANSWER_CLI_OLLAMA=1 to execute the live answer CLI")
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


def test_live_answer_cli_uses_reviewed_nvidia_inputs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _require_opt_in_and_artifacts()

    exit_code = main(
        [
            "answer",
            "What was NVIDIA's FY2026 net profit margin?",
            "--company",
            "NVIDIA Corporation",
            "--target-period-end",
            "2026-01-25",
            "--evidence-directory",
            str(EVIDENCE_DIRECTORY),
            "--financial-facts",
            str(FACTS_PATH),
            "--ollama",
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code in {0, 3}
    assert payload["selected_fact_ids"] == [
        "nvidia-fy2026-net-income",
        "nvidia-fy2026-revenue",
    ]
    assert payload["calculation"] is not None
    assert payload["answer"] is not None
    assert payload["answer"]["provenance"]["model"]["model"] == "qwen3:4b-instruct"
    if exit_code == 0:
        assert {item["marker"] for item in payload["answer"]["citations"]} == {"[PDF p. 139]"}
    else:
        assert payload["abstention_reason"] == "answer_generation_abstained"
