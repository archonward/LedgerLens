from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass

import pytest

from ledgerlens.generation import (
    AnswerStatus,
    AnswerTask,
    ExplanationValidationStatus,
    GeneratedAnswer,
    GenerationProvenance,
    GroundingDiagnostic,
    GroundingRejectionCode,
    InferenceConfig,
    ModelIdentity,
    RejectedModelOutput,
)
from ledgerlens.generation.benchmark import (
    AcceptanceCase,
    BenchmarkError,
    CaseKind,
    render_benchmark_report,
    run_acceptance_benchmark,
    write_benchmark_artifacts,
    write_rejected_output_debug_file,
)

SCRIPT = "scripts/benchmark_ollama_generation.py"


@dataclass(frozen=True)
class FakeFact:
    fact_id: str


@dataclass(frozen=True)
class FakeCalculation:
    calculation_id: str


@dataclass(frozen=True)
class FakeRequest:
    question: str
    task: AnswerTask
    calculation: FakeCalculation
    facts: tuple[FakeFact, ...]


@dataclass
class SequenceGenerator:
    answers: list[GeneratedAnswer]

    def generate(self, _request) -> GeneratedAnswer:
        return self.answers.pop(0)


MODEL = ModelIdentity("ollama", "qwen3:4b-instruct", "sha256-" + "a" * 64)
CONFIG = InferenceConfig(96, 0.0)


def _request(question: str, task: AnswerTask = AnswerTask.REVENUE_GROWTH) -> FakeRequest:
    return FakeRequest(question, task, FakeCalculation("sha256-calculation"), (FakeFact("fact"),))


def _answer(
    status: AnswerStatus,
    validation: ExplanationValidationStatus,
    *,
    reason: str | None = None,
    latency: str | None = "12",
    prompt_hash: str | None = "prompt-hash",
    output_hash: str | None = "output-hash",
) -> GeneratedAnswer:
    diagnostic = None
    if validation is ExplanationValidationStatus.PASSED:
        diagnostic = GroundingDiagnostic(True, ())
    elif validation is ExplanationValidationStatus.FAILED:
        diagnostic = GroundingDiagnostic(
            False,
            (GroundingRejectionCode.DISALLOWED_VOCABULARY,),
            disallowed_words=("unsupported",),
        )
    return GeneratedAnswer(
        status=status,
        question="question",
        task=AnswerTask.REVENUE_GROWTH,
        answer_text="answer" if status is AnswerStatus.ANSWERED else "abstained",
        reported_values=(),
        computed_value=None,
        model_generated_explanation=(
            "Revenue increased from the prior fiscal year."
            if status is AnswerStatus.ANSWERED
            else None
        ),
        citations=(),
        provenance=GenerationProvenance(
            MODEL,
            CONFIG,
            prompt_hash,
            output_hash,
            latency,
            validation,
            diagnostic,
        ),
        abstention_reason=reason,
    )


def _cases() -> tuple[AcceptanceCase, ...]:
    return (
        AcceptanceCase(
            "positive-growth",
            CaseKind.POSITIVE,
            _request("How much did revenue grow?"),  # type: ignore[arg-type]
            repeats=2,
        ),
        AcceptanceCase(
            "unsupported-question",
            CaseKind.NEGATIVE,
            _request("Why did expenses change?"),  # type: ignore[arg-type]
            expected_abstention_reason=("question_is_not_supported_by_revenue_growth_evidence"),
        ),
    )


def test_benchmark_separates_acceptance_rejection_and_structured_abstention() -> None:
    generator = SequenceGenerator(
        [
            _answer(AnswerStatus.ANSWERED, ExplanationValidationStatus.PASSED),
            _answer(
                AnswerStatus.ABSTAINED,
                ExplanationValidationStatus.FAILED,
                reason="model_output_failed_grounding",
                latency="18",
            ),
            _answer(
                AnswerStatus.ABSTAINED,
                ExplanationValidationStatus.NOT_RUN,
                reason="question_is_not_supported_by_revenue_growth_evidence",
                latency=None,
                prompt_hash=None,
                output_hash=None,
            ),
        ]
    )

    payload = run_acceptance_benchmark(
        generator,  # type: ignore[arg-type]
        _cases(),
        model=MODEL,
        inference=CONFIG,
        source={"document_id": "controlled"},
        created_at_utc="2026-10-26T00:00:00Z",
    )

    assert payload["schema_version"] == "1.1"
    assert payload["configuration_id"].startswith("sha256-")  # type: ignore[union-attr]
    summary = payload["summary"]
    assert summary == {
        "total_runs": 3,
        "positive_runs": 2,
        "negative_runs": 1,
        "expected_structured_abstentions": 1,
        "unexpected_negative_outcomes": 0,
        "accepted_positive_runs": 1,
        "grounding_rejected_positive_runs": 1,
        "inference_failed_positive_runs": 0,
        "grounding_rejection_categories": {"disallowed_vocabulary": 1},
        "acceptance_rate": "0.5",
        "acceptance_rate_denominator": 2,
        "harness_expectations_passed": True,
        "latency_ms": {"minimum": "12", "median": "15", "maximum": "18"},
    }
    runs = payload["runs"]
    assert runs[0]["prompt_sha256"] == "prompt-hash"  # type: ignore[index]
    assert runs[1]["model_output_sha256"] == "output-hash"  # type: ignore[index]
    assert runs[2]["harness_expectation_passed"] is True  # type: ignore[index]


def test_benchmark_payload_and_report_are_reproducible_for_fixed_observations(tmp_path) -> None:
    answers = [
        _answer(AnswerStatus.ANSWERED, ExplanationValidationStatus.PASSED),
        _answer(AnswerStatus.ANSWERED, ExplanationValidationStatus.PASSED),
        _answer(
            AnswerStatus.ABSTAINED,
            ExplanationValidationStatus.NOT_RUN,
            reason="question_is_not_supported_by_revenue_growth_evidence",
            latency=None,
            prompt_hash=None,
            output_hash=None,
        ),
    ]
    payload = run_acceptance_benchmark(
        SequenceGenerator(answers),  # type: ignore[arg-type]
        _cases(),
        model=MODEL,
        inference=CONFIG,
        source={"document_id": "controlled"},
        created_at_utc="2026-10-26T00:00:00Z",
    )
    destination = tmp_path / "benchmark"

    written = write_benchmark_artifacts(payload, destination)

    assert written == destination
    assert json.loads((destination / "benchmark.json").read_text(encoding="utf-8")) == payload
    report = (destination / "report.md").read_text(encoding="utf-8")
    assert report == render_benchmark_report(payload)
    assert "not an independent semantic-correctness judgment" in report
    assert "only two positive NVIDIA questions" in report
    with pytest.raises(BenchmarkError, match="already exists"):
        write_benchmark_artifacts(payload, destination)
    assert write_benchmark_artifacts(payload, destination, overwrite=True) == destination


def test_raw_rejected_output_requires_separate_explicit_debug_file(tmp_path) -> None:
    record = RejectedModelOutput(
        "How much did revenue grow?",
        AnswerTask.REVENUE_GROWTH,
        "prompt-hash",
        "output-hash",
        "Revenue increased with unsupported wording.",
        GroundingDiagnostic(
            False,
            (GroundingRejectionCode.DISALLOWED_VOCABULARY,),
            disallowed_words=("unsupported", "wording"),
        ),
    )
    destination = tmp_path / "rejected.json"

    with pytest.raises(BenchmarkError, match="allow_raw_output=True"):
        write_rejected_output_debug_file((record,), destination, model=MODEL, inference=CONFIG)
    written = write_rejected_output_debug_file(
        (record,),
        destination,
        model=MODEL,
        inference=CONFIG,
        allow_raw_output=True,
    )

    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["records"][0]["raw_output"] == record.raw_output
    assert "do not commit or publish" in payload["warning"]


def test_benchmark_detects_harness_mismatch_and_model_identity_change() -> None:
    wrong_negative = _answer(
        AnswerStatus.ABSTAINED,
        ExplanationValidationStatus.NOT_RUN,
        reason="different_reason",
        latency=None,
        prompt_hash=None,
        output_hash=None,
    )
    answers = [
        _answer(AnswerStatus.ANSWERED, ExplanationValidationStatus.PASSED),
        _answer(AnswerStatus.ANSWERED, ExplanationValidationStatus.PASSED),
        wrong_negative,
    ]
    payload = run_acceptance_benchmark(
        SequenceGenerator(answers),  # type: ignore[arg-type]
        _cases(),
        model=MODEL,
        inference=CONFIG,
        source={},
    )
    assert payload["summary"]["harness_expectations_passed"] is False  # type: ignore[index]

    changed = _answer(AnswerStatus.ANSWERED, ExplanationValidationStatus.PASSED)
    changed = GeneratedAnswer(
        **{
            **changed.__dict__,
            "provenance": GenerationProvenance(
                ModelIdentity("ollama", "qwen3:4b-instruct", "sha256-" + "b" * 64),
                CONFIG,
                "prompt",
                "output",
                "1",
                ExplanationValidationStatus.PASSED,
                GroundingDiagnostic(True, ()),
            ),
        }
    )
    with pytest.raises(BenchmarkError, match="model identity changed"):
        run_acceptance_benchmark(
            SequenceGenerator([changed]),  # type: ignore[arg-type]
            (AcceptanceCase("positive", CaseKind.POSITIVE, _request("growth")),),  # type: ignore[arg-type]
            model=MODEL,
            inference=CONFIG,
            source={},
        )


def test_positive_inference_failure_is_not_treated_as_harness_success() -> None:
    failed = _answer(
        AnswerStatus.ABSTAINED,
        ExplanationValidationStatus.NOT_RUN,
        reason="model_inference_failed",
        output_hash=None,
    )
    payload = run_acceptance_benchmark(
        SequenceGenerator([failed]),  # type: ignore[arg-type]
        (AcceptanceCase("positive", CaseKind.POSITIVE, _request("growth")),),  # type: ignore[arg-type]
        model=MODEL,
        inference=CONFIG,
        source={},
    )

    summary = payload["summary"]
    assert summary["harness_expectations_passed"] is False  # type: ignore[index]
    assert summary["inference_failed_positive_runs"] == 1  # type: ignore[index]
    assert summary["acceptance_rate_denominator"] == 0  # type: ignore[index]


@pytest.mark.parametrize(
    "case",
    [
        lambda: AcceptanceCase("BAD", CaseKind.POSITIVE, _request("growth")),
        lambda: AcceptanceCase("case", CaseKind.POSITIVE, _request("growth"), repeats=0),
        lambda: AcceptanceCase(
            "case", CaseKind.NEGATIVE, _request("growth"), expected_abstention_reason=None
        ),
    ],
)
def test_invalid_case_definitions_are_rejected(case) -> None:
    with pytest.raises(BenchmarkError):
        case()


def test_live_benchmark_script_requires_explicit_acknowledgement() -> None:
    completed = subprocess.run(
        [sys.executable, SCRIPT],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "--run-live is required" in completed.stderr
