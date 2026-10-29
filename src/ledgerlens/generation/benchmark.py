"""Reproducible acceptance benchmarking for the strict answer-generation boundary."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from ledgerlens.generation import (
    GROUNDING_POLICY_VERSION,
    AnswerRequest,
    AnswerStatus,
    ExplanationValidationStatus,
    GeneratedAnswer,
    InferenceConfig,
    ModelIdentity,
    RejectedModelOutput,
)

BENCHMARK_SCHEMA_VERSION = "1.2"


class BenchmarkError(ValueError):
    """The benchmark definition or observed result is invalid."""


class CaseKind(StrEnum):
    """Whether a case measures model acceptance or deterministic abstention."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class AnswerGeneratorLike(Protocol):
    def generate(self, request: AnswerRequest) -> GeneratedAnswer: ...


@dataclass(frozen=True)
class AcceptanceCase:
    """One fixed request and its expected harness-level behavior."""

    case_id: str
    kind: CaseKind
    request: AnswerRequest
    repeats: int = 1
    expected_abstention_reason: str | None = None

    def __post_init__(self) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789-_"
        if not self.case_id or any(character not in allowed for character in self.case_id):
            raise BenchmarkError(
                "case_id must use lowercase letters, digits, hyphens, or underscores"
            )
        if (
            not isinstance(self.repeats, int)
            or isinstance(self.repeats, bool)
            or not 1 <= self.repeats <= 20
        ):
            raise BenchmarkError("case repeats must be an integer between 1 and 20")
        if self.kind is CaseKind.POSITIVE and self.expected_abstention_reason is not None:
            raise BenchmarkError("positive cases cannot declare an expected abstention")
        if self.kind is CaseKind.NEGATIVE and not self.expected_abstention_reason:
            raise BenchmarkError("negative cases require an expected abstention reason")

    def definition(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "kind": self.kind.value,
            "task": self.request.task.value,
            "question": self.request.question,
            "calculation_id": self.request.calculation.calculation_id,
            "input_fact_ids": [fact.fact_id for fact in self.request.facts],
            "repeats": self.repeats,
            "expected_abstention_reason": self.expected_abstention_reason,
        }


def run_acceptance_benchmark(
    generator: AnswerGeneratorLike,
    cases: tuple[AcceptanceCase, ...],
    *,
    model: ModelIdentity,
    inference: InferenceConfig,
    source: dict[str, object],
    created_at_utc: str | None = None,
) -> dict[str, object]:
    """Run fixed cases and return a machine-readable acceptance record."""
    if not cases or len({case.case_id for case in cases}) != len(cases):
        raise BenchmarkError("benchmark cases must be non-empty and have unique IDs")
    if model.revision is None:
        raise BenchmarkError("benchmark model identity must include the exact installed digest")
    created = created_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    definitions = [case.definition() for case in cases]
    configuration = {
        "model": model.to_dict(),
        "inference": inference.to_dict(),
        "grounding_policy_version": GROUNDING_POLICY_VERSION,
        "cases": definitions,
        "source": source,
    }
    configuration_id = (
        "sha256-"
        + hashlib.sha256(
            json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    runs: list[dict[str, object]] = []
    for case in cases:
        for repeat in range(1, case.repeats + 1):
            answer = generator.generate(case.request)
            if answer.provenance.model != model:
                raise BenchmarkError("model identity changed during benchmark")
            if answer.provenance.inference != inference:
                raise BenchmarkError("inference configuration changed during benchmark")
            harness_passed = _harness_passed(case, answer)
            runs.append(
                {
                    "run_id": f"{case.case_id}-{repeat}",
                    "case_id": case.case_id,
                    "kind": case.kind.value,
                    "repeat": repeat,
                    "answer_status": answer.status.value,
                    "accepted": answer.status is AnswerStatus.ANSWERED,
                    "explanation_validation": answer.provenance.explanation_validation.value,
                    "abstention_reason": answer.abstention_reason,
                    "response_latency_ms": answer.provenance.response_latency_ms,
                    "prompt_sha256": answer.provenance.prompt_sha256,
                    "model_output_sha256": answer.provenance.model_output_sha256,
                    "grounding_diagnostic": (
                        answer.provenance.grounding_diagnostic.to_dict()
                        if answer.provenance.grounding_diagnostic
                        else None
                    ),
                    "calculation_id": case.request.calculation.calculation_id,
                    "physical_pdf_pages": sorted(
                        {citation.physical_page_number for citation in answer.citations}
                    ),
                    "harness_expectation_passed": harness_passed,
                }
            )
    summary = _summarize(runs)
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "ledgerlens_ollama_generation_acceptance",
        "configuration_id": configuration_id,
        "created_at_utc": created,
        **configuration,
        "runs": runs,
        "summary": summary,
        "interpretation": {
            "test_harness_correctness": (
                "harness_expectation_passed checks routing, strict validation, and expected "
                "structured abstentions; it is independent of model acceptance"
            ),
            "model_acceptance_rate": (
                "accepted positive responses divided by positive responses that reached "
                "LedgerLens explanation validation"
            ),
            "semantic_correctness_limitation": (
                "passing the lexical grounding policy is not independent semantic correctness "
                "or financial-answer quality evaluation"
            ),
            "sample_size_limitation": (
                "only two NVIDIA questions are tested; repeated runs measure stability, not "
                "coverage of financial question answering"
            ),
        },
    }


def _harness_passed(case: AcceptanceCase, answer: GeneratedAnswer) -> bool:
    if case.kind is CaseKind.NEGATIVE:
        return (
            answer.status is AnswerStatus.ABSTAINED
            and answer.abstention_reason == case.expected_abstention_reason
            and answer.provenance.prompt_sha256 is None
            and answer.provenance.explanation_validation is ExplanationValidationStatus.NOT_RUN
        )
    has_model_record = (
        answer.provenance.prompt_sha256 is not None
        and answer.provenance.model_output_sha256 is not None
        and answer.provenance.grounding_diagnostic is not None
    )
    if answer.provenance.explanation_validation is ExplanationValidationStatus.PASSED:
        return (
            has_model_record
            and answer.status is AnswerStatus.ANSWERED
            and answer.abstention_reason is None
            and answer.provenance.grounding_diagnostic.accepted
        )
    if answer.provenance.explanation_validation is ExplanationValidationStatus.FAILED:
        return (
            has_model_record
            and answer.status is AnswerStatus.ABSTAINED
            and answer.abstention_reason == "model_output_failed_grounding"
            and not answer.provenance.grounding_diagnostic.accepted
        )
    return False


def _summarize(runs: list[dict[str, object]]) -> dict[str, object]:
    positives = [run for run in runs if run["kind"] == CaseKind.POSITIVE.value]
    negatives = [run for run in runs if run["kind"] == CaseKind.NEGATIVE.value]
    evaluated = [
        run
        for run in positives
        if run["explanation_validation"]
        in {ExplanationValidationStatus.PASSED.value, ExplanationValidationStatus.FAILED.value}
    ]
    accepted = sum(run["accepted"] is True for run in evaluated)
    rejected = sum(
        run["explanation_validation"] == ExplanationValidationStatus.FAILED.value
        for run in evaluated
    )
    failures = sum(run["abstention_reason"] == "model_inference_failed" for run in positives)
    latencies = sorted(
        Decimal(value) for run in positives if (value := run["response_latency_ms"]) is not None
    )
    rejection_categories: Counter[str] = Counter()
    for run in evaluated:
        diagnostic = run["grounding_diagnostic"]
        if isinstance(diagnostic, dict):
            codes = diagnostic.get("rejection_codes", [])
            if isinstance(codes, list):
                rejection_categories.update(code for code in codes if isinstance(code, str))
    return {
        "total_runs": len(runs),
        "positive_runs": len(positives),
        "negative_runs": len(negatives),
        "expected_structured_abstentions": sum(
            run["harness_expectation_passed"] is True for run in negatives
        ),
        "unexpected_negative_outcomes": sum(
            run["harness_expectation_passed"] is not True for run in negatives
        ),
        "accepted_positive_runs": accepted,
        "grounding_rejected_positive_runs": rejected,
        "inference_failed_positive_runs": failures,
        "grounding_rejection_categories": dict(sorted(rejection_categories.items())),
        "acceptance_rate": str(Decimal(accepted) / Decimal(len(evaluated))) if evaluated else None,
        "acceptance_rate_denominator": len(evaluated),
        "harness_expectations_passed": all(run["harness_expectation_passed"] for run in runs),
        "latency_ms": _latency_summary(latencies),
    }


def _latency_summary(values: list[Decimal]) -> dict[str, str] | None:
    if not values:
        return None
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {"minimum": str(values[0]), "median": str(median), "maximum": str(values[-1])}


def render_benchmark_report(payload: dict[str, object]) -> str:
    """Render a concise Markdown report from the machine-readable payload."""
    summary = payload["summary"]
    assert isinstance(summary, dict)
    runs = payload["runs"]
    assert isinstance(runs, list)
    lines = [
        "# LedgerLens Ollama generation acceptance benchmark",
        "",
        f"- Configuration: `{payload['configuration_id']}`",
        f"- Created: `{payload['created_at_utc']}`",
        f"- Model: `{payload['model']}`",
        f"- Inference: `{payload['inference']}`",
        f"- Grounding policy: `{payload['grounding_policy_version']}`",
        f"- Harness expectations passed: `{summary['harness_expectations_passed']}`",
        f"- Positive acceptance: `{summary['accepted_positive_runs']}/"
        f"{summary['acceptance_rate_denominator']}` (`{summary['acceptance_rate']}`)",
        f"- Grounding rejections: `{summary['grounding_rejected_positive_runs']}`",
        f"- Rejection categories: `{summary['grounding_rejection_categories']}`",
        f"- Inference failures: `{summary['inference_failed_positive_runs']}`",
        f"- Expected structured abstentions: "
        f"`{summary['expected_structured_abstentions']}/{summary['negative_runs']}`",
        f"- Latency milliseconds: `{summary['latency_ms']}`",
        "",
        "## Runs",
        "",
        "| Run | Kind | Status | Validation | Grounding codes | Reason | Latency ms | "
        "Prompt hash | Output hash |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for run in runs:
        assert isinstance(run, dict)
        diagnostic = run["grounding_diagnostic"]
        codes = diagnostic.get("rejection_codes", []) if isinstance(diagnostic, dict) else []
        lines.append(
            f"| {run['run_id']} | {run['kind']} | {run['answer_status']} | "
            f"{run['explanation_validation']} | {', '.join(codes)} | "
            f"{run['abstention_reason'] or ''} | "
            f"{run['response_latency_ms'] or ''} | {run['prompt_sha256'] or ''} | "
            f"{run['model_output_sha256'] or ''} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The harness result verifies case routing and expected structured abstentions. Model "
            "acceptance measures only whether output passed LedgerLens's unchanged lexical "
            "grounding validator. It is not an independent semantic-correctness judgment. This "
            "benchmark has only two positive NVIDIA questions; repeated runs measure stability "
            "but do not establish broader financial-question coverage.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_artifacts(
    payload: dict[str, object], output_directory: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically publish JSON and Markdown without silently replacing prior results."""
    destination = Path(output_directory)
    if destination.exists() and not overwrite:
        raise BenchmarkError(
            f"benchmark output already exists: {destination}; use --overwrite to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "benchmark.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (temporary / "report.md").write_text(
            render_benchmark_report(payload), encoding="utf-8", newline="\n"
        )
        if not destination.exists():
            os.replace(temporary, destination)
            return destination
        backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
        os.replace(destination, backup)
        try:
            os.replace(temporary, destination)
        except OSError:
            os.replace(backup, destination)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return destination
    except OSError as exc:
        shutil.rmtree(temporary, ignore_errors=True)
        raise BenchmarkError(f"could not publish benchmark artifacts: {exc}") from exc


def write_rejected_output_debug_file(
    records: tuple[RejectedModelOutput, ...],
    output_path: str | Path,
    *,
    model: ModelIdentity,
    inference: InferenceConfig,
    allow_raw_output: bool = False,
    overwrite: bool = False,
) -> Path:
    """Write raw rejected responses only after an explicit safety acknowledgement."""
    if not allow_raw_output:
        raise BenchmarkError("allow_raw_output=True is required to write rejected model text")
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise BenchmarkError(
            f"debug output already exists: {destination}; use --overwrite to replace it"
        )
    payload = {
        "schema_version": "1.0",
        "warning": (
            "Local debugging artifact containing raw rejected model output; do not commit or "
            "publish this file"
        ),
        "model": model.to_dict(),
        "inference": inference.to_dict(),
        "records": [record.to_dict() for record in records],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_path, destination)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise BenchmarkError(f"could not write rejected-output debug file: {exc}") from exc
    return destination


__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "AcceptanceCase",
    "BenchmarkError",
    "CaseKind",
    "render_benchmark_report",
    "run_acceptance_benchmark",
    "write_benchmark_artifacts",
    "write_rejected_output_debug_file",
]
