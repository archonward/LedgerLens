"""Reproducible reference-based evaluation for the multi-document workflow."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ledgerlens.generation import GROUNDING_POLICY_VERSION, LocalModelBackend
from ledgerlens.multi_document_answer_generation import (
    MULTI_DOCUMENT_INFERENCE_CONFIG,
    MultiDocumentAnswerStatus,
)
from ledgerlens.multi_document_workflow import (
    PERCENTAGE_PRESENTATION_POLICY_VERSION,
    MultiDocumentWorkflowRequest,
    run_multi_document_workflow,
    workflow_request_from_dict,
)

MULTI_DOCUMENT_EVALUATION_SCHEMA_VERSION = "1.0"


class MultiDocumentEvaluationError(ValueError):
    """An evaluation definition or protected output operation is invalid."""


@dataclass(frozen=True)
class ReferenceProvenance:
    """Caller-supplied provenance for one non-generated reference label."""

    reviewed_by: str
    reviewed_at_utc: str
    method: str
    note: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "reviewed_by": self.reviewed_by,
            "reviewed_at_utc": self.reviewed_at_utc,
            "method": self.method,
            "note": self.note,
        }


@dataclass(frozen=True)
class ExpectedCitation:
    role: str
    document_id: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "document_id": self.document_id}


@dataclass(frozen=True)
class MultiDocumentEvaluationCase:
    case_id: str
    request: MultiDocumentWorkflowRequest
    expected_computed_value: Decimal
    expected_citations: tuple[ExpectedCitation, ...]
    reference_provenance: ReferenceProvenance

    def definition(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "request": {
                "question": self.request.question,
                "task": self.request.task.value,
                "company": self.request.company,
                "sources": [
                    {
                        "document_id": source.document_id,
                        "source_sha256": source.source_sha256,
                    }
                    for source in self.request.sources
                ],
                "selections": [item.to_dict() for item in self.request.selections],
                "selection_policy": self.request.selection_policy.to_dict(),
            },
            "reference": {
                "expected_computed_value": str(self.expected_computed_value),
                "expected_citations": [item.to_dict() for item in self.expected_citations],
                "provenance": self.reference_provenance.to_dict(),
            },
        }


@dataclass(frozen=True)
class MultiDocumentEvaluationSet:
    dataset_id: str
    path: Path
    sha256: str
    cases: tuple[MultiDocumentEvaluationCase, ...]


def _text(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MultiDocumentEvaluationError(f"{context} must be a non-empty string")
    return value


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MultiDocumentEvaluationError(f"{context} must be an object")
    return value


def _reference_provenance(value: object, context: str) -> ReferenceProvenance:
    record = _object(value, context)
    note = record.get("note")
    if note is not None and (not isinstance(note, str) or not note.strip()):
        raise MultiDocumentEvaluationError(f"{context}.note must be null or non-empty")
    reviewed_at = _text(record.get("reviewed_at_utc"), f"{context}.reviewed_at_utc")
    try:
        parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MultiDocumentEvaluationError(
            f"{context}.reviewed_at_utc must be an ISO UTC timestamp"
        ) from exc
    if not reviewed_at.endswith("Z") or parsed.tzinfo != UTC:
        raise MultiDocumentEvaluationError(
            f"{context}.reviewed_at_utc must be an ISO UTC timestamp ending in Z"
        )
    return ReferenceProvenance(
        _text(record.get("reviewed_by"), f"{context}.reviewed_by"),
        reviewed_at,
        _text(record.get("method"), f"{context}.method"),
        note,
    )


def load_multi_document_evaluation_set(
    path: str | Path,
) -> MultiDocumentEvaluationSet:
    """Load explicit reference cases; never derive or approve expected labels."""
    source_path = Path(path)
    try:
        raw = source_path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except OSError as exc:
        raise MultiDocumentEvaluationError(f"could not read evaluation set: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiDocumentEvaluationError(
            f"evaluation set is not valid UTF-8 JSON: {exc}"
        ) from exc
    record = _object(value, "evaluation set")
    if record.get("schema_version") != MULTI_DOCUMENT_EVALUATION_SCHEMA_VERSION:
        raise MultiDocumentEvaluationError("unsupported evaluation set schema_version")
    dataset_id = _text(record.get("dataset_id"), "dataset_id")
    allowed_id_characters = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_")
    if any(character not in allowed_id_characters for character in dataset_id):
        raise MultiDocumentEvaluationError(
            "dataset_id must use lowercase letters, digits, hyphens, or underscores"
        )
    source_values = record.get("sources")
    if not isinstance(source_values, list) or not source_values:
        raise MultiDocumentEvaluationError("sources must be a non-empty array")
    case_values = record.get("cases")
    if not isinstance(case_values, list) or not case_values:
        raise MultiDocumentEvaluationError("cases must be a non-empty array")
    cases: list[MultiDocumentEvaluationCase] = []
    for index, case_value in enumerate(case_values):
        case = _object(case_value, f"cases[{index}]")
        reference = _object(case.get("reference"), f"cases[{index}].reference")
        try:
            expected = Decimal(
                _text(
                    reference.get("expected_computed_value"),
                    f"cases[{index}].reference.expected_computed_value",
                )
            )
        except (InvalidOperation, ValueError) as exc:
            raise MultiDocumentEvaluationError(
                f"cases[{index}] expected_computed_value must be a finite Decimal string"
            ) from exc
        if not expected.is_finite():
            raise MultiDocumentEvaluationError(
                f"cases[{index}] expected_computed_value must be finite"
            )
        citation_values = reference.get("expected_citations")
        if not isinstance(citation_values, list) or not citation_values:
            raise MultiDocumentEvaluationError(
                f"cases[{index}].reference.expected_citations must be a non-empty array"
            )
        citations = tuple(
            ExpectedCitation(
                _text(item.get("role"), f"cases[{index}].expected_citations[{position}].role"),
                _text(
                    item.get("document_id"),
                    f"cases[{index}].expected_citations[{position}].document_id",
                ),
            )
            for position, item in enumerate(
                _object(item, f"cases[{index}].expected_citations[{position}]")
                for position, item in enumerate(citation_values)
            )
        )
        request_value = {
            "schema_version": "1.0",
            "question": case.get("question"),
            "task": case.get("task"),
            "company": case.get("company"),
            "sources": source_values,
            "selections": case.get("selections"),
            "selection_policy": case.get("selection_policy", {"single_source": "abstain"}),
        }
        try:
            request = workflow_request_from_dict(request_value, base_directory=source_path.parent)
        except ValueError as exc:
            raise MultiDocumentEvaluationError(f"invalid case {index}: {exc}") from exc
        case_id = _text(case.get("case_id"), f"cases[{index}].case_id")
        if any(character not in allowed_id_characters for character in case_id):
            raise MultiDocumentEvaluationError(
                f"cases[{index}].case_id contains unsupported characters"
            )
        if len(set((item.role, item.document_id) for item in citations)) != len(citations):
            raise MultiDocumentEvaluationError(
                f"cases[{index}] expected citations must be unique role-document pairs"
            )
        cases.append(
            MultiDocumentEvaluationCase(
                case_id,
                request,
                expected,
                citations,
                _reference_provenance(
                    reference.get("provenance"), f"cases[{index}].reference.provenance"
                ),
            )
        )
    if len({case.case_id for case in cases}) != len(cases):
        raise MultiDocumentEvaluationError("case IDs must be unique")
    return MultiDocumentEvaluationSet(
        dataset_id,
        source_path,
        hashlib.sha256(raw).hexdigest(),
        tuple(cases),
    )


def _observed_citations(answer: object) -> tuple[tuple[str, str], ...]:
    citations = getattr(answer, "citations", ())
    return tuple(sorted((item.role, item.annual_report.document_id) for item in citations))


def run_multi_document_evaluation(
    evaluation_set: MultiDocumentEvaluationSet,
    backend: LocalModelBackend,
    *,
    created_at_utc: str | None = None,
) -> dict[str, object]:
    """Run fixed reference cases and separately score deterministic and model outcomes."""
    configuration = {
        "dataset_id": evaluation_set.dataset_id,
        "dataset_sha256": evaluation_set.sha256,
        "inference": MULTI_DOCUMENT_INFERENCE_CONFIG.to_dict(),
        "grounding_policy_version": GROUNDING_POLICY_VERSION,
        "presentation_policy_version": PERCENTAGE_PRESENTATION_POLICY_VERSION,
        "cases": [case.definition() for case in evaluation_set.cases],
    }
    records: list[dict[str, object]] = []
    for case in evaluation_set.cases:
        workflow = run_multi_document_workflow(case.request, backend)
        answer = workflow.answer
        computed = answer.computed_value if answer else None
        actual_value = Decimal(computed.computed_value) if computed else None
        numerical_correct = (
            actual_value == case.expected_computed_value if actual_value is not None else False
        )
        actual_citations = _observed_citations(answer)
        expected_citations = tuple(
            sorted((item.role, item.document_id) for item in case.expected_citations)
        )
        citation_correct = actual_citations == expected_citations
        provenance = answer.provenance if answer else None
        generation = answer.to_dict()["generation_outcome"] if answer else {}
        records.append(
            {
                "case_id": case.case_id,
                "status": workflow.status,
                "expected": {
                    "computed_value": str(case.expected_computed_value),
                    "citations": [item.to_dict() for item in case.expected_citations],
                    "reference_provenance": case.reference_provenance.to_dict(),
                },
                "observed": {
                    "computed_value": computed.computed_value if computed else None,
                    "calculation_id": computed.calculation_id if computed else None,
                    "citations": [
                        {"role": role, "document_id": document_id}
                        for role, document_id in actual_citations
                    ],
                    "presentation": (
                        workflow.presentation.to_dict() if workflow.presentation else None
                    ),
                },
                "checks": {
                    "numerical_correct": numerical_correct,
                    "citation_correct": citation_correct,
                },
                "model": provenance.model.to_dict() if provenance else None,
                "inference": provenance.inference.to_dict() if provenance else None,
                "response_latency_ms": provenance.response_latency_ms if provenance else None,
                "model_invocation_attempted": generation.get("model_invocation_attempted", False),
                "model_invocation_completed": generation.get("model_invocation_completed", False),
                "grounding_accepted": generation.get("explanation_accepted", False),
                "grounding_diagnostic": (
                    provenance.grounding_diagnostic.to_dict()
                    if provenance and provenance.grounding_diagnostic
                    else None
                ),
                "abstention_reason": (
                    answer.abstention_reason
                    if answer
                    else (
                        workflow.orchestration.abstention_reason.value
                        if workflow.orchestration.abstention_reason
                        else None
                    )
                ),
                "selection": (
                    workflow.orchestration.selection.to_dict()
                    if workflow.orchestration.selection
                    else None
                ),
            }
        )
    observed_models = {
        json.dumps(item["model"], sort_keys=True, separators=(",", ":"))
        for item in records
        if item["model"] is not None
    }
    if len(observed_models) > 1:
        raise MultiDocumentEvaluationError("model identity changed during evaluation")
    configuration["model"] = (
        json.loads(next(iter(observed_models))) if observed_models else backend.identity.to_dict()
    )
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    configuration_id = "sha256-" + hashlib.sha256(canonical).hexdigest()
    summary = {
        "cases": len(records),
        "validated_calculations": sum(
            item["observed"]["computed_value"] is not None for item in records
        ),  # type: ignore[index]
        "numerically_correct": sum(item["checks"]["numerical_correct"] is True for item in records),  # type: ignore[index]
        "citation_correct": sum(item["checks"]["citation_correct"] is True for item in records),  # type: ignore[index]
        "model_invocation_attempted": sum(
            item["model_invocation_attempted"] is True for item in records
        ),
        "model_invocation_completed": sum(
            item["model_invocation_completed"] is True for item in records
        ),
        "grounding_accepted": sum(item["grounding_accepted"] is True for item in records),
        "grounding_rejected": sum(
            item["status"] == MultiDocumentAnswerStatus.MODEL_OUTPUT_REJECTED.value
            for item in records
        ),
        "validation_abstained": sum(item["status"] == "validation_abstained" for item in records),
        "inference_failed": sum(
            item["status"] == MultiDocumentAnswerStatus.INFERENCE_FAILED.value for item in records
        ),
        "reference_checks_passed": all(
            item["checks"]["numerical_correct"] is True  # type: ignore[index]
            and item["checks"]["citation_correct"] is True  # type: ignore[index]
            for item in records
        ),
    }
    return {
        "schema_version": MULTI_DOCUMENT_EVALUATION_SCHEMA_VERSION,
        "evaluation": "ledgerlens_multi_document_end_to_end",
        "configuration_id": configuration_id,
        "created_at_utc": created_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "configuration": configuration,
        "cases": records,
        "summary": summary,
        "interpretation": {
            "reference_labels": (
                "expected values and citation identities were explicitly supplied with review "
                "provenance; LedgerLens did not generate or independently approve them"
            ),
            "calculation_accuracy": "exact Decimal equality against the supplied reference",
            "citation_correctness": "exact role-to-document identity equality",
            "grounding_limitation": (
                "grounding acceptance measures policy 1.1 compliance, not independent semantic "
                "correctness"
            ),
        },
    }


def write_multi_document_evaluation(
    payload: dict[str, object],
    output_root: str | Path = "artifacts/evaluation",
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically publish one fingerprinted evaluation without silent replacement."""
    configuration_id = payload.get("configuration_id")
    if not isinstance(configuration_id, str) or not configuration_id.startswith("sha256-"):
        raise MultiDocumentEvaluationError("evaluation payload lacks a configuration fingerprint")
    destination = Path(output_root) / configuration_id
    if destination.exists() and not overwrite:
        raise MultiDocumentEvaluationError(
            f"evaluation output already exists: {destination}; use --overwrite to replace it"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        (temporary / "evaluation.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
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
        raise MultiDocumentEvaluationError(f"could not publish evaluation: {exc}") from exc


__all__ = [
    "MULTI_DOCUMENT_EVALUATION_SCHEMA_VERSION",
    "ExpectedCitation",
    "MultiDocumentEvaluationCase",
    "MultiDocumentEvaluationError",
    "MultiDocumentEvaluationSet",
    "ReferenceProvenance",
    "load_multi_document_evaluation_set",
    "run_multi_document_evaluation",
    "write_multi_document_evaluation",
]
