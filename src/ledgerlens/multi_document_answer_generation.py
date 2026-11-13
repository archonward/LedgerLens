"""Grounded qualitative generation over separately validated annual reports."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ledgerlens.calculations import CalculationResult
from ledgerlens.financial_facts import FinancialFact
from ledgerlens.generation import (
    ComputedValue,
    ExplanationValidationStatus,
    GenerationProvenance,
    GroundingDiagnostic,
    InferenceConfig,
    LocalModelBackend,
    RejectedModelOutput,
    ReportedValue,
    _diagnose_explanation,
    _grounded_conclusion,
)
from ledgerlens.multi_document_generation import (
    MULTI_DOCUMENT_GENERATION_SCHEMA_VERSION,
    MultiDocumentGenerationRequest,
    MultiDocumentGenerationValidationResult,
    MultiDocumentGenerationValidationStatus,
    MultiDocumentGenerationValidator,
    SourceSpecificCitation,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus

MULTI_DOCUMENT_ANSWER_SCHEMA_VERSION = "1.0"
MULTI_DOCUMENT_INFERENCE_CONFIG = InferenceConfig(max_new_tokens=96, temperature=0.0)


class MultiDocumentAnswerStatus(StrEnum):
    """Outcome at the validation, inference, and grounding boundaries."""

    ANSWERED = "answered"
    MODEL_OUTPUT_REJECTED = "model_output_rejected"
    VALIDATION_ABSTAINED = "validation_abstained"
    INFERENCE_FAILED = "inference_failed"


@dataclass(frozen=True)
class MultiDocumentGeneratedAnswer:
    """Deterministic financial material plus an accepted explanation or safe failure."""

    status: MultiDocumentAnswerStatus
    validation: MultiDocumentGenerationValidationResult
    answer_text: str
    reported_values: tuple[ReportedValue, ...]
    computed_value: ComputedValue | None
    model_generated_explanation: str | None
    citations: tuple[SourceSpecificCitation, ...]
    provenance: GenerationProvenance
    abstention_reason: str | None = None

    def __post_init__(self) -> None:
        validated = self.validation.status is MultiDocumentGenerationValidationStatus.VALIDATED
        if self.status is MultiDocumentAnswerStatus.ANSWERED:
            if (
                not validated
                or self.computed_value is None
                or self.model_generated_explanation is None
                or self.abstention_reason is not None
                or not self.citations
                or self.provenance.explanation_validation is not ExplanationValidationStatus.PASSED
            ):
                raise ValueError("answered results require validated financial and model output")
        elif self.status is MultiDocumentAnswerStatus.MODEL_OUTPUT_REJECTED:
            if (
                not validated
                or self.computed_value is None
                or self.model_generated_explanation is not None
                or not self.citations
                or self.abstention_reason != "model_output_failed_grounding"
                or self.provenance.explanation_validation is not ExplanationValidationStatus.FAILED
            ):
                raise ValueError("rejected results require validated financial material")
        elif self.status is MultiDocumentAnswerStatus.INFERENCE_FAILED:
            if (
                not validated
                or self.computed_value is None
                or self.model_generated_explanation is not None
                or not self.citations
                or self.abstention_reason != "model_inference_failed"
            ):
                raise ValueError("inference failures require validated financial material")
        elif (
            validated
            or self.reported_values
            or self.computed_value is not None
            or self.model_generated_explanation is not None
            or self.citations
            or self.abstention_reason is None
        ):
            raise ValueError("validation abstentions must not expose unvalidated material")

    @property
    def request(self) -> MultiDocumentGenerationRequest:
        return self.validation.request

    def to_dict(self) -> dict[str, object]:
        model_attempted = self.status is not MultiDocumentAnswerStatus.VALIDATION_ABSTAINED
        model_completed = self.status in {
            MultiDocumentAnswerStatus.ANSWERED,
            MultiDocumentAnswerStatus.MODEL_OUTPUT_REJECTED,
        }
        return {
            "schema_version": MULTI_DOCUMENT_ANSWER_SCHEMA_VERSION,
            "validation_schema_version": MULTI_DOCUMENT_GENERATION_SCHEMA_VERSION,
            "status": self.status.value,
            "question": self.request.question,
            "task": self.request.task.value,
            "answer_text": self.answer_text,
            "reported_values": [item.to_dict() for item in self.reported_values],
            "computed_value": self.computed_value.to_dict() if self.computed_value else None,
            "model_generated_explanation": self.model_generated_explanation,
            "citations": [citation.to_dict() for citation in self.citations],
            "selection": self.request.selection.to_dict(),
            "validation": self.validation.to_dict(),
            "generation_outcome": {
                "validation_succeeded": model_attempted,
                "model_invocation_attempted": model_attempted,
                "model_invocation_completed": model_completed,
                "explanation_accepted": self.status is MultiDocumentAnswerStatus.ANSWERED,
            },
            "provenance": self.provenance.to_dict(),
            "abstention_reason": self.abstention_reason,
        }


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _elapsed_ms(started_ns: int) -> str:
    return str((time.perf_counter_ns() - started_ns) // 1_000_000)


def _provenance(
    backend: LocalModelBackend,
    prompt_sha256: str | None,
    output_sha256: str | None,
    latency_ms: str | None = None,
    validation: ExplanationValidationStatus = ExplanationValidationStatus.NOT_RUN,
    diagnostic: GroundingDiagnostic | None = None,
) -> GenerationProvenance:
    return GenerationProvenance(
        backend.identity,
        MULTI_DOCUMENT_INFERENCE_CONFIG,
        prompt_sha256,
        output_sha256,
        latency_ms,
        validation,
        diagnostic,
    )


def _reported_value(fact: FinancialFact, marker: str) -> ReportedValue:
    assert fact.reporting_period.label and fact.currency.code and fact.units.label
    assert fact.verification.reviewed_by
    assert fact.verification.reviewed_at_utc
    assert fact.verification.method
    return ReportedValue(
        fact.fact_id,
        fact.metric,
        fact.reported_value,
        fact.company,
        fact.reporting_period.label,
        fact.currency.code,
        fact.units.label,
        marker,
        fact.verification.status.value,
        fact.verification.reviewed_by,
        fact.verification.reviewed_at_utc,
        fact.verification.method,
        fact.verification.note,
        fact.warnings,
        fact.limitations,
    )


def _computed_value(
    calculation: CalculationResult,
    citations: tuple[SourceSpecificCitation, ...],
) -> ComputedValue:
    return ComputedValue(
        calculation.calculation_id,
        calculation.calculation_type.value,
        calculation.formula,
        tuple(item.fact.fact_id for item in calculation.inputs),
        str(calculation.computed_value),
        calculation.unit.label,
        tuple(citation.marker for citation in citations),
    )


def _render_deterministic_material(
    reported: tuple[ReportedValue, ...],
    computed: ComputedValue,
) -> str:
    facts = "\n".join(f"- {item.render()}" for item in reported)
    return f"Source-reported values:\n{facts}\n\nComputed result:\n- {computed.render()}"


def _render_answer(
    reported: tuple[ReportedValue, ...],
    computed: ComputedValue,
    explanation: str,
) -> str:
    return (
        f"{_render_deterministic_material(reported, computed)}\n\n"
        f"Model-generated explanation:\n- {explanation}"
    )


def _build_prompt(
    request: MultiDocumentGenerationRequest,
    calculation: CalculationResult,
    citations: tuple[SourceSpecificCitation, ...],
    conclusion: str,
    allowed_words: frozenset[str],
) -> str:
    by_fact_id = {binding.fact.fact_id: binding.fact for binding in request.bindings}
    facts = "\n".join(
        f"- {citation.fact_id}: {by_fact_id[citation.fact_id].metric}; "
        f"{by_fact_id[citation.fact_id].reported_value}; "
        f"{by_fact_id[citation.fact_id].reporting_period.label}; "
        f"annual report {citation.annual_report.document_id}; "
        f"PDF physical page {citation.physical_page_number}"
        for citation in citations
    )
    return (
        "You are LedgerLens's qualitative explanation component. Return one plain-text "
        "sentence. Do not emit numbers, currency, percentages, citations, causes, forecasts, "
        "or facts beyond the allowed conclusion. The source facts come from separately "
        "validated annual reports; do not compare or rank the reports.\n"
        f"Question: {request.question.strip()}\nValidated source facts:\n{facts}\n"
        f"Validated calculation: {calculation.calculation_type.value}; "
        f"{calculation.computed_value} {calculation.unit.label}.\n"
        f"Allowed conclusion: {conclusion}\n"
        f"Allowed words: {', '.join(sorted(allowed_words))}\nExplanation:"
    )


def _validated_material(
    validation: MultiDocumentGenerationValidationResult,
) -> tuple[tuple[ReportedValue, ...], ComputedValue]:
    calculation = validation.reproduced_calculation
    assert calculation is not None
    facts = {binding.fact.fact_id: binding.fact for binding in validation.request.bindings}
    reported = tuple(
        _reported_value(facts[citation.fact_id], citation.marker)
        for citation in validation.citations
    )
    return reported, _computed_value(calculation, validation.citations)


class MultiDocumentAnswerGenerator:
    """Revalidate a raw request, then obtain and ground one qualitative sentence."""

    def __init__(
        self,
        backend: LocalModelBackend,
        *,
        rejected_output_handler: Callable[[RejectedModelOutput], None] | None = None,
    ) -> None:
        self._backend = backend
        self._rejected_output_handler = rejected_output_handler

    def generate(
        self,
        request: MultiDocumentGenerationRequest | MultiDocumentGenerationValidationResult,
        corpora: Mapping[str, EvidenceCorpus],
    ) -> MultiDocumentGeneratedAnswer:
        """Generate only after fresh validation; never trust a supplied validated status."""
        raw_request = (
            request.request
            if isinstance(request, MultiDocumentGenerationValidationResult)
            else request
        )
        validation = MultiDocumentGenerationValidator().validate(raw_request, corpora)
        if validation.status is MultiDocumentGenerationValidationStatus.ABSTAINED:
            reason = validation.abstention_reason
            return MultiDocumentGeneratedAnswer(
                MultiDocumentAnswerStatus.VALIDATION_ABSTAINED,
                validation,
                "I cannot answer from the supplied validated evidence.",
                (),
                None,
                None,
                (),
                _provenance(self._backend, None, None),
                reason.value if reason else "multi_document_validation_failed",
            )
        calculation = validation.reproduced_calculation
        assert calculation is not None
        conclusion, allowed, required = _grounded_conclusion(raw_request)  # type: ignore[arg-type]
        prompt = _build_prompt(raw_request, calculation, validation.citations, conclusion, allowed)
        prompt_hash = _sha256(prompt)
        started_ns = time.perf_counter_ns()
        try:
            raw_output = self._backend.generate(prompt, MULTI_DOCUMENT_INFERENCE_CONFIG)
        except Exception:
            reported, computed = _validated_material(validation)
            return MultiDocumentGeneratedAnswer(
                MultiDocumentAnswerStatus.INFERENCE_FAILED,
                validation,
                _render_deterministic_material(reported, computed),
                reported,
                computed,
                None,
                validation.citations,
                _provenance(
                    self._backend,
                    prompt_hash,
                    None,
                    _elapsed_ms(started_ns),
                ),
                "model_inference_failed",
            )
        latency_ms = _elapsed_ms(started_ns)
        output_hash = _sha256(raw_output) if isinstance(raw_output, str) else None
        explanation, diagnostic = _diagnose_explanation(raw_output, allowed, required)
        reported, computed = _validated_material(validation)
        if explanation is None:
            if self._rejected_output_handler is not None and isinstance(raw_output, str):
                self._rejected_output_handler(
                    RejectedModelOutput(
                        raw_request.question,
                        raw_request.task,
                        prompt_hash,
                        output_hash or _sha256(raw_output),
                        raw_output,
                        diagnostic,
                    )
                )
            return MultiDocumentGeneratedAnswer(
                MultiDocumentAnswerStatus.MODEL_OUTPUT_REJECTED,
                validation,
                _render_deterministic_material(reported, computed),
                reported,
                computed,
                None,
                validation.citations,
                _provenance(
                    self._backend,
                    prompt_hash,
                    output_hash,
                    latency_ms,
                    ExplanationValidationStatus.FAILED,
                    diagnostic,
                ),
                "model_output_failed_grounding",
            )
        return MultiDocumentGeneratedAnswer(
            MultiDocumentAnswerStatus.ANSWERED,
            validation,
            _render_answer(reported, computed, explanation),
            reported,
            computed,
            explanation,
            validation.citations,
            _provenance(
                self._backend,
                prompt_hash,
                output_hash,
                latency_ms,
                ExplanationValidationStatus.PASSED,
                diagnostic,
            ),
        )


__all__ = [
    "MULTI_DOCUMENT_ANSWER_SCHEMA_VERSION",
    "MULTI_DOCUMENT_INFERENCE_CONFIG",
    "MultiDocumentAnswerGenerator",
    "MultiDocumentAnswerStatus",
    "MultiDocumentGeneratedAnswer",
]
