"""Evidence-grounded answers over verified facts and deterministic calculations.

Models are untrusted presentation components: deterministic code renders every
financial value, formula, citation, and provenance record.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ledgerlens.calculations import (
    CalculationError,
    CalculationResult,
    CalculationType,
    FactOperand,
    profit_margin,
    year_over_year_growth,
)
from ledgerlens.financial_facts import FinancialFact, VerificationStatus
from ledgerlens.retrieval.bm25 import EvidenceCorpus

GENERATION_SCHEMA_VERSION = "1.0"
GROUNDING_POLICY_VERSION = "1.0"
_WORDS = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_FORBIDDEN = frozenset("0123456789$%€£¥[]{}<>")
_COMMON_WORDS = frozenset(
    "a and as compared current fiscal for from in indicates of period prior reported result "
    "shows the this to was with year".split()
)


class AnswerTask(StrEnum):
    """Initially supported, explicitly selected answer tasks."""

    REVENUE_GROWTH = "revenue_growth"
    NET_PROFIT_MARGIN = "net_profit_margin"


class AnswerStatus(StrEnum):
    """Whether supplied material supported a grounded answer."""

    ANSWERED = "answered"
    ABSTAINED = "abstained"


@dataclass(frozen=True)
class ModelIdentity:
    """Auditable identity of a configured local open-model backend."""

    backend: str
    model: str
    revision: str | None = None

    def __post_init__(self) -> None:
        for field, value in (("backend", self.backend), ("model", self.model)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision must be null or a non-empty string")

    def to_dict(self) -> dict[str, str | None]:
        return {"backend": self.backend, "model": self.model, "revision": self.revision}


@dataclass(frozen=True)
class InferenceConfig:
    """Backend-neutral local generation settings."""

    max_new_tokens: int = 96
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or not 1 <= self.max_new_tokens <= 512
        ):
            raise ValueError("max_new_tokens must be an integer between 1 and 512")
        if (
            not isinstance(self.temperature, int | float)
            or isinstance(self.temperature, bool)
            or not 0 <= self.temperature <= 2
        ):
            raise ValueError("temperature must be numeric and between 0 and 2")

    def to_dict(self) -> dict[str, int | float]:
        return {"max_new_tokens": self.max_new_tokens, "temperature": self.temperature}


class LocalModelBackend(Protocol):
    """Minimal interface for an in-process local open-model runtime."""

    @property
    def identity(self) -> ModelIdentity: ...

    def generate(self, prompt: str, config: InferenceConfig) -> str: ...


@dataclass(frozen=True)
class CallableLocalModelBackend:
    """Dependency-free adapter for a caller-owned local model runtime."""

    identity: ModelIdentity
    generate_text: Callable[[str, InferenceConfig], str]

    def generate(self, prompt: str, config: InferenceConfig) -> str:
        return self.generate_text(prompt, config)


@dataclass(frozen=True)
class AnswerRequest:
    """Explicit question, task, facts, result, and validating corpus."""

    question: str
    task: AnswerTask
    facts: tuple[FinancialFact, ...]
    calculation: CalculationResult
    corpus: EvidenceCorpus

    def __post_init__(self) -> None:
        if not isinstance(self.question, str) or not self.question.strip():
            raise ValueError("question must be a non-empty string")


@dataclass(frozen=True)
class EvidenceCitation:
    """Physical-page citation with evidence and artifact integrity metadata."""

    document_id: str
    source_sha256: str
    physical_page_number: int
    supporting_text: str
    supporting_text_sha256: str
    selected_text_sha256: str
    evidence_jsonl_sha256: str
    evidence_manifest_sha256: str
    screening_warnings: tuple[str, ...]
    known_limitations: tuple[str, ...]
    review_provenance: dict[str, object] | None

    @property
    def marker(self) -> str:
        return f"[PDF p. {self.physical_page_number}]"

    @classmethod
    def from_fact(cls, fact: FinancialFact) -> EvidenceCitation:
        evidence = fact.evidence
        return cls(
            document_id=evidence.document_id,
            source_sha256=evidence.source_sha256,
            physical_page_number=evidence.physical_page_number,
            supporting_text=evidence.supporting_text,
            supporting_text_sha256=evidence.supporting_text_sha256,
            selected_text_sha256=evidence.selected_text_sha256,
            evidence_jsonl_sha256=evidence.evidence_jsonl_sha256,
            evidence_manifest_sha256=evidence.evidence_manifest_sha256,
            screening_warnings=evidence.screening_warnings,
            known_limitations=evidence.known_limitations,
            review_provenance=evidence.review_provenance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "marker": self.marker,
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "physical_page_number": self.physical_page_number,
            "supporting_text": self.supporting_text,
            "supporting_text_sha256": self.supporting_text_sha256,
            "selected_text_sha256": self.selected_text_sha256,
            "evidence_jsonl_sha256": self.evidence_jsonl_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "screening_warnings": list(self.screening_warnings),
            "known_limitations": list(self.known_limitations),
            "review_provenance": self.review_provenance,
        }


@dataclass(frozen=True)
class ReportedValue:
    """Deterministically rendered source-reported value."""

    fact_id: str
    metric: str
    reported_value: str
    company: str
    reporting_period: str
    currency: str
    units: str
    citation_marker: str
    verification_status: str
    reviewed_by: str
    reviewed_at_utc: str
    review_method: str
    verification_note: str | None
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]

    def render(self) -> str:
        return (
            f"{self.company} reported {self.metric.replace('_', ' ')} of "
            f"{self.reported_value} ({self.currency}; {self.units}) for "
            f"{self.reporting_period} {self.citation_marker}."
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "metric": self.metric,
            "reported_value": self.reported_value,
            "company": self.company,
            "reporting_period": self.reporting_period,
            "currency": self.currency,
            "units": self.units,
            "citation_marker": self.citation_marker,
            "verification": {
                "status": self.verification_status,
                "reviewed_by": self.reviewed_by,
                "reviewed_at_utc": self.reviewed_at_utc,
                "method": self.review_method,
                "note": self.verification_note,
            },
            "warnings": list(self.warnings),
            "limitations": list(self.limitations),
            "text": self.render(),
        }


@dataclass(frozen=True)
class ComputedValue:
    """Deterministically rendered result, never arithmetic from the model."""

    calculation_id: str
    calculation_type: str
    formula: str
    input_fact_ids: tuple[str, ...]
    computed_value: str
    unit: str
    citation_markers: tuple[str, ...]

    def render(self) -> str:
        citations = " ".join(self.citation_markers)
        return (
            f"Deterministic {self.calculation_type.replace('_', ' ')}: "
            f"{self.computed_value} {self.unit}. Formula: {self.formula}. {citations}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "calculation_id": self.calculation_id,
            "calculation_type": self.calculation_type,
            "formula": self.formula,
            "input_fact_ids": list(self.input_fact_ids),
            "computed_value": self.computed_value,
            "unit": self.unit,
            "citation_markers": list(self.citation_markers),
            "text": self.render(),
        }


@dataclass(frozen=True)
class GenerationProvenance:
    """Model identity, settings, and hashes of exact model I/O."""

    model: ModelIdentity
    inference: InferenceConfig
    prompt_sha256: str | None
    model_output_sha256: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model.to_dict(),
            "inference": self.inference.to_dict(),
            "grounding_policy_version": GROUNDING_POLICY_VERSION,
            "prompt_sha256": self.prompt_sha256,
            "model_output_sha256": self.model_output_sha256,
        }


@dataclass(frozen=True)
class GeneratedAnswer:
    """Structured answer or conservative abstention."""

    status: AnswerStatus
    question: str
    task: AnswerTask
    answer_text: str
    reported_values: tuple[ReportedValue, ...]
    computed_value: ComputedValue | None
    model_generated_explanation: str | None
    citations: tuple[EvidenceCitation, ...]
    provenance: GenerationProvenance
    abstention_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GENERATION_SCHEMA_VERSION,
            "status": self.status.value,
            "question": self.question,
            "task": self.task.value,
            "answer_text": self.answer_text,
            "reported_values": [item.to_dict() for item in self.reported_values],
            "computed_value": self.computed_value.to_dict() if self.computed_value else None,
            "model_generated_explanation": self.model_generated_explanation,
            "citations": [item.to_dict() for item in self.citations],
            "provenance": self.provenance.to_dict(),
            "abstention_reason": self.abstention_reason,
        }


class AnswerGenerator:
    """Validate inputs, obtain a qualitative explanation, and render an answer."""

    def __init__(
        self, backend: LocalModelBackend, *, inference_config: InferenceConfig | None = None
    ) -> None:
        self._backend = backend
        self._config = inference_config or InferenceConfig()

    def generate(self, request: AnswerRequest) -> GeneratedAnswer:
        """Answer, or abstain on unsupported, unverified, or stale material."""
        error = self._validate_request(request)
        if error:
            return self._abstain(request, error)
        conclusion, allowed, required = _grounded_conclusion(request)
        prompt = _build_prompt(request, conclusion, allowed)
        prompt_hash = _sha256(prompt)
        try:
            raw_output = self._backend.generate(prompt, self._config)
        except Exception:  # A local runtime failure must not leak financial output.
            return self._abstain(request, "model_inference_failed", prompt_hash)
        output_hash = _sha256(raw_output) if isinstance(raw_output, str) else None
        explanation = _checked_explanation(raw_output, allowed, required)
        if explanation is None:
            return self._abstain(request, "model_output_failed_grounding", prompt_hash, output_hash)

        citations = _unique_citations(request.facts)
        reported = tuple(_reported_value(fact) for fact in request.facts)
        computed = _computed_value(request.calculation, citations)
        return GeneratedAnswer(
            status=AnswerStatus.ANSWERED,
            question=request.question,
            task=request.task,
            answer_text=_render_answer(reported, computed, explanation),
            reported_values=reported,
            computed_value=computed,
            model_generated_explanation=explanation,
            citations=citations,
            provenance=self._provenance(prompt_hash, output_hash),
        )

    def _validate_request(self, request: AnswerRequest) -> str | None:
        if len(request.facts) != 2 or len({fact.fact_id for fact in request.facts}) != 2:
            return "calculation_inputs_must_be_two_unique_facts"
        if any(
            fact.verification.status is not VerificationStatus.VERIFIED for fact in request.facts
        ):
            return "financial_facts_are_not_verified"
        try:
            for fact in request.facts:
                FactOperand(fact, request.corpus)
        except CalculationError:
            return "financial_fact_evidence_is_invalid_or_stale"
        inputs = {item.fact.fact_id: item.fact for item in request.calculation.inputs}
        supplied = {fact.fact_id: fact for fact in request.facts}
        if inputs != supplied:
            return "calculation_inputs_do_not_match_supplied_facts"
        roles = {item.role: item.fact for item in request.calculation.inputs}
        try:
            reproduced = self._reproduce(request, roles)
        except CalculationError:
            return "calculation_could_not_be_reproduced"
        if isinstance(reproduced, str):
            return reproduced
        if reproduced.to_dict() != request.calculation.to_dict():
            return "calculation_result_is_not_reproducible"
        return None

    def _reproduce(
        self, request: AnswerRequest, roles: dict[str, FinancialFact]
    ) -> CalculationResult | str:
        if request.task is AnswerTask.REVENUE_GROWTH:
            if not _question_supports_growth(request.question):
                return "question_is_not_supported_by_revenue_growth_evidence"
            if request.calculation.calculation_type is not CalculationType.YEAR_OVER_YEAR_GROWTH:
                return "calculation_type_does_not_match_answer_task"
            if set(roles) != {"current", "prior"} or any(
                fact.metric.casefold() != "revenue" for fact in roles.values()
            ):
                return "calculation_inputs_do_not_support_revenue_growth"
            return year_over_year_growth(
                FactOperand(roles["current"], request.corpus),
                FactOperand(roles["prior"], request.corpus),
            )
        if not _question_supports_margin(request.question):
            return "question_is_not_supported_by_net_profit_margin_evidence"
        if request.calculation.calculation_type is not CalculationType.PROFIT_MARGIN:
            return "calculation_type_does_not_match_answer_task"
        if (
            set(roles) != {"profit", "revenue"}
            or roles.get("profit", roles["revenue"]).metric.casefold() != "net_income"
            or roles["revenue"].metric.casefold() != "revenue"
        ):
            return "calculation_inputs_do_not_support_net_profit_margin"
        return profit_margin(
            FactOperand(roles["profit"], request.corpus),
            FactOperand(roles["revenue"], request.corpus),
        )

    def _provenance(self, prompt: str | None, output: str | None) -> GenerationProvenance:
        return GenerationProvenance(self._backend.identity, self._config, prompt, output)

    def _abstain(
        self,
        request: AnswerRequest,
        reason: str,
        prompt: str | None = None,
        output: str | None = None,
    ) -> GeneratedAnswer:
        return GeneratedAnswer(
            AnswerStatus.ABSTAINED,
            request.question,
            request.task,
            "I cannot answer from the supplied validated evidence.",
            (),
            None,
            None,
            (),
            self._provenance(prompt, output),
            reason,
        )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _question_supports_growth(question: str) -> bool:
    words = {word.casefold() for word in _WORDS.findall(question)}
    return "revenue" in words and bool(
        words & {"change", "changed", "grew", "grow", "growth", "increase", "increased"}
    )


def _question_supports_margin(question: str) -> bool:
    words = {word.casefold() for word in _WORDS.findall(question)}
    return "margin" in words and bool(words & {"income", "net", "profit"})


def _grounded_conclusion(
    request: AnswerRequest,
) -> tuple[str, frozenset[str], tuple[frozenset[str], ...]]:
    if request.task is AnswerTask.REVENUE_GROWTH:
        value = request.calculation.computed_value
        direction = "increased" if value > 0 else "decreased" if value < 0 else "unchanged"
        return (
            f"Revenue {direction} from the prior fiscal year.",
            _COMMON_WORDS | {"revenue", direction},
            (frozenset({"revenue"}), frozenset({direction})),
        )
    return (
        "Net income represented a portion of revenue for the fiscal year.",
        _COMMON_WORDS | {"income", "margin", "net", "portion", "represented", "revenue"},
        (
            frozenset({"revenue"}),
            frozenset({"margin", "portion", "represented"}),
            frozenset({"income", "net"}),
        ),
    )


def _build_prompt(request: AnswerRequest, conclusion: str, allowed_words: frozenset[str]) -> str:
    facts = "\n".join(
        f"- {fact.fact_id}: {fact.metric}; {fact.reported_value}; "
        f"{fact.reporting_period.label}; PDF physical page {fact.evidence.physical_page_number}"
        for fact in request.facts
    )
    return (
        "You are LedgerLens's qualitative explanation component. Return one plain-text "
        "sentence. Do not emit numbers, currency, percentages, citations, causes, forecasts, "
        "or facts beyond the allowed conclusion.\n"
        f"Question: {request.question.strip()}\nValidated source facts:\n{facts}\n"
        f"Validated calculation: {request.calculation.calculation_type.value}; "
        f"{request.calculation.computed_value} {request.calculation.unit.label}.\n"
        f"Allowed conclusion: {conclusion}\n"
        f"Allowed words: {', '.join(sorted(allowed_words))}\nExplanation:"
    )


def _checked_explanation(
    output: object,
    allowed: frozenset[str],
    required: tuple[frozenset[str], ...],
) -> str | None:
    if not isinstance(output, str):
        return None
    explanation = " ".join(output.split())
    if not explanation or len(explanation) > 300 or "\n" in output:
        return None
    if any(character in _FORBIDDEN for character in explanation):
        return None
    words = {word.casefold() for word in _WORDS.findall(explanation)}
    if not words or not words <= allowed or any(not words & group for group in required):
        return None
    if len(re.findall(r"[.!?]", explanation)) > 1:
        return None
    return explanation


def _unique_citations(facts: tuple[FinancialFact, ...]) -> tuple[EvidenceCitation, ...]:
    result: list[EvidenceCitation] = []
    seen: set[tuple[str, int, str]] = set()
    for fact in facts:
        citation = EvidenceCitation.from_fact(fact)
        key = (citation.document_id, citation.physical_page_number, citation.supporting_text_sha256)
        if key not in seen:
            seen.add(key)
            result.append(citation)
    return tuple(result)


def _reported_value(fact: FinancialFact) -> ReportedValue:
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
        EvidenceCitation.from_fact(fact).marker,
        fact.verification.status.value,
        fact.verification.reviewed_by,
        fact.verification.reviewed_at_utc,
        fact.verification.method,
        fact.verification.note,
        fact.warnings,
        fact.limitations,
    )


def _computed_value(
    calculation: CalculationResult, citations: tuple[EvidenceCitation, ...]
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


def _render_answer(
    reported: tuple[ReportedValue, ...], computed: ComputedValue, explanation: str
) -> str:
    facts = "\n".join(f"- {item.render()}" for item in reported)
    return (
        f"Source-reported values:\n{facts}\n\nComputed result:\n- {computed.render()}\n\n"
        f"Model-generated explanation:\n- {explanation}"
    )


__all__ = [
    "GENERATION_SCHEMA_VERSION",
    "GROUNDING_POLICY_VERSION",
    "AnswerGenerator",
    "AnswerRequest",
    "AnswerStatus",
    "AnswerTask",
    "CallableLocalModelBackend",
    "ComputedValue",
    "EvidenceCitation",
    "GeneratedAnswer",
    "GenerationProvenance",
    "InferenceConfig",
    "LocalModelBackend",
    "ModelIdentity",
    "ReportedValue",
]
