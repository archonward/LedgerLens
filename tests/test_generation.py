from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal

from ledgerlens.calculations import FactOperand, profit_margin, year_over_year_growth
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
from ledgerlens.generation import (
    AnswerGenerator,
    AnswerRequest,
    AnswerStatus,
    AnswerTask,
    ExplanationValidationStatus,
    GenerationProvenance,
    GroundingRejectionCode,
    InferenceConfig,
    ModelIdentity,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _period(start: date, end: date, label: str) -> ReportingPeriod:
    return ReportingPeriod(
        ResolutionStatus.KNOWN,
        PeriodType.DURATION,
        start,
        end,
        label,
    )


def _fixture() -> tuple[EvidenceCorpus, dict[str, FinancialFact]]:
    rows = (
        ("Revenue", "$150"),
        ("Revenue", "$100"),
        ("Net income", "$30"),
    )
    pages = []
    for number, (metric, value) in enumerate(rows, 1):
        text = f"Controlled statement\n{metric}: {value}"
        pages.append(
            EvidencePage(
                document_id="sha256-" + "a" * 64,
                company="Synthetic Company",
                report_year=2026,
                physical_page_number=number,
                text=text,
                text_sha256=_sha256(text),
                method="canonical_extraction",
                provenance={"source_sha256": "a" * 64},
                screening_findings={"canonical": {"status": "pass", "warnings": []}},
                review_provenance=None,
                known_limitations=("Controlled fixture only.",),
            )
        )
    corpus = EvidenceCorpus(
        document_id="sha256-" + "a" * 64,
        source_sha256="a" * 64,
        pages=tuple(pages),
        excluded_pages=(),
        manifest={},
        evidence_jsonl_sha256=_sha256("evidence"),
        manifest_sha256=_sha256("manifest"),
    )
    current = _period(date(2025, 1, 1), date(2025, 12, 31), "FY2025")
    prior = _period(date(2024, 1, 1), date(2024, 12, 31), "FY2024")

    def fact(
        page: int, fact_id: str, metric: str, value: str, normalized: str, period: ReportingPeriod
    ) -> FinancialFact:
        evidence_page = corpus.pages[page - 1]
        return FinancialFact(
            fact_id=fact_id,
            metric=metric,
            reported_value=value,
            normalized_value=Decimal(normalized),
            company=evidence_page.company,
            reporting_period=period,
            currency=Currency(ResolutionStatus.KNOWN, "USD"),
            units=Units(
                ResolutionStatus.KNOWN,
                UnitKind.MONETARY,
                Decimal("1000000"),
                "USD millions",
            ),
            table_column=TableColumn(ResolutionStatus.KNOWN, period.label),
            evidence=SupportingEvidence.from_evidence(corpus, evidence_page, evidence_page.text),
            verification=Verification(
                VerificationStatus.VERIFIED,
                "fixture reviewer",
                "2026-10-24T00:00:00Z",
                "controlled offline fixture",
            ),
            limitations=("Controlled fixture only.",),
        )

    return corpus, {
        "current_revenue": fact(1, "revenue-2025", "revenue", "$150", "150", current),
        "prior_revenue": fact(2, "revenue-2024", "revenue", "$100", "100", prior),
        "net_income": fact(3, "net-income-2025", "net_income", "$30", "30", current),
    }


@dataclass
class MockBackend:
    response: object
    calls: int = 0
    prompt: str = ""
    identity: ModelIdentity = ModelIdentity("mock-local", "controlled-open-model", "test")

    def generate(self, prompt: str, config: InferenceConfig) -> str:
        self.calls += 1
        self.prompt = prompt
        if isinstance(self.response, Exception):
            raise self.response
        return self.response  # type: ignore[return-value]


def _growth_request(corpus: EvidenceCorpus, facts: dict[str, FinancialFact]) -> AnswerRequest:
    current = facts["current_revenue"]
    prior = facts["prior_revenue"]
    calculation = year_over_year_growth(FactOperand(current, corpus), FactOperand(prior, corpus))
    return AnswerRequest(
        "How much did revenue grow in the current fiscal year?",
        AnswerTask.REVENUE_GROWTH,
        (current, prior),
        calculation,
        corpus,
    )


def test_growth_answer_separates_reported_computed_and_model_content() -> None:
    corpus, facts = _fixture()
    backend = MockBackend("Revenue increased from the prior fiscal year.")

    answer = AnswerGenerator(backend).generate(_growth_request(corpus, facts))
    payload = answer.to_dict()

    assert answer.status is AnswerStatus.ANSWERED
    assert "Source-reported values:" in answer.answer_text
    assert "$150" in answer.answer_text and "$100" in answer.answer_text
    assert "Computed result:" in answer.answer_text
    assert "50.0 percent" in answer.answer_text
    assert "Model-generated explanation:" in answer.answer_text
    assert answer.model_generated_explanation == "Revenue increased from the prior fiscal year."
    assert {item.marker for item in answer.citations} == {"[PDF p. 1]", "[PDF p. 2]"}
    assert payload["computed_value"]["computed_value"] == "50.0"  # type: ignore[index]
    assert payload["citations"][0]["source_sha256"] == "a" * 64  # type: ignore[index]
    assert payload["reported_values"][0]["verification"]["reviewed_by"] == (  # type: ignore[index]
        "fixture reviewer"
    )
    assert payload["reported_values"][0]["limitations"] == [  # type: ignore[index]
        "Controlled fixture only."
    ]
    assert payload["provenance"]["prompt_sha256"]  # type: ignore[index]
    assert payload["provenance"]["model_output_sha256"]  # type: ignore[index]
    assert answer.provenance.response_latency_ms is not None
    assert answer.provenance.explanation_validation is ExplanationValidationStatus.PASSED
    assert "$150" in backend.prompt and "50.0 percent" in backend.prompt


def test_equivalent_noncausal_growth_wording_is_accepted() -> None:
    corpus, facts = _fixture()
    request = _growth_request(corpus, facts)

    output = "Revenue for the current fiscal year shows an increase compared to the prior year."
    answer = AnswerGenerator(MockBackend(output)).generate(request)

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.provenance.grounding_diagnostic is not None
    assert answer.provenance.grounding_diagnostic.accepted is True
    assert answer.provenance.to_dict()["grounding_policy_version"] == "1.1"


def test_net_profit_margin_answer_uses_validated_inputs() -> None:
    corpus, facts = _fixture()
    profit = facts["net_income"]
    revenue = facts["current_revenue"]
    calculation = profit_margin(FactOperand(profit, corpus), FactOperand(revenue, corpus))
    request = AnswerRequest(
        "What was the net profit margin?",
        AnswerTask.NET_PROFIT_MARGIN,
        (profit, revenue),
        calculation,
        corpus,
    )

    answer = AnswerGenerator(
        MockBackend("Net income represented a portion of revenue for the fiscal year.")
    ).generate(request)

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.computed_value is not None
    assert answer.computed_value.computed_value == "20.0"
    assert answer.computed_value.input_fact_ids == ("net-income-2025", "revenue-2025")


def test_numeric_or_causal_model_claim_causes_abstention() -> None:
    corpus, facts = _fixture()
    request = _growth_request(corpus, facts)

    numeric = AnswerGenerator(MockBackend("Revenue increased by 50 percent.")).generate(request)
    causal = AnswerGenerator(MockBackend("Revenue increased because of demand.")).generate(request)

    for answer in (numeric, causal):
        assert answer.status is AnswerStatus.ABSTAINED
        assert answer.abstention_reason == "model_output_failed_grounding"
        assert answer.reported_values == ()
        assert answer.computed_value is None
        assert answer.citations == ()
        assert answer.provenance.response_latency_ms is not None
        assert answer.provenance.explanation_validation is ExplanationValidationStatus.FAILED

    assert numeric.provenance.grounding_diagnostic is not None
    assert numeric.provenance.grounding_diagnostic.rejection_codes == (
        GroundingRejectionCode.FORBIDDEN_CHARACTER,
        GroundingRejectionCode.DISALLOWED_VOCABULARY,
    )
    assert numeric.provenance.grounding_diagnostic.forbidden_characters == ("0", "5")
    assert numeric.provenance.grounding_diagnostic.disallowed_words == ("by", "percent")
    assert causal.provenance.grounding_diagnostic is not None
    assert causal.provenance.grounding_diagnostic.rejection_codes == (
        GroundingRejectionCode.DISALLOWED_VOCABULARY,
    )
    assert causal.provenance.grounding_diagnostic.disallowed_words == ("because", "demand")


def test_wrong_direction_and_multiline_diagnostics_do_not_change_rejection() -> None:
    corpus, facts = _fixture()
    request = _growth_request(corpus, facts)

    wrong = AnswerGenerator(MockBackend("Revenue decreased from the prior fiscal year.")).generate(
        request
    )
    multiline = AnswerGenerator(
        MockBackend("Revenue increased from the prior fiscal year.\nThis result shows revenue.")
    ).generate(request)

    assert wrong.abstention_reason == "model_output_failed_grounding"
    assert wrong.provenance.grounding_diagnostic is not None
    assert wrong.provenance.grounding_diagnostic.rejection_codes == (
        GroundingRejectionCode.DISALLOWED_VOCABULARY,
        GroundingRejectionCode.MISSING_REQUIRED_TERM_GROUP,
    )
    assert wrong.provenance.grounding_diagnostic.disallowed_words == ("decreased",)
    assert wrong.provenance.grounding_diagnostic.missing_required_term_groups == (
        ("increase", "increased"),
    )
    assert multiline.abstention_reason == "model_output_failed_grounding"
    assert multiline.provenance.grounding_diagnostic is not None
    assert GroundingRejectionCode.MULTILINE_OUTPUT in (
        multiline.provenance.grounding_diagnostic.rejection_codes
    )
    assert GroundingRejectionCode.MULTIPLE_SENTENCES in (
        multiline.provenance.grounding_diagnostic.rejection_codes
    )


def test_nominal_growth_wording_still_rejects_wrong_direction_forecasts_and_citations() -> None:
    corpus, facts = _fixture()
    request = _growth_request(corpus, facts)
    rejected = (
        "Revenue for the current fiscal year shows a decrease compared to the prior year.",
        "Revenue for the current fiscal year shows an increase and forecasts more growth.",
        "Revenue for the current fiscal year shows an increase [PDF p. 139].",
    )

    answers = tuple(AnswerGenerator(MockBackend(output)).generate(request) for output in rejected)

    assert all(answer.status is AnswerStatus.ABSTAINED for answer in answers)
    assert all(answer.abstention_reason == "model_output_failed_grounding" for answer in answers)
    assert answers[0].provenance.grounding_diagnostic is not None
    assert answers[0].provenance.grounding_diagnostic.disallowed_words == ("decrease",)
    assert answers[1].provenance.grounding_diagnostic is not None
    assert answers[1].provenance.grounding_diagnostic.disallowed_words == (
        "forecasts",
        "growth",
        "more",
    )
    assert answers[2].provenance.grounding_diagnostic is not None
    assert GroundingRejectionCode.FORBIDDEN_CHARACTER in (
        answers[2].provenance.grounding_diagnostic.rejection_codes
    )


def test_raw_rejected_output_is_available_only_through_opt_in_handler() -> None:
    corpus, facts = _fixture()
    request = _growth_request(corpus, facts)
    records = []
    generator = AnswerGenerator(
        MockBackend("Revenue increased because of demand."),
        rejected_output_handler=records.append,
    )

    answer = generator.generate(request)

    assert answer.status is AnswerStatus.ABSTAINED
    assert len(records) == 1
    assert records[0].raw_output == "Revenue increased because of demand."
    assert records[0].model_output_sha256 == answer.provenance.model_output_sha256
    assert "raw_output" not in answer.to_dict()["provenance"]  # type: ignore[operator]


def test_unsupported_question_abstains_before_model_call() -> None:
    corpus, facts = _fixture()
    original = _growth_request(corpus, facts)
    request = replace(original, question="Why did operating expenses change?")
    backend = MockBackend("Revenue increased from the prior fiscal year.")

    answer = AnswerGenerator(backend).generate(request)

    assert answer.status is AnswerStatus.ABSTAINED
    assert answer.abstention_reason == "question_is_not_supported_by_revenue_growth_evidence"
    assert backend.calls == 0


def test_stale_evidence_and_tampered_calculation_abstain() -> None:
    corpus, facts = _fixture()
    request = _growth_request(corpus, facts)
    backend = MockBackend("Revenue increased from the prior fiscal year.")

    stale = replace(request, corpus=replace(corpus, manifest_sha256="b" * 64))
    stale_answer = AnswerGenerator(backend).generate(stale)
    tampered = replace(
        request,
        calculation=replace(request.calculation, computed_value=Decimal("99")),
    )
    tampered_answer = AnswerGenerator(backend).generate(tampered)

    assert stale_answer.abstention_reason == "financial_fact_evidence_is_invalid_or_stale"
    assert tampered_answer.abstention_reason == "calculation_result_is_not_reproducible"
    assert backend.calls == 0


def test_unverified_fact_and_backend_failure_abstain() -> None:
    corpus, facts = _fixture()
    request = _growth_request(corpus, facts)
    unverified = replace(
        facts["current_revenue"],
        verification=Verification(VerificationStatus.UNVERIFIED),
    )
    unverified_request = replace(
        request,
        facts=(unverified, facts["prior_revenue"]),
    )

    unverified_answer = AnswerGenerator(MockBackend("unused")).generate(unverified_request)
    failed_answer = AnswerGenerator(MockBackend(RuntimeError("offline runtime failed"))).generate(
        request
    )

    assert unverified_answer.abstention_reason == "financial_facts_are_not_verified"
    assert (
        unverified_answer.provenance.explanation_validation is ExplanationValidationStatus.NOT_RUN
    )
    assert failed_answer.abstention_reason == "model_inference_failed"
    assert failed_answer.provenance.prompt_sha256 is not None
    assert failed_answer.provenance.model_output_sha256 is None
    assert failed_answer.provenance.response_latency_ms is not None
    assert failed_answer.provenance.explanation_validation is ExplanationValidationStatus.NOT_RUN


def test_existing_generation_provenance_constructor_remains_compatible() -> None:
    provenance = GenerationProvenance(
        ModelIdentity("mock-local", "controlled-open-model"),
        InferenceConfig(),
        None,
        None,
    )

    assert provenance.response_latency_ms is None
    assert provenance.explanation_validation is ExplanationValidationStatus.NOT_RUN
