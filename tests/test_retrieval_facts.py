from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from pathlib import Path

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
from ledgerlens.generation import AnswerGenerator, InferenceConfig, ModelIdentity
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationRequest,
    OrchestrationStatus,
)
from ledgerlens.retrieval import (
    EmbeddingModelInfo,
    HybridRetriever,
    RetrievalError,
    build_embedding_index,
    load_embedding_index,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage
from ledgerlens.retrieval_facts import (
    AssociationOutcome,
    RetrievalAnswerAbstentionReason,
    RetrievalFinancialAnswerOrchestrator,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


@dataclass
class FakeEmbeddingBackend:
    info: EmbeddingModelInfo = EmbeddingModelInfo(
        "controlled-hybrid",
        "controlled/hybrid",
        "revision",
        "controlled",
        "1",
        2,
        128,
        "controlled",
        "controlled",
    )

    def embed_passages(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[tuple[float, ...], ...]:
        assert batch_size > 0
        vectors = []
        for text in texts:
            if "CURRENT STATEMENT" in text:
                vectors.append((1.0, 0.0))
            elif "PRIOR STATEMENT" in text:
                vectors.append((0.9, 0.1))
            else:
                vectors.append((0.0, 1.0))
        return tuple(vectors)

    def embed_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


@dataclass
class MockModelBackend:
    calls: int = 0
    identity: ModelIdentity = ModelIdentity("mock", "retrieval-fact-test")

    def generate(self, prompt: str, _config: InferenceConfig) -> str:
        self.calls += 1
        if "net profit margin" in prompt.casefold():
            return "Net income represented a portion of revenue for the fiscal year."
        return "Revenue increased from the prior fiscal year."


def _period(start: date, end: date, label: str) -> ReportingPeriod:
    return ReportingPeriod(ResolutionStatus.KNOWN, PeriodType.DURATION, start, end, label)


def _fixture(tmp_path: Path) -> tuple[EvidenceCorpus, tuple[FinancialFact, ...], HybridRetriever]:
    document_id = "sha256-" + "a" * 64
    source_hash = "a" * 64
    texts = (
        "CURRENT STATEMENT\nRevenue $150\nNet income $30",
        "PRIOR STATEMENT\nRevenue $100",
        "UNRELATED DISCLOSURE\nShare information",
    )
    pages = tuple(
        EvidencePage(
            document_id,
            "Synthetic Company",
            2025,
            number,
            text,
            _sha256(text),
            "canonical_extraction",
            {"source_sha256": source_hash},
            {"canonical": {"status": "pass", "warnings": []}},
            None,
            ("Controlled fixture only.",),
        )
        for number, text in enumerate(texts, start=1)
    )
    corpus = EvidenceCorpus(
        document_id,
        source_hash,
        pages,
        (),
        {},
        _sha256("evidence"),
        _sha256("manifest"),
    )
    current = _period(date(2025, 1, 1), date(2025, 12, 31), "FY2025")
    prior = _period(date(2024, 1, 1), date(2024, 12, 31), "FY2024")

    def fact(
        page_number: int,
        fact_id: str,
        metric: str,
        reported: str,
        normalized: str,
        period: ReportingPeriod,
    ) -> FinancialFact:
        page = pages[page_number - 1]
        return FinancialFact(
            fact_id,
            metric,
            reported,
            Decimal(normalized),
            page.company,
            period,
            Currency(ResolutionStatus.KNOWN, "USD"),
            Units(ResolutionStatus.KNOWN, UnitKind.MONETARY, Decimal("1000000"), "USD millions"),
            TableColumn(ResolutionStatus.KNOWN, period.label),
            SupportingEvidence.from_evidence(corpus, page, page.text),
            Verification(
                VerificationStatus.VERIFIED,
                "fixture reviewer",
                "2026-10-31T00:00:00Z",
                "controlled fixture",
            ),
        )

    facts = (
        fact(1, "revenue-2025", "revenue", "$150", "150", current),
        fact(1, "net-income-2025", "net_income", "$30", "30", current),
        fact(2, "revenue-2024", "revenue", "$100", "100", prior),
    )
    embedding_backend = FakeEmbeddingBackend()
    built = build_embedding_index(corpus, embedding_backend, output_root=tmp_path / "embeddings")
    index = load_embedding_index(built.output_directory, corpus)
    return corpus, facts, HybridRetriever(corpus, index, embedding_backend)


def _request(
    corpus: EvidenceCorpus,
    facts: tuple[FinancialFact, ...],
    question: str,
) -> OrchestrationRequest:
    return OrchestrationRequest(
        question,
        "Synthetic Company",
        date(2025, 12, 31),
        facts,
        corpus,
    )


def _integration(
    retriever: HybridRetriever, backend: MockModelBackend | None = None
) -> tuple[RetrievalFinancialAnswerOrchestrator, MockModelBackend]:
    model = backend or MockModelBackend()
    orchestrator = FinancialAnswerOrchestrator(AnswerGenerator(model))
    return RetrievalFinancialAnswerOrchestrator(orchestrator, retriever), model


def test_hybrid_retrieval_associates_distinct_facts_and_answers_supported_tasks(
    tmp_path: Path,
) -> None:
    corpus, facts, retriever = _fixture(tmp_path)
    integration, backend = _integration(retriever)

    growth = integration.answer(_request(corpus, facts, "How much did revenue grow?"), top_k=2)
    margin = integration.answer(_request(corpus, facts, "What was the net profit margin?"), top_k=1)

    assert growth.status is OrchestrationStatus.ANSWERED
    assert growth.orchestration is not None
    assert growth.orchestration.selected_fact_ids == ("revenue-2025", "revenue-2024")
    assert growth.orchestration.calculation is not None
    assert str(growth.orchestration.calculation.computed_value) == "50.0"
    assert growth.association.retrieved_pages[0].associated_fact_ids == (
        "net-income-2025",
        "revenue-2025",
    )
    assert margin.status is OrchestrationStatus.ANSWERED
    assert margin.orchestration is not None
    assert margin.orchestration.selected_fact_ids == ("net-income-2025", "revenue-2025")
    assert margin.orchestration.calculation is not None
    assert str(margin.orchestration.calculation.computed_value) == "20.0"
    assert backend.calls == 2


def test_retrieval_cutoff_never_falls_back_to_all_supplied_facts(tmp_path: Path) -> None:
    corpus, facts, retriever = _fixture(tmp_path)
    integration, backend = _integration(retriever)

    result = integration.answer(_request(corpus, facts, "How much did revenue grow?"), top_k=1)

    assert result.status is OrchestrationStatus.ABSTAINED
    assert result.abstention_reason is (
        RetrievalAnswerAbstentionReason.INSUFFICIENT_ASSOCIATED_FACTS
    )
    assert result.orchestration is not None
    assert result.orchestration.abstention_reason.value == "missing_fact"
    decisions = {item.fact_id: item for item in result.association.fact_decisions}
    assert decisions["revenue-2025"].outcome is AssociationOutcome.ASSOCIATED
    assert decisions["revenue-2024"].outcome is AssociationOutcome.EVIDENCE_NOT_RETRIEVED
    assert backend.calls == 0


def test_stale_unverified_and_ambiguous_associations_abstain_deterministically(
    tmp_path: Path,
) -> None:
    corpus, facts, retriever = _fixture(tmp_path)
    integration, backend = _integration(retriever)
    current, net_income, prior = facts
    stale = replace(
        current,
        evidence=replace(current.evidence, selected_text_sha256="0" * 64),
    )
    unverified = replace(
        current,
        verification=Verification(VerificationStatus.UNVERIFIED, note="controlled negative"),
    )
    duplicate = replace(current, fact_id="revenue-2025-duplicate")

    stale_result = integration.answer(
        _request(corpus, (stale, net_income, prior), "How much did revenue grow?"), top_k=2
    )
    unverified_result = integration.answer(
        _request(corpus, (unverified, net_income, prior), "How much did revenue grow?"), top_k=2
    )
    ambiguous_result = integration.answer(
        _request(corpus, (*facts, duplicate), "How much did revenue grow?"), top_k=2
    )

    assert stale_result.abstention_reason is (
        RetrievalAnswerAbstentionReason.INVALID_OR_STALE_ASSOCIATION
    )
    assert stale_result.association.invalid_fact_ids == ("revenue-2025",)
    assert unverified_result.abstention_reason is (
        RetrievalAnswerAbstentionReason.UNVERIFIED_ASSOCIATED_FACT
    )
    assert ambiguous_result.abstention_reason is (
        RetrievalAnswerAbstentionReason.AMBIGUOUS_ASSOCIATED_FACTS
    )
    assert backend.calls == 0


def test_tampered_retrieval_identity_and_retrieval_error_abstain(tmp_path: Path) -> None:
    corpus, facts, retriever = _fixture(tmp_path)
    all_results = retriever.search("How much did revenue grow?", top_k=3)
    results = all_results[:2]
    tampered = (replace(results[0], evidence_jsonl_sha256="0" * 64), results[1])
    unrelated = (replace(all_results[2], rank=1),)

    @dataclass
    class ControlledRetriever:
        results: tuple = ()
        error: bool = False

        def search(self, _query: str, *, top_k: int = 5, excerpt_characters: int = 500):
            assert top_k > 0 and excerpt_characters > 0
            if self.error:
                raise RetrievalError("controlled retrieval failure")
            return self.results

    invalid_integration = RetrievalFinancialAnswerOrchestrator(
        FinancialAnswerOrchestrator(AnswerGenerator(MockModelBackend())),
        ControlledRetriever(tampered),
    )
    failed_integration = RetrievalFinancialAnswerOrchestrator(
        FinancialAnswerOrchestrator(AnswerGenerator(MockModelBackend())),
        ControlledRetriever(error=True),
    )
    empty_integration = RetrievalFinancialAnswerOrchestrator(
        FinancialAnswerOrchestrator(AnswerGenerator(MockModelBackend())),
        ControlledRetriever(),
    )
    unrelated_integration = RetrievalFinancialAnswerOrchestrator(
        FinancialAnswerOrchestrator(AnswerGenerator(MockModelBackend())),
        ControlledRetriever(unrelated),
    )

    invalid = invalid_integration.answer(_request(corpus, facts, "How much did revenue grow?"))
    failed = failed_integration.answer(_request(corpus, facts, "How much did revenue grow?"))
    empty = empty_integration.answer(_request(corpus, facts, "How much did revenue grow?"))
    no_facts = unrelated_integration.answer(_request(corpus, facts, "How much did revenue grow?"))

    assert invalid.abstention_reason is (RetrievalAnswerAbstentionReason.INVALID_OR_STALE_RETRIEVAL)
    assert "stale evidence.jsonl hash" in (invalid.detail or "")
    assert failed.abstention_reason is RetrievalAnswerAbstentionReason.RETRIEVAL_FAILED
    assert failed.detail == "controlled retrieval failure"
    assert empty.abstention_reason is RetrievalAnswerAbstentionReason.NO_RETRIEVED_EVIDENCE
    assert no_facts.abstention_reason is (
        RetrievalAnswerAbstentionReason.NO_ASSOCIATED_REVIEWED_FACTS
    )
