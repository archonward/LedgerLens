"""Conservative association of hybrid-retrieved evidence with reviewed financial facts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from ledgerlens.financial_facts import (
    FinancialFact,
    FinancialFactValidationError,
    VerificationStatus,
)
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationAbstentionReason,
    OrchestrationRequest,
    OrchestrationResult,
    OrchestrationStatus,
)
from ledgerlens.retrieval.bm25 import EvidenceCorpus, EvidencePage, RetrievalError
from ledgerlens.retrieval.hybrid import HybridSearchResult

RETRIEVAL_FACT_SCHEMA_VERSION = "1.0"


class AssociationOutcome(StrEnum):
    """Outcome of comparing one reviewed fact with retrieved evidence identities."""

    ASSOCIATED = "associated"
    ASSOCIATED_UNVERIFIED = "associated_unverified"
    EVIDENCE_NOT_RETRIEVED = "evidence_not_retrieved"
    INVALID_OR_STALE = "invalid_or_stale"


class RetrievalAnswerAbstentionReason(StrEnum):
    """Retrieval-specific reasons that prevent or constrain orchestration."""

    RETRIEVAL_FAILED = "retrieval_failed"
    NO_RETRIEVED_EVIDENCE = "no_retrieved_evidence"
    INVALID_OR_STALE_RETRIEVAL = "invalid_or_stale_retrieval"
    NO_ASSOCIATED_REVIEWED_FACTS = "no_associated_reviewed_facts"
    INVALID_OR_STALE_ASSOCIATION = "invalid_or_stale_association"
    INSUFFICIENT_ASSOCIATED_FACTS = "insufficient_associated_facts"
    AMBIGUOUS_ASSOCIATED_FACTS = "ambiguous_associated_facts"
    UNVERIFIED_ASSOCIATED_FACT = "unverified_associated_fact"
    DOWNSTREAM_ABSTENTION = "downstream_abstention"


@dataclass(frozen=True)
class RetrievedPageAssociation:
    """One retrieved page and all exact-identity fact associations on that page."""

    rank: int
    score: float
    physical_page_number: int
    document_id: str
    source_sha256: str
    selected_text_sha256: str
    evidence_jsonl_sha256: str
    evidence_manifest_sha256: str
    evidence_method: str
    associated_fact_ids: tuple[str, ...]
    retrieval_details: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "score": self.score,
            "physical_page_number": self.physical_page_number,
            "document_id": self.document_id,
            "source_sha256": self.source_sha256,
            "selected_text_sha256": self.selected_text_sha256,
            "evidence_jsonl_sha256": self.evidence_jsonl_sha256,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "evidence_method": self.evidence_method,
            "associated_fact_ids": list(self.associated_fact_ids),
            "retrieval_details": self.retrieval_details,
        }


@dataclass(frozen=True)
class FactAssociationDecision:
    """Auditable outcome for one supplied financial fact."""

    fact_id: str
    outcome: AssociationOutcome
    verification_status: VerificationStatus
    physical_page_number: int
    selected_text_sha256: str
    retrieval_rank: int | None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "fact_id": self.fact_id,
            "outcome": self.outcome.value,
            "verification_status": self.verification_status.value,
            "physical_page_number": self.physical_page_number,
            "selected_text_sha256": self.selected_text_sha256,
            "retrieval_rank": self.retrieval_rank,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FactAssociationResult:
    """Validated retrieved pages, fact decisions, and eligible associated facts."""

    query: str
    top_k: int
    retrieved_pages: tuple[RetrievedPageAssociation, ...]
    fact_decisions: tuple[FactAssociationDecision, ...]
    associated_facts: tuple[FinancialFact, ...]
    error: RetrievalAnswerAbstentionReason | None = None
    detail: str | None = None

    @property
    def invalid_fact_ids(self) -> tuple[str, ...]:
        return tuple(
            decision.fact_id
            for decision in self.fact_decisions
            if decision.outcome is AssociationOutcome.INVALID_OR_STALE
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "top_k": self.top_k,
            "retrieved_page_count": len(self.retrieved_pages),
            "retrieved_pages": [item.to_dict() for item in self.retrieved_pages],
            "fact_decisions": [item.to_dict() for item in self.fact_decisions],
            "associated_fact_ids": [fact.fact_id for fact in self.associated_facts],
            "invalid_fact_ids": list(self.invalid_fact_ids),
            "error": self.error.value if self.error else None,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class RetrievalFinancialAnswerResult:
    """Retrieval association plus the unchanged downstream orchestration result."""

    status: OrchestrationStatus
    question: str
    association: FactAssociationResult
    orchestration: OrchestrationResult | None
    abstention_reason: RetrievalAnswerAbstentionReason | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RETRIEVAL_FACT_SCHEMA_VERSION,
            "status": self.status.value,
            "question": self.question,
            "retrieval_association": self.association.to_dict(),
            "orchestration": self.orchestration.to_dict() if self.orchestration else None,
            "abstention_reason": self.abstention_reason.value if self.abstention_reason else None,
            "detail": self.detail,
        }


class HybridRetrieverLike(Protocol):
    """Search boundary shared by the real hybrid retriever and controlled tests."""

    def search(
        self, query: str, *, top_k: int = 5, excerpt_characters: int = 500
    ) -> tuple[HybridSearchResult, ...]: ...


class RetrievalFinancialAnswerOrchestrator:
    """Retrieve evidence, associate reviewed facts, then invoke the existing orchestrator."""

    def __init__(
        self,
        orchestrator: FinancialAnswerOrchestrator,
        retriever: HybridRetrieverLike,
    ) -> None:
        self._orchestrator = orchestrator
        self._retriever = retriever

    def answer(
        self, request: OrchestrationRequest, *, top_k: int = 5
    ) -> RetrievalFinancialAnswerResult:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        try:
            results = self._retriever.search(request.question, top_k=top_k)
        except RetrievalError as exc:
            association = FactAssociationResult(
                request.question,
                top_k,
                (),
                (),
                (),
                RetrievalAnswerAbstentionReason.RETRIEVAL_FAILED,
                str(exc),
            )
            return _retrieval_abstention(request.question, association, association.error, str(exc))

        association = associate_retrieved_facts(
            request.question,
            top_k,
            request.corpus,
            request.facts,
            results,
        )
        if association.error is not None:
            return _retrieval_abstention(
                request.question,
                association,
                association.error,
                association.detail,
            )
        if not association.retrieved_pages:
            return _retrieval_abstention(
                request.question,
                association,
                RetrievalAnswerAbstentionReason.NO_RETRIEVED_EVIDENCE,
            )
        if not association.associated_facts:
            reason = (
                RetrievalAnswerAbstentionReason.INVALID_OR_STALE_ASSOCIATION
                if association.invalid_fact_ids
                else RetrievalAnswerAbstentionReason.NO_ASSOCIATED_REVIEWED_FACTS
            )
            return _retrieval_abstention(request.question, association, reason)

        downstream = self._orchestrator.answer(
            OrchestrationRequest(
                request.question,
                request.company,
                request.target_period_end,
                association.associated_facts,
                request.corpus,
            )
        )
        if downstream.status is OrchestrationStatus.ANSWERED:
            return RetrievalFinancialAnswerResult(
                OrchestrationStatus.ANSWERED,
                request.question,
                association,
                downstream,
            )
        reason = _map_downstream_reason(downstream.abstention_reason, association)
        return RetrievalFinancialAnswerResult(
            OrchestrationStatus.ABSTAINED,
            request.question,
            association,
            downstream,
            reason,
            downstream.detail,
        )


def associate_retrieved_facts(
    query: str,
    top_k: int,
    corpus: EvidenceCorpus,
    facts: tuple[FinancialFact, ...],
    results: tuple[HybridSearchResult, ...],
) -> FactAssociationResult:
    """Associate facts only when their complete selected-evidence identity was retrieved."""
    retrieval_error = _validate_retrieval_results(corpus, results)
    if retrieval_error is not None:
        return FactAssociationResult(
            query,
            top_k,
            (),
            (),
            (),
            RetrievalAnswerAbstentionReason.INVALID_OR_STALE_RETRIEVAL,
            retrieval_error,
        )

    rank_by_identity = {_page_identity(corpus, result.page): result.rank for result in results}
    decisions: list[FactAssociationDecision] = []
    associated: list[FinancialFact] = []
    for fact in facts:
        try:
            fact.validate_against(corpus)
        except FinancialFactValidationError as exc:
            decisions.append(
                FactAssociationDecision(
                    fact.fact_id,
                    AssociationOutcome.INVALID_OR_STALE,
                    fact.verification.status,
                    fact.evidence.physical_page_number,
                    fact.evidence.selected_text_sha256,
                    None,
                    str(exc),
                )
            )
            continue
        rank = rank_by_identity.get(_fact_identity(fact))
        if rank is None:
            decisions.append(
                FactAssociationDecision(
                    fact.fact_id,
                    AssociationOutcome.EVIDENCE_NOT_RETRIEVED,
                    fact.verification.status,
                    fact.evidence.physical_page_number,
                    fact.evidence.selected_text_sha256,
                    None,
                )
            )
            continue
        outcome = (
            AssociationOutcome.ASSOCIATED
            if fact.verification.status is VerificationStatus.VERIFIED
            else AssociationOutcome.ASSOCIATED_UNVERIFIED
        )
        decisions.append(
            FactAssociationDecision(
                fact.fact_id,
                outcome,
                fact.verification.status,
                fact.evidence.physical_page_number,
                fact.evidence.selected_text_sha256,
                rank,
            )
        )
        associated.append(fact)

    associated_ids_by_rank: dict[int, list[str]] = {result.rank: [] for result in results}
    for decision in decisions:
        if decision.retrieval_rank is not None:
            associated_ids_by_rank[decision.retrieval_rank].append(decision.fact_id)
    pages = tuple(
        RetrievedPageAssociation(
            result.rank,
            result.score,
            result.page.physical_page_number,
            result.page.document_id,
            corpus.source_sha256,
            result.page.text_sha256,
            result.evidence_jsonl_sha256,
            result.manifest_sha256,
            result.page.method,
            tuple(sorted(associated_ids_by_rank[result.rank])),
            result.retrieval_details,
        )
        for result in results
    )
    return FactAssociationResult(
        query,
        top_k,
        pages,
        tuple(decisions),
        tuple(associated),
    )


def _page_identity(corpus: EvidenceCorpus, page: EvidencePage) -> tuple[object, ...]:
    return (
        page.document_id,
        corpus.source_sha256,
        corpus.evidence_jsonl_sha256,
        corpus.manifest_sha256,
        page.physical_page_number,
        page.text_sha256,
        page.method,
    )


def _fact_identity(fact: FinancialFact) -> tuple[object, ...]:
    evidence = fact.evidence
    return (
        evidence.document_id,
        evidence.source_sha256,
        evidence.evidence_jsonl_sha256,
        evidence.evidence_manifest_sha256,
        evidence.physical_page_number,
        evidence.selected_text_sha256,
        evidence.evidence_method,
    )


def _validate_retrieval_results(
    corpus: EvidenceCorpus, results: tuple[HybridSearchResult, ...]
) -> str | None:
    if tuple(result.rank for result in results) != tuple(range(1, len(results) + 1)):
        return "retrieval ranks must be unique and contiguous from one"
    seen: set[tuple[object, ...]] = set()
    for result in results:
        if result.evidence_jsonl_sha256 != corpus.evidence_jsonl_sha256:
            return f"retrieval rank {result.rank} has a stale evidence.jsonl hash"
        if result.manifest_sha256 != corpus.manifest_sha256:
            return f"retrieval rank {result.rank} has a stale evidence manifest hash"
        if result.page not in corpus.pages:
            return f"retrieval rank {result.rank} page is not in the supplied evidence corpus"
        identity = _page_identity(corpus, result.page)
        if identity in seen:
            return "retrieval results contain a duplicate selected-evidence identity"
        seen.add(identity)
    return None


def _map_downstream_reason(
    reason: OrchestrationAbstentionReason | None,
    association: FactAssociationResult,
) -> RetrievalAnswerAbstentionReason:
    if reason in {
        OrchestrationAbstentionReason.MISSING_FACT,
        OrchestrationAbstentionReason.UNRESOLVED_FACT_CONTEXT,
    }:
        if association.invalid_fact_ids:
            return RetrievalAnswerAbstentionReason.INVALID_OR_STALE_ASSOCIATION
        return RetrievalAnswerAbstentionReason.INSUFFICIENT_ASSOCIATED_FACTS
    if reason in {
        OrchestrationAbstentionReason.AMBIGUOUS_FACT,
        OrchestrationAbstentionReason.DUPLICATE_FACT_ID,
    }:
        return RetrievalAnswerAbstentionReason.AMBIGUOUS_ASSOCIATED_FACTS
    if reason is OrchestrationAbstentionReason.UNVERIFIED_FACT:
        return RetrievalAnswerAbstentionReason.UNVERIFIED_ASSOCIATED_FACT
    if reason is OrchestrationAbstentionReason.INVALID_OR_STALE_EVIDENCE:
        return RetrievalAnswerAbstentionReason.INVALID_OR_STALE_ASSOCIATION
    return RetrievalAnswerAbstentionReason.DOWNSTREAM_ABSTENTION


def _retrieval_abstention(
    question: str,
    association: FactAssociationResult,
    reason: RetrievalAnswerAbstentionReason | None,
    detail: str | None = None,
) -> RetrievalFinancialAnswerResult:
    return RetrievalFinancialAnswerResult(
        OrchestrationStatus.ABSTAINED,
        question,
        association,
        None,
        reason,
        detail,
    )


__all__ = [
    "RETRIEVAL_FACT_SCHEMA_VERSION",
    "AssociationOutcome",
    "FactAssociationDecision",
    "FactAssociationResult",
    "RetrievalAnswerAbstentionReason",
    "RetrievalFinancialAnswerOrchestrator",
    "RetrievalFinancialAnswerResult",
    "RetrievedPageAssociation",
    "associate_retrieved_facts",
]
