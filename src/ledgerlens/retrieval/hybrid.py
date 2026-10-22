"""Reciprocal-rank fusion of page-level BM25 and semantic retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ledgerlens.retrieval.bm25 import (
    EXCERPT_CHARACTERS_DEFAULT,
    BM25Retriever,
    EvidenceCorpus,
    EvidencePage,
    RetrievalError,
    SearchResult,
)
from ledgerlens.retrieval.semantic import EmbeddingBackend, EmbeddingIndex, SemanticRetriever

RRF_K_DEFAULT = 60
BM25_WEIGHT_DEFAULT = 1.0
SEMANTIC_WEIGHT_DEFAULT = 1.0


@dataclass(frozen=True)
class FusionComponent:
    """One retriever's rank, score, and contribution for a fused page."""

    rank: int | None
    raw_score: float | None
    weight: float
    contribution: float

    def to_dict(self) -> dict[str, int | float | None]:
        """Return serializable component evidence."""
        return {
            "rank": self.rank,
            "raw_score": self.raw_score,
            "weight": self.weight,
            "rrf_contribution": self.contribution,
        }


@dataclass(frozen=True)
class HybridSearchResult:
    """A fused physical-page result retaining both component rankings."""

    rank: int
    score: float
    page: EvidencePage
    excerpt: str
    evidence_jsonl_sha256: str
    manifest_sha256: str
    rrf_k: int
    bm25: FusionComponent
    semantic: FusionComponent
    excerpt_source: str

    retrieval_details_key = "fusion"

    @property
    def retrieval_details(self) -> dict[str, Any]:
        """Return the exact fusion inputs for evaluation reports."""
        return {
            "method": "weighted_reciprocal_rank_fusion",
            "rrf_k": self.rrf_k,
            "tie_breaker": "ascending_physical_page_number",
            "excerpt_source": self.excerpt_source,
            "components": {
                "bm25": self.bm25.to_dict(),
                "page_semantic": self.semantic.to_dict(),
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return selected evidence plus transparent fusion details."""
        payload = SearchResult(
            rank=self.rank,
            score=self.score,
            page=self.page,
            excerpt=self.excerpt,
            evidence_jsonl_sha256=self.evidence_jsonl_sha256,
            manifest_sha256=self.manifest_sha256,
        ).to_dict()
        payload["fusion"] = self.retrieval_details
        return payload


def _validate_weight(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RetrievalError(f"{label} must be a finite number greater than zero")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise RetrievalError(f"{label} must be a finite number greater than zero")
    return value


class HybridRetriever:
    """Fuse existing page-level BM25 and semantic rankings with weighted RRF."""

    def __init__(
        self,
        corpus: EvidenceCorpus,
        semantic_index: EmbeddingIndex,
        backend: EmbeddingBackend,
        *,
        rrf_k: int = RRF_K_DEFAULT,
        bm25_weight: float = BM25_WEIGHT_DEFAULT,
        semantic_weight: float = SEMANTIC_WEIGHT_DEFAULT,
    ) -> None:
        if not isinstance(rrf_k, int) or isinstance(rrf_k, bool) or rrf_k < 1:
            raise RetrievalError("RRF rank constant must be a positive integer")
        self.rrf_k = rrf_k
        self.bm25_weight = _validate_weight(bm25_weight, "BM25 weight")
        self.semantic_weight = _validate_weight(semantic_weight, "Semantic weight")
        evidence_checks = {
            "document_id": corpus.document_id,
            "source_sha256": corpus.source_sha256,
            "evidence_jsonl_sha256": corpus.evidence_jsonl_sha256,
            "evidence_manifest_sha256": corpus.manifest_sha256,
        }
        for key, expected in evidence_checks.items():
            if semantic_index.manifest.get(key) != expected:
                raise RetrievalError(
                    f"Semantic embedding index {key} does not match the hybrid evidence corpus"
                )
        self.corpus = corpus
        self.bm25 = BM25Retriever(corpus)
        self.semantic = SemanticRetriever(corpus, semantic_index, backend)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        excerpt_characters: int = EXCERPT_CHARACTERS_DEFAULT,
    ) -> tuple[HybridSearchResult, ...]:
        """Fuse full component rankings and return deterministic physical pages."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise RetrievalError("top-k must be a positive integer")
        if excerpt_characters < 1:
            raise RetrievalError("excerpt characters must be a positive integer")
        if not query.strip() or not self.corpus.pages:
            return ()

        depth = len(self.corpus.pages)
        bm25_results = self.bm25.search(query, top_k=depth, excerpt_characters=excerpt_characters)
        semantic_results = self.semantic.search(
            query, top_k=depth, excerpt_characters=excerpt_characters
        )
        bm25_by_page = {result.page.physical_page_number: result for result in bm25_results}
        semantic_by_page = {result.page.physical_page_number: result for result in semantic_results}

        fused: list[
            tuple[
                float,
                EvidencePage,
                SearchResult | None,
                SearchResult | None,
            ]
        ] = []
        for page in self.corpus.pages:
            page_number = page.physical_page_number
            bm25_result = bm25_by_page.get(page_number)
            semantic_result = semantic_by_page.get(page_number)
            score = 0.0
            if bm25_result is not None:
                score += self.bm25_weight / (self.rrf_k + bm25_result.rank)
            if semantic_result is not None:
                score += self.semantic_weight / (self.rrf_k + semantic_result.rank)
            if score > 0:
                fused.append((score, page, bm25_result, semantic_result))
        fused.sort(key=lambda item: (-item[0], item[1].physical_page_number))

        results: list[HybridSearchResult] = []
        for rank, (score, page, bm25_result, semantic_result) in enumerate(fused[:top_k], start=1):
            excerpt_result = bm25_result if bm25_result is not None else semantic_result
            if excerpt_result is None:  # Defensive: a positive fusion score requires a component.
                raise RetrievalError("Fused page has no component ranking")
            results.append(
                HybridSearchResult(
                    rank=rank,
                    score=score,
                    page=page,
                    excerpt=excerpt_result.excerpt,
                    evidence_jsonl_sha256=self.corpus.evidence_jsonl_sha256,
                    manifest_sha256=self.corpus.manifest_sha256,
                    rrf_k=self.rrf_k,
                    bm25=self._component(bm25_result, self.bm25_weight),
                    semantic=self._component(semantic_result, self.semantic_weight),
                    excerpt_source=(
                        "bm25_matched_passage"
                        if bm25_result is not None
                        else "page_semantic_page_start"
                    ),
                )
            )
        return tuple(results)

    def _component(self, result: SearchResult | None, weight: float) -> FusionComponent:
        if result is None:
            return FusionComponent(rank=None, raw_score=None, weight=weight, contribution=0.0)
        return FusionComponent(
            rank=result.rank,
            raw_score=result.score,
            weight=weight,
            contribution=weight / (self.rrf_k + result.rank),
        )
