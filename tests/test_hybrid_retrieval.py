"""Offline tests for reciprocal-rank fusion retrieval and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

import ledgerlens.cli as cli
from ledgerlens.evaluation import evaluate_hybrid, load_evaluation_set
from ledgerlens.retrieval import (
    BM25Retriever,
    EmbeddingModelInfo,
    EvidenceCorpus,
    EvidencePage,
    HybridRetriever,
    RetrievalError,
    SemanticRetriever,
    build_embedding_index,
    load_embedding_index,
)

SOURCE_HASH = "7" * 64
DOCUMENT_ID = f"sha256-{SOURCE_HASH}"


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _page(
    number: int,
    text: str,
    *,
    method: str = "canonical_extraction",
    limitations: tuple[str, ...] = (),
) -> EvidencePage:
    return EvidencePage(
        document_id=DOCUMENT_ID,
        company="Fictional Accelerators Ltd.",
        report_year=2033,
        physical_page_number=number,
        text=text,
        text_sha256=_hash(text),
        method=method,
        provenance={"source_sha256": SOURCE_HASH, "fixture": True},
        screening_findings={"quality": {"status": "pass", "warnings": []}},
        review_provenance={"authorized": True} if method == "reviewed_ocr_recovery" else None,
        known_limitations=limitations,
    )


@pytest.fixture
def corpus() -> EvidenceCorpus:
    return EvidenceCorpus(
        document_id=DOCUMENT_ID,
        source_sha256=SOURCE_HASH,
        pages=(
            _page(1, "target target revenue details"),
            _page(
                2,
                "target conceptually related segment discussion",
                method="reviewed_ocr_recovery",
                limitations=("Flattened table headers require source review.",),
            ),
            _page(3, "unrelated governance discussion"),
        ),
        excluded_pages=(4,),
        manifest={"schema_version": "1.0"},
        evidence_jsonl_sha256="8" * 64,
        manifest_sha256="9" * 64,
    )


class FakeBackend:
    """Deterministic semantic model boundary with no external access."""

    def __init__(self) -> None:
        self._info = EmbeddingModelInfo(
            model_name="fictional/hybrid-model",
            model_repository="fictional/hybrid-model-onnx",
            model_revision="a" * 40,
            backend="fake",
            backend_version="1.0",
            dimension=2,
            max_input_tokens=512,
            document_encoding="fake.passage",
            query_encoding="fake.query",
        )
        self.vectors = {
            "target target revenue details": (0.8, 0.6),
            "target conceptually related segment discussion": (1.0, 0.0),
            "unrelated governance discussion": (0.0, 1.0),
        }

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_passages(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[tuple[float, ...], ...]:
        assert batch_size > 0
        return tuple(self.vectors[text] for text in texts)

    def embed_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0) for _ in texts)

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


@pytest.fixture
def hybrid_inputs(
    tmp_path: Path, corpus: EvidenceCorpus
) -> tuple[FakeBackend, Path, HybridRetriever]:
    backend = FakeBackend()
    built = build_embedding_index(corpus, backend, output_root=tmp_path / "embeddings")
    index = load_embedding_index(built.output_directory, corpus)
    return backend, built.output_directory, HybridRetriever(corpus, index, backend)


def test_equal_weight_rrf_uses_component_ranks_and_physical_page_ties(
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever],
) -> None:
    _, _, retriever = hybrid_inputs
    results = retriever.search("target", top_k=3)

    assert [result.page.physical_page_number for result in results] == [1, 2, 3]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert results[1].score == pytest.approx(results[0].score)
    assert results[0].bm25.rank == 1
    assert results[0].semantic.rank == 2
    assert results[1].bm25.rank == 2
    assert results[1].semantic.rank == 1


def test_weights_change_fusion_without_changing_component_rankings(
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever], corpus: EvidenceCorpus
) -> None:
    backend, directory, _ = hybrid_inputs
    index = load_embedding_index(directory, corpus)
    bm25_heavy = HybridRetriever(corpus, index, backend, bm25_weight=2, semantic_weight=1).search(
        "target", top_k=3
    )
    semantic_heavy = HybridRetriever(
        corpus, index, backend, bm25_weight=1, semantic_weight=2
    ).search("target", top_k=3)

    assert bm25_heavy[0].page.physical_page_number == 1
    assert semantic_heavy[0].page.physical_page_number == 2
    assert bm25_heavy[0].bm25.rank == BM25Retriever(corpus).search("target", top_k=3)[0].rank
    semantic = SemanticRetriever(corpus, index, backend).search("target", top_k=3)
    page_two = next(item for item in semantic_heavy if item.page.physical_page_number == 2)
    assert page_two.semantic.rank == semantic[0].rank


def test_semantic_only_match_and_selected_evidence_provenance_survive(
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever],
) -> None:
    _, _, retriever = hybrid_inputs
    results = retriever.search("paraphrase absent from every page", top_k=3)

    assert [item.page.physical_page_number for item in results] == [2, 1, 3]
    assert all(item.bm25.rank is None for item in results)
    assert results[0].excerpt in results[0].page.text
    assert results[0].excerpt_source == "page_semantic_page_start"
    assert results[0].page.method == "reviewed_ocr_recovery"
    assert results[0].page.review_provenance == {"authorized": True}
    assert results[0].page.known_limitations == ("Flattened table headers require source review.",)
    assert results[0].to_dict()["fusion"]["components"]["bm25"]["rank"] is None


def test_bm25_excerpt_is_an_exact_selected_text_substring(
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever],
) -> None:
    _, _, retriever = hybrid_inputs
    result = retriever.search("target", top_k=1)[0]

    assert result.excerpt_source == "bm25_matched_passage"
    assert result.excerpt in result.page.text
    assert "target" in result.excerpt


def test_empty_query_and_invalid_fusion_settings_are_rejected(
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever], corpus: EvidenceCorpus
) -> None:
    backend, directory, retriever = hybrid_inputs
    index = load_embedding_index(directory, corpus)
    assert retriever.search("   ") == ()
    with pytest.raises(RetrievalError, match="top-k"):
        retriever.search("target", top_k=0)
    with pytest.raises(RetrievalError, match="rank constant"):
        HybridRetriever(corpus, index, backend, rrf_k=0)
    with pytest.raises(RetrievalError, match="BM25 weight"):
        HybridRetriever(corpus, index, backend, bm25_weight=0)
    with pytest.raises(RetrievalError, match="Semantic weight"):
        HybridRetriever(corpus, index, backend, semantic_weight=float("nan"))


def test_hybrid_rejects_an_index_bound_to_different_evidence(
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever], corpus: EvidenceCorpus
) -> None:
    backend, directory, _ = hybrid_inputs
    index = load_embedding_index(directory, corpus)
    changed = replace(corpus, evidence_jsonl_sha256="0" * 64)

    with pytest.raises(RetrievalError, match="evidence_jsonl_sha256"):
        HybridRetriever(changed, index, backend)


def test_hybrid_evaluation_records_fusion_and_cache_provenance(
    tmp_path: Path,
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever],
    corpus: EvidenceCorpus,
) -> None:
    backend, directory, _ = hybrid_inputs
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "fictional-hybrid-evaluation",
                "document_id": DOCUMENT_ID,
                "source_sha256": SOURCE_HASH,
                "questions": [
                    {
                        "question_id": "dev-segment",
                        "question": "target",
                        "split": "development",
                        "status": "verified",
                        "leakage_group": "segment",
                        "relevant_pages": [2],
                        "verification": {
                            "method": "manual_source_review",
                            "reviewed_by": "Fictional reviewer",
                            "reviewed_on": "2033-01-01",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    report = evaluate_hybrid(
        corpus,
        load_embedding_index(directory, corpus),
        backend,
        load_evaluation_set(dataset_path),
        split="development",
        cutoffs=(1, 3),
        semantic_weight=2,
    ).payload

    assert report["aggregate_metrics"]["hit_at_k"] == {"1": 1.0, "3": 1.0}
    assert report["retriever"]["name"] == "hybrid_rrf"
    assert report["retriever"]["fusion"]["semantic_weight"] == 2.0
    assert report["retriever"]["components"]["page_semantic"]["embedding_index"][
        "vectors_jsonl_sha256"
    ]
    assert (
        report["questions"][0]["retrieved_pages"][0]["fusion"]["components"]["page_semantic"][
            "rank"
        ]
        == 1
    )


def test_cli_runs_hybrid_retrieval_without_a_real_model(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hybrid_inputs: tuple[FakeBackend, Path, HybridRetriever],
    corpus: EvidenceCorpus,
) -> None:
    backend, directory, _ = hybrid_inputs
    monkeypatch.setattr(cli, "load_evidence_export", lambda _: corpus)
    monkeypatch.setattr(cli, "create_fastembed_backend", lambda *args, **kwargs: backend)

    assert (
        cli.main(
            [
                "hybrid-retrieve",
                "fictional-evidence",
                str(directory),
                "--query",
                "target",
                "--top-k",
                "2",
                "--semantic-weight",
                "2",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Physical PDF page 2" in output
    assert "BM25: rank 2" in output
    assert "Page semantic: rank 1" in output
    assert "Flattened table headers require source review." in output
