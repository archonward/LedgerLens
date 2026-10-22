"""Offline tests for semantic page retrieval, caching, and evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

import ledgerlens.cli as cli
from ledgerlens.evaluation import evaluate_semantic, load_evaluation_set
from ledgerlens.retrieval import (
    EmbeddingModelInfo,
    EvidenceCorpus,
    EvidencePage,
    RetrievalError,
    SemanticRetriever,
    build_embedding_index,
    load_embedding_index,
)

SOURCE_HASH = "a" * 64
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
        company="Fictional Semiconductor Ltd.",
        report_year=2030,
        physical_page_number=number,
        text=text,
        text_sha256=_hash(text),
        method=method,
        provenance={"source_sha256": SOURCE_HASH, "fixture": True},
        screening_findings={"quality": {"status": "pass", "warnings": []}},
        review_provenance={"reviewed": True} if method == "reviewed_ocr_recovery" else None,
        known_limitations=limitations,
    )


@pytest.fixture
def corpus() -> EvidenceCorpus:
    pages = (
        _page(1, "Revenue increased because cloud demand accelerated."),
        _page(2, "Manufacturing capacity and supplier lead times create risks."),
        _page(
            3,
            "The recovered segment discussion describes compute growth.",
            method="reviewed_ocr_recovery",
            limitations=("Flattened table headings require source review.",),
        ),
    )
    return EvidenceCorpus(
        document_id=DOCUMENT_ID,
        source_sha256=SOURCE_HASH,
        pages=pages,
        excluded_pages=(4,),
        manifest={"schema_version": "1.0"},
        evidence_jsonl_sha256="b" * 64,
        manifest_sha256="c" * 64,
    )


class FakeBackend:
    """Deterministic test double; no model or network access."""

    def __init__(self) -> None:
        self._info = EmbeddingModelInfo(
            model_name="fictional/three-dimensional",
            model_repository="fictional/three-dimensional-onnx",
            model_revision="1" * 40,
            backend="fake",
            backend_version="1.0",
            dimension=3,
            max_input_tokens=512,
            document_encoding="fake.passage",
            query_encoding="fake.query",
        )
        self.passages = {
            "Revenue increased because cloud demand accelerated.": (1.0, 0.0, 0.0),
            "Manufacturing capacity and supplier lead times create risks.": (0.0, 1.0, 0.0),
            "The recovered segment discussion describes compute growth.": (0.0, 0.0, 1.0),
        }
        self.queries = {
            "Why did sales improve?": (1.0, 0.0, 0.0),
            "What could delay production?": (0.0, 1.0, 0.0),
            "Which division expanded?": (0.0, 0.0, 1.0),
            "tie": (1.0, 1.0, 0.0),
        }
        self.passage_calls = 0

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_passages(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[tuple[float, ...], ...]:
        assert batch_size > 0
        self.passage_calls += 1
        return tuple(self.passages[text] for text in texts)

    def embed_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self.queries[text] for text in texts)

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return simple whitespace-token offsets for protocol completeness."""
        return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))


@pytest.fixture
def semantic_inputs(
    tmp_path: Path, corpus: EvidenceCorpus
) -> tuple[FakeBackend, Path, SemanticRetriever]:
    backend = FakeBackend()
    result = build_embedding_index(corpus, backend, output_root=tmp_path / "embeddings")
    index = load_embedding_index(result.output_directory, corpus)
    return backend, result.output_directory, SemanticRetriever(corpus, index, backend)


def test_embedding_cache_is_hash_bound_and_reusable(
    semantic_inputs: tuple[FakeBackend, Path, SemanticRetriever], corpus: EvidenceCorpus
) -> None:
    backend, directory, _ = semantic_inputs

    index = load_embedding_index(directory, corpus)

    assert backend.passage_calls == 1
    assert index.pages == (1, 2, 3)
    assert index.model == backend.info
    assert index.manifest["evidence_jsonl_sha256"] == corpus.evidence_jsonl_sha256
    assert index.manifest["evidence_manifest_sha256"] == corpus.manifest_sha256
    assert "512-token input limit" in index.manifest["limitations"][0]


def test_semantic_ranking_is_deterministic_and_preserves_provenance(
    semantic_inputs: tuple[FakeBackend, Path, SemanticRetriever],
) -> None:
    _, _, retriever = semantic_inputs

    result = retriever.search("Which division expanded?", top_k=3)

    assert [item.page.physical_page_number for item in result] == [3, 1, 2]
    assert result[0].page.method == "reviewed_ocr_recovery"
    assert result[0].page.review_provenance == {"reviewed": True}
    assert result[0].page.known_limitations == ("Flattened table headings require source review.",)
    assert result[0].excerpt in result[0].page.text
    assert result[0].evidence_jsonl_sha256 == "b" * 64


def test_equal_cosine_scores_use_physical_page_order(
    semantic_inputs: tuple[FakeBackend, Path, SemanticRetriever],
) -> None:
    _, _, retriever = semantic_inputs

    result = retriever.search("tie", top_k=3)

    assert [item.page.physical_page_number for item in result] == [1, 2, 3]


def test_empty_query_and_invalid_top_k_are_handled(
    semantic_inputs: tuple[FakeBackend, Path, SemanticRetriever],
) -> None:
    _, _, retriever = semantic_inputs

    assert retriever.search("   ") == ()
    with pytest.raises(RetrievalError, match="positive integer"):
        retriever.search("tie", top_k=0)


def test_existing_index_is_protected_and_overwrite_is_explicit(
    tmp_path: Path, corpus: EvidenceCorpus
) -> None:
    backend = FakeBackend()
    root = tmp_path / "embeddings"
    first = build_embedding_index(corpus, backend, output_root=root)
    original_manifest = (first.output_directory / "manifest.json").read_bytes()

    with pytest.raises(RetrievalError, match="already exists"):
        build_embedding_index(corpus, backend, output_root=root)
    assert (first.output_directory / "manifest.json").read_bytes() == original_manifest

    replacement = build_embedding_index(corpus, backend, output_root=root, overwrite=True)
    assert replacement.output_directory == first.output_directory


def test_stale_evidence_and_tampered_vectors_are_rejected(
    semantic_inputs: tuple[FakeBackend, Path, SemanticRetriever], corpus: EvidenceCorpus
) -> None:
    _, directory, _ = semantic_inputs
    stale = replace(corpus, evidence_jsonl_sha256="d" * 64)

    with pytest.raises(RetrievalError, match="evidence_jsonl_sha256"):
        load_embedding_index(directory, stale)

    vectors_path = directory / "vectors.jsonl"
    vectors_path.write_text(vectors_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RetrievalError, match="vectors hash"):
        load_embedding_index(directory, corpus)


def test_backend_configuration_must_match_cached_index(
    semantic_inputs: tuple[FakeBackend, Path, SemanticRetriever], corpus: EvidenceCorpus
) -> None:
    backend, directory, _ = semantic_inputs
    index = load_embedding_index(directory, corpus)
    backend._info = replace(backend.info, model_revision="2" * 40)

    with pytest.raises(RetrievalError, match="configuration does not match"):
        SemanticRetriever(corpus, index, backend)


def test_semantic_evaluation_uses_existing_judgments(
    tmp_path: Path,
    corpus: EvidenceCorpus,
    semantic_inputs: tuple[FakeBackend, Path, SemanticRetriever],
) -> None:
    backend, directory, _ = semantic_inputs
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "fictional-semantic-evaluation",
                "document_id": DOCUMENT_ID,
                "source_sha256": SOURCE_HASH,
                "questions": [
                    {
                        "question_id": "dev-revenue",
                        "question": "Why did sales improve?",
                        "split": "development",
                        "status": "verified",
                        "leakage_group": "revenue",
                        "relevant_pages": [1],
                        "verification": {
                            "method": "manual_source_review",
                            "reviewed_by": "Fictional reviewer",
                            "reviewed_on": "2030-01-01",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index = load_embedding_index(directory, corpus)

    report = evaluate_semantic(
        corpus,
        index,
        backend,
        load_evaluation_set(dataset_path),
        split="development",
        cutoffs=(1, 3),
    ).payload

    assert report["aggregate_metrics"]["hit_at_k"] == {"1": 1.0, "3": 1.0}
    assert report["retriever"]["name"] == "page_level_semantic"
    assert report["retriever"]["model"]["model_revision"] == "1" * 40
    assert report["retriever"]["embedding_index"]["vectors_jsonl_sha256"]


def test_cli_builds_and_queries_cache_without_real_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    corpus: EvidenceCorpus,
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(cli, "load_evidence_export", lambda _: corpus)
    monkeypatch.setattr(cli, "create_fastembed_backend", lambda *args, **kwargs: backend)

    assert (
        cli.main(
            [
                "embed-evidence",
                "fictional-evidence",
                "--output-root",
                str(tmp_path / "embeddings"),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Physical pages embedded: 3" in output
    index_directory = next((tmp_path / "embeddings" / DOCUMENT_ID).iterdir())

    assert (
        cli.main(
            [
                "semantic-retrieve",
                "fictional-evidence",
                str(index_directory),
                "--query",
                "Which division expanded?",
                "--top-k",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Physical PDF page 3" in output
    assert "reviewed_ocr_recovery" in output
    assert "Flattened table headings require source review." in output
