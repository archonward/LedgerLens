"""Offline tests for chunk-aware semantic retrieval and caching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

import ledgerlens.cli as cli
from ledgerlens.evaluation import evaluate_chunk_semantic, load_evaluation_set
from ledgerlens.retrieval import (
    ChunkingConfig,
    ChunkSemanticRetriever,
    EmbeddingModelInfo,
    EvidenceCorpus,
    EvidencePage,
    RetrievalError,
    build_chunk_embedding_index,
    chunk_page_text,
    load_chunk_embedding_index,
)

SOURCE_HASH = "d" * 64
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
        company="Fictional Circuits Ltd.",
        report_year=2032,
        physical_page_number=number,
        text=text,
        text_sha256=_hash(text),
        method=method,
        provenance={"source_sha256": SOURCE_HASH, "fixture": True},
        screening_findings={"quality": {"status": "pass", "warnings": []}},
        review_provenance={"authorized_by": "fixture user"}
        if method == "reviewed_ocr_recovery"
        else None,
        known_limitations=limitations,
    )


@pytest.fixture
def corpus() -> EvidenceCorpus:
    pages = (
        _page(1, "overview neutral words\n\nlate target catalyst grew strongly"),
        _page(2, "target appears once then ordinary context remains"),
        _page(
            3,
            "reviewed target segment table",
            method="reviewed_ocr_recovery",
            limitations=("Flattened table headers require source review.",),
        ),
    )
    return EvidenceCorpus(
        document_id=DOCUMENT_ID,
        source_sha256=SOURCE_HASH,
        pages=pages,
        excluded_pages=(4,),
        manifest={"schema_version": "1.0"},
        evidence_jsonl_sha256="e" * 64,
        manifest_sha256="f" * 64,
    )


class FakeChunkBackend:
    """Deterministic model double with exact whitespace offsets."""

    def __init__(self) -> None:
        self._info = EmbeddingModelInfo(
            model_name="fictional/chunk-model",
            model_repository="fictional/chunk-model-onnx",
            model_revision="2" * 40,
            backend="fake",
            backend_version="1.0",
            dimension=3,
            max_input_tokens=12,
            document_encoding="fake.passage",
            query_encoding="fake.query",
        )
        self.passage_calls = 0

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        return tuple((match.start(), match.end()) for match in re.finditer(r"\S+", text))

    def embed_passages(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[tuple[float, ...], ...]:
        assert batch_size > 0
        self.passage_calls += 1
        return tuple(self._vector(text) for text in texts)

    def embed_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, 0.0, 0.0) if text == "target" else (0.0, 1.0, 0.0) for text in texts)

    @staticmethod
    def _vector(text: str) -> tuple[float, float, float]:
        if "late target" in text:
            return (1.0, 0.0, 0.0)
        if "target" in text:
            return (0.8, 0.6, 0.0)
        return (0.0, 1.0, 0.0)


@pytest.fixture
def chunk_inputs(
    tmp_path: Path, corpus: EvidenceCorpus
) -> tuple[FakeChunkBackend, Path, ChunkSemanticRetriever]:
    backend = FakeChunkBackend()
    config = ChunkingConfig(max_tokens=5, overlap_tokens=2, boundary_window_tokens=2)
    result = build_chunk_embedding_index(
        corpus, backend, output_root=tmp_path / "chunk-embeddings", config=config
    )
    index = load_chunk_embedding_index(result.output_directory, corpus)
    return backend, result.output_directory, ChunkSemanticRetriever(corpus, index, backend)


def test_chunks_are_exact_page_slices_with_offsets_and_overlap(corpus: EvidenceCorpus) -> None:
    backend = FakeChunkBackend()
    page = corpus.pages[0]
    offsets = backend.token_offsets(page.text)
    chunks = chunk_page_text(
        page,
        offsets,
        ChunkingConfig(max_tokens=5, overlap_tokens=2, boundary_window_tokens=2),
    )

    assert len(chunks) >= 2
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(page.text)
    assert all(page.text[item.char_start : item.char_end] == item.text for item in chunks)
    assert all(item.token_count <= 5 for item in chunks)
    assert chunks[1].token_start <= chunks[0].token_end
    assert chunks[0].text.endswith("\n\n")


def test_cache_is_hash_bound_configured_and_protected(
    chunk_inputs: tuple[FakeChunkBackend, Path, ChunkSemanticRetriever],
    corpus: EvidenceCorpus,
) -> None:
    backend, directory, _ = chunk_inputs
    index = load_chunk_embedding_index(directory, corpus)

    assert index.chunking.max_tokens == 5
    assert index.manifest["evidence_jsonl_sha256"] == corpus.evidence_jsonl_sha256
    assert index.manifest["chunk_count"] == len(index.chunks)
    assert backend.passage_calls == 1
    with pytest.raises(RetrievalError, match="already exists"):
        build_chunk_embedding_index(
            corpus,
            backend,
            output_root=directory.parents[1],
            config=index.chunking,
        )

    stale = replace(corpus, manifest_sha256="0" * 64)
    with pytest.raises(RetrievalError, match="evidence_manifest_sha256"):
        load_chunk_embedding_index(directory, stale)
    chunks_path = directory / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RetrievalError, match="vectors hash"):
        load_chunk_embedding_index(directory, corpus)


def test_page_aggregation_is_deterministic_and_preserves_provenance(
    chunk_inputs: tuple[FakeChunkBackend, Path, ChunkSemanticRetriever],
    corpus: EvidenceCorpus,
) -> None:
    backend, directory, max_retriever = chunk_inputs
    maximum = max_retriever.search("target", top_k=3)

    assert maximum[0].page.physical_page_number == 1
    assert maximum[0].excerpt in maximum[0].page.text
    assert maximum[0].retrieval_details["best_chunk"]["char_start"] >= 0
    reviewed = next(item for item in maximum if item.page.physical_page_number == 3)
    assert reviewed.page.method == "reviewed_ocr_recovery"
    assert reviewed.page.review_provenance == {"authorized_by": "fixture user"}
    assert reviewed.page.known_limitations == ("Flattened table headers require source review.",)

    index = load_chunk_embedding_index(directory, corpus)
    mean = ChunkSemanticRetriever(
        corpus, index, backend, aggregation="mean_top_n", aggregation_top_n=2
    ).search("target", top_k=3)
    assert [item.page.physical_page_number for item in mean] == [1, 3, 2]
    assert mean[0].score < maximum[0].score
    assert mean[1].score > mean[2].score
    assert mean[0].retrieval_details["aggregation"] == "mean_top_n"
    assert mean[0].retrieval_details["aggregation_top_n"] == 2


def test_invalid_settings_and_empty_queries_are_handled(
    chunk_inputs: tuple[FakeChunkBackend, Path, ChunkSemanticRetriever],
    corpus: EvidenceCorpus,
) -> None:
    backend, directory, retriever = chunk_inputs
    assert retriever.search("   ") == ()
    with pytest.raises(RetrievalError, match="positive integer"):
        retriever.search("target", top_k=0)
    with pytest.raises(RetrievalError, match="smaller than max tokens"):
        ChunkingConfig(max_tokens=5, overlap_tokens=5).validate(backend.info.max_input_tokens)
    with pytest.raises(RetrievalError, match="Unsupported chunk aggregation"):
        ChunkSemanticRetriever(
            corpus,
            load_chunk_embedding_index(directory, corpus),
            backend,
            aggregation="sum",
        )


def test_chunk_evaluation_records_configuration_and_exact_chunk_details(
    tmp_path: Path,
    chunk_inputs: tuple[FakeChunkBackend, Path, ChunkSemanticRetriever],
    corpus: EvidenceCorpus,
) -> None:
    backend, directory, _ = chunk_inputs
    dataset_path = tmp_path / "questions.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_id": "fictional-chunk-evaluation",
                "document_id": DOCUMENT_ID,
                "source_sha256": SOURCE_HASH,
                "questions": [
                    {
                        "question_id": "dev-late-target",
                        "question": "target",
                        "split": "development",
                        "status": "verified",
                        "leakage_group": "target",
                        "relevant_pages": [1],
                        "verification": {
                            "method": "manual_source_review",
                            "reviewed_by": "Fictional reviewer",
                            "reviewed_on": "2032-01-01",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = evaluate_chunk_semantic(
        corpus,
        load_chunk_embedding_index(directory, corpus),
        backend,
        load_evaluation_set(dataset_path),
        split="development",
        cutoffs=(1, 3),
    ).payload

    assert report["aggregate_metrics"]["hit_at_k"] == {"1": 1.0, "3": 1.0}
    assert report["retriever"]["name"] == "chunk_aware_page_semantic"
    assert report["retriever"]["chunking"]["max_tokens"] == 5
    best = report["questions"][0]["retrieved_pages"][0]
    assert best["chunk_retrieval"]["best_chunk"]["char_start"] >= 0


def test_cli_builds_and_queries_chunks_without_a_real_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    corpus: EvidenceCorpus,
) -> None:
    backend = FakeChunkBackend()
    monkeypatch.setattr(cli, "load_evidence_export", lambda _: corpus)
    monkeypatch.setattr(cli, "create_fastembed_backend", lambda *args, **kwargs: backend)

    assert (
        cli.main(
            [
                "embed-evidence-chunks",
                "fictional-evidence",
                "--output-root",
                str(tmp_path / "chunk-embeddings"),
                "--max-tokens",
                "5",
                "--overlap-tokens",
                "2",
                "--boundary-window-tokens",
                "2",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Physical pages represented: 3" in output
    index_directory = next((tmp_path / "chunk-embeddings" / DOCUMENT_ID).iterdir())

    assert (
        cli.main(
            [
                "chunk-semantic-retrieve",
                "fictional-evidence",
                str(index_directory),
                "--query",
                "target",
                "--top-k",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Physical PDF page 1" in output
    assert "chars [" in output
    assert "Exact source excerpt:" in output
    assert "page-000001-chunk-" in output


def test_invalid_chunk_cache_record_is_rejected(
    chunk_inputs: tuple[FakeChunkBackend, Path, ChunkSemanticRetriever],
    corpus: EvidenceCorpus,
) -> None:
    _, directory, _ = chunk_inputs
    chunks_path = directory / "chunks.jsonl"
    records = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines()]
    records[0]["char_end"] += 1
    content = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records)
    chunks_path.write_text(content, encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks_jsonl_sha256"] = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RetrievalError, match="exact selected-text slice"):
        load_chunk_embedding_index(directory, corpus)
