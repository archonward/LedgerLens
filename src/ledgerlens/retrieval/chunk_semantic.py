"""Token-aware chunk embeddings aggregated into physical-page rankings."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ledgerlens.retrieval.bm25 import (
    EXCERPT_CHARACTERS_DEFAULT,
    EvidenceCorpus,
    EvidencePage,
    RetrievalError,
    SearchResult,
)
from ledgerlens.retrieval.semantic import (
    DEFAULT_BATCH_SIZE,
    EmbeddingBackend,
    EmbeddingModelInfo,
    _load_model_info,
    _normalize,
    _publish,
    _sha256,
)

CHUNK_INDEX_SCHEMA_VERSION = "1.0"
DEFAULT_CHUNK_TOKENS = 384
DEFAULT_OVERLAP_TOKENS = 64
DEFAULT_BOUNDARY_WINDOW_TOKENS = 64
SUPPORTED_AGGREGATIONS = {"max", "mean_top_n"}


@dataclass(frozen=True)
class ChunkingConfig:
    """Deterministic token-window and boundary-selection settings."""

    max_tokens: int = DEFAULT_CHUNK_TOKENS
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS
    boundary_window_tokens: int = DEFAULT_BOUNDARY_WINDOW_TOKENS
    boundary_preference: str = "paragraph_then_line_then_token"

    def validate(self, model_max_tokens: int) -> None:
        if not isinstance(self.max_tokens, int) or isinstance(self.max_tokens, bool):
            raise RetrievalError("Chunk max tokens must be an integer")
        if self.max_tokens < 1 or self.max_tokens > model_max_tokens - 2:
            raise RetrievalError(f"Chunk max tokens must be between 1 and {model_max_tokens - 2}")
        if (
            not isinstance(self.overlap_tokens, int)
            or isinstance(self.overlap_tokens, bool)
            or not 0 <= self.overlap_tokens < self.max_tokens
        ):
            raise RetrievalError("Chunk overlap must be non-negative and smaller than max tokens")
        if (
            not isinstance(self.boundary_window_tokens, int)
            or isinstance(self.boundary_window_tokens, bool)
            or not 0 <= self.boundary_window_tokens < self.max_tokens
        ):
            raise RetrievalError(
                "Chunk boundary window must be non-negative and smaller than max tokens"
            )
        if self.boundary_preference != "paragraph_then_line_then_token":
            raise RetrievalError("Unsupported chunk boundary preference")


@dataclass(frozen=True)
class EvidenceChunk:
    """An exact selected-text slice from one physical page."""

    chunk_id: str
    physical_page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    token_count: int
    text: str
    text_sha256: str


@dataclass(frozen=True)
class ChunkEmbeddingIndex:
    """Validated chunk vectors and reproducibility metadata."""

    directory: Path
    model: EmbeddingModelInfo
    chunking: ChunkingConfig
    chunks: tuple[EvidenceChunk, ...]
    vectors: tuple[tuple[float, ...], ...]
    manifest: dict[str, Any]
    manifest_sha256: str
    chunks_jsonl_sha256: str


@dataclass(frozen=True)
class ChunkEmbeddingIndexResult:
    """Summary of a published chunk-embedding cache."""

    output_directory: Path
    document_id: str
    page_count: int
    chunk_count: int
    dimension: int
    model: EmbeddingModelInfo
    chunking: ChunkingConfig


@dataclass(frozen=True)
class ChunkSearchResult:
    """A page ranking with the highest-scoring source chunk attached."""

    rank: int
    score: float
    page: EvidencePage
    excerpt: str
    evidence_jsonl_sha256: str
    manifest_sha256: str
    best_chunk: EvidenceChunk
    best_chunk_score: float
    aggregation: str
    aggregation_top_n: int
    considered_chunk_count: int

    @property
    def retrieval_details(self) -> dict[str, Any]:
        """Return chunk provenance for evaluation reports."""
        excerpt_end = self.best_chunk.char_start + len(self.excerpt)
        return {
            "aggregation": self.aggregation,
            "aggregation_top_n": self.aggregation_top_n,
            "considered_chunk_count": self.considered_chunk_count,
            "best_chunk": {
                "chunk_id": self.best_chunk.chunk_id,
                "chunk_index": self.best_chunk.chunk_index,
                "score": self.best_chunk_score,
                "char_start": self.best_chunk.char_start,
                "char_end": self.best_chunk.char_end,
                "token_start": self.best_chunk.token_start,
                "token_end": self.best_chunk.token_end,
                "token_count": self.best_chunk.token_count,
                "text_sha256": self.best_chunk.text_sha256,
                "excerpt_char_start": self.best_chunk.char_start,
                "excerpt_char_end": excerpt_end,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return page evidence plus exact chunk provenance."""
        payload = SearchResult(
            rank=self.rank,
            score=self.score,
            page=self.page,
            excerpt=self.excerpt,
            evidence_jsonl_sha256=self.evidence_jsonl_sha256,
            manifest_sha256=self.manifest_sha256,
        ).to_dict()
        payload["chunk_retrieval"] = self.retrieval_details
        return payload


def _chunk_key(model: EmbeddingModelInfo, config: ChunkingConfig) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", model.model_name.casefold()).strip("-")
    return (
        f"{name}-{model.model_revision[:12]}-t{config.max_tokens}"
        f"-o{config.overlap_tokens}-b{config.boundary_window_tokens}"
    )


def default_chunk_index_directory(
    output_root: Path,
    corpus: EvidenceCorpus,
    model: EmbeddingModelInfo,
    config: ChunkingConfig,
) -> Path:
    """Return a stable path derived from evidence, model, and chunking settings."""
    return Path(output_root) / corpus.document_id / _chunk_key(model, config)


def _boundary_end(
    text: str,
    offsets: tuple[tuple[int, int], ...],
    start_token: int,
    proposed_end_token: int,
    boundary_window_tokens: int,
) -> tuple[int, int]:
    if proposed_end_token == len(offsets) or boundary_window_tokens == 0:
        end_char = len(text) if proposed_end_token == len(offsets) else offsets[-1][1]
        return proposed_end_token, end_char
    earliest_token = max(start_token + 1, proposed_end_token - boundary_window_tokens)
    search_start = offsets[earliest_token - 1][1]
    search_end = offsets[proposed_end_token - 1][1]
    region = text[search_start:search_end]
    paragraph = region.rfind("\n\n")
    line = region.rfind("\n")
    relative_boundary = paragraph + 2 if paragraph >= 0 else line + 1 if line >= 0 else -1
    if relative_boundary < 0:
        return proposed_end_token, search_end
    boundary_char = search_start + relative_boundary
    end_token = max(
        (index + 1 for index, (_, end) in enumerate(offsets) if end <= boundary_char),
        default=proposed_end_token,
    )
    if end_token <= start_token:
        return proposed_end_token, search_end
    return end_token, boundary_char


def chunk_page_text(
    page: EvidencePage,
    token_offsets: tuple[tuple[int, int], ...],
    config: ChunkingConfig,
) -> tuple[EvidenceChunk, ...]:
    """Create overlapping exact-text chunks without crossing a physical page."""
    if any(start < 0 or end <= start or end > len(page.text) for start, end in token_offsets):
        raise RetrievalError(
            f"Tokenizer returned invalid offsets for physical page {page.physical_page_number}"
        )
    if token_offsets != tuple(sorted(token_offsets)):
        raise RetrievalError(
            f"Tokenizer offsets are not ordered for physical page {page.physical_page_number}"
        )
    if not token_offsets:
        text_hash = _sha256(page.text.encode("utf-8"))
        return (
            EvidenceChunk(
                chunk_id=f"page-{page.physical_page_number:06d}-chunk-0001",
                physical_page_number=page.physical_page_number,
                chunk_index=1,
                char_start=0,
                char_end=len(page.text),
                token_start=0,
                token_end=0,
                token_count=0,
                text=page.text,
                text_sha256=text_hash,
            ),
        )

    chunks: list[EvidenceChunk] = []
    start_token = 0
    while start_token < len(token_offsets):
        proposed_end = min(start_token + config.max_tokens, len(token_offsets))
        end_token, end_char = _boundary_end(
            page.text,
            token_offsets,
            start_token,
            proposed_end,
            config.boundary_window_tokens,
        )
        start_char = 0 if start_token == 0 else token_offsets[start_token][0]
        if end_token == len(token_offsets):
            end_char = len(page.text)
        chunk_text = page.text[start_char:end_char]
        chunk_index = len(chunks) + 1
        chunks.append(
            EvidenceChunk(
                chunk_id=f"page-{page.physical_page_number:06d}-chunk-{chunk_index:04d}",
                physical_page_number=page.physical_page_number,
                chunk_index=chunk_index,
                char_start=start_char,
                char_end=end_char,
                token_start=start_token,
                token_end=end_token,
                token_count=end_token - start_token,
                text=chunk_text,
                text_sha256=_sha256(chunk_text.encode("utf-8")),
            )
        )
        if end_token == len(token_offsets):
            break
        start_token = max(start_token + 1, end_token - config.overlap_tokens)
    return tuple(chunks)


def build_chunk_embedding_index(
    corpus: EvidenceCorpus,
    backend: EmbeddingBackend,
    *,
    output_root: Path = Path("artifacts/chunk-embeddings"),
    config: ChunkingConfig = ChunkingConfig(),
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
) -> ChunkEmbeddingIndexResult:
    """Tokenize, chunk, embed, and atomically publish selected page evidence."""
    config.validate(backend.info.max_input_tokens)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise RetrievalError("Embedding batch size must be a positive integer")
    output_directory = default_chunk_index_directory(
        Path(output_root), corpus, backend.info, config
    )
    if output_directory.exists() and not overwrite:
        raise RetrievalError(
            f"Chunk embedding index already exists: {output_directory}. "
            "Use --overwrite to replace it."
        )

    chunks = tuple(
        chunk
        for page in corpus.pages
        for chunk in chunk_page_text(page, backend.token_offsets(page.text), config)
    )
    raw_vectors = backend.embed_passages(
        tuple(chunk.text for chunk in chunks), batch_size=batch_size
    )
    if len(raw_vectors) != len(chunks):
        raise RetrievalError(
            "Embedding backend returned a different number of vectors than evidence chunks"
        )
    vectors = tuple(
        _normalize(vector, backend.info.dimension, chunk.chunk_id)
        for chunk, vector in zip(chunks, raw_vectors, strict=True)
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        chunks_path = temp_directory / "chunks.jsonl"
        with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
            for chunk, vector in zip(chunks, vectors, strict=True):
                record = {**asdict(chunk), "vector": vector}
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        chunks_hash = _sha256(chunks_path.read_bytes())
        manifest = {
            "schema_version": CHUNK_INDEX_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "document_id": corpus.document_id,
            "source_sha256": corpus.source_sha256,
            "evidence_jsonl_sha256": corpus.evidence_jsonl_sha256,
            "evidence_manifest_sha256": corpus.manifest_sha256,
            "page_count": len(corpus.pages),
            "physical_pages": [page.physical_page_number for page in corpus.pages],
            "chunk_count": len(chunks),
            "model": asdict(backend.info),
            "chunking": asdict(config),
            "batch_size": batch_size,
            "retrieval_unit": "selected_physical_page_chunk",
            "page_aggregation": "configured_at_query_time",
            "chunks_file": "chunks.jsonl",
            "chunks_jsonl_sha256": chunks_hash,
            "limitations": [
                "Chunks preserve exact source-text slices but do not reconstruct table structure.",
                "Overlapping chunks can repeat headings, rows, and narrative text.",
                "Page aggregation is a ranking signal, not evidence verification or financial "
                "accuracy.",
            ],
        }
        (temp_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _publish(temp_directory, output_directory, overwrite)
    except Exception:
        shutil.rmtree(temp_directory, ignore_errors=True)
        raise
    return ChunkEmbeddingIndexResult(
        output_directory=output_directory,
        document_id=corpus.document_id,
        page_count=len(corpus.pages),
        chunk_count=len(chunks),
        dimension=backend.info.dimension,
        model=backend.info,
        chunking=config,
    )


def _load_chunking(value: Any, model: EmbeddingModelInfo) -> ChunkingConfig:
    if not isinstance(value, dict):
        raise RetrievalError("Chunk embedding manifest: 'chunking' must be an object")
    try:
        config = ChunkingConfig(**value)
    except TypeError as exc:
        raise RetrievalError(f"Invalid chunking metadata: {exc}") from exc
    config.validate(model.max_input_tokens)
    return config


def load_chunk_embedding_index(directory: Path, corpus: EvidenceCorpus) -> ChunkEmbeddingIndex:
    """Validate a chunk cache against selected evidence and exact source slices."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    chunks_path = root / "chunks.jsonl"
    if not root.is_dir():
        raise RetrievalError(f"Chunk embedding index directory does not exist: {root}")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except OSError as exc:
        raise RetrievalError(
            f"Could not read chunk embedding manifest at {manifest_path}: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Malformed chunk embedding manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RetrievalError("Chunk embedding manifest must be a JSON object")
    if manifest.get("schema_version") != CHUNK_INDEX_SCHEMA_VERSION:
        raise RetrievalError("Chunk embedding manifest uses an unsupported schema_version")
    checks = {
        "document_id": corpus.document_id,
        "source_sha256": corpus.source_sha256,
        "evidence_jsonl_sha256": corpus.evidence_jsonl_sha256,
        "evidence_manifest_sha256": corpus.manifest_sha256,
        "page_count": len(corpus.pages),
        "physical_pages": [page.physical_page_number for page in corpus.pages],
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RetrievalError(f"Chunk embedding manifest {key} does not match evidence")
    if manifest.get("chunks_file") != "chunks.jsonl":
        raise RetrievalError("Chunk embedding manifest references an unsupported chunks file")
    model = _load_model_info(manifest.get("model"))
    chunking = _load_chunking(manifest.get("chunking"), model)
    expected_hash = manifest.get("chunks_jsonl_sha256")
    if not isinstance(expected_hash, str) or not expected_hash:
        raise RetrievalError("Chunk embedding manifest has no chunks hash")
    try:
        chunks_raw = chunks_path.read_bytes()
        lines = chunks_raw.decode("utf-8").splitlines()
    except OSError as exc:
        raise RetrievalError(f"Could not read chunk vectors at {chunks_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise RetrievalError("Chunk vectors are not valid UTF-8") from exc
    if _sha256(chunks_raw) != expected_hash:
        raise RetrievalError("Chunk vectors hash does not match the manifest")
    if manifest.get("chunk_count") != len(lines):
        raise RetrievalError("Chunk count does not match the manifest")

    page_by_number = {page.physical_page_number: page for page in corpus.pages}
    chunks: list[EvidenceChunk] = []
    vectors: list[tuple[float, ...]] = []
    previous_key = (0, 0)
    per_page_indices: dict[int, int] = {}
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalError(f"Malformed chunks.jsonl line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise RetrievalError(f"chunks.jsonl line {line_number} must be an object")
        try:
            chunk = EvidenceChunk(
                **{key: record[key] for key in EvidenceChunk.__dataclass_fields__}
            )
        except (KeyError, TypeError) as exc:
            raise RetrievalError(f"Invalid chunk metadata on line {line_number}: {exc}") from exc
        page = page_by_number.get(chunk.physical_page_number)
        if page is None:
            raise RetrievalError(f"Chunk line {line_number} references an unavailable page")
        key = (chunk.physical_page_number, chunk.chunk_index)
        expected_index = per_page_indices.get(chunk.physical_page_number, 0) + 1
        if key <= previous_key or chunk.chunk_index != expected_index:
            raise RetrievalError("Chunks must be unique and ordered within physical pages")
        per_page_indices[chunk.physical_page_number] = chunk.chunk_index
        previous_key = key
        if not 0 <= chunk.char_start <= chunk.char_end <= len(page.text):
            raise RetrievalError(f"Invalid character offsets for chunk {chunk.chunk_id}")
        if page.text[chunk.char_start : chunk.char_end] != chunk.text:
            raise RetrievalError(f"Chunk {chunk.chunk_id} is not an exact selected-text slice")
        if _sha256(chunk.text.encode("utf-8")) != chunk.text_sha256:
            raise RetrievalError(f"Text hash mismatch for chunk {chunk.chunk_id}")
        if (
            chunk.token_start < 0
            or chunk.token_end < chunk.token_start
            or chunk.token_count != chunk.token_end - chunk.token_start
            or chunk.token_count > chunking.max_tokens
        ):
            raise RetrievalError(f"Invalid token offsets for chunk {chunk.chunk_id}")
        raw_vector = record.get("vector")
        if not isinstance(raw_vector, list) or any(
            not isinstance(value, int | float) or isinstance(value, bool) for value in raw_vector
        ):
            raise RetrievalError(f"Invalid vector for chunk {chunk.chunk_id}")
        chunks.append(chunk)
        vectors.append(
            _normalize(tuple(float(value) for value in raw_vector), model.dimension, chunk.chunk_id)
        )
    if set(per_page_indices) != set(page_by_number):
        raise RetrievalError("Chunk cache does not represent every selected physical page")
    return ChunkEmbeddingIndex(
        directory=root,
        model=model,
        chunking=chunking,
        chunks=tuple(chunks),
        vectors=tuple(vectors),
        manifest=manifest,
        manifest_sha256=_sha256(manifest_raw),
        chunks_jsonl_sha256=_sha256(chunks_raw),
    )


class ChunkSemanticRetriever:
    """Rank chunks with cosine similarity and aggregate them to physical pages."""

    def __init__(
        self,
        corpus: EvidenceCorpus,
        index: ChunkEmbeddingIndex,
        backend: EmbeddingBackend,
        *,
        aggregation: str = "max",
        aggregation_top_n: int = 2,
    ) -> None:
        if backend.info != index.model:
            raise RetrievalError("Embedding backend configuration does not match the chunk index")
        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise RetrievalError(f"Unsupported chunk aggregation: {aggregation}")
        if (
            not isinstance(aggregation_top_n, int)
            or isinstance(aggregation_top_n, bool)
            or aggregation_top_n < 1
        ):
            raise RetrievalError("Chunk aggregation top-N must be a positive integer")
        self.corpus = corpus
        self.index = index
        self.backend = backend
        self.aggregation = aggregation
        self.aggregation_top_n = aggregation_top_n

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        excerpt_characters: int = EXCERPT_CHARACTERS_DEFAULT,
    ) -> tuple[ChunkSearchResult, ...]:
        """Rank pages using their maximum or top-N mean chunk similarity."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise RetrievalError("top-k must be a positive integer")
        if excerpt_characters < 1:
            raise RetrievalError("excerpt characters must be a positive integer")
        if not query.strip() or not self.corpus.pages:
            return ()
        embedded = self.backend.embed_queries((query,))
        if len(embedded) != 1:
            raise RetrievalError("Embedding backend did not return exactly one query vector")
        query_vector = _normalize(embedded[0], self.index.model.dimension, "query")
        by_page: dict[int, list[tuple[float, EvidenceChunk]]] = {}
        for chunk, vector in zip(self.index.chunks, self.index.vectors, strict=True):
            score = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            by_page.setdefault(chunk.physical_page_number, []).append((score, chunk))

        pages = {page.physical_page_number: page for page in self.corpus.pages}
        ranked: list[tuple[float, EvidencePage, float, EvidenceChunk, int]] = []
        for page_number, chunk_scores in by_page.items():
            chunk_scores.sort(key=lambda item: (-item[0], item[1].chunk_index))
            best_score, best_chunk = chunk_scores[0]
            if self.aggregation == "max":
                page_score = best_score
                used = 1
            else:
                used = min(self.aggregation_top_n, len(chunk_scores))
                page_score = sum(score for score, _ in chunk_scores[:used]) / used
            ranked.append((page_score, pages[page_number], best_score, best_chunk, used))
        ranked.sort(key=lambda item: (-item[0], item[1].physical_page_number))

        return tuple(
            ChunkSearchResult(
                rank=rank,
                score=page_score,
                page=page,
                excerpt=best_chunk.text[:excerpt_characters],
                evidence_jsonl_sha256=self.corpus.evidence_jsonl_sha256,
                manifest_sha256=self.corpus.manifest_sha256,
                best_chunk=best_chunk,
                best_chunk_score=best_score,
                aggregation=self.aggregation,
                aggregation_top_n=used,
                considered_chunk_count=len(by_page[page.physical_page_number]),
            )
            for rank, (page_score, page, best_score, best_chunk, used) in enumerate(
                ranked[:top_k], start=1
            )
        )
