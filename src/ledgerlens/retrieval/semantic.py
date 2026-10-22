"""Page-level semantic retrieval over provenance-validated selected evidence."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ledgerlens.retrieval.bm25 import (
    EXCERPT_CHARACTERS_DEFAULT,
    EvidenceCorpus,
    RetrievalError,
    SearchResult,
)

EMBEDDING_INDEX_SCHEMA_VERSION = "1.0"
DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MODEL_REPOSITORY = "Qdrant/bge-small-en-v1.5-onnx-Q"
DEFAULT_MODEL_REVISION = "c32e6154d1bb7a0e47c5e745fd895e7700f44385"
DEFAULT_DIMENSION = 384
DEFAULT_MAX_INPUT_TOKENS = 512
DEFAULT_BATCH_SIZE = 32
MODEL_FILES = (
    "config.json",
    "model_optimized.onnx",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


@dataclass(frozen=True)
class EmbeddingModelInfo:
    """Identity and configuration required to reproduce embeddings."""

    model_name: str
    model_repository: str
    model_revision: str
    backend: str
    backend_version: str
    dimension: int
    max_input_tokens: int
    document_encoding: str
    query_encoding: str
    normalization: str = "l2"
    similarity: str = "cosine"


class EmbeddingBackend(Protocol):
    """Minimal model boundary used by indexing, retrieval, and offline tests."""

    @property
    def info(self) -> EmbeddingModelInfo: ...

    def embed_passages(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[tuple[float, ...], ...]: ...

    def embed_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]: ...

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]: ...


@dataclass(frozen=True)
class EmbeddingIndex:
    """Validated page vectors bound to one selected-evidence export."""

    directory: Path
    model: EmbeddingModelInfo
    vectors: tuple[tuple[float, ...], ...]
    pages: tuple[int, ...]
    manifest: dict[str, Any]
    manifest_sha256: str
    vectors_jsonl_sha256: str


@dataclass(frozen=True)
class EmbeddingIndexResult:
    """Summary of a newly published embedding cache."""

    output_directory: Path
    document_id: str
    page_count: int
    dimension: int
    model: EmbeddingModelInfo


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _normalize(vector: tuple[float, ...], dimension: int, context: str) -> tuple[float, ...]:
    if len(vector) != dimension:
        raise RetrievalError(
            f"{context}: expected embedding dimension {dimension}, found {len(vector)}"
        )
    if any(not math.isfinite(value) for value in vector):
        raise RetrievalError(f"{context}: embedding contains a non-finite value")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise RetrievalError(f"{context}: embedding has zero magnitude")
    return tuple(value / norm for value in vector)


def _model_key(info: EmbeddingModelInfo) -> str:
    name = re.sub(r"[^a-z0-9]+", "-", info.model_name.casefold()).strip("-")
    return f"{name}-{info.model_revision[:12]}"


def default_embedding_index_directory(
    output_root: Path, corpus: EvidenceCorpus, model: EmbeddingModelInfo
) -> Path:
    """Return the stable cache path for an evidence export and pinned model."""
    return Path(output_root) / corpus.document_id / _model_key(model)


def _publish(temp_directory: Path, output_directory: Path, overwrite: bool) -> None:
    if not output_directory.exists():
        temp_directory.replace(output_directory)
        return
    if not overwrite:
        raise RetrievalError(
            f"Embedding index already exists: {output_directory}. Use --overwrite to replace it."
        )
    backup = output_directory.with_name(f".{output_directory.name}.backup-{uuid.uuid4().hex}")
    output_directory.replace(backup)
    try:
        temp_directory.replace(output_directory)
    except OSError:
        backup.replace(output_directory)
        raise
    shutil.rmtree(backup, ignore_errors=True)


class FastEmbedBackend:
    """Pinned, CPU-only FastEmbed adapter for the initial semantic baseline."""

    def __init__(self, model: Any, tokenizer: Any, info: EmbeddingModelInfo) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._info = info

    @property
    def info(self) -> EmbeddingModelInfo:
        return self._info

    def embed_passages(
        self, texts: tuple[str, ...], *, batch_size: int
    ) -> tuple[tuple[float, ...], ...]:
        try:
            return tuple(
                tuple(float(value) for value in vector)
                for vector in self._model.passage_embed(texts, batch_size=batch_size)
            )
        except Exception as exc:
            raise RetrievalError(f"Passage embedding failed: {exc}") from exc

    def embed_queries(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        try:
            return tuple(
                tuple(float(value) for value in vector) for vector in self._model.query_embed(texts)
            )
        except Exception as exc:
            raise RetrievalError(f"Query embedding failed: {exc}") from exc

    def token_offsets(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return model-token character spans without special tokens or truncation."""
        try:
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
        except Exception as exc:
            raise RetrievalError(f"Model tokenization failed: {exc}") from exc
        return tuple((start, end) for start, end in encoding.offsets if end > start)


def create_fastembed_backend(
    model_cache: Path,
    *,
    allow_download: bool,
    threads: int | None = None,
) -> FastEmbedBackend:
    """Load the pinned ONNX model, optionally downloading it into an ignored cache."""
    try:
        from fastembed import TextEmbedding
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RetrievalError(
            "Semantic retrieval dependencies are unavailable. "
            "Run 'uv sync --locked --extra semantic'."
        ) from exc

    model_cache = Path(model_cache)
    model_cache.mkdir(parents=True, exist_ok=True)
    try:
        snapshot = snapshot_download(
            repo_id=DEFAULT_MODEL_REPOSITORY,
            revision=DEFAULT_MODEL_REVISION,
            cache_dir=model_cache,
            local_files_only=not allow_download,
            allow_patterns=list(MODEL_FILES),
            token=False,
        )
        model = TextEmbedding(
            model_name=DEFAULT_MODEL_NAME,
            specific_model_path=snapshot,
            providers=["CPUExecutionProvider"],
            threads=threads,
        )
        tokenizer = Tokenizer.from_file(str(Path(snapshot) / "tokenizer.json"))
    except Exception as exc:
        mode = "download or load" if allow_download else "load from the local model cache"
        raise RetrievalError(
            f"Could not {mode} {DEFAULT_MODEL_NAME} at revision {DEFAULT_MODEL_REVISION}: {exc}"
        ) from exc

    return FastEmbedBackend(
        model,
        tokenizer,
        EmbeddingModelInfo(
            model_name=DEFAULT_MODEL_NAME,
            model_repository=DEFAULT_MODEL_REPOSITORY,
            model_revision=DEFAULT_MODEL_REVISION,
            backend="fastembed",
            backend_version=importlib.metadata.version("fastembed"),
            dimension=DEFAULT_DIMENSION,
            max_input_tokens=DEFAULT_MAX_INPUT_TOKENS,
            document_encoding="fastembed.passage_embed",
            query_encoding="fastembed.query_embed",
        ),
    )


def build_embedding_index(
    corpus: EvidenceCorpus,
    backend: EmbeddingBackend,
    *,
    output_root: Path = Path("artifacts/embeddings"),
    batch_size: int = DEFAULT_BATCH_SIZE,
    overwrite: bool = False,
) -> EmbeddingIndexResult:
    """Embed selected physical pages and atomically publish a hash-bound cache."""
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
        raise RetrievalError("Embedding batch size must be a positive integer")
    info = backend.info
    if info.dimension < 1 or info.max_input_tokens < 1:
        raise RetrievalError("Embedding model dimension and input limit must be positive")
    output_directory = default_embedding_index_directory(Path(output_root), corpus, info)
    if output_directory.exists() and not overwrite:
        raise RetrievalError(
            f"Embedding index already exists: {output_directory}. Use --overwrite to replace it."
        )

    texts = tuple(page.text for page in corpus.pages)
    raw_vectors = backend.embed_passages(texts, batch_size=batch_size)
    if len(raw_vectors) != len(corpus.pages):
        raise RetrievalError(
            "Embedding backend returned a different number of vectors than selected pages"
        )
    vectors = tuple(
        _normalize(vector, info.dimension, f"physical page {page.physical_page_number}")
        for page, vector in zip(corpus.pages, raw_vectors, strict=True)
    )

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temp_directory = Path(
        tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent)
    )
    try:
        vectors_path = temp_directory / "vectors.jsonl"
        with vectors_path.open("w", encoding="utf-8", newline="\n") as handle:
            for page, vector in zip(corpus.pages, vectors, strict=True):
                record = {
                    "physical_page_number": page.physical_page_number,
                    "text_sha256": page.text_sha256,
                    "vector": vector,
                }
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        vectors_hash = _sha256(vectors_path.read_bytes())
        manifest = {
            "schema_version": EMBEDDING_INDEX_SCHEMA_VERSION,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "document_id": corpus.document_id,
            "source_sha256": corpus.source_sha256,
            "evidence_jsonl_sha256": corpus.evidence_jsonl_sha256,
            "evidence_manifest_sha256": corpus.manifest_sha256,
            "page_count": len(corpus.pages),
            "physical_pages": [page.physical_page_number for page in corpus.pages],
            "model": asdict(info),
            "batch_size": batch_size,
            "retrieval_unit": "selected_physical_page",
            "vectors_file": "vectors.jsonl",
            "vectors_jsonl_sha256": vectors_hash,
            "limitations": [
                f"Page text beyond the model's {info.max_input_tokens}-token input limit is "
                "truncated by the embedding backend.",
                "Embedding similarity is a ranking signal, not evidence verification or "
                "financial accuracy.",
            ],
        }
        (temp_directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        _publish(temp_directory, output_directory, overwrite)
    except Exception:
        shutil.rmtree(temp_directory, ignore_errors=True)
        raise
    return EmbeddingIndexResult(
        output_directory=output_directory,
        document_id=corpus.document_id,
        page_count=len(corpus.pages),
        dimension=info.dimension,
        model=info,
    )


def _require_string(value: dict[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise RetrievalError(f"{context}: {key!r} must be a non-empty string")
    return item


def _load_model_info(value: Any) -> EmbeddingModelInfo:
    if not isinstance(value, dict):
        raise RetrievalError("Embedding manifest: 'model' must be an object")
    try:
        info = EmbeddingModelInfo(**value)
    except (TypeError, ValueError) as exc:
        raise RetrievalError(f"Embedding manifest has invalid model metadata: {exc}") from exc
    if info.dimension < 1 or info.max_input_tokens < 1:
        raise RetrievalError("Embedding manifest model dimensions must be positive")
    if info.normalization != "l2" or info.similarity != "cosine":
        raise RetrievalError("Embedding index uses an unsupported normalization or similarity")
    return info


def load_embedding_index(directory: Path, corpus: EvidenceCorpus) -> EmbeddingIndex:
    """Load and validate cached vectors against the selected evidence and manifest hashes."""
    root = Path(directory)
    manifest_path = root / "manifest.json"
    vectors_path = root / "vectors.jsonl"
    if not root.is_dir():
        raise RetrievalError(f"Embedding index directory does not exist: {root}")
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except OSError as exc:
        raise RetrievalError(
            f"Could not read embedding manifest at {manifest_path}: {exc}"
        ) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RetrievalError(f"Malformed embedding manifest at {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RetrievalError("Embedding manifest must be a JSON object")
    if manifest.get("schema_version") != EMBEDDING_INDEX_SCHEMA_VERSION:
        raise RetrievalError("Embedding manifest uses an unsupported schema_version")
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
            raise RetrievalError(f"Embedding manifest {key} does not match selected evidence")
    if manifest.get("vectors_file") != "vectors.jsonl":
        raise RetrievalError("Embedding manifest references an unsupported vectors file")
    info = _load_model_info(manifest.get("model"))
    expected_vectors_hash = _require_string(manifest, "vectors_jsonl_sha256", "Embedding manifest")
    try:
        vectors_raw = vectors_path.read_bytes()
        lines = vectors_raw.decode("utf-8").splitlines()
    except OSError as exc:
        raise RetrievalError(f"Could not read embedding vectors at {vectors_path}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise RetrievalError(f"Embedding vectors are not valid UTF-8: {vectors_path}") from exc
    if _sha256(vectors_raw) != expected_vectors_hash:
        raise RetrievalError("Embedding vectors hash does not match the manifest")
    if len(lines) != len(corpus.pages):
        raise RetrievalError("Embedding vector count does not match selected evidence")

    vectors: list[tuple[float, ...]] = []
    pages: list[int] = []
    for line_number, (line, page) in enumerate(zip(lines, corpus.pages, strict=True), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RetrievalError(f"Malformed vectors.jsonl line {line_number}: {exc}") from exc
        if not isinstance(record, dict):
            raise RetrievalError(f"vectors.jsonl line {line_number} must be an object")
        if record.get("physical_page_number") != page.physical_page_number:
            raise RetrievalError(
                f"Embedding page order mismatch at physical page {page.physical_page_number}"
            )
        if record.get("text_sha256") != page.text_sha256:
            raise RetrievalError(
                f"Embedding text hash mismatch at physical page {page.physical_page_number}"
            )
        raw_vector = record.get("vector")
        if not isinstance(raw_vector, list) or any(
            not isinstance(value, int | float) or isinstance(value, bool) for value in raw_vector
        ):
            raise RetrievalError(f"vectors.jsonl line {line_number} has an invalid vector")
        vector = _normalize(
            tuple(float(value) for value in raw_vector),
            info.dimension,
            f"vectors.jsonl line {line_number}",
        )
        vectors.append(vector)
        pages.append(page.physical_page_number)

    return EmbeddingIndex(
        directory=root,
        model=info,
        vectors=tuple(vectors),
        pages=tuple(pages),
        manifest=manifest,
        manifest_sha256=_sha256(manifest_raw),
        vectors_jsonl_sha256=_sha256(vectors_raw),
    )


def _excerpt(text: str, max_characters: int) -> str:
    excerpt = text[:max_characters]
    if len(text) > max_characters:
        boundary = excerpt.rfind(" ")
        if boundary > 0:
            excerpt = excerpt[:boundary]
    return excerpt.strip()


class SemanticRetriever:
    """Deterministic cosine-similarity ranking over cached page embeddings."""

    def __init__(
        self,
        corpus: EvidenceCorpus,
        index: EmbeddingIndex,
        backend: EmbeddingBackend,
    ) -> None:
        if index.pages != tuple(page.physical_page_number for page in corpus.pages):
            raise RetrievalError("Embedding index page order does not match selected evidence")
        if backend.info != index.model:
            raise RetrievalError("Embedding backend configuration does not match the index")
        self.corpus = corpus
        self.index = index
        self.backend = backend

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        excerpt_characters: int = EXCERPT_CHARACTERS_DEFAULT,
    ) -> tuple[SearchResult, ...]:
        """Rank selected physical pages with cosine similarity."""
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
        scored = [
            (sum(left * right for left, right in zip(query_vector, vector, strict=True)), page)
            for page, vector in zip(self.corpus.pages, self.index.vectors, strict=True)
        ]
        scored.sort(key=lambda item: (-item[0], item[1].physical_page_number))
        return tuple(
            SearchResult(
                rank=rank,
                score=score,
                page=page,
                excerpt=_excerpt(page.text, excerpt_characters),
                evidence_jsonl_sha256=self.corpus.evidence_jsonl_sha256,
                manifest_sha256=self.corpus.manifest_sha256,
            )
            for rank, (score, page) in enumerate(scored[:top_k], start=1)
        )
