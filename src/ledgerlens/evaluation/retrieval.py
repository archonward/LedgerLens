"""Dataset validation, retrieval metrics, and reproducible report writing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from ledgerlens import __version__
from ledgerlens.retrieval import BM25Retriever, EvidenceCorpus, SearchResult

EVALUATION_SCHEMA_VERSION = "1.0"
REPORT_SCHEMA_VERSION = "1.0"
SUPPORTED_SPLITS = {"development", "held_out"}
SUPPORTED_STATUSES = {"verified", "provisional"}


class EvaluationError(Exception):
    """Raised when judgments or evaluation settings are invalid."""


@dataclass(frozen=True)
class EvaluationQuestion:
    """A question and independently supplied page-level relevance judgment."""

    question_id: str
    question: str
    split: str
    status: str
    leakage_group: str
    relevant_pages: tuple[int, ...]
    candidate_pages: tuple[int, ...]
    verification: dict[str, Any] | None
    notes: str | None


@dataclass(frozen=True)
class EvaluationSet:
    """A validated evaluation definition bound to one source document."""

    dataset_id: str
    document_id: str
    source_sha256: str
    questions: tuple[EvaluationQuestion, ...]
    path: Path
    sha256: str


@dataclass(frozen=True)
class EvaluationReport:
    """A structured retrieval evaluation result."""

    payload: dict[str, Any]

    @property
    def verified_question_count(self) -> int:
        """Return the number of questions included in official metrics."""
        return self.payload["counts"]["verified_evaluated"]

    @property
    def provisional_question_count(self) -> int:
        """Return the number of displayed questions excluded from official metrics."""
        return self.payload["counts"]["provisional_excluded"]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_string(value: dict[str, Any], key: str, context: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise EvaluationError(f"{context}: {key!r} must be a non-empty string")
    return item


def _page_list(value: Any, context: str, *, allow_empty: bool) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in value
    ):
        raise EvaluationError(f"{context} must be a list of positive physical page numbers")
    if not allow_empty and not value:
        raise EvaluationError(f"{context} must contain at least one physical page number")
    if len(value) != len(set(value)):
        raise EvaluationError(f"{context} contains duplicate physical page numbers")
    return tuple(value)


def _validate_verification(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{context}: verified questions require a verification object")
    if value.get("method") != "manual_source_review":
        raise EvaluationError(f"{context}: verification method must be 'manual_source_review'")
    _require_string(value, "reviewed_by", f"{context} verification")
    reviewed_on = _require_string(value, "reviewed_on", f"{context} verification")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", reviewed_on):
        raise EvaluationError(
            f"{context}: verification reviewed_on must be an ISO date (YYYY-MM-DD)"
        )
    try:
        date.fromisoformat(reviewed_on)
    except ValueError as exc:
        raise EvaluationError(
            f"{context}: verification reviewed_on must be an ISO date (YYYY-MM-DD)"
        ) from exc
    return value


def _normalized_question(question: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", question.casefold(), re.UNICODE))


def load_evaluation_set(path: str | Path) -> EvaluationSet:
    """Load and validate judgments without consulting retrieval output."""
    evaluation_path = Path(path)
    try:
        raw = evaluation_path.read_bytes()
    except OSError as exc:
        raise EvaluationError(f"Could not read evaluation set at {evaluation_path}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"Malformed evaluation set at {evaluation_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError("Evaluation set must be a JSON object")
    if payload.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        raise EvaluationError("Evaluation set uses an unsupported schema_version")
    dataset_id = _require_string(payload, "dataset_id", "Evaluation set")
    document_id = _require_string(payload, "document_id", "Evaluation set")
    source_sha256 = _require_string(payload, "source_sha256", "Evaluation set")
    records = payload.get("questions")
    if not isinstance(records, list):
        raise EvaluationError("Evaluation set: 'questions' must be a list")

    questions: list[EvaluationQuestion] = []
    seen_ids: set[str] = set()
    leakage_splits: dict[str, str] = {}
    exact_splits: dict[str, str] = {}
    for index, record in enumerate(records, start=1):
        context = f"question {index}"
        if not isinstance(record, dict):
            raise EvaluationError(f"{context} must be a JSON object")
        question_id = _require_string(record, "question_id", context)
        if question_id in seen_ids:
            raise EvaluationError(f"Duplicate question_id: {question_id}")
        seen_ids.add(question_id)
        question_text = _require_string(record, "question", context)
        split = _require_string(record, "split", context)
        if split not in SUPPORTED_SPLITS:
            raise EvaluationError(f"{context}: split must be development or held_out")
        status = _require_string(record, "status", context)
        if status not in SUPPORTED_STATUSES:
            raise EvaluationError(f"{context}: status must be verified or provisional")
        leakage_group = _require_string(record, "leakage_group", context)
        previous_split = leakage_splits.setdefault(leakage_group, split)
        if previous_split != split:
            raise EvaluationError(
                f"Leakage group {leakage_group!r} occurs in both development and held_out"
            )
        normalized = _normalized_question(question_text)
        previous_exact_split = exact_splits.setdefault(normalized, split)
        if previous_exact_split != split:
            raise EvaluationError("Exact or punctuation-only question duplicate crosses splits")

        if status == "verified":
            relevant_pages = _page_list(
                record.get("relevant_pages"), f"{context} relevant_pages", allow_empty=False
            )
            if "candidate_pages" in record:
                raise EvaluationError(f"{context}: verified questions must not use candidate_pages")
            candidate_pages: tuple[int, ...] = ()
            verification = _validate_verification(record.get("verification"), context)
        else:
            if "relevant_pages" in record:
                raise EvaluationError(
                    f"{context}: provisional questions must not declare verified relevant_pages"
                )
            relevant_pages = ()
            candidate_pages = _page_list(
                record.get("candidate_pages", []),
                f"{context} candidate_pages",
                allow_empty=True,
            )
            if record.get("verification") is not None:
                raise EvaluationError(
                    f"{context}: provisional verification must be null or omitted"
                )
            verification = None
        notes = record.get("notes")
        if notes is not None and not isinstance(notes, str):
            raise EvaluationError(f"{context}: notes must be a string or null")
        questions.append(
            EvaluationQuestion(
                question_id=question_id,
                question=question_text,
                split=split,
                status=status,
                leakage_group=leakage_group,
                relevant_pages=relevant_pages,
                candidate_pages=candidate_pages,
                verification=verification,
                notes=notes,
            )
        )

    return EvaluationSet(
        dataset_id=dataset_id,
        document_id=document_id,
        source_sha256=source_sha256,
        questions=tuple(questions),
        path=evaluation_path,
        sha256=_sha256(raw),
    )


def _validate_cutoffs(cutoffs: tuple[int, ...]) -> tuple[int, ...]:
    if not cutoffs or any(
        not isinstance(cutoff, int) or isinstance(cutoff, bool) or cutoff < 1 for cutoff in cutoffs
    ):
        raise EvaluationError("Evaluation cutoffs must be positive integers")
    return tuple(sorted(set(cutoffs)))


def _retrieved_page(result: SearchResult) -> dict[str, Any]:
    return {
        "rank": result.rank,
        "score": result.score,
        "physical_page_number": result.page.physical_page_number,
        "method": result.page.method,
        "text_sha256": result.page.text_sha256,
        "source_sha256": result.page.provenance.get("source_sha256"),
        "known_limitations": list(result.page.known_limitations),
    }


def evaluate_bm25(
    corpus: EvidenceCorpus,
    evaluation_set: EvaluationSet,
    *,
    split: str,
    cutoffs: tuple[int, ...] = (1, 3, 5, 10),
) -> EvaluationReport:
    """Evaluate BM25 on one split, excluding provisional questions from metrics."""
    if split not in SUPPORTED_SPLITS:
        raise EvaluationError("Evaluation split must be development or held_out")
    cutoffs = _validate_cutoffs(cutoffs)
    if evaluation_set.document_id != corpus.document_id:
        raise EvaluationError("Evaluation-set document_id does not match selected evidence")
    if evaluation_set.source_sha256 != corpus.source_sha256:
        raise EvaluationError("Evaluation-set source_sha256 does not match selected evidence")
    selected_questions = tuple(q for q in evaluation_set.questions if q.split == split)
    available_pages = {page.physical_page_number for page in corpus.pages}
    for question in selected_questions:
        if question.status == "verified":
            missing = sorted(set(question.relevant_pages) - available_pages)
            if missing:
                raise EvaluationError(
                    f"Verified question {question.question_id!r} references unavailable selected "
                    f"evidence pages: {', '.join(str(page) for page in missing)}"
                )

    retriever = BM25Retriever(corpus)
    per_question: list[dict[str, Any]] = []
    verified_metrics: list[dict[str, Any]] = []
    result_depth = max(cutoffs)
    full_depth = max(len(corpus.pages), 1)
    for question in selected_questions:
        full_results = retriever.search(question.question, top_k=full_depth)
        displayed_results = full_results[:result_depth]
        record: dict[str, Any] = {
            "question_id": question.question_id,
            "question": question.question,
            "split": question.split,
            "status": question.status,
            "leakage_group": question.leakage_group,
            "included_in_official_metrics": question.status == "verified",
            "retrieved_pages": [_retrieved_page(result) for result in displayed_results],
            "notes": question.notes,
        }
        if question.status == "provisional":
            record.update(
                {
                    "candidate_pages": list(question.candidate_pages),
                    "metrics": None,
                    "exclusion_reason": "Supporting pages have not been manually verified.",
                }
            )
        else:
            relevant = set(question.relevant_pages)
            ranked_pages = [result.page.physical_page_number for result in full_results]
            first_rank = next(
                (rank for rank, page in enumerate(ranked_pages, start=1) if page in relevant), None
            )
            metrics = {
                "hit_at_k": {
                    str(cutoff): int(bool(relevant.intersection(ranked_pages[:cutoff])))
                    for cutoff in cutoffs
                },
                "recall_at_k": {
                    str(cutoff): len(relevant.intersection(ranked_pages[:cutoff])) / len(relevant)
                    for cutoff in cutoffs
                },
                "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
                "first_relevant_rank": first_rank,
            }
            record.update(
                {
                    "relevant_pages": list(question.relevant_pages),
                    "verification": question.verification,
                    "metrics": metrics,
                    "exclusion_reason": None,
                }
            )
            verified_metrics.append(metrics)
        per_question.append(record)

    verified_count = len(verified_metrics)
    aggregate: dict[str, Any] | None
    if verified_count:
        aggregate = {
            "hit_at_k": {
                str(cutoff): sum(metric["hit_at_k"][str(cutoff)] for metric in verified_metrics)
                / verified_count
                for cutoff in cutoffs
            },
            "recall_at_k": {
                str(cutoff): sum(metric["recall_at_k"][str(cutoff)] for metric in verified_metrics)
                / verified_count
                for cutoff in cutoffs
            },
            "mrr": sum(metric["reciprocal_rank"] for metric in verified_metrics) / verified_count,
        }
    else:
        aggregate = None

    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "ledgerlens_version": __version__,
        "dataset": {
            "dataset_id": evaluation_set.dataset_id,
            "path": str(evaluation_set.path),
            "sha256": evaluation_set.sha256,
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "split": split,
        },
        "evidence": {
            "document_id": corpus.document_id,
            "source_sha256": corpus.source_sha256,
            "evidence_jsonl_sha256": corpus.evidence_jsonl_sha256,
            "manifest_sha256": corpus.manifest_sha256,
        },
        "retriever": {
            "name": "page_level_bm25",
            "k1": retriever.k1,
            "b": retriever.b,
            "unit": "selected_physical_page",
            "tie_breaker": "ascending_physical_page_number",
        },
        "cutoffs": list(cutoffs),
        "counts": {
            "questions_in_split": len(selected_questions),
            "verified_evaluated": verified_count,
            "provisional_excluded": sum(q.status == "provisional" for q in selected_questions),
        },
        "aggregate_metrics": aggregate,
        "questions": per_question,
        "metric_notes": {
            "hit_at_k": "One when any verified relevant page appears in the first K results.",
            "recall_at_k": "Fraction of all verified relevant pages appearing in the first K.",
            "mrr": "Mean reciprocal rank of the first verified relevant page over full ranking.",
        },
    }
    return EvaluationReport(payload=payload)


def write_evaluation_report(
    report: EvaluationReport,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a report without silently replacing an existing result."""
    destination = Path(output_path)
    if destination.exists() and not overwrite:
        raise EvaluationError(
            f"Evaluation output already exists: {destination}. Use --overwrite to replace it."
        )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(report.payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, destination)
    except OSError as exc:
        if "temporary_path" in locals():
            temporary_path.unlink(missing_ok=True)
        raise EvaluationError(f"Could not write evaluation report to {destination}: {exc}") from exc
    return destination
