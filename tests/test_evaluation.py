"""Offline tests for retrieval evaluation datasets, metrics, and CLI output."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ledgerlens.evaluation import (
    EvaluationError,
    evaluate_bm25,
    load_evaluation_set,
    write_evaluation_report,
)
from ledgerlens.retrieval import load_evidence_export

DOCUMENT_ID = "sha256-" + "d" * 64
SOURCE_HASH = "d" * 64


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_evidence(root: Path) -> Path:
    root.mkdir()
    texts = {
        1: "Revenue overview for the fictional company.",
        2: "Revenue growth came from accelerated computing demand.",
        3: "Cash flows and liquidity discussion.",
        4: "General corporate information.",
    }
    records = []
    decisions = []
    for page, text in texts.items():
        text_hash = _hash(text)
        records.append(
            {
                "schema_version": "1.0",
                "document_id": DOCUMENT_ID,
                "company": "Fictional Computing Ltd.",
                "report_year": 2030,
                "physical_page_number": page,
                "text": text,
                "text_sha256": text_hash,
                "method": "canonical_extraction",
                "provenance": {
                    "source_sha256": SOURCE_HASH,
                    "canonical_text_sha256": text_hash,
                },
                "screening_findings": {"quality": {"status": "pass", "warnings": []}},
                "review_provenance": None,
                "known_limitations": [],
            }
        )
        decisions.append(
            {
                "physical_page_number": page,
                "method": "canonical_extraction",
                "selected_text_sha256": text_hash,
                "known_limitations": [],
            }
        )
    manifest = {
        "schema_version": "1.0",
        "document_id": DOCUMENT_ID,
        "source_sha256": SOURCE_HASH,
        "counts": {
            "selected_pages": 4,
            "canonical_pages": 4,
            "reviewed_recoveries": 0,
            "excluded_pages": 0,
            "pages_with_limitations": 0,
        },
        "selection_decisions": decisions,
        "excluded_pages": [],
        "pages_with_limitations": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "evidence.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return root


def _verified(
    question_id: str,
    question: str,
    relevant_pages: list[int],
    *,
    split: str = "development",
    leakage_group: str | None = None,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "split": split,
        "status": "verified",
        "leakage_group": leakage_group or question_id,
        "relevant_pages": relevant_pages,
        "verification": {
            "method": "manual_source_review",
            "reviewed_by": "Fictional reviewer",
            "reviewed_on": "2030-04-05",
        },
    }


def _provisional(question_id: str, question: str) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "split": "development",
        "status": "provisional",
        "leakage_group": question_id,
        "candidate_pages": [3],
        "notes": "Candidate only; source review is pending.",
    }


def _write_dataset(
    path: Path,
    questions: list[dict[str, Any]],
    *,
    document_id: str = DOCUMENT_ID,
    source_hash: str = SOURCE_HASH,
) -> Path:
    payload = {
        "schema_version": "1.0",
        "dataset_id": "fictional-retrieval-v1",
        "document_id": document_id,
        "source_sha256": source_hash,
        "questions": questions,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def evaluation_inputs(tmp_path: Path) -> tuple[Path, Path]:
    evidence = _write_evidence(tmp_path / "evidence")
    dataset = _write_dataset(
        tmp_path / "evaluation.json",
        [
            _verified("single", "accelerated computing demand", [2]),
            _verified("multiple", "revenue growth", [1, 2]),
            _verified("miss", "xylophone", [3]),
            _provisional("candidate", "cash flows"),
            _verified("held", "liquidity", [3], split="held_out"),
        ],
    )
    return evidence, dataset


def test_loader_separates_verified_and_provisional_judgments(
    evaluation_inputs: tuple[Path, Path],
) -> None:
    _, dataset_path = evaluation_inputs
    dataset = load_evaluation_set(dataset_path)

    assert dataset.questions[0].relevant_pages == (2,)
    assert dataset.questions[3].relevant_pages == ()
    assert dataset.questions[3].candidate_pages == (3,)
    assert dataset.sha256 == hashlib.sha256(dataset_path.read_bytes()).hexdigest()


def test_metrics_handle_single_multiple_and_missed_relevant_pages(
    evaluation_inputs: tuple[Path, Path],
) -> None:
    evidence_path, dataset_path = evaluation_inputs
    report = evaluate_bm25(
        load_evidence_export(evidence_path),
        load_evaluation_set(dataset_path),
        split="development",
        cutoffs=(1, 2),
    ).payload
    by_id = {record["question_id"]: record for record in report["questions"]}

    assert by_id["single"]["metrics"]["hit_at_k"] == {"1": 1, "2": 1}
    assert by_id["single"]["metrics"]["reciprocal_rank"] == 1.0
    assert by_id["multiple"]["metrics"]["recall_at_k"] == {"1": 0.5, "2": 1.0}
    assert by_id["miss"]["metrics"]["first_relevant_rank"] is None
    assert by_id["miss"]["metrics"]["reciprocal_rank"] == 0.0
    assert report["aggregate_metrics"]["hit_at_k"] == {"1": 2 / 3, "2": 2 / 3}
    assert report["aggregate_metrics"]["recall_at_k"] == {"1": 0.5, "2": 2 / 3}
    assert report["aggregate_metrics"]["mrr"] == 2 / 3


def test_provisional_questions_are_run_but_excluded_from_metrics(
    evaluation_inputs: tuple[Path, Path],
) -> None:
    evidence_path, dataset_path = evaluation_inputs
    report = evaluate_bm25(
        load_evidence_export(evidence_path),
        load_evaluation_set(dataset_path),
        split="development",
    ).payload
    candidate = next(q for q in report["questions"] if q["question_id"] == "candidate")

    assert report["counts"] == {
        "questions_in_split": 4,
        "verified_evaluated": 3,
        "provisional_excluded": 1,
    }
    assert candidate["included_in_official_metrics"] is False
    assert candidate["metrics"] is None
    assert candidate["retrieved_pages"][0]["physical_page_number"] == 3


def test_exact_split_selection_and_report_provenance(
    evaluation_inputs: tuple[Path, Path],
) -> None:
    evidence_path, dataset_path = evaluation_inputs
    corpus = load_evidence_export(evidence_path)
    dataset = load_evaluation_set(dataset_path)
    report = evaluate_bm25(corpus, dataset, split="held_out", cutoffs=(1,)).payload

    assert [question["question_id"] for question in report["questions"]] == ["held"]
    assert report["dataset"]["sha256"] == dataset.sha256
    assert report["evidence"]["evidence_jsonl_sha256"] == corpus.evidence_jsonl_sha256
    assert report["evidence"]["manifest_sha256"] == corpus.manifest_sha256
    assert report["aggregate_metrics"]["hit_at_k"] == {"1": 1.0}


def test_split_with_only_provisional_questions_has_no_official_metrics(tmp_path: Path) -> None:
    evidence = _write_evidence(tmp_path / "evidence")
    dataset = _write_dataset(tmp_path / "evaluation.json", [_provisional("p1", "cash")])

    report = evaluate_bm25(
        load_evidence_export(evidence),
        load_evaluation_set(dataset),
        split="development",
    ).payload

    assert report["aggregate_metrics"] is None
    assert report["counts"]["verified_evaluated"] == 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "draft", "status must be verified or provisional"),
        ("split", "test", "split must be development or held_out"),
        ("verification", None, "require a verification object"),
    ],
)
def test_invalid_judgment_fields_are_rejected(
    tmp_path: Path, field: str, value: Any, message: str
) -> None:
    question = _verified("q1", "revenue", [1])
    question[field] = value
    dataset = _write_dataset(tmp_path / "evaluation.json", [question])

    with pytest.raises(EvaluationError, match=message):
        load_evaluation_set(dataset)


def test_provisional_question_cannot_declare_relevant_pages(tmp_path: Path) -> None:
    question = _provisional("q1", "cash")
    question["relevant_pages"] = [3]
    dataset = _write_dataset(tmp_path / "evaluation.json", [question])

    with pytest.raises(EvaluationError, match="must not declare verified relevant_pages"):
        load_evaluation_set(dataset)


def test_duplicate_and_leakage_checks_prevent_cross_split_contamination(tmp_path: Path) -> None:
    questions = [
        _verified("q1", "Revenue growth?", [2], leakage_group="revenue-growth"),
        _verified(
            "q2",
            "How did revenue grow?",
            [2],
            split="held_out",
            leakage_group="revenue-growth",
        ),
    ]
    dataset = _write_dataset(tmp_path / "evaluation.json", questions)

    with pytest.raises(EvaluationError, match="occurs in both"):
        load_evaluation_set(dataset)


def test_exact_question_duplicate_cannot_cross_splits(tmp_path: Path) -> None:
    questions = [
        _verified("q1", "What was revenue growth?", [2]),
        _verified(
            "q2",
            "WHAT WAS REVENUE GROWTH!",
            [2],
            split="held_out",
            leakage_group="different-label",
        ),
    ]
    dataset = _write_dataset(tmp_path / "evaluation.json", questions)

    with pytest.raises(EvaluationError, match="question duplicate crosses splits"):
        load_evaluation_set(dataset)


def test_document_and_source_mismatches_are_rejected(
    evaluation_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    evidence_path, _ = evaluation_inputs
    corpus = load_evidence_export(evidence_path)
    wrong_document = _write_dataset(
        tmp_path / "wrong-document.json",
        [_verified("q1", "revenue", [1])],
        document_id="sha256-" + "e" * 64,
    )
    wrong_source = _write_dataset(
        tmp_path / "wrong-source.json",
        [_verified("q2", "revenue", [1])],
        source_hash="e" * 64,
    )

    with pytest.raises(EvaluationError, match="document_id"):
        evaluate_bm25(corpus, load_evaluation_set(wrong_document), split="development")
    with pytest.raises(EvaluationError, match="source_sha256"):
        evaluate_bm25(corpus, load_evaluation_set(wrong_source), split="development")


def test_unavailable_verified_page_is_rejected(evaluation_inputs: tuple[Path, Path]) -> None:
    evidence_path, dataset_path = evaluation_inputs
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    payload["questions"][0]["relevant_pages"] = [99]
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(EvaluationError, match="unavailable selected evidence pages: 99"):
        evaluate_bm25(
            load_evidence_export(evidence_path),
            load_evaluation_set(dataset_path),
            split="development",
        )


@pytest.mark.parametrize("cutoffs", [(), (0,), (-1, 5), (True,)])
def test_invalid_cutoffs_are_rejected(
    evaluation_inputs: tuple[Path, Path], cutoffs: tuple[int, ...]
) -> None:
    evidence_path, dataset_path = evaluation_inputs

    with pytest.raises(EvaluationError, match="positive integers"):
        evaluate_bm25(
            load_evidence_export(evidence_path),
            load_evaluation_set(dataset_path),
            split="development",
            cutoffs=cutoffs,
        )


def test_report_write_is_atomic_and_protects_existing_output(
    evaluation_inputs: tuple[Path, Path], tmp_path: Path
) -> None:
    evidence_path, dataset_path = evaluation_inputs
    report = evaluate_bm25(
        load_evidence_export(evidence_path),
        load_evaluation_set(dataset_path),
        split="held_out",
    )
    output = tmp_path / "reports" / "result.json"

    assert write_evaluation_report(report, output) == output
    assert json.loads(output.read_text(encoding="utf-8"))["dataset"]["split"] == "held_out"
    with pytest.raises(EvaluationError, match="already exists"):
        write_evaluation_report(report, output)
    write_evaluation_report(report, output, overwrite=True)


@pytest.mark.parametrize("entry_point", [[sys.executable, "-m", "ledgerlens"], ["ledgerlens"]])
def test_cli_writes_fictional_evaluation_report(
    evaluation_inputs: tuple[Path, Path], tmp_path: Path, entry_point: list[str]
) -> None:
    evidence_path, dataset_path = evaluation_inputs
    output = tmp_path / "evaluation-result.json"
    completed = subprocess.run(
        [
            *entry_point,
            "evaluate-retrieval",
            str(evidence_path),
            str(dataset_path),
            "--split",
            "development",
            "--k",
            "1",
            "2",
            "--output",
            str(output),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Verified questions evaluated: 3" in completed.stdout
    assert "Provisional questions excluded from official metrics: 1" in completed.stdout
    assert "Hit@1: 0.6667" in completed.stdout
    assert output.is_file()


def test_cli_refuses_to_replace_judgment_input(
    evaluation_inputs: tuple[Path, Path],
) -> None:
    evidence_path, dataset_path = evaluation_inputs
    original = dataset_path.read_bytes()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledgerlens",
            "evaluate-retrieval",
            str(evidence_path),
            str(dataset_path),
            "--split",
            "development",
            "--output",
            str(dataset_path),
            "--overwrite",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 2
    assert "must be separate" in completed.stderr
    assert dataset_path.read_bytes() == original
