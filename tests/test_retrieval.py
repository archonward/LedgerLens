"""Offline tests for selected-evidence loading and BM25 page retrieval."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ledgerlens.retrieval import (
    BM25Retriever,
    RetrievalError,
    load_evidence_export,
    tokenize,
)

DOCUMENT_ID = "sha256-" + "a" * 64
SOURCE_HASH = "a" * 64


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record(
    page: int,
    text: str,
    *,
    method: str = "canonical_extraction",
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    review = None
    provenance: dict[str, Any] = {
        "source_sha256": SOURCE_HASH,
        "canonical_pages_sha256": "b" * 64,
        "canonical_text_sha256": _hash_text(text),
    }
    screening: dict[str, Any] = {
        "automatic_index_eligible": True,
        "quality": {"status": "pass", "warnings": []},
    }
    if method == "reviewed_ocr_recovery":
        review = {
            "review_type": "conversational_ai_comparison_with_user_authorization",
            "permission": "retrieval_prototype",
        }
        provenance = {
            "source_sha256": SOURCE_HASH,
            "recovery_engine": "tesseract",
            "review_record_sha256": "c" * 64,
        }
        screening = {
            "canonical": {
                "automatic_index_eligible": False,
                "quality": {"status": "review_required", "warnings": ["CID tokens found"]},
            },
            "selected_recovery": {"quality": {"status": "pass", "warnings": []}},
        }
    return {
        "schema_version": "1.0",
        "document_id": DOCUMENT_ID,
        "company": "Fictional Chips Ltd.",
        "report_year": 2030,
        "physical_page_number": page,
        "text": text,
        "text_sha256": _hash_text(text),
        "method": method,
        "provenance": provenance,
        "screening_findings": screening,
        "review_provenance": review,
        "known_limitations": limitations or [],
    }


def _write_export(
    root: Path,
    records: list[dict[str, Any]],
    *,
    excluded: list[int] | None = None,
    document_id: str = DOCUMENT_ID,
) -> Path:
    root.mkdir()
    excluded = excluded or []
    decisions = [
        {
            "physical_page_number": record["physical_page_number"],
            "method": record["method"],
            "selected_text_sha256": record["text_sha256"],
            "known_limitations": record["known_limitations"],
        }
        for record in records
    ]
    manifest = {
        "schema_version": "1.0",
        "document_id": document_id,
        "source_sha256": SOURCE_HASH,
        "counts": {
            "selected_pages": len(records),
            "canonical_pages": sum(r["method"] == "canonical_extraction" for r in records),
            "reviewed_recoveries": sum(r["method"] == "reviewed_ocr_recovery" for r in records),
            "excluded_pages": len(excluded),
            "pages_with_limitations": sum(bool(r["known_limitations"]) for r in records),
        },
        "selection_decisions": decisions,
        "excluded_pages": excluded,
        "pages_with_limitations": [
            r["physical_page_number"] for r in records if r["known_limitations"]
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "evidence.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return root


@pytest.fixture
def evidence_export(tmp_path: Path) -> Path:
    return _write_export(
        tmp_path / "evidence",
        [
            _record(1, "Market overview and corporate governance."),
            _record(2, "Data Center revenue grew from strong accelerated computing demand."),
            _record(
                3,
                "Reportable segments include Compute & Networking revenue of $1,250.",
                method="reviewed_ocr_recovery",
                limitations=["Table headers are flattened; inspect the source image."],
            ),
            _record(4, "Liquidity and working capital discussion."),
        ],
        excluded=[5],
    )


def test_loads_selected_evidence_and_preserves_provenance(evidence_export: Path) -> None:
    corpus = load_evidence_export(evidence_export)

    assert corpus.document_id == DOCUMENT_ID
    assert [page.physical_page_number for page in corpus.pages] == [1, 2, 3, 4]
    assert corpus.excluded_pages == (5,)
    reviewed = corpus.pages[2]
    assert reviewed.method == "reviewed_ocr_recovery"
    assert reviewed.review_provenance is not None
    assert reviewed.provenance["recovery_engine"] == "tesseract"
    assert reviewed.known_limitations == ("Table headers are flattened; inspect the source image.",)
    assert reviewed.warnings == ("CID tokens found",)


def test_relevant_page_ranks_above_irrelevant_page(evidence_export: Path) -> None:
    results = BM25Retriever(load_evidence_export(evidence_export)).search(
        "Data Center revenue growth", top_k=4
    )

    assert results[0].page.physical_page_number == 2
    assert all(result.score > 0 for result in results)
    assert 1 not in [result.page.physical_page_number for result in results]


def test_ties_use_physical_page_order(tmp_path: Path) -> None:
    export = _write_export(
        tmp_path / "ties",
        [_record(7, "identical search phrase"), _record(9, "identical search phrase")],
    )

    results = BM25Retriever(load_evidence_export(export)).search("search")

    assert [result.page.physical_page_number for result in results] == [7, 9]
    assert results[0].score == results[1].score


def test_tokenization_is_case_insensitive_and_normalizes_numeric_commas() -> None:
    assert tokenize("REVENUE was $1,250 and AI-driven.") == (
        "revenue",
        "was",
        "1250",
        "and",
        "ai",
        "driven",
    )


def test_financial_terms_and_numbers_are_retrievable(evidence_export: Path) -> None:
    results = BM25Retriever(load_evidence_export(evidence_export)).search("SEGMENTS 1250")

    assert results[0].page.physical_page_number == 3
    assert results[0].page.method == "reviewed_ocr_recovery"


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_empty_query_returns_no_results(evidence_export: Path, query: str) -> None:
    assert BM25Retriever(load_evidence_export(evidence_export)).search(query) == ()


def test_no_positive_match_returns_no_results(evidence_export: Path) -> None:
    assert BM25Retriever(load_evidence_export(evidence_export)).search("xylophone") == ()


@pytest.mark.parametrize("top_k", [0, -1, True])
def test_invalid_top_k_is_rejected(evidence_export: Path, top_k: Any) -> None:
    with pytest.raises(RetrievalError, match="top-k"):
        BM25Retriever(load_evidence_export(evidence_export)).search("revenue", top_k=top_k)


def test_missing_evidence_files_fail_clearly(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(RetrievalError, match="manifest is missing"):
        load_evidence_export(empty)


def test_malformed_evidence_record_fails_clearly(tmp_path: Path) -> None:
    export = _write_export(tmp_path / "malformed", [_record(1, "valid")])
    (export / "evidence.jsonl").write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(RetrievalError, match="Malformed evidence.jsonl line 1"):
        load_evidence_export(export)


def test_manifest_document_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    export = _write_export(
        tmp_path / "mismatch", [_record(1, "valid")], document_id="sha256-" + "f" * 64
    )

    with pytest.raises(RetrievalError, match="document_id does not match"):
        load_evidence_export(export)


def test_stale_text_and_manifest_hashes_are_rejected(tmp_path: Path) -> None:
    record = _record(1, "original")
    export = _write_export(tmp_path / "stale", [record])
    changed = deepcopy(record)
    changed["text"] = "changed"
    (export / "evidence.jsonl").write_text(json.dumps(changed) + "\n", encoding="utf-8")

    with pytest.raises(RetrievalError, match="text_sha256 does not match"):
        load_evidence_export(export)


def test_excluded_pages_never_appear(evidence_export: Path) -> None:
    results = BM25Retriever(load_evidence_export(evidence_export)).search("revenue", top_k=20)

    assert 5 not in [result.page.physical_page_number for result in results]


def test_excerpt_is_an_exact_selected_text_substring(evidence_export: Path) -> None:
    result = BM25Retriever(load_evidence_export(evidence_export)).search("accelerated", top_k=1)[0]

    assert "accelerated" in result.excerpt
    assert result.excerpt in result.page.text


def test_cli_returns_reviewed_result_as_json(evidence_export: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledgerlens",
            "retrieve",
            str(evidence_export),
            "--query",
            "reportable segments 1250",
            "--top-k",
            "1",
            "--json",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    assert output["result_count"] == 1
    assert output["results"][0]["physical_page_number"] == 3
    assert output["results"][0]["method"] == "reviewed_ocr_recovery"
    assert output["results"][0]["known_limitations"]
