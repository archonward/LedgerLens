"""Behavior tests for reviewed recovery selection and evidence export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ledgerlens.evidence import (
    EvidenceSelectionError,
    create_reviewed_recovery,
    prepare_evidence,
    select_page_evidence,
)

DOCUMENT_HASH = "a" * 64
DOCUMENT_ID = f"sha256-{DOCUMENT_HASH}"
EXPECTED_PHRASE = "fictional computing and Al."
CORRECTED_PHRASE = "fictional computing and AI."


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def quality(*, eligible: bool) -> dict[str, object]:
    return {
        "status": "no_detected_issues" if eligible else "issues_detected",
        "unresolved_character_code_count": 0 if eligible else 2,
        "replacement_character_count": 0,
        "unexpected_control_character_count": 0,
        "requires_review": not eligible,
        "automatic_indexing_eligible": eligible,
        "warnings": [] if eligible else ["Fictional extraction defect."],
    }


def make_canonical(tmp_path: Path) -> Path:
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    write_json(
        canonical / "document.json",
        {
            "schema_version": "1.2",
            "document_id": DOCUMENT_ID,
            "sha256": DOCUMENT_HASH,
            "company": "Fictional Company",
            "report_year": 2026,
            "page_count": 3,
            "extractor": {"name": "pdfplumber", "version": "test"},
            "extraction_settings": {"method": "Page.extract_text"},
        },
    )
    pages = [
        {
            "schema_version": "1.2",
            "document_id": DOCUMENT_ID,
            "physical_page_number": 1,
            "text": "Canonical page one.",
            "text_quality": quality(eligible=True),
        },
        {
            "schema_version": "1.2",
            "document_id": DOCUMENT_ID,
            "physical_page_number": 2,
            "text": "(cid:10)(cid:11)",
            "text_quality": quality(eligible=False),
        },
        {
            "schema_version": "1.2",
            "document_id": DOCUMENT_ID,
            "physical_page_number": 3,
            "text": "Canonical notes page.",
            "text_quality": quality(eligible=True),
        },
    ]
    with (canonical / "pages.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for page in pages:
            stream.write(json.dumps(page) + "\n")
    (canonical / "inspection.md").write_text("fictional inspection\n", encoding="utf-8")
    return canonical


def make_recovery(
    tmp_path: Path, text: str = f"Recovered {EXPECTED_PHRASE}\n"
) -> tuple[Path, Path]:
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    candidate = recovery / "candidate.txt"
    metadata = recovery / "recovery.json"
    candidate.write_text(text, encoding="utf-8", newline="")
    write_json(
        metadata,
        {
            "schema_version": "1.1",
            "document_id": DOCUMENT_ID,
            "source_sha256": DOCUMENT_HASH,
            "physical_page_number": 2,
            "source_page_count": 3,
            "engine": "tesseract",
            "extractor": {"name": "tesseract", "version": "test"},
            "extraction_settings": {"dpi": 300, "language": "eng"},
            "text": text,
            "text_quality": quality(eligible=True),
        },
    )
    return candidate, metadata


def make_review_and_manifest(
    tmp_path: Path,
    canonical: Path,
    *,
    page_limitations: list[dict[str, object]] | None = None,
) -> tuple[Path, Path, Path, Path]:
    candidate, recovery_metadata = make_recovery(tmp_path)
    reviewed = tmp_path / "reviewed"
    create_reviewed_recovery(
        candidate,
        recovery_metadata,
        output_directory=reviewed,
        expected_phrase=EXPECTED_PHRASE,
        corrected_phrase=CORRECTED_PHRASE,
        correction_reason="The source uses the abbreviation AI.",
        review_provenance={
            "review_type": "conversational_ai_assistant_review",
            "user_authorized_prototype_use": True,
        },
        review_scope=["fictional narrative and table checks"],
        limitations=["Flattened table headers require source review."],
    )
    corrected = reviewed / "corrected.txt"
    review_record = reviewed / "review.json"
    manifest = tmp_path / "review-manifest.json"
    write_json(
        manifest,
        {
            "schema_version": "1.0",
            "document_id": DOCUMENT_ID,
            "source_sha256": DOCUMENT_HASH,
            "reviewed_recoveries": [
                {
                    "physical_page_number": 2,
                    "selected_for_retrieval_prototype": True,
                    "corrected_text_path": str(corrected),
                    "review_record_path": str(review_record),
                    "original_candidate_path": str(candidate),
                    "recovery_metadata_path": str(recovery_metadata),
                    "original_candidate_sha256": hash_file(candidate),
                    "recovery_metadata_sha256": hash_file(recovery_metadata),
                    "corrected_text_sha256": hash_file(corrected),
                    "review_record_sha256": hash_file(review_record),
                }
            ],
            "page_limitations": page_limitations or [],
        },
    )
    assert canonical.is_dir()
    return manifest, candidate, recovery_metadata, reviewed


def test_selects_eligible_canonical_text_and_excludes_flagged_page(tmp_path: Path) -> None:
    canonical = make_canonical(tmp_path)

    result = select_page_evidence(canonical)

    assert [page["physical_page_number"] for page in result.pages] == [1, 3]
    assert all(page["method"] == "canonical_extraction" for page in result.pages)
    assert result.canonical_count == 2
    assert result.reviewed_recovery_count == 0
    assert [page["physical_page_number"] for page in result.excluded_pages] == [2]


def test_reviewed_recovery_is_selected_once_in_stable_page_order(
    tmp_path: Path,
) -> None:
    canonical = make_canonical(tmp_path)
    manifest, _, _, _ = make_review_and_manifest(
        tmp_path,
        canonical,
        page_limitations=[
            {
                "physical_page_number": 3,
                "limitations": ["Table-value associations require source review."],
            }
        ],
    )

    result = select_page_evidence(canonical, manifest)

    assert [page["physical_page_number"] for page in result.pages] == [1, 2, 3]
    assert len({page["physical_page_number"] for page in result.pages}) == 3
    assert result.canonical_count == 2
    assert result.reviewed_recovery_count == 1
    assert result.excluded_pages == ()
    recovered = result.pages[1]
    assert recovered["method"] == "reviewed_ocr_recovery"
    assert CORRECTED_PHRASE in recovered["text"]
    assert EXPECTED_PHRASE not in recovered["text"]
    assert recovered["review_provenance"]["user_authorized_prototype_use"] is True
    assert result.pages_with_limitations == (2, 3)
    assert result.pages[2]["known_limitations"] == [
        "Table-value associations require source review."
    ]


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("source", "Manifest source hash mismatch"),
        ("document", "Manifest document ID mismatch"),
        ("page", "Review page number mismatch"),
        ("candidate_hash", "Manifest original_candidate_sha256 mismatch"),
        ("recovery_hash", "Manifest recovery_metadata_sha256 mismatch"),
        ("corrected_hash", "Manifest corrected_text_sha256 mismatch"),
        ("review_hash", "Manifest review_record_sha256 mismatch"),
        ("duplicate", "Duplicate or conflicting recovery selections"),
    ],
)
def test_invalid_stale_conflicting_or_mismatched_reviews_are_rejected(
    tmp_path: Path, mutation: str, error: str
) -> None:
    canonical = make_canonical(tmp_path)
    manifest_path, _, _, _ = make_review_and_manifest(tmp_path, canonical)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    selection = manifest["reviewed_recoveries"][0]
    if mutation == "source":
        manifest["source_sha256"] = "b" * 64
    elif mutation == "document":
        manifest["document_id"] = f"sha256-{'b' * 64}"
    elif mutation == "page":
        selection["physical_page_number"] = 1
    elif mutation == "candidate_hash":
        selection["original_candidate_sha256"] = "0" * 64
    elif mutation == "recovery_hash":
        selection["recovery_metadata_sha256"] = "0" * 64
    elif mutation == "corrected_hash":
        selection["corrected_text_sha256"] = "0" * 64
    elif mutation == "review_hash":
        selection["review_record_sha256"] = "0" * 64
    elif mutation == "duplicate":
        manifest["reviewed_recoveries"].append(dict(selection))
    write_json(manifest_path, manifest)

    with pytest.raises(EvidenceSelectionError, match=error):
        select_page_evidence(canonical, manifest_path)


@pytest.mark.parametrize(
    "text",
    [
        "The expected phrase is absent.",
        f"{EXPECTED_PHRASE} repeated {EXPECTED_PHRASE}",
    ],
)
def test_reviewed_correction_requires_exactly_one_occurrence(tmp_path: Path, text: str) -> None:
    candidate, metadata = make_recovery(tmp_path, text)

    with pytest.raises(EvidenceSelectionError, match="exactly once"):
        create_reviewed_recovery(
            candidate,
            metadata,
            output_directory=tmp_path / "reviewed",
            expected_phrase=EXPECTED_PHRASE,
            corrected_phrase=CORRECTED_PHRASE,
            correction_reason="Fictional correction.",
            review_provenance={"review_type": "test"},
            review_scope=["test"],
            limitations=[],
        )


def test_export_preserves_limitations_protects_output_and_keeps_inputs_unchanged(
    tmp_path: Path,
) -> None:
    canonical = make_canonical(tmp_path)
    manifest, candidate, recovery_metadata, reviewed = make_review_and_manifest(
        tmp_path,
        canonical,
        page_limitations=[
            {
                "physical_page_number": 3,
                "limitations": ["Table-value associations require source review."],
            }
        ],
    )
    protected_paths = [
        canonical / "document.json",
        canonical / "pages.jsonl",
        canonical / "inspection.md",
        candidate,
        recovery_metadata,
        reviewed / "corrected.txt",
        reviewed / "review.json",
        manifest,
    ]
    before = {path: hash_file(path) for path in protected_paths}

    result = prepare_evidence(
        canonical,
        review_manifest_path=manifest,
        output_root=tmp_path / "evidence",
    )
    records = [
        json.loads(line)
        for line in (result.output_directory / "evidence.jsonl").read_text("utf-8").splitlines()
    ]
    export_manifest = json.loads((result.output_directory / "manifest.json").read_text("utf-8"))

    assert [record["physical_page_number"] for record in records] == [1, 2, 3]
    assert records[1]["method"] == "reviewed_ocr_recovery"
    assert records[2]["known_limitations"] == ["Table-value associations require source review."]
    assert export_manifest["counts"] == {
        "selected_pages": 3,
        "canonical_pages": 2,
        "reviewed_recoveries": 1,
        "excluded_pages": 0,
        "pages_with_limitations": 2,
    }
    assert export_manifest["pages_with_limitations"] == [2, 3]
    assert before == {path: hash_file(path) for path in protected_paths}

    with pytest.raises(EvidenceSelectionError, match="--overwrite"):
        prepare_evidence(
            canonical,
            review_manifest_path=manifest,
            output_root=tmp_path / "evidence",
        )
