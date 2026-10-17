"""Behavior tests for local PDF ingestion."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest
from reportlab.lib.pdfencrypt import StandardEncryption
from reportlab.pdfgen import canvas

from ledgerlens.ingestion import IngestionError, assess_text_quality, ingest_pdf, recover_page
from ledgerlens.ingestion import recovery as recovery_module


def make_fictional_pdf(path: Path, *, blank_middle_page: bool = True) -> None:
    pdf = canvas.Canvas(str(path))
    pdf.drawString(72, 720, "Example Company annual report narrative.")
    pdf.drawString(72, 700, "Revenue increased; this text is fictional.")
    pdf.showPage()
    if blank_middle_page:
        pdf.showPage()
    pdf.drawString(72, 720, "Fictional Statement of Operations")
    pdf.drawString(72, 700, "Revenue 2024 1,250")
    pdf.drawString(72, 680, "Operating result -75")
    pdf.save()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class FakePypdfPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class FakePypdfReader:
    def __init__(self, _: Path) -> None:
        self.pages = [FakePypdfPage("Alternative candidate text without detected CID tokens.")]


class FakePdfReadError(Exception):
    pass


def test_multi_page_pdf_produces_ordered_page_records_and_provenance(tmp_path: Path) -> None:
    source = tmp_path / "fictional-report.pdf"
    output_root = tmp_path / "processed"
    make_fictional_pdf(source)

    result = ingest_pdf(
        source,
        company="Example Company",
        report_year=2024,
        source_url="https://example.test/fictional-report.pdf",
        output_root=output_root,
    )

    metadata = json.loads((result.output_directory / "document.json").read_text("utf-8"))
    pages = read_jsonl(result.output_directory / "pages.jsonl")
    inspection = (result.output_directory / "inspection.md").read_text("utf-8")

    assert metadata["document_id"] == result.document_id
    assert metadata["sha256"] == result.document_id.removeprefix("sha256-")
    assert metadata["original_filename"] == source.name
    assert metadata["company"] == "Example Company"
    assert metadata["report_year"] == 2024
    assert metadata["source_url"] == "https://example.test/fictional-report.pdf"
    assert metadata["page_count"] == 3
    assert metadata["schema_version"] == "1.1"
    assert metadata["quality_summary"]["status"] == "issues_detected"
    assert metadata["quality_summary"]["pages_requiring_review"] == [2]
    assert metadata["extractor"]["name"] == "pdfplumber"
    assert [page["physical_page_number"] for page in pages] == [1, 2, 3]
    assert all(page["document_id"] == result.document_id for page in pages)
    assert "annual report narrative" in pages[0]["text"]
    assert "Revenue 2024 1,250" in pages[2]["text"]
    assert "Operating result -75" in pages[2]["text"]
    assert pages[2]["character_count"] == len(pages[2]["text"])
    assert pages[2]["text_quality"]["status"] == "no_detected_issues"
    assert pages[2]["text_quality"]["automatic_indexing_eligible"] is True
    assert "Physical page 3" in inspection
    assert "Currency and units have not yet been interpreted" in inspection


def test_blank_page_is_retained_and_flagged(tmp_path: Path) -> None:
    source = tmp_path / "with-blank-page.pdf"
    make_fictional_pdf(source)

    result = ingest_pdf(source, company="Example Company", report_year=2024, output_root=tmp_path)
    pages = read_jsonl(result.output_directory / "pages.jsonl")

    assert result.pages_without_text == (2,)
    assert pages[1]["physical_page_number"] == 2
    assert pages[1]["text"] == ""
    assert pages[1]["character_count"] == 0
    assert "may be blank or may require OCR" in pages[1]["warnings"][0]
    assert pages[1]["text_quality"]["requires_review"] is True
    assert pages[1]["text_quality"]["automatic_indexing_eligible"] is False


def test_quality_detection_counts_multiple_cid_tokens_and_preserves_text() -> None:
    corrupted = "Revenue (cid:62)(cid:75) and repeated (cid:62); value -75."

    quality = assess_text_quality(corrupted)

    assert quality.unresolved_character_code_count == 3
    assert quality.status == "issues_detected"
    assert quality.requires_review is True
    assert quality.automatic_indexing_eligible is False
    assert "(cid:<digits>)" in quality.warnings[0]
    assert corrupted == "Revenue (cid:62)(cid:75) and repeated (cid:62); value -75."


def test_clean_text_has_no_detected_issues_but_is_not_labelled_verified() -> None:
    quality = assess_text_quality("Fictional revenue was 100.0% in 2024.")

    assert quality.status == "no_detected_issues"
    assert quality.unresolved_character_code_count == 0
    assert quality.warnings == ()
    assert "verified" not in quality.status


def test_cid_findings_appear_in_page_document_and_inspection_summaries(tmp_path: Path) -> None:
    source = tmp_path / "cid-text.pdf"
    pdf = canvas.Canvas(str(source))
    corrupted = "Heading (cid:62)(cid:75) amount -75 (cid:62)"
    pdf.drawString(72, 720, corrupted)
    pdf.save()

    result = ingest_pdf(source, company="Fictional", report_year=2024, output_root=tmp_path / "out")
    metadata = json.loads((result.output_directory / "document.json").read_text("utf-8"))
    page = read_jsonl(result.output_directory / "pages.jsonl")[0]
    inspection = (result.output_directory / "inspection.md").read_text("utf-8")

    assert page["text"] == corrupted
    assert page["text_quality"]["unresolved_character_code_count"] == 3
    assert page["text_quality"]["requires_review"] is True
    assert page["text_quality"]["automatic_indexing_eligible"] is False
    assert metadata["quality_summary"]["unresolved_character_codes"] == {
        "total_count": 3,
        "pages": [{"physical_page_number": 1, "count": 3}],
    }
    assert metadata["quality_summary"]["pages_requiring_review"] == [1]
    assert result.unresolved_character_code_findings == ((1, 3),)
    assert "Unresolved character-code tokens: 3" in inspection
    assert "Pages with unresolved character codes: 1 (3 tokens)" in inspection
    assert "Eligible for automatic indexing: no" in inspection
    assert corrupted in inspection


def test_identical_pdf_content_has_same_document_id(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "renamed-copy.pdf"
    make_fictional_pdf(first, blank_middle_page=False)
    shutil.copyfile(first, second)

    first_result = ingest_pdf(
        first,
        company="Example",
        report_year=2024,
        output_root=tmp_path / "a",
    )
    second_result = ingest_pdf(
        second,
        company="Example",
        report_year=2024,
        output_root=tmp_path / "b",
    )

    assert first_result.document_id == second_result.document_id


@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_bad_inputs_fail_clearly_without_outputs(tmp_path: Path, kind: str) -> None:
    source = tmp_path / f"{kind}.pdf"
    if kind == "malformed":
        source.write_bytes(b"This is not a PDF.")
    output_root = tmp_path / "processed"

    with pytest.raises(IngestionError, match="does not exist|malformed|invalid"):
        ingest_pdf(source, company="Example", report_year=2024, output_root=output_root)

    assert not list(output_root.glob("sha256-*")) if output_root.exists() else True


def test_password_protected_pdf_fails_clearly(tmp_path: Path) -> None:
    source = tmp_path / "protected.pdf"
    encryption = StandardEncryption("secret", canCopy=0, canModify=0, canPrint=0)
    pdf = canvas.Canvas(str(source), encrypt=encryption)
    pdf.drawString(72, 720, "Protected fictional content")
    pdf.save()

    with pytest.raises(IngestionError, match="password-protected"):
        ingest_pdf(source, company="Example", report_year=2024, output_root=tmp_path / "out")


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "fictional.pdf"
    output_root = tmp_path / "processed"
    make_fictional_pdf(source, blank_middle_page=False)
    first = ingest_pdf(source, company="Old Company", report_year=2023, output_root=output_root)
    marker = first.output_directory / "old-output-marker"
    marker.write_text("preserve until overwrite", encoding="utf-8")

    with pytest.raises(IngestionError, match="--overwrite"):
        ingest_pdf(source, company="New Company", report_year=2024, output_root=output_root)
    assert marker.exists()

    replaced = ingest_pdf(
        source,
        company="New Company",
        report_year=2024,
        output_root=output_root,
        overwrite=True,
    )
    metadata = json.loads((replaced.output_directory / "document.json").read_text("utf-8"))
    assert metadata["company"] == "New Company"
    assert not marker.exists()


@pytest.mark.parametrize("entry_point", [[sys.executable, "-m", "ledgerlens"], ["ledgerlens"]])
def test_ingest_command_works_for_both_entry_points(tmp_path: Path, entry_point: list[str]) -> None:
    source = tmp_path / "fictional.pdf"
    make_fictional_pdf(source, blank_middle_page=False)
    output_root = tmp_path / entry_point[-1].replace("ledgerlens", "module")

    result = subprocess.run(
        [
            *entry_point,
            "ingest",
            str(source),
            "--company",
            "Example Company",
            "--report-year",
            "2024",
            "--output-root",
            str(output_root),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "Document ID: sha256-" in result.stdout
    assert "Pages: 2" in result.stdout
    assert "Pages with no extracted text: none" in result.stdout
    assert "Extraction warnings: none" in result.stdout
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_cli_reports_input_errors_without_tracebacks(tmp_path: Path, kind: str) -> None:
    source = tmp_path / f"{kind}.pdf"
    if kind == "malformed":
        source.write_text("not a PDF", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ledgerlens",
            "ingest",
            str(source),
            "--company",
            "Example Company",
            "--report-year",
            "2024",
            "--output-root",
            str(tmp_path / "processed"),
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert result.stderr.startswith("error:")
    assert "Traceback" not in result.stderr


def test_recovery_validates_physical_page_numbers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fictional.pdf"
    make_fictional_pdf(source, blank_middle_page=False)
    monkeypatch.setattr(
        recovery_module,
        "_load_pypdf",
        lambda: (FakePypdfReader, FakePdfReadError, "test-version"),
    )

    with pytest.raises(IngestionError, match="at least 1"):
        recover_page(source, physical_page_number=0, output_root=tmp_path / "recovery")
    with pytest.raises(IngestionError, match="out of range"):
        recover_page(source, physical_page_number=2, output_root=tmp_path / "recovery")


def test_recovery_preserves_canonical_ingestion_and_protects_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fictional.pdf"
    make_fictional_pdf(source, blank_middle_page=False)
    ingestion = ingest_pdf(
        source,
        company="Fictional Company",
        report_year=2024,
        output_root=tmp_path / "processed",
    )
    canonical_hashes = {
        name: file_hash(ingestion.output_directory / name)
        for name in ("document.json", "pages.jsonl", "inspection.md")
    }
    monkeypatch.setattr(
        recovery_module,
        "_load_pypdf",
        lambda: (FakePypdfReader, FakePdfReadError, "test-version"),
    )

    candidate = recover_page(
        source,
        physical_page_number=1,
        output_root=tmp_path / "recovery",
    )
    record = json.loads((candidate.output_directory / "recovery.json").read_text("utf-8"))

    assert record["candidate_status"] == "requires_review"
    assert record["automatic_indexing_eligible"] is False
    assert record["approved_for_evidence"] is False
    assert record["approved_for_canonical_replacement"] is False
    assert record["canonical_ingestion_modified"] is False
    assert record["text_quality"]["status"] == "no_detected_issues"
    assert (candidate.output_directory / "candidate.txt").read_text("utf-8") == record["text"]
    assert canonical_hashes == {
        name: file_hash(ingestion.output_directory / name)
        for name in ("document.json", "pages.jsonl", "inspection.md")
    }

    with pytest.raises(IngestionError, match="--overwrite"):
        recover_page(source, physical_page_number=1, output_root=tmp_path / "recovery")


def test_missing_optional_recovery_dependency_has_helpful_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fictional.pdf"
    make_fictional_pdf(source, blank_middle_page=False)
    monkeypatch.setitem(sys.modules, "pypdf", None)

    with pytest.raises(IngestionError, match="uv sync --extra recovery --group dev"):
        recover_page(source, physical_page_number=1, output_root=tmp_path / "recovery")
