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
from ledgerlens.ingestion import pdf as pdf_module
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
    assert metadata["schema_version"] == "1.2"
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


def test_quality_detection_counts_replacement_and_unexpected_controls_without_mutation() -> None:
    source = "Heading\tvalue\r\nBad replacement: \ufffd controls: \x00\x0b"

    quality = assess_text_quality(source)

    assert quality.replacement_character_count == 1
    assert quality.unexpected_control_character_count == 2
    assert quality.requires_review is True
    assert source == "Heading\tvalue\r\nBad replacement: \ufffd controls: \x00\x0b"


def test_quality_detection_allows_ordinary_whitespace_and_preserves_source() -> None:
    source = "Heading\tvalue\r\nSecond line"

    quality = assess_text_quality(source)

    assert quality.replacement_character_count == 0
    assert quality.unexpected_control_character_count == 0
    assert quality.status == "no_detected_issues"
    assert source == "Heading\tvalue\r\nSecond line"


def test_expanded_quality_findings_surface_without_changing_page_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fictional.pdf"
    source.write_bytes(b"fictional local PDF bytes")
    extracted = "Exact\ttext\r\nReplacement \ufffd and control \x00"

    class FakePage:
        def extract_text(self, **_: object) -> str:
            return extracted

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self) -> FakePdf:
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(pdf_module.pdfplumber, "open", lambda _: FakePdf())

    result = ingest_pdf(source, company="Fictional", report_year=2026, output_root=tmp_path / "out")
    metadata = json.loads((result.output_directory / "document.json").read_text("utf-8"))
    page = read_jsonl(result.output_directory / "pages.jsonl")[0]
    inspection = (result.output_directory / "inspection.md").read_text("utf-8")

    assert page["text"] == extracted
    assert page["text_quality"]["replacement_character_count"] == 1
    assert page["text_quality"]["unexpected_control_character_count"] == 1
    assert metadata["quality_summary"]["replacement_characters"]["total_count"] == 1
    assert metadata["quality_summary"]["unexpected_control_characters"]["total_count"] == 1
    assert result.replacement_character_findings == ((1, 1),)
    assert result.unexpected_control_character_findings == ((1, 1),)
    assert "Pages with Unicode replacement characters: 1 (1 character)" in inspection
    assert "Pages with unexpected control characters: 1 (1 character)" in inspection


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


def test_tesseract_executable_resolution_supports_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"fictional executable")
    monkeypatch.setattr(shutil, "which", lambda _: None)

    assert recovery_module.resolve_tesseract_executable(executable) == executable.resolve()

    with pytest.raises(IngestionError, match="does not exist"):
        recovery_module.resolve_tesseract_executable(tmp_path / "missing.exe")


def test_missing_tesseract_has_helpful_error_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fictional.pdf"
    make_fictional_pdf(source, blank_middle_page=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    output_root = tmp_path / "recovery"

    with pytest.raises(IngestionError, match="Install Tesseract.*--tesseract-path"):
        recover_page(
            source,
            physical_page_number=1,
            engine="tesseract",
            output_root=output_root,
        )

    assert not list(output_root.rglob("recovery.json"))


def test_pdfium_renderer_uses_requested_physical_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    accessed: list[int] = []
    rendered_scales: list[float] = []

    class FakeImage:
        def save(self, path: Path, *, format: str) -> None:
            assert format == "PNG"
            path.write_bytes(b"png")

    class FakeBitmap:
        def to_pil(self) -> FakeImage:
            return FakeImage()

        def close(self) -> None:
            pass

    class FakePage:
        def render(self, *, scale: float) -> FakeBitmap:
            rendered_scales.append(scale)
            return FakeBitmap()

        def close(self) -> None:
            pass

    class FakeDocument:
        def __init__(self, _: Path) -> None:
            pass

        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> FakePage:
            accessed.append(index)
            return FakePage()

        def close(self) -> None:
            pass

    class FakePdfium:
        PdfDocument = FakeDocument

    monkeypatch.setattr(recovery_module, "_load_pdfium", lambda: (FakePdfium, "test-version"))
    output = tmp_path / "page.png"

    page_count, renderer_version = recovery_module._render_page_with_pdfium(
        tmp_path / "source.pdf",
        physical_page_number=4,
        dpi=300,
        output_path=output,
    )

    assert page_count == 5
    assert renderer_version == "test-version"
    assert accessed == [3]
    assert rendered_scales == [pytest.approx(300 / 72)]
    assert output.read_bytes() == b"png"


def test_external_ocr_failure_and_timeout_are_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args[0], 1, stdout="", stderr="fictional OCR failure")

    monkeypatch.setattr(subprocess, "run", failed_run)
    with pytest.raises(IngestionError, match="exit code 1: fictional OCR failure"):
        recovery_module._run_external(["tesseract"], timeout_seconds=2, purpose="OCR")

    def timed_out_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args[0], 2)

    monkeypatch.setattr(subprocess, "run", timed_out_run)
    with pytest.raises(IngestionError, match="timed out after 2 seconds"):
        recovery_module._run_external(["tesseract"], timeout_seconds=2, purpose="OCR")


def test_tesseract_invocation_preserves_exact_text_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_bytes = b"Line 1\r\nLine 2\n"

    def fake_run_external(
        command: list[str], *, timeout_seconds: int, purpose: str
    ) -> subprocess.CompletedProcess[str]:
        assert timeout_seconds == 30
        output_base = Path(command[2])
        if command[-1] == "tsv":
            output_base.with_suffix(".tsv").write_text(
                "level\tconf\ttext\n5\t94.5\tLine\n", encoding="utf-8"
            )
        else:
            output_base.with_suffix(".txt").write_bytes(source_bytes)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(recovery_module, "_run_external", fake_run_external)
    image = tmp_path / "page.png"
    image.write_bytes(b"png")

    text, words_path, word_count, confidence = recovery_module._invoke_tesseract(
        executable=tmp_path / "tesseract.exe",
        image_path=image,
        work_directory=tmp_path,
        language="eng",
        psm=3,
        oem=1,
        dpi=300,
        timeout_seconds=30,
    )

    assert text == "Line 1\r\nLine 2\n"
    assert (tmp_path / "candidate.txt").read_bytes() == source_bytes
    assert words_path == tmp_path / "words.tsv"
    assert word_count == 1
    assert confidence == 94.5


def test_tesseract_candidate_is_separate_protected_and_preserves_canonical(
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
    executable = tmp_path / "tesseract.exe"
    executable.write_bytes(b"fictional executable")

    def fake_render(
        pdf_path: Path, *, physical_page_number: int, dpi: int, output_path: Path
    ) -> tuple[int, str]:
        assert pdf_path == source
        assert physical_page_number == 1
        assert dpi == 300
        output_path.write_bytes(b"fictional png")
        return 2, "test-pdfium"

    def fake_ocr(**kwargs: object) -> tuple[str, Path, int, float]:
        assert kwargs["executable"] == executable.resolve()
        assert kwargs["language"] == "eng"
        assert kwargs["psm"] == 3
        assert kwargs["oem"] == 1
        work_directory = Path(str(kwargs["work_directory"]))
        words = work_directory / "words.tsv"
        words.write_text("level\tconf\ttext\n5\t96\tRevenue\n", encoding="utf-8")
        return "OCR candidate text -75\n", words, 4, 91.5

    monkeypatch.setattr(recovery_module, "_render_page_with_pdfium", fake_render)
    monkeypatch.setattr(recovery_module, "_tesseract_version", lambda _: "5.test")
    monkeypatch.setattr(recovery_module, "_invoke_tesseract", fake_ocr)

    candidate = recover_page(
        source,
        physical_page_number=1,
        engine="tesseract",
        tesseract_path=executable,
        output_root=tmp_path / "recovery",
    )
    record = json.loads((candidate.output_directory / "recovery.json").read_text("utf-8"))

    assert candidate.output_directory.name == "tesseract"
    assert record["engine"] == "tesseract"
    assert record["extraction_settings"]["dpi"] == 300
    assert record["ocr_word_count"] == 4
    assert record["approved_for_evidence"] is False
    assert (candidate.output_directory / "rendered-page.png").is_file()
    assert (candidate.output_directory / "words.tsv").is_file()
    assert (candidate.output_directory / "candidate.txt").read_text("utf-8") == (
        "OCR candidate text -75\n"
    )
    with pytest.raises(IngestionError, match="--overwrite"):
        recover_page(
            source,
            physical_page_number=1,
            engine="tesseract",
            tesseract_path=executable,
            output_root=tmp_path / "recovery",
        )
    assert canonical_hashes == {
        name: file_hash(ingestion.output_directory / name)
        for name in ("document.json", "pages.jsonl", "inspection.md")
    }


def test_unknown_recovery_engine_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "fictional.pdf"
    make_fictional_pdf(source, blank_middle_page=False)

    with pytest.raises(IngestionError, match="Unsupported recovery engine"):
        recover_page(
            source,
            physical_page_number=1,
            engine="unknown",
            output_root=tmp_path / "recovery",
        )
