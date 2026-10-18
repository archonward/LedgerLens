"""Extract one page with an optional alternative engine for manual review."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ledgerlens.ingestion.pdf import IngestionError, _publish, compute_sha256
from ledgerlens.ingestion.quality import TextQuality, assess_text_quality

RECOVERY_SCHEMA_VERSION = "1.1"
DEFAULT_OCR_DPI = 300
DEFAULT_OCR_LANGUAGE = "eng"
DEFAULT_OCR_PSM = 3
DEFAULT_OCR_OEM = 1
DEFAULT_OCR_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class RecoveryResult:
    """Summary of an alternative extraction candidate."""

    document_id: str
    physical_page_number: int
    page_count: int
    character_count: int
    quality: TextQuality
    output_directory: Path
    engine: str = "pypdf"
    ocr_word_count: int | None = None
    mean_word_confidence: float | None = None


def _load_pypdf() -> tuple[Any, type[Exception], str]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError:
        raise IngestionError(
            "The optional recovery dependency is not installed. "
            "Run 'uv sync --extra recovery --group dev' and try again."
        ) from None
    return PdfReader, PdfReadError, version("pypdf")


def _load_pdfium() -> tuple[Any, str]:
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise IngestionError(
            "The optional OCR renderer is not installed. "
            "Run 'uv sync --extra ocr --group dev' and try again."
        ) from None
    return pdfium, version("pypdfium2")


def resolve_tesseract_executable(explicit_path: Path | None = None) -> Path:
    """Resolve Tesseract without installing it or changing PATH."""
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser()
        if not candidate.is_file():
            raise IngestionError(f"Tesseract executable does not exist: '{candidate}'.")
        return candidate.resolve()

    discovered = shutil.which("tesseract")
    if discovered is None:
        raise IngestionError(
            "Tesseract was not found on PATH. Install Tesseract with English language data, "
            "then retry or pass --tesseract-path with the executable's full path."
        )
    return Path(discovered).resolve()


def _run_external(
    command: list[str], *, timeout_seconds: int, purpose: str
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            encoding="utf-8",
            errors="replace",
            shell=False,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        raise IngestionError(
            f"{purpose} timed out after {timeout_seconds} seconds. "
            "No recovery output was published."
        ) from None
    except OSError as exc:
        raise IngestionError(f"Could not start {purpose}: {exc}") from None

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if len(detail) > 500:
            detail = detail[:500] + "..."
        raise IngestionError(
            f"{purpose} failed with exit code {completed.returncode}"
            + (f": {detail}" if detail else ".")
        )
    return completed


def _render_page_with_pdfium(
    pdf_path: Path,
    *,
    physical_page_number: int,
    dpi: int,
    output_path: Path,
) -> tuple[int, str]:
    pdfium, renderer_version = _load_pdfium()
    document = None
    page = None
    bitmap = None
    try:
        document = pdfium.PdfDocument(pdf_path)
        page_count = len(document)
        if physical_page_number > page_count:
            raise IngestionError(
                f"Physical page {physical_page_number} is out of range; "
                f"the PDF has {page_count} pages."
            )
        page = document[physical_page_number - 1]
        bitmap = page.render(scale=dpi / 72)
        bitmap.to_pil().save(output_path, format="PNG")
        return page_count, renderer_version
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(
            f"Could not render physical page {physical_page_number} with pypdfium2: {exc}"
        ) from None
    finally:
        for resource in (bitmap, page, document):
            close = getattr(resource, "close", None)
            if callable(close):
                close()


def _tesseract_version(executable: Path) -> str:
    result = _run_external(
        [str(executable), "--version"],
        timeout_seconds=15,
        purpose="Tesseract version check",
    )
    first_line = (result.stdout or result.stderr).splitlines()
    return first_line[0].strip() if first_line else "unknown"


def _invoke_tesseract(
    *,
    executable: Path,
    image_path: Path,
    work_directory: Path,
    language: str,
    psm: int,
    oem: int,
    dpi: int,
    timeout_seconds: int,
) -> tuple[str, Path, int, float | None]:
    common_options = [
        "-l",
        language,
        "--oem",
        str(oem),
        "--psm",
        str(psm),
        "--dpi",
        str(dpi),
        "-c",
        "page_separator=",
    ]
    text_base = work_directory / "ocr-text"
    words_base = work_directory / "ocr-words"
    _run_external(
        [str(executable), str(image_path), str(text_base), *common_options],
        timeout_seconds=timeout_seconds,
        purpose="Tesseract text OCR",
    )
    _run_external(
        [str(executable), str(image_path), str(words_base), *common_options, "tsv"],
        timeout_seconds=timeout_seconds,
        purpose="Tesseract word-level OCR",
    )

    text_path = text_base.with_suffix(".txt")
    tsv_path = words_base.with_suffix(".tsv")
    if not text_path.is_file() or not tsv_path.is_file():
        raise IngestionError(
            "Tesseract completed without producing the expected text and TSV files."
        )

    with text_path.open("r", encoding="utf-8", newline="") as stream:
        text = stream.read()
    confidences: list[float] = []
    word_count = 0
    with tsv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row.get("level") != "5" or not row.get("text", "").strip():
                continue
            word_count += 1
            try:
                confidence = float(row.get("conf", "-1"))
            except ValueError:
                continue
            if confidence >= 0:
                confidences.append(confidence)

    mean_confidence = sum(confidences) / len(confidences) if confidences else None
    final_tsv_path = work_directory / "words.tsv"
    tsv_path.replace(final_tsv_path)
    text_path.replace(work_directory / "candidate.txt")
    return text, final_tsv_path, word_count, mean_confidence


def _candidate_record(
    *,
    document_id: str,
    full_hash: str,
    path: Path,
    physical_page_number: int,
    page_count: int,
    engine: str,
    extractor: dict[str, object],
    settings: dict[str, object],
    text: str,
    quality: TextQuality,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    candidate_warning = (
        "Alternative extraction candidate requires visual review and must not automatically "
        "replace canonical ingestion text or be treated as verified evidence."
    )
    record: dict[str, object] = {
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "document_id": document_id,
        "source_sha256": full_hash,
        "original_filename": path.name,
        "physical_page_number": physical_page_number,
        "source_page_count": page_count,
        "extracted_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "engine": engine,
        "extractor": extractor,
        "extraction_settings": settings,
        "candidate_status": "requires_review",
        "automatic_indexing_eligible": False,
        "approved_for_evidence": False,
        "approved_for_canonical_replacement": False,
        "canonical_ingestion_modified": False,
        "text": text,
        "character_count": len(text),
        "text_quality": quality.as_dict(),
        "warnings": [candidate_warning, *quality.warnings],
    }
    if extra:
        record.update(extra)
    return record


def _recover_with_pypdf(
    path: Path, *, physical_page_number: int
) -> tuple[int, str, dict[str, object], dict[str, object]]:
    PdfReader, PdfReadError, extractor_version = _load_pypdf()
    try:
        reader = PdfReader(path)
        page_count = len(reader.pages)
        if physical_page_number > page_count:
            raise IngestionError(
                f"Physical page {physical_page_number} is out of range; "
                f"the PDF has {page_count} pages."
            )
        text = reader.pages[physical_page_number - 1].extract_text() or ""
    except IngestionError:
        raise
    except PdfReadError as exc:
        raise IngestionError(f"Could not parse PDF '{path}' with pypdf: {exc}") from None
    except OSError as exc:
        raise IngestionError(f"Could not read PDF '{path}': {exc}") from None
    except Exception as exc:
        raise IngestionError(
            f"Could not recover physical page {physical_page_number} with pypdf: {exc}"
        ) from None
    return (
        page_count,
        text,
        {"name": "pypdf", "version": extractor_version},
        {
            "method": "PageObject.extract_text",
            "arguments": {},
            "physical_page_number_basis": 1,
        },
    )


def recover_page(
    pdf_path: Path,
    *,
    physical_page_number: int,
    engine: str = "pypdf",
    output_root: Path = Path("artifacts/recovery"),
    overwrite: bool = False,
    tesseract_path: Path | None = None,
    dpi: int = DEFAULT_OCR_DPI,
    language: str = DEFAULT_OCR_LANGUAGE,
    psm: int = DEFAULT_OCR_PSM,
    oem: int = DEFAULT_OCR_OEM,
    timeout_seconds: int = DEFAULT_OCR_TIMEOUT_SECONDS,
) -> RecoveryResult:
    """Extract one physical page as an unreviewed candidate."""
    path = Path(pdf_path)
    if not path.exists():
        raise IngestionError(f"PDF does not exist: '{path}'.")
    if not path.is_file():
        raise IngestionError(f"PDF path is not a file: '{path}'.")
    if physical_page_number < 1:
        raise IngestionError("Physical page number must be at least 1.")
    if engine not in {"pypdf", "tesseract"}:
        raise IngestionError(f"Unsupported recovery engine: '{engine}'.")
    if dpi < 72:
        raise IngestionError("OCR DPI must be at least 72.")
    if timeout_seconds < 1:
        raise IngestionError("OCR timeout must be at least 1 second.")

    full_hash = compute_sha256(path)
    document_id = f"sha256-{full_hash}"
    page_directory = Path(output_root) / document_id / f"page-{physical_page_number}"
    output_directory = page_directory if engine == "pypdf" else page_directory / "tesseract"
    if output_directory.exists() and not overwrite:
        raise IngestionError(
            f"Recovery output already exists at '{output_directory}'. "
            "Use --overwrite to replace it."
        )

    try:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temp_directory = Path(
            tempfile.mkdtemp(
                prefix=f".{engine}-page-{physical_page_number}.", dir=output_directory.parent
            )
        )
    except OSError as exc:
        raise IngestionError(f"Cannot prepare recovery output: {exc}") from None

    word_count: int | None = None
    mean_confidence: float | None = None
    try:
        if engine == "pypdf":
            page_count, text, extractor, settings = _recover_with_pypdf(
                path, physical_page_number=physical_page_number
            )
            extra: dict[str, object] = {}
        else:
            executable = resolve_tesseract_executable(tesseract_path)
            rendered_path = temp_directory / "rendered-page.png"
            page_count, renderer_version = _render_page_with_pdfium(
                path,
                physical_page_number=physical_page_number,
                dpi=dpi,
                output_path=rendered_path,
            )
            ocr_version = _tesseract_version(executable)
            text, _, word_count, mean_confidence = _invoke_tesseract(
                executable=executable,
                image_path=rendered_path,
                work_directory=temp_directory,
                language=language,
                psm=psm,
                oem=oem,
                dpi=dpi,
                timeout_seconds=timeout_seconds,
            )
            extractor = {"name": "tesseract", "version": ocr_version}
            settings = {
                "renderer": {"name": "pypdfium2", "version": renderer_version},
                "dpi": dpi,
                "language": language,
                "page_segmentation_mode": psm,
                "ocr_engine_mode": oem,
                "timeout_seconds_per_invocation": timeout_seconds,
                "physical_page_number_basis": 1,
            }
            extra = {
                "rendered_page_image": "rendered-page.png",
                "word_level_output": "words.tsv",
                "ocr_word_count": word_count,
                "mean_word_confidence": mean_confidence,
                "confidence_notice": "OCR confidence is not proof of correctness.",
            }

        quality = assess_text_quality(text)
        record = _candidate_record(
            document_id=document_id,
            full_hash=full_hash,
            path=path,
            physical_page_number=physical_page_number,
            page_count=page_count,
            engine=engine,
            extractor=extractor,
            settings=settings,
            text=text,
            quality=quality,
            extra=extra,
        )
        (temp_directory / "recovery.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        candidate_path = temp_directory / "candidate.txt"
        if not candidate_path.exists():
            candidate_path.write_text(text, encoding="utf-8")
        try:
            _publish(temp_directory, output_directory, overwrite)
        except OSError as exc:
            raise IngestionError(f"Could not publish recovery output: {exc}") from None
    except IngestionError:
        raise
    except OSError as exc:
        raise IngestionError(f"Could not write recovery output: {exc}") from None
    finally:
        if temp_directory.exists():
            shutil.rmtree(temp_directory, ignore_errors=True)

    return RecoveryResult(
        document_id=document_id,
        physical_page_number=physical_page_number,
        page_count=page_count,
        character_count=len(text),
        quality=quality,
        output_directory=output_directory.resolve(),
        engine=engine,
        ocr_word_count=word_count,
        mean_word_confidence=mean_confidence,
    )
