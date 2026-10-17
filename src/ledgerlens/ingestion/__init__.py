"""PDF text extraction, quality checks, and candidate page recovery."""

from ledgerlens.ingestion.pdf import ExtractionResult, IngestionError, ingest_pdf
from ledgerlens.ingestion.quality import TextQuality, assess_text_quality
from ledgerlens.ingestion.recovery import RecoveryResult, recover_page

__all__ = [
    "ExtractionResult",
    "IngestionError",
    "RecoveryResult",
    "TextQuality",
    "assess_text_quality",
    "ingest_pdf",
    "recover_page",
]
