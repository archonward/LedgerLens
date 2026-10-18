# Reviewed recovery and evidence selection

LedgerLens keeps four representations separate:

1. **Canonical extraction** is the unchanged `document.json`, `pages.jsonl`, and `inspection.md`
   produced by `ingest`. Its page screening controls default automatic-indexing eligibility.
2. **Recovery candidates** are immutable alternative outputs such as Tesseract text, rendering,
   metadata, and word-level TSV. Their existence never grants permission to select them.
3. **Reviewed derivatives** contain a narrowly corrected copy plus a review record. The original
   candidate remains unchanged. Review records describe who or what reviewed the candidate, the
   authorized scope, the exact correction, and known limitations.
4. **Selected evidence** contains at most one text representation per physical page. It uses
   eligible canonical text unless an explicit manifest selects a validated reviewed derivative.

These artifacts are ignored by Git because they contain derived report content.

## Review record

`review.json` uses schema version `1.0` and records:

- document ID, source PDF SHA-256, and physical page;
- original candidate, recovery metadata, and corrected-text hashes;
- one exact expected phrase, its correction, occurrence count, and reason;
- creation timestamp, review provenance, and review scope;
- explicit permission for retrieval-prototype use;
- known limitations and exclusions from approval.

`create_reviewed_recovery` requires the expected phrase to occur exactly once. The corrected text
must equal the original candidate with only that single replacement. An absent or ambiguous phrase
fails without publishing output.

Conversational review by an AI assistant followed by user authorization must be recorded as such.
It must not be described as independent human verification.

## Review selection manifest

A manifest uses this shape:

```json
{
  "schema_version": "1.0",
  "document_id": "sha256-<source-hash>",
  "source_sha256": "<source-hash>",
  "reviewed_recoveries": [
    {
      "physical_page_number": 128,
      "selected_for_retrieval_prototype": true,
      "corrected_text_path": "page-128/tesseract/corrected.txt",
      "review_record_path": "page-128/tesseract/review.json",
      "original_candidate_path": "../../recovery/<document-id>/page-128/tesseract/candidate.txt",
      "recovery_metadata_path": "../../recovery/<document-id>/page-128/tesseract/recovery.json",
      "original_candidate_sha256": "<sha256>",
      "recovery_metadata_sha256": "<sha256>",
      "corrected_text_sha256": "<sha256>",
      "review_record_sha256": "<sha256>"
    }
  ],
  "page_limitations": [
    {
      "physical_page_number": 153,
      "limitations": [
        "Table-value associations require source review."
      ]
    }
  ]
}
```

Relative paths are resolved from the manifest's directory. Absolute paths are also accepted.
LedgerLens validates the manifest, review record, recovery metadata, and actual file hashes. It
rejects changed artifacts, mismatched identity or page provenance, missing prototype permission,
and duplicate or conflicting selections. It never scans recovery directories to infer approval.

Page limitations do not exclude otherwise eligible text. They remain attached to the selected
page record and export manifest.

## Prepare evidence

Windows PowerShell:

```powershell
uv run ledgerlens prepare-evidence `
  "data/processed/sha256-<source-hash>" `
  --review-manifest "artifacts/reviewed-recovery/sha256-<source-hash>/review-manifest.json"
```

Linux:

```bash
uv run ledgerlens prepare-evidence \
  "data/processed/sha256-<source-hash>" \
  --review-manifest \
  "artifacts/reviewed-recovery/sha256-<source-hash>/review-manifest.json"
```

Output defaults to `artifacts/evidence/<document-id>/`:

- `evidence.jsonl` contains selected pages in physical page order.
- `manifest.json` records canonical and review-manifest input hashes, each selection decision,
  selected-text hashes, limitations, and exclusions.

Existing output is protected unless `--overwrite` is supplied. Publication is atomic, and the
command does not modify canonical extraction, recovery candidates, reviewed derivatives, or the
review manifest.

## Prototype limitations

- Passing automatic screening is not verification.
- Conversational review and user authorization are not independent human validation.
- Flattened table headers can make column associations ambiguous.
- Selected OCR may support page-level retrieval while remaining unsuitable for unattended
  numerical interpretation or structured financial calculations.
- Sampling cannot establish that every page in a report is accurate.
