# NVIDIA three-report corpus

Milestone 8A adds two independently identified and hash-bound local sources to the existing
FY2026 corpus. It does not reconcile facts across documents or verify any new financial value.
The FY2026 source, evidence, review records, financial facts, embeddings, and benchmark artifacts
were not regenerated or modified.

## Corpus inventory

| Annual review | Fiscal year end | Physical pages | SHA-256 / document ID | Preparation status |
| --- | --- | ---: | --- | --- |
| NVIDIA 2024 | January 28, 2024 | 174 | `b5751f4bb240eeb63d119211d6cc495e1d623897f1f2af1887e387bad1404c84` | Canonical evidence export prepared; primary statement pages 137-141 carry table-association limitations |
| NVIDIA 2025 | January 26, 2025 | 181 | `757b344ab62ba282f6e44e0702e8ce5f190d88fb23b5e0a2beaf4cb277a395cc` | Canonical evidence export prepared; primary statement pages 144-148 carry table-association limitations |
| NVIDIA 2026 | January 25, 2026 | Existing validated source | `0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c` | Existing reviewed FY2026 workflow; unchanged by Milestone 8A |

The document ID is `sha256-<full source hash>`. Physical page numbering starts at the first PDF
page and is intentionally independent of printed page labels.

## Quality findings

Both new PDFs are readable with the existing native pdfplumber pipeline. Every physical page was
processed. The implemented checks found no empty extraction, literal `(cid:<digits>)` token,
U+FFFD replacement character, or unexpected Unicode control on either report. A separate UTF-8
audit over every page confirmed those results.

Source-render inspection covered covers, low-text or graphic-led pages, Form 10-K identity pages,
the five primary financial-statement pages in each report, and short pages near the end. Native
text was legible and present, so targeted OCR or alternative extraction was not justified. The
financial statements remain flattened text: table geometry and exact column/value associations
are not validated by successful extraction. The affected primary statement pages carry explicit
limitations in their evidence exports.

This was AI-assisted visual and extraction-quality review, not independent human verification.
No FY2024 or FY2025 financial figure was approved, and selected evidence must not be treated as a
reviewed financial fact.

Detailed quality reports, review manifests, rendered inspection images, processed text, and
evidence exports are local generated artifacts under ignored `data/processed/`, `artifacts/`, and
`tmp/` paths. They must not be committed.

## Reproduce ingestion

First verify that the local files are the intended sources.

Windows PowerShell:

```powershell
Get-FileHash "data/raw/nvidia-2024-annual-report.pdf" -Algorithm SHA256
Get-FileHash "data/raw/nvidia-2025-Annual-Report.pdf" -Algorithm SHA256

uv run ledgerlens ingest "data/raw/nvidia-2024-annual-report.pdf" `
  --company "NVIDIA" --report-year 2024
uv run ledgerlens ingest "data/raw/nvidia-2025-Annual-Report.pdf" `
  --company "NVIDIA" --report-year 2025
```

Linux:

```bash
sha256sum data/raw/nvidia-2024-annual-report.pdf
sha256sum data/raw/nvidia-2025-Annual-Report.pdf

uv run ledgerlens ingest data/raw/nvidia-2024-annual-report.pdf \
  --company NVIDIA --report-year 2024
uv run ledgerlens ingest data/raw/nvidia-2025-Annual-Report.pdf \
  --company NVIDIA --report-year 2025
```

The output directories are derived from source hashes, and the CLI refuses to overwrite an
existing extraction unless `--overwrite` is explicitly supplied. Do not use that option for the
prepared corpus without first archiving and reviewing the baseline.

After source-render inspection, create a separate schema 1.0 review manifest for each document.
The Milestone 8A manifests select no recovery candidate and attach limitations to the five primary
statement pages. Prepare each export independently:

```powershell
$fy2024 = "sha256-b5751f4bb240eeb63d119211d6cc495e1d623897f1f2af1887e387bad1404c84"
$fy2025 = "sha256-757b344ab62ba282f6e44e0702e8ce5f190d88fb23b5e0a2beaf4cb277a395cc"

uv run ledgerlens prepare-evidence "data/processed/$fy2024" `
  --review-manifest "artifacts/reviewed-recovery/$fy2024/review-manifest.json"
uv run ledgerlens prepare-evidence "data/processed/$fy2025" `
  --review-manifest "artifacts/reviewed-recovery/$fy2025/review-manifest.json"
```

Do not silently reuse these manifests for a different source hash or add a reviewed recovery
without completing the hash-bound review workflow described in [the evidence guide](evidence.md).

## Next validation steps

1. Independently review candidate fact pages against the source rendering.
2. Bind each approved fact to its exact document identity, physical page, metric, fiscal period,
   currency, scale, sign, and table-column association.
3. Record reviewer identity and method without describing AI-assisted review as independent human
   verification.
4. Add multi-document reconciliation and orchestration only in Milestone 8B; retrieval relevance
   alone must never promote a page into a verified financial fact.
