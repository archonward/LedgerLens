# Local PDF ingestion

Milestone 1 provides a CPU-only baseline for extracting text from a user-supplied PDF. It does not
download reports, use OCR, reconstruct tables, or validate financial information.

## Add a local report

Place the PDF in `data/raw/`. Files in that directory are ignored by Git; keep the explanatory
README in place and do not commit annual reports.

### Windows PowerShell

```powershell
Copy-Item "C:\path\to\annual-report.pdf" "data\raw\annual-report.pdf"
uv run ledgerlens ingest "data/raw/annual-report.pdf" `
  --company "Example Company" `
  --report-year 2024 `
  --source-url "https://example.com/annual-report.pdf"
```

### Linux

```bash
cp /path/to/annual-report.pdf data/raw/annual-report.pdf
uv run ledgerlens ingest data/raw/annual-report.pdf \
  --company "Example Company" \
  --report-year 2024 \
  --source-url "https://example.com/annual-report.pdf"
```

The source URL is optional provenance metadata and is not fetched. Output defaults to
`data/processed/<document-id>/`, where the document ID is `sha256-` followed by the full SHA-256
content hash. Re-running the same PDF protects the existing output; use `--overwrite` only when
you intentionally want to replace that extraction.

## Outputs

### `document.json`

Document-level metadata contains:

- schema version, stable document ID, and full SHA-256 hash;
- original filename, supplied company, report year, and optional source URL;
- physical PDF page count and UTC extraction timestamp;
- extractor name and installed version;
- extraction settings and aggregated page warnings.

The report year describes the document as a whole. A page may contain comparative figures for
several years.

### `pages.jsonl`

Each UTF-8 JSON line represents one physical PDF page in source order and contains:

- schema version and document ID;
- `physical_page_number`, starting at 1;
- extracted text and its character count;
- page-specific extraction warnings.

Blank pages remain in the file. A page with no extracted text is flagged as potentially blank or
requiring OCR; this is not a conclusive scanned-page diagnosis.

### `inspection.md`

This report summarizes metadata, warnings, limitations, character counts, and a short text preview
for every page. It is an inspection aid, not a substitute for checking the source PDF.

## Inspect against the source

1. Open the source PDF and `inspection.md` side by side. Remember that LedgerLens numbers physical
   PDF pages from 1; these numbers can differ from page labels printed in the report.
2. Choose a narrative page. Compare sentences, punctuation, negative signs, headings, footnotes,
   and reading order with that physical source page. Use the full text in `pages.jsonl` when the
   preview is truncated.
3. Choose a financial-statement page. Compare row labels, column headings, year columns,
   parentheses or minus signs, footnote markers, and values. Treat apparent rows and columns as
   unverified because this milestone extracts text rather than table structure.
4. Review every warning and spot-check pages with very low character counts. Keep a record of any
   omissions, reordered text, merged columns, or broken symbols before building later stages.

## Known limitations

- `pdfplumber` works best on machine-generated PDFs. No OCR is performed.
- A no-text warning means only that text extraction returned nothing; the page may be blank,
  image-only, scanned, malformed, or otherwise difficult to extract.
- Tables are retained only as text emitted by the extractor. Cell structure, reading order, and
  visual relationships require later inspection and implementation.
- Printed page labels, document sections, reporting periods within pages, currencies, and units
  are not inferred.
- Extracted text may differ from visual layout and does not constitute validated financial data.
- Password-protected PDFs are rejected; this command does not accept or store passwords.
