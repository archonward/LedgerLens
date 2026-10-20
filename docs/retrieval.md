# Page-level BM25 retrieval

LedgerLens implements a small, CPU-only lexical retrieval baseline over a selected-evidence
export. It does not read the PDF, canonical extraction, archives, or recovery candidate folders.
The evidence-selection step remains the only place that chooses between canonical and explicitly
reviewed recovery text.

## Why BM25 first

BM25 rewards query terms that occur on a page, gives more weight to rarer terms, accounts for term
frequency, and normalizes for page length. It is deterministic, needs no model download or GPU, and
provides a useful baseline against which a later embedding retriever can be evaluated. A high score
means a stronger lexical match within this corpus. It is not a probability, confidence estimate,
financial correctness assessment, or evidence verification result.

## Run a query

Windows PowerShell:

```powershell
uv run ledgerlens retrieve `
  "artifacts/evidence/sha256-<hash>" `
  --query "What drove Data Center revenue growth?" `
  --top-k 5
```

Linux:

```bash
uv run ledgerlens retrieve \
  "artifacts/evidence/sha256-<hash>" \
  --query "What drove Data Center revenue growth?" \
  --top-k 5
```

`uv run python -m ledgerlens retrieve ...` is equivalent. Add `--json` for machine-readable output.
If no page has a positive lexical score, the command reports an explicit no-match outcome instead
of returning arbitrary zero-score pages.

Each result includes its rank, BM25 score, document ID, physical PDF page number, selected
extraction method, exact-text excerpt, selected-text and source hashes, screening warnings, known
limitations, and review provenance when present. Physical PDF page numbers can differ from printed
page labels. The excerpt is a bounded substring of the selected page text and is never generated or
paraphrased.

## Input validation

The loader requires `evidence.jsonl` and `manifest.json` from `prepare-evidence`. It verifies:

- manifest, record, document, and source identities;
- every selected-text SHA-256 hash;
- unique records in physical-page order;
- selection-decision method, hash, and limitation agreement;
- selected, recovery, exclusion, and limitation counts; and
- that an excluded page is not present in selected evidence.

These checks detect malformed, stale, or internally inconsistent exports. Hash agreement proves
file identity, not extraction completeness or financial accuracy. Candidate files are never
discovered or approved merely because they exist.

## Tokenisation and scoring

The baseline uses the standard Okapi BM25 formula with `k1 = 1.5` and `b = 0.75`. Tokenisation is
Unicode-aware and case-insensitive. Underscores and surrounding punctuation delimit tokens;
letters and numbers stay indexable, internal periods and commas are retained, and commas are
removed from all-numeric tokens so `1,250` matches `1250`. There is no stemming, stop-word removal,
synonym expansion, financial interpretation, query rewriting, text splitting, or table
reconstruction. Repeated query terms contribute repeatedly. Score ties are resolved by ascending
physical page number.

There is one retrieval document per selected physical page. Empty pages and empty queries have no
indexable match. Pages excluded during evidence selection are absent from the index. Reviewed OCR
text is searchable only when already present in the selected-evidence export, and its review record
and limitations travel with every result.

## Current limitations and evaluation

Lexical retrieval can miss conceptually related wording and can over-rank repeated boilerplate. A
page is a coarse retrieval unit, and extracted table reading order may obscure relationships among
labels, columns, signs, periods, currencies, and units. A matched page must still be inspected
before using its contents, especially when it carries a limitation. Retrieval performs no
calculation and makes no numerical conclusion.

No benchmark quality claim has been made. The implemented [evaluation framework](evaluation.md)
measures the baseline on separately supplied, manually verified page judgments while excluding
provisional questions from official metrics. A later embedding baseline can be compared on the
same held-out split while preserving the selected evidence and provenance contract.
