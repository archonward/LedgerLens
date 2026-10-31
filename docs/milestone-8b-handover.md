# Milestone 8B Phase 1 handover

Milestone 8B Phase 1 reviewed NVIDIA revenue and net income across the FY2024, FY2025,
and FY2026 annual reports. It produced source-specific proposals and a reconciliation audit.
The user subsequently attested that they independently inspected and approved all 14 proposals;
separate verified outputs now record that confirmation. No calculation or multi-document
orchestration was added, and the existing FY2026 records remain unchanged.

## Current state

- Fourteen original financial-fact proposals remain unchanged with
  `verification.status: unverified`.
- Fourteen separate promoted records have `verification.status: verified` and bind the
  user-confirmation record hash in their verification notes.
- Six proposals are bound to the FY2024 report and cover FY2022-FY2024 revenue and net income.
- Six proposals are bound to the FY2025 report and cover FY2023-FY2025 revenue and net income.
- Two proposals are bound to the FY2026 report's FY2024 comparative column.
- The four existing verified FY2026-source facts for FY2025 and FY2026 were read for comparison
  and remain unchanged.
- Every proposal retains its source document ID, source and evidence hashes, physical PDF page,
  exact selected text, reporting period, USD currency, million-unit scale, sign, and review
  metadata. Proposal files reload through `load_financial_facts` only with their corresponding
  validated evidence corpus.

The proposal review was performed by the OpenAI Codex automated coding agent using source
renderings and selected evidence. Promotion records the later user attestation as independent
review. The user's personal identity and original inspection timestamp were not provided and are
not inferred; the recorded UTC timestamp is the confirmation-receipt and promotion time.

## Source pages and comparison

| Source report | Income statement | Fiscal-year support | Currency support |
| --- | ---: | ---: | ---: |
| FY2024 | physical PDF page 137 | page 142 | page 144 |
| FY2025 | physical PDF page 144 | page 149 | page 151 |
| FY2026 | physical PDF page 139 | page 144 | page 145 |

All reviewed values are positive amounts reported in USD millions. The sources describe 52-week
years ending on the last Sunday in January. Phase 1 found:

| Fiscal year | Revenue | Net income | Comparison |
| --- | ---: | ---: | --- |
| FY2022 | `$ 26,914` | `$ 9,752` | FY2024 report only; not cross-report comparable |
| FY2023 | `$ 26,974` | `$ 4,368` | FY2024 and FY2025 reports agree |
| FY2024 | `$ 60,922` | `$ 29,760` | FY2024, FY2025, and FY2026 reports agree |
| FY2025 | `$ 130,497` | `$ 72,880` | FY2025 and FY2026 reports agree |
| FY2026 | `$ 215,938` | `$ 120,067` | FY2026 report only; not cross-report comparable |

No numerical discrepancy was observed in an overlapping column. Agreement is a comparison result,
not a reason to merge records, prefer a newer report, or change verification status.

## Local artifacts

All paths below are generated, Git-ignored local artifacts and must not be committed:

- Reconciliation report:
  `artifacts/reconciliation/milestone-8b-phase1/reconciliation-report.md`
- Machine-readable review and comparison record:
  `artifacts/reconciliation/milestone-8b-phase1/review.json`
- FY2024-source proposals:
  `artifacts/reconciliation/milestone-8b-phase1/sha256-b5751f4bb240eeb63d119211d6cc495e1d623897f1f2af1887e387bad1404c84/proposed-facts.jsonl`
- FY2025-source proposals:
  `artifacts/reconciliation/milestone-8b-phase1/sha256-757b344ab62ba282f6e44e0702e8ce5f190d88fb23b5e0a2beaf4cb277a395cc/proposed-facts.jsonl`
- FY2026-source FY2024 proposals:
  `artifacts/reconciliation/milestone-8b-phase1/sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c/proposed-facts.jsonl`
- User-confirmation record:
  `artifacts/financial-facts/milestone-8b-user-confirmation.json`
- Promotion manifest:
  `artifacts/financial-facts/milestone-8b-promotion-manifest.json`
- FY2024-source verified output:
  `artifacts/financial-facts/sha256-b5751f4bb240eeb63d119211d6cc495e1d623897f1f2af1887e387bad1404c84/milestone-8b-verified-facts.jsonl`
- FY2025-source verified output:
  `artifacts/financial-facts/sha256-757b344ab62ba282f6e44e0702e8ce5f190d88fb23b5e0a2beaf4cb277a395cc/milestone-8b-verified-facts.jsonl`
- FY2026-source FY2024 verified output:
  `artifacts/financial-facts/sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c/milestone-8b-verified-facts.jsonl`

The canonical evidence corpora remain under `artifacts/evidence/<document-id>/`. Existing FY2026
facts and their review record remain under
`artifacts/financial-facts/sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c/`.

## Recorded confirmation and provenance limits

The user confirmed against the original PDF renderings:

1. The `Revenue` or `Net income` row is the intended metric.
2. The value belongs to the stated reporting-period column.
3. The sign is positive and the transcription is exact.
4. The statement scale is millions and the currency interpretation is USD.
5. The displayed year end and 52-week disclosure support the proposed inclusive period.
6. The source document, physical page, and evidence hashes still match.
7. Review metadata must distinguish user confirmation from prior Codex-assisted review.

The confirmation record truthfully identifies the reviewer only as
`LedgerLens user (personal identity not provided)`. LedgerLens did not independently authenticate
the reviewer or the inspection time. Cross-report agreement did not replace the user's checks.
FY2022 and FY2026 still have no second-report comparison within the current three-report
statements.

## Recommended next steps

1. Use only the separate verified outputs when a downstream step requires the promoted facts;
   retain the original proposal files as the immutable pre-confirmation state.
2. Verify the confirmation-record and promotion-manifest hashes before consuming a verified file.
3. Re-run `load_financial_facts` against the matching source corpus before deterministic
   calculations are enabled.
4. Add focused tests only when a reusable promotion or reconciliation workflow is implemented.
5. Design multi-document reconciliation or orchestration in a later phase without silently
   deduplicating equivalent facts or selecting sources by recency.
