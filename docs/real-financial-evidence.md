# NVIDIA FY2026 reviewed financial evidence demonstration

This milestone demonstrates the financial-fact and deterministic-calculation layers against a
small, genuinely report-backed set of NVIDIA facts. The source PDF, selected evidence, reviewed
fact records, review record, and calculation output remain local under Git-ignored `data/` and
`artifacts/` paths. No report content or generated financial artifact is committed.

## Source and review scope

The source is NVIDIA's FY2026 annual report with document ID
`sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c`.
The selected-evidence export used by the demonstration has these hashes:

- `evidence.jsonl`: `adaebb1e8bcafa49e403608a9644e7620ee4ca27f5206ec12c9e72d1d1d57f77`
- `manifest.json`: `a6058cfcf64d44de35af4aac466da418a361b0187422a1993aad69db2969ee25`

Reviewer identity: **OpenAI Codex automated coding agent**.

Review method: Codex visually inspected original PDF renderings and compared physical page 139
with its exact selected-evidence text. Supporting pages were inspected to avoid silently inferring
period length or currency. This was Codex-assisted review requested by the user; it was **not
independent human verification**, and no claim of independent human approval is made.

The reviewed physical PDF pages were:

| Physical page | Printed page | Review purpose |
| ---: | ---: | --- |
| 22 | 3 | Defines Fiscal 20__ as ending on the last Sunday in January of the stated year. |
| 69 | 50 | States that Fiscal 2026, 2025, and 2024 were 52-week years. |
| 139 | 51 | Consolidated Statements of Income; values, units, periods, and column geometry. |
| 145 | 57 | States the U.S.-dollar functional-currency and remeasurement policy. |

The ignored `review.json` records the renderer, per-page render hashes, observations, derivation
method, reviewer identity, timestamp, and limitations. Its SHA-256 is
`79ace3d1e3282991aa11d2989830cfafc5e1408db2f513821f49ee32c0cc99e5`. Every fact's verification
note includes that hash, so substituting a changed review record causes the demonstration to fail.

## Reviewed facts

All four values occur on physical PDF page 139, printed page 51, in the Consolidated Statements of
Income. The page labels the table `(In millions, except per share data)`. Visual inspection
confirmed that each value is under the stated year-end column.

| Fact ID | Metric | Period | Exact reported value | Normalized value | Currency and scale |
| --- | --- | --- | ---: | ---: | --- |
| `nvidia-fy2026-revenue` | Revenue | Year ended Jan. 25, 2026 | `$ 215,938` | `215938` | USD millions |
| `nvidia-fy2025-revenue` | Revenue | Year ended Jan. 26, 2025 | `$ 130,497` | `130497` | USD millions |
| `nvidia-fy2026-net-income` | Net income | Year ended Jan. 25, 2026 | `$ 120,067` | `120067` | USD millions |
| `nvidia-fy2025-net-income` | Net income | Year ended Jan. 26, 2025 | `$ 72,880` | `72880` | USD millions |

The table prints year-end dates, not start dates. Start dates are recorded as explicit derivations,
not extracted claims: each reviewed year is 52 weeks, so the inclusive start date is 363 days
before its displayed end date. This yields Jan. 27, 2025 for FY2026 and Jan. 29, 2024 for FY2025.

The ignored `facts.jsonl` has SHA-256
`33584ce1e3537c177d8704b65c3fa3c4992e32d0d41ad19bcb4a27179c998a68`. Loading it through
`load_financial_facts` validates every fact against the unchanged selected-evidence export before
calculation.

## Reproduced calculations

The demonstration uses the existing `year_over_year_growth` and `profit_margin` functions without
modifying their formulas or validation rules.

| Calculation | Reproducible Decimal result | Readable approximation |
| --- | ---: | ---: |
| FY2026 revenue growth versus FY2025 | `65.47353579009479145114447075411695%` | 65.47% |
| FY2026 net profit margin | `55.60253406070260908223656790375015%` | 55.60% |
| FY2025 net profit margin | `55.84802715771243783381993455788256%` | 55.85% |

The readable approximations are explanatory only. LedgerLens retains the full result produced by
its fixed Decimal context and does not yet implement a presentation-rounding policy.

The ignored demonstration output is `artifacts/calculations/nvidia-fy2026-reviewed.json`, with
SHA-256 `bcfe9a13b7f288e98cdbd7192d4cb0addd6b2aba0bc2bf82d76017a5fb8b319c`. It embeds the complete
fact snapshots, exact formulas, evidence hashes, physical page, review metadata, arithmetic
settings, and content-derived calculation IDs.

## Reproduce locally

The source report, selected-evidence export, reviewed facts, and review record must already exist
in their ignored local paths. The script never modifies those inputs:

```powershell
uv run python scripts/demonstrate_nvidia_fy2026.py
```

To publish the deterministic JSON under ignored artifacts:

```powershell
uv run python scripts/demonstrate_nvidia_fy2026.py `
  --output artifacts/calculations/nvidia-fy2026-reviewed.json `
  --overwrite
```

The focused integration test runs completely offline. It skips when the ignored local report
artifacts are absent, and otherwise verifies the exact facts, review identity, provenance, and
calculation results:

```powershell
uv run pytest tests/test_real_evidence_integration.py
```

## Limitations and API findings

- The review was performed by Codex and is not independent human verification.
- Only four values and the context needed for three calculations were reviewed. No conclusion is
  made about other report values or pages.
- `SupportingEvidence` currently binds one physical page. Period-length and currency support from
  pages 22, 69, and 145 therefore live in a separate review record whose hash is carried in the
  fact verification note. Native multi-page fact evidence would make this relationship stronger.
- `Verification` records reviewer identity, method, timestamp, and note as free text but has no
  structured reviewer-type or `independent_human_verification_claimed` field. The local review
  record supplies that missing distinction explicitly.
- Financial facts have a JSONL loader but no first-class atomic publisher or fact-set manifest.
  The ignored review record and facts file are milestone-specific artifacts.
- Calculation results have structured serialization but no loader or atomic publisher. The
  demonstration script supplies a protected optional output path.
- The script intentionally uses fixed reviewed fact IDs; it performs no extraction, input
  discovery, metric interpretation, or answer generation.

