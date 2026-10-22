# Financial fact representation

LedgerLens represents a reported financial fact separately from page retrieval, calculations, and
answer generation. The representation is deliberately conservative: it records a reviewed
interpretation but never reconstructs a table, chooses a period, assigns a currency, or infers a
scale automatically.

The implementation is in `ledgerlens.financial_facts`. It accepts a validated `EvidenceCorpus`
from the existing selected-evidence loader and binds each fact to an exact substring on one
physical PDF page.

## Model

A `FinancialFact` records:

- a stable fact ID, metric name, exact `reported_value`, and optional decimal
  `normalized_value`;
- company and an explicit reporting period;
- currency plus unit kind, label, and scale;
- the table-column association, including `not_applicable` for prose evidence;
- source document identity, company, report year, and SHA-256; physical PDF page; exact supporting
  text and its hash; selected-page text hash; evidence export hashes; and extraction method;
- inherited screening warnings, evidence limitations, and recovery review provenance;
- fact-level warnings and limitations; and
- verification status and, when reviewed, reviewer, timestamp, method, and note.

Decimals serialize as strings. `normalized_value` is the locale-independent numeric form of the
reported value in the **reported unit**. The unit scale is retained separately and is not applied.
For example, reported `"$60,922"` with normalized value `"60922"` and scale `"1000000"` means
60,922 units of USD millions. Producing a base-unit value would be a calculation and is outside
this milestone.

## Resolution and verification

Reporting period, currency, units, and table-column association carry a `status` rather than using
an unexplained null:

- `known` means an interpretation was explicitly selected and its required fields are present;
- `missing` means the selected evidence does not provide it;
- `ambiguous` means multiple interpretations remain plausible; and
- `not_applicable` is allowed only for currency or table-column association where the concept
  genuinely does not apply.

Unresolved fields require a note, must not contain a silently selected value, and require a
fact-level warning or limitation. A `verified` fact requires a normalized value, complete human
review metadata, a known reporting period and units, and no missing or ambiguous critical context.
Automatic text-quality screening and reviewed OCR selection do not verify a financial fact.

The other fact states are:

- `ambiguous`: review found a material unresolved interpretation; and
- `unverified`: the record has not received the review needed to claim correctness.

`reviewed_by` identifies the actual reviewer; it does not imply that the reviewer is human or
independent. Automated-agent review must say so explicitly in the reviewer identity, method, and
note. Where that distinction matters, keep a separate review record with a structured
`independent_human_verification_claimed` field and bind its hash into the fact metadata. See the
[NVIDIA FY2026 reviewed evidence demonstration](real-financial-evidence.md).

## Evidence binding

Use `SupportingEvidence.from_evidence(corpus, page, supporting_text)` to require that supporting
text occurs verbatim in selected page text. Call `fact.validate_against(corpus)` before consuming a
fact. It rejects a different document, source hash, export hash, page, company, extraction method,
selected-text hash, or supporting excerpt.

`load_financial_facts(path, corpus)` reads schema-version `1.0` JSONL, rejects blank or malformed
records and duplicate fact IDs, reconstructs decimal values without floats, and validates every
record against the selected-evidence corpus.

```python
from datetime import date
from decimal import Decimal

from ledgerlens.financial_facts import (
    Currency,
    FinancialFact,
    PeriodType,
    ReportingPeriod,
    ResolutionStatus,
    SupportingEvidence,
    TableColumn,
    UnitKind,
    Units,
    Verification,
    VerificationStatus,
)

page = corpus.pages[0]  # corpus was returned by load_evidence_export(...)
evidence = SupportingEvidence.from_evidence(
    corpus, page, "Year Ended December 31, 2025\nRevenue $1,234"
)
fact = FinancialFact(
    fact_id="example-2025-revenue",
    metric="revenue",
    reported_value="$1,234",
    normalized_value=Decimal("1234"),
    company=page.company,
    reporting_period=ReportingPeriod(
        status=ResolutionStatus.KNOWN,
        period_type=PeriodType.DURATION,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
        label="Year Ended December 31, 2025",
    ),
    currency=Currency(status=ResolutionStatus.KNOWN, code="USD"),
    units=Units(
        status=ResolutionStatus.KNOWN,
        kind=UnitKind.MONETARY,
        scale=Decimal("1000000"),
        label="USD in millions",
    ),
    table_column=TableColumn(
        status=ResolutionStatus.KNOWN,
        label="Year Ended December 31, 2025",
    ),
    evidence=evidence,
    verification=Verification(
        status=VerificationStatus.VERIFIED,
        reviewed_by="reviewer-id",
        reviewed_at_utc="2026-10-22T02:00:00Z",
        method="manual comparison with the rendered source page",
    ),
)
fact.validate_against(corpus)
payload = fact.to_dict()
```

The numbers above are illustrative placeholders, not extracted financial evidence. The offline
tests use a controlled NVIDIA-shaped fixture to exercise provenance and ambiguity handling; they
do not assert that bundled report evidence exists.

## Current limitations

- There is no automatic extraction or table reconstruction.
- The model stores one exact excerpt on one physical page; multi-page support is not implemented.
- Metric names are project-controlled strings, not a financial taxonomy.
- ISO currency codes are structurally checked but not looked up against an external registry.
- Normalization is represented and validated, but LedgerLens does not yet perform it.
- Facts are loaded and validated as JSONL; publishing a separate atomic fact artifact and CLI
  commands are not yet implemented.
- The initial deterministic calculation layer consumes verified facts; automatic input selection
  and question answering remain unimplemented.

