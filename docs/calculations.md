# Deterministic financial calculations

LedgerLens provides a small calculation layer in `ledgerlens.calculations`. It accepts only
financial facts paired with the selected-evidence corpus that validates them. It does not extract
facts, choose inputs, interpret metrics, or generate answers.

## Validation boundary

Wrap each input as `FactOperand(fact, corpus)`. Construction and every calculation call require
the fact to be `verified` and rerun `FinancialFact.validate_against(corpus)`. This rejects stale
evidence exports, changed selected text, missing pages, and mismatched source or review metadata
before arithmetic begins. Ambiguous and unverified facts are rejected even if they contain a
numeric value.

Compatibility rules are calculation-specific:

- `year_over_year_growth(current, prior)` requires the same company, metric, currency, and unit
  kind. Both periods must be known durations of 350-380 inclusive days, ordered as contiguous
  consecutive fiscal years. The prior base value must be nonzero.
- `profit_margin(profit, revenue)` requires the same company and resolved duration period, two
  monetary facts, and compatible currencies. Revenue must be nonzero.
- `financial_ratio(numerator, denominator, ratio_name=...)` requires the same company, resolved
  period, currency, and supported unit kind. The denominator must be nonzero. `ratio_name` labels
  the result; it does not assert that the selected metrics form a recognized accounting ratio.

Compatible facts may use different scales. Each calculation converts a normalized value to its
base magnitude using `normalized_value * units.scale` before applying the formula. Currency
conversion is not performed: currency codes must already match. Reporting-period compatibility is
based on status, type, and dates rather than display labels.

## Arithmetic and results

All arithmetic uses Python `Decimal` under a local, fixed context of 34 significant digits with
`ROUND_HALF_EVEN`, exponent bounds of -999999 and 999999, and clamp disabled. Results are not
silently rounded for display. A serialized result records:

- schema version and a content-derived SHA-256 calculation ID;
- calculation type and the exact symbolic formula;
- the computed decimal as a string;
- result kind, label, and scale;
- Decimal precision and rounding mode; and
- each input role plus the complete financial-fact snapshot, including reported value, normalized
  value, period, currency, units, verification, source hashes, physical PDF page, exact evidence,
  review metadata, warnings, and limitations.

The initial formulas are:

```text
year-over-year growth = ((current * current scale) - (prior * prior scale))
                        / (prior * prior scale) * 100

profit margin = (profit * profit scale) / (revenue * revenue scale) * 100

financial ratio = (numerator * numerator scale) / (denominator * denominator scale)
```

## Controlled example

Assume `current_fact` and `prior_fact` are controlled, verified synthetic fixtures and their
respective evidence exports have already been loaded:

```python
from ledgerlens.calculations import FactOperand, year_over_year_growth

result = year_over_year_growth(
    FactOperand(current_fact, current_corpus),
    FactOperand(prior_fact, prior_corpus),
)

print(result.computed_value)
print(result.unit.label)
payload = result.to_dict()
```

For a current synthetic value of 1.5 in millions and a prior synthetic value of 1,000 in
thousands, the base values are 1,500,000 and 1,000,000. The result is `50.0` percent. These are
controlled placeholders, not extracted company figures.

Profit margin and a named ratio use the same boundary:

```python
from ledgerlens.calculations import FactOperand, financial_ratio, profit_margin

margin = profit_margin(
    FactOperand(profit_fact, corpus),
    FactOperand(revenue_fact, corpus),
)
ratio = financial_ratio(
    FactOperand(current_assets_fact, corpus),
    FactOperand(current_liabilities_fact, corpus),
    ratio_name="current ratio",
)
```

## Current limitations

- The API is independent of the CLI and has no calculation artifact publisher or loader.
- There is no automatic fact selection, metric taxonomy, currency conversion, aggregation,
  period alignment, trailing-period construction, or multi-input formula engine.
- Profit and revenue roles, and caller-supplied ratio names, are not semantically inferred or
  checked against an accounting taxonomy.
- Year-over-year validation uses a conservative annual-duration window, contiguous dates, and a
  consecutive end-year rule; it does not model every possible fiscal-calendar transition.
- Results retain full precision from the fixed arithmetic context. Presentation rounding and
  significant-figure policies are not implemented.
- Results are report-derived calculations, not report statements or model conclusions.

See the [NVIDIA FY2026 reviewed evidence demonstration](real-financial-evidence.md) for an offline,
report-backed use of this API.

