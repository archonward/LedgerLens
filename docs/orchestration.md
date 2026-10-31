# Deterministic retrieval-to-answer orchestration

LedgerLens provides a narrow Python orchestration layer for two supported financial questions:

- year-over-year revenue growth; and
- net-profit margin for one reporting period.

The coordinator does not extract facts, reconstruct tables, search for new values, ask a model to
select inputs, or perform model-generated arithmetic. It consumes an already validated
selected-evidence corpus and a caller-supplied collection of reviewed `FinancialFact` records.

## Explicit request scope

An `OrchestrationRequest` requires the question, exact company name, exact target-period end date,
reviewed facts, and the evidence corpus. The exact date is deliberately caller supplied: the
coordinator does not silently translate a fiscal-year phrase into a reporting period.

```python
from ledgerlens.orchestration import (
    FinancialAnswerOrchestrator,
    OrchestrationRequest,
)

orchestrator = FinancialAnswerOrchestrator(answer_generator)
result = orchestrator.answer(
    OrchestrationRequest(
        question="How much did NVIDIA revenue grow in FY2026?",
        company="NVIDIA",
        target_period_end=date(2026, 1, 25),
        facts=reviewed_facts,
        corpus=evidence_corpus,
    )
)
```

Task recognition is deterministic and limited to the existing revenue-growth and net-profit-margin
vocabularies. A question matching neither task abstains as unsupported; one matching both abstains
as ambiguous.

## Fact selection and calculation

Selection uses exact, auditable rules:

- the company must match the explicit request case-insensitively, with no alias expansion;
- the metric must be exactly `revenue` or `net_income` as required by the task;
- the current fact must have a known period ending on the supplied date;
- revenue growth selects the revenue fact whose period ends one day before the current period
  starts; and
- profit margin selects net income and revenue with the same supplied period end.

Every role records its metric, period rule, all candidate fact IDs, selected fact ID, and outcome.
The selected facts are then required to be verified, revalidated against the unchanged evidence
corpus, and passed through the existing `FactOperand` and Decimal calculation functions. The
existing answer generator revalidates and reproduces the calculation again before inference.

This deliberately duplicates validation at trust boundaries. Selection does not make evidence or
calculation validation optional.

## Structured abstention

The coordinator returns `OrchestrationResult` rather than throwing for expected insufficiency. Its
structured reasons cover:

- unsupported or ambiguous tasks;
- duplicate fact IDs;
- missing, ambiguous, or unresolved fact matches;
- unverified facts;
- invalid or stale evidence;
- calculation incompatibility; and
- downstream answer-generation abstention.

If model output fails grounding, the result retains the selected fact IDs, deterministic
calculation, complete generated-answer abstention, and grounding diagnostics. No source-reported or
computed value is replaced by model output.

## NVIDIA integration

The reviewed NVIDIA FY2026 integration verifies that orchestration selects:

- `nvidia-fy2026-revenue` and `nvidia-fy2025-revenue` for revenue growth; and
- `nvidia-fy2026-net-income` and `nvidia-fy2026-revenue` for FY2026 net-profit margin.

It reproduces the existing exact Decimal results and page-139 citations. The test skips when the
Git-ignored reviewed artifacts are absent. A separately opt-in Ollama integration uses the existing
model and inference settings:

```powershell
$env:LEDGERLENS_RUN_ORCHESTRATION_OLLAMA = "1"
uv run pytest -s tests/test_orchestration_ollama.py
```

The default suite uses controlled mock backends and never contacts Ollama or executes a model.

## Limitations

- Only two task patterns and exact metric names are supported.
- Company aliases, period inference, quarter selection, and multi-document fact stores are not
  implemented.
- The fact collection is supplied explicitly. The optional
  [retrieval association layer](retrieval-fact-association.md) can restrict it to facts whose exact
  selected-evidence identities were retrieved, but ranked retrieval never discovers new facts.
- Duplicate valid facts cause abstention rather than preference or ranking.
- The narrow [`ledgerlens answer`](answer-cli.md) command is a direct adapter over this API; no
  conversational routing is provided.
