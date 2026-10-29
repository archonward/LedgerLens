# Financial answer CLI

`ledgerlens answer` is a narrow user-facing adapter over `FinancialAnswerOrchestrator`. It supports
only year-over-year revenue growth and net-profit margin using local, explicitly reviewed facts
that remain bound to a validated selected-evidence export.

The command does not retrieve new facts, extract tables, infer fiscal periods, expand company
aliases, perform model arithmetic, or download a model. The question, exact company name, exact
reporting-period end date, evidence directory, and financial-facts JSONL file are all explicit.

## Usage

Ollama execution is mandatory but explicitly opted in with `--ollama`. The adapter remains fixed
to the existing `qwen3:4b-instruct` backend and the existing inference configuration of 96 maximum
new tokens and temperature 0.0. The exact tag must already be installed in the local loopback
Ollama service; LedgerLens does not pull it or fall back to another model.

```powershell
uv run ledgerlens answer "How much did NVIDIA revenue grow in FY2026?" `
  --company "NVIDIA" `
  --target-period-end 2026-01-25 `
  --evidence-directory "artifacts/evidence/sha256-<document-hash>" `
  --financial-facts "artifacts/financial-facts/sha256-<document-hash>/facts.jsonl" `
  --ollama
```

The module entry point is equivalent:

```bash
uv run python -m ledgerlens answer "What was NVIDIA's FY2026 net profit margin?" \
  --company "NVIDIA" \
  --target-period-end 2026-01-25 \
  --evidence-directory "artifacts/evidence/sha256-<document-hash>" \
  --financial-facts "artifacts/financial-facts/sha256-<document-hash>/facts.jsonl" \
  --ollama --json
```

`--target-period-end` must be an exact `YYYY-MM-DD` date. A fiscal-year label in the question is
never converted into a date. `--company` is compared using the orchestrator's case-insensitive
exact match and does not apply aliases.

## Output and exit status

The default terminal view separates the answer, selection audit, evidence citations, source and
artifact hashes, review provenance, warnings, limitations, and generation provenance. `--json`
emits the unchanged structured orchestration result, including the deterministic calculation and
complete answer-generation record.

Exit statuses are:

- `0`: an answer passed all input, calculation, and grounding checks;
- `2`: command syntax or a local evidence/fact input was invalid; and
- `3`: LedgerLens conservatively abstained.

Expected insufficiency is represented using the orchestrator's structured abstention reasons,
such as `unsupported_task`, `missing_fact`, `ambiguous_fact`, `unverified_fact`,
`invalid_or_stale_evidence`, or `answer_generation_abstained`. A downstream grounding abstention
retains the deterministic calculation and safe grounding diagnostics in JSON; it does not expose
raw model output.

Local input failures use a separate `invalid_local_input` CLI error because a malformed or missing
artifact could not be converted into a validated orchestration request.

## Tests and optional live coverage

The offline suite writes controlled evidence and financial-fact fixtures, loads them through the
production validators, and uses a mock model backend. It never contacts Ollama:

```bash
uv run pytest tests/test_cli_answer.py
```

The live CLI test is disabled unless explicitly enabled and the ignored reviewed NVIDIA artifacts
are present:

```powershell
$env:LEDGERLENS_RUN_ANSWER_CLI_OLLAMA = "1"
uv run pytest -s tests/test_cli_answer_ollama.py
```

The live test accepts either a grounded answer or the existing structured grounding abstention;
it does not weaken validation to make the model output pass.

## Limitations

- Only the two orchestration task patterns are supported.
- Local files must already conform to the selected-evidence and financial-fact schemas.
- Fact selection is exact and deterministic; duplicate candidates abstain instead of being ranked.
- Ollama is the only user-facing backend and the exact model and inference settings are fixed.
- Qualitative model text remains subject to the existing strict lexical grounding policy.
- The full Decimal result is retained because presentation rounding remains undefined.
- Reviewed NVIDIA source data, facts, model files, and generated outputs remain Git-ignored.
