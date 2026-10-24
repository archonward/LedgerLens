# Evidence-grounded financial answer generation

LedgerLens can now produce a narrowly scoped answer from explicitly supplied, verified financial
facts and an already-computed deterministic result. The initial tasks are year-over-year revenue
growth and net profit margin. There is no fact extraction, result selection, retrieval orchestration,
or conversational interface in this layer.

## Trust boundary

`AnswerGenerator` treats model text as untrusted. Before inference it:

1. revalidates both facts against the selected-evidence corpus;
2. requires the facts to be verified and to exactly match the immutable snapshots in the result;
3. checks that fact roles and metrics match the explicitly selected `AnswerTask`; and
4. recomputes the calculation and requires the complete serialized result to match.

Deterministic code then owns all reported values, computed values, formulas, units, and physical
PDF page markers. The model is asked only for one qualitative sentence. Its response must contain
no digits, currency or percentage symbols, citation syntax, causal claims, forecasts, or words
outside a small task-specific vocabulary. It must also state the direction supported by a growth
result, or the supported income-to-revenue relationship for a margin. A failed input check, backend
failure, or rejected explanation produces a structured abstention with no financial values or
citations in the displayed answer.

This lexical policy is intentionally restrictive. It is a safety boundary for the first milestone,
not a general semantic-entailment system.

## Local model interface

`LocalModelBackend` is the dependency-free protocol. A backend exposes a `ModelIdentity` and a
`generate(prompt, config)` method. `CallableLocalModelBackend` adapts an in-process callable from a
locally installed open-model runtime without making that runtime a core dependency:

```python
from ledgerlens.generation import (
    AnswerGenerator,
    CallableLocalModelBackend,
    InferenceConfig,
    ModelIdentity,
)


def run_local_model(prompt: str, config: InferenceConfig) -> str:
    # Call a locally installed CPU-compatible runtime here. Loading models and choosing
    # runtime-specific decoding arguments stay outside LedgerLens's validation boundary.
    return local_runtime.generate(
        prompt,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )


backend = CallableLocalModelBackend(
    identity=ModelIdentity(
        backend="local-runtime-name",
        model="open-model-name",
        revision="pinned-revision-or-file-hash",
    ),
    generate_text=run_local_model,
)
generator = AnswerGenerator(
    backend,
    inference_config=InferenceConfig(max_new_tokens=96, temperature=0.0),
)
```

LedgerLens does not install, download, or initialize a model. Pinning a model revision or local
model-file hash remains the caller's responsibility. Offline tests use mock backends and never
execute a model, inspect credentials, use a GPU, or access the network.

With a validated `corpus` and indexed `facts`, the revenue-growth call is explicit:

```python
from ledgerlens.calculations import FactOperand, year_over_year_growth
from ledgerlens.generation import AnswerRequest, AnswerTask

current = facts["nvidia-fy2026-revenue"]
prior = facts["nvidia-fy2025-revenue"]
growth = year_over_year_growth(FactOperand(current, corpus), FactOperand(prior, corpus))
answer = generator.generate(
    AnswerRequest(
        question="How much did NVIDIA revenue grow in FY2026?",
        task=AnswerTask.REVENUE_GROWTH,
        facts=(current, prior),
        calculation=growth,
        corpus=corpus,
    )
)
```

## Answer request and result

The caller explicitly constructs an `AnswerRequest` with a question, `AnswerTask`, two facts, one
`CalculationResult`, and the evidence corpus that validates those facts. This is intentionally not
autonomous selection. A successful `GeneratedAnswer` has three visibly separate sections:

- **Source-reported values** use each fact's exact `reported_value`, reporting-period label,
  currency, units, and deterministic `[PDF p. N]` marker.
- **Computed result** uses the exact Decimal string, formula, unit, input fact IDs, calculation ID,
  and citations inherited from its inputs.
- **Model-generated explanation** contains only the checked qualitative model sentence.

The structured citations retain source, selected-text, evidence-export, and manifest hashes;
supporting text; warnings; limitations; and review provenance. Generation provenance records the
backend, model, revision, inference settings, grounding-policy version, prompt hash, and raw model
output hash. Raw model output is not promoted into a financial value.

## Reviewed NVIDIA demonstration

The existing ignored NVIDIA FY2026 fact set supports these two requests:

| Task | Source-reported inputs | Exact deterministic result | Citation |
| --- | --- | ---: | --- |
| Revenue growth | FY2026 revenue `$ 215,938`; FY2025 revenue `$ 130,497` (USD millions) | `65.47353579009479145114447075411695%` | Physical PDF page 139 |
| Net profit margin | FY2026 net income `$ 120,067`; FY2026 revenue `$ 215,938` (USD millions) | `55.60253406070260908223656790375015%` | Physical PDF page 139 |

The integration test loads the unchanged local evidence and fact artifacts, recomputes both
results, supplies controlled mock explanations, and checks the resulting page citations:

```powershell
uv run pytest tests/test_real_evidence_integration.py
```

It skips if the Git-ignored report artifacts are absent. The financial review remains the earlier
Codex-assisted visual review and is **not independent human verification**. The mock backend proves
the validation and rendering workflow, not the quality of any particular open model.

## Current limitations

- Only explicitly selected revenue-growth and net-profit-margin requests are supported.
- Question support uses conservative keyword checks; valid paraphrases can abstain.
- Explanation grounding is lexical and template-like. It rejects unsupported vocabulary but does
  not provide general semantic entailment.
- The model cannot add quantitative detail, causal analysis, management commentary, forecasts, or
  extra citations, even when such evidence might exist elsewhere.
- Results retain the calculation layer's full Decimal value because presentation rounding remains
  undefined.
- One fact still binds one physical page. Multi-page context referenced by review metadata is not
  converted into additional answer citations.
- No concrete model runtime is a project dependency. CPU-local inference is opt-in through the
  backend protocol and must be configured and tested by the caller.
