# Ollama model-acceptance benchmark

This benchmark measures whether the selected local `qwen3:4b-instruct` model can produce text that
passes LedgerLens's existing answer-generation boundary. It does not change prompts, facts,
calculations, numeric rendering, citations, or grounding rules to improve acceptance.

## Fixed case set

The live benchmark first validates the original NVIDIA PDF, review record, selected-evidence
export, and financial facts through the existing real-evidence demonstration. It then uses the
existing deterministic calculation functions and these cases:

| Case | Kind | Runs | Expected harness behavior |
| --- | --- | ---: | --- |
| `revenue-growth` | Positive | 3 by default | Invoke the model; record grounding acceptance or rejection. |
| `net-profit-margin` | Positive | 3 by default | Invoke the model; record grounding acceptance or rejection. |
| `unsupported-question` | Negative | 1 | Abstain before inference with the expected question-support reason. |
| `task-calculation-mismatch` | Negative | 1 | Abstain before inference because the calculation type does not match. |
| `unverified-input` | Negative | 1 | Abstain before inference because verification was deliberately removed in memory. |

The negative transformations are controlled benchmark cases. They do not modify the reviewed fact
file, selected evidence, review record, source PDF, or calculations. Only the two positive cases
enter the model-acceptance-rate denominator. Repeats can be set from 1 through 20; changing the
repeat count changes the configuration ID.

## Run the live benchmark

The command requires an explicit `--run-live` acknowledgement. It uses fixed
`max_new_tokens=96` and `temperature=0.0`, verifies the exact installed Ollama tag and digest before
inference, and never pulls or substitutes a model.

```powershell
uv run python scripts/benchmark_ollama_generation.py --run-live
```

```bash
uv run python scripts/benchmark_ollama_generation.py --run-live
```

The protected default destination is:

```text
artifacts/evaluation/ollama-generation-qwen3-4b-instruct/
  benchmark.json
  report.md
```

Use `--overwrite` only when intentionally replacing that ignored output. The source-derived inputs
and generated benchmark files remain ignored by Git. The command validates all inputs before
contacting the model and publishes both files atomically only after all runs complete.

## Machine-readable record

`benchmark.json` includes:

- schema and deterministic configuration IDs;
- exact Ollama model tag and SHA-256 digest;
- fixed inference configuration;
- source-document, evidence-export, review-record, fact, calculation, and physical-page references;
- case definitions and repeat counts;
- per-run answer status, grounding outcome, abstention reason, client-observed latency, prompt hash,
  model-output hash, calculation ID, and emitted citation pages; and
- aggregate acceptance, grounding-rejection, inference-failure, negative-abstention, harness, and
  latency statistics.

Raw model output is intentionally not copied into the artifact. Its hash makes repeated outputs
comparable without promoting rejected text into an answer or retaining potentially unsuitable
generated prose.

## Reading the report correctly

The generated `report.md` separates four ideas:

1. **Test-harness correctness** means positive cases reached inference and negative cases produced
   their exact expected structured abstentions without invoking the model.
2. **Model acceptance rate** is accepted positive outputs divided by positive outputs that reached
   LedgerLens explanation validation. Inference failures are reported separately and excluded.
3. **Semantic correctness remains limited.** Passing the current strict lexical validator is not
   an independent human judgment of meaning, completeness, or financial-answer quality.
4. **The sample is small.** Two questions with repeated runs can reveal acceptance stability but
   cannot establish coverage across companies, reports, metrics, or question styles.

An accepted response still receives deterministic financial values and physical-page citations
from LedgerLens. A rejected response remains an abstention. The benchmark never treats model text
as a source of arithmetic or financial facts.

## Offline validation

The default test suite does not contact Ollama or execute a model. Controlled outcomes test case
validation, positive acceptance, grounding rejection, structured abstention, aggregation,
configuration hashing, report rendering, model-identity drift detection, and protected artifact
publication:

```powershell
uv run pytest tests/test_generation.py tests/test_generation_benchmark.py `
  tests/test_ollama_backend.py
```

The earlier single-pass real-model smoke test remains independently opt-in. The benchmark script is
the reproducible multi-run evaluation entry point.
