# Safe multi-document answer generation

Milestone 8B Phase 3B connects the Phase 3A multi-document validation contract to the existing
local-model interface. `MultiDocumentAnswerGenerator` asks a model for only one short qualitative
sentence. Deterministic code continues to own every financial value, percentage, formula, unit,
calculation, and citation.

This is a Python API only. It adds no CLI, retrieval, extraction, fiscal-period inference, source
selection, evidence merging, or automatic model installation.

## Validation-to-generation trust boundary

Generation accepts the raw `MultiDocumentGenerationRequest` and the mapping of exact document IDs
to separately loaded `EvidenceCorpus` objects. It constructs a concrete
`MultiDocumentGenerationValidator` internally on every call. That validator rechecks the preserved
reconciliation selection, source-specific bindings, evidence hashes and pages, verification state,
task and metric rules, question compatibility, and complete independently reproduced Decimal
calculation.

A caller-supplied `MultiDocumentGenerationValidationResult` is not trusted as an authorization
token. If one is supplied, the generator discards its claimed status, citations, and reproduced
calculation and revalidates its underlying raw request. Constructing or modifying an object so that
it says `validated` therefore cannot bypass evidence or calculation validation.

No backend call occurs when fresh validation abstains. Unvalidated financial material and
citations are not exposed in that result.

## Model boundary

The component uses the existing `LocalModelBackend` protocol. Production local inference uses the
unchanged `OllamaBackend`, which accepts only an unauthenticated loopback origin, verifies the exact
already-installed `qwen3:4b-instruct` tag and digest before every call, never pulls a model, and
uses non-streaming generation with thinking disabled.

Phase 3B fixes inference to `max_new_tokens=96` and `temperature=0.0`; callers cannot override that
configuration. Model provenance records the backend, exact model identity and revision, inference
settings, prompt hash, output hash, response latency, grounding-policy version, and explanation
validation diagnostics.

The prompt contains only the validated facts and reproduced calculation. It tells the model to
return one qualitative sentence without numbers, currency, percentages, citations, causes,
forecasts, report rankings, or facts beyond the allowed deterministic conclusion.

## Deterministic rendering and citations

Each reported value is rendered from its validated `FinancialFact`. The calculation section uses
the independently reproduced `CalculationResult`, including its exact Decimal string, formula,
unit, input fact IDs, and content-derived calculation ID.

Every input retains its separate `SourceSpecificCitation`. Markers name the company, annual-report
year, full document identity, and physical PDF page. Structured citations additionally retain the
source PDF, supporting-text, selected-text, evidence JSONL, and evidence manifest hashes; evidence
method; warnings and limitations; evidence review provenance; and fact verification provenance.
Reports are never merged into one citation or corpus.

## Grounding and result states

Model output is processed by grounding policy 1.1 without modification. The existing lexical,
forbidden-character, length, sentence-count, required-term, and direction checks reject invented
figures, causal or forecast language, opposite relationships, unsupported vocabulary, citations,
and output-contract violations. Rejected raw text is never presented as the accepted explanation
and is absent from normal provenance. The existing explicit rejected-output handler remains
available for local Git-ignored debugging.

`MultiDocumentGeneratedAnswer.status` distinguishes four outcomes:

- `answered`: fresh validation succeeded and the model sentence passed grounding;
- `model_output_rejected`: the financial calculation and citations remain available, but the
  model sentence failed grounding and is omitted;
- `validation_abstained`: validation failed before inference, so no calculation or citations are
  presented as validated output; and
- `inference_failed`: validation succeeded, the deterministic calculation and citations remain
  available, but the backend call failed and no model explanation is shown.

The selection audit and complete Phase 3A validation record remain embedded in every result.

## Offline and optional real-model tests

The default suite uses controlled mock backends and distinct synthetic corpora. It checks accepted
explanations, incorrect source mappings, tampered calculations, fabricated validated results,
unsupported model statements, backend failures, provenance, citations, and the fixed inference
configuration. It never contacts Ollama.

Run the focused offline tests with:

```powershell
uv run pytest tests/test_multi_document_orchestration.py -k "multi_document_answer_generation"
```

The real NVIDIA multi-document test is separately opt-in. It validates FY2025 revenue growth using
FY2024- and FY2025-report facts, and FY2024 net-profit margin using facts from distinct reports.
It verifies exact Decimal results and source-specific citations. A completed Qwen invocation may
still produce either `answered` or `model_output_rejected`; invocation success is not treated as
grounding acceptance.

```powershell
$env:LEDGERLENS_RUN_MULTI_DOCUMENT_OLLAMA = "1"
uv run pytest -s tests/test_multi_document_orchestration.py -k "real_ollama_multi_document"
```

```bash
LEDGERLENS_RUN_MULTI_DOCUMENT_OLLAMA=1 \
  uv run pytest -s tests/test_multi_document_orchestration.py \
  -k "real_ollama_multi_document"
```

Without that exact environment variable, the test skips before contacting Ollama. It writes no
benchmark artifacts and does not overwrite prior outputs. When explicitly enabled, missing local
artifacts skip the test; an unavailable Ollama server or missing exact Qwen tag fails safely rather
than selecting or downloading another model.

## Remaining limitations

Only the explicit revenue-growth and net-profit-margin tasks are supported. Question matching and
model grounding remain deliberately narrow and lexical, so safe paraphrases can be rejected. The
model cannot add quantitative details, causes, forecasts, management commentary, or extra evidence.
Presentation rounding remains undefined, and the API retains the calculation layer's full Decimal
result. Phase 3B does not add a user-facing workflow or make the model a trusted financial reasoner.

Phase 3C adds a separate request-file CLI, deterministic presentation layer, and reference-case
evaluation adapter without changing this generator. See
[multi-document-cli.md](multi-document-cli.md) and
[multi-document-evaluation.md](multi-document-evaluation.md).
