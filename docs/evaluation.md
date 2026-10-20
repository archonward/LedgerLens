# Retrieval evaluation

LedgerLens implements an offline framework for evaluating page-level BM25 retrieval against
independently supplied relevance judgments. No NVIDIA benchmark result is included in the
repository, and no benchmark claim should be made until supporting pages have been manually
verified against the source report.

Evaluation definitions and retrieval results are separate files. The evaluator never turns a
high-ranked BM25 page into ground truth.

## Evaluation-set format

An evaluation set is UTF-8 JSON using schema version `1.0`:

```json
{
  "schema_version": "1.0",
  "dataset_id": "example-company-retrieval-v1",
  "document_id": "sha256-<source-hash>",
  "source_sha256": "<source-hash>",
  "questions": [
    {
      "question_id": "dev-001",
      "question": "What drove ExampleCo service revenue growth?",
      "split": "development",
      "status": "verified",
      "leakage_group": "service-revenue-drivers",
      "relevant_pages": [10, 11],
      "verification": {
        "method": "manual_source_review",
        "reviewed_by": "<reviewer identifier>",
        "reviewed_on": "YYYY-MM-DD"
      },
      "notes": "Optional review notes."
    },
    {
      "question_id": "dev-002",
      "question": "A candidate question awaiting source review",
      "split": "development",
      "status": "provisional",
      "leakage_group": "candidate-topic",
      "candidate_pages": [10],
      "notes": "Candidate pages are not relevance judgments."
    }
  ]
}
```

The example values above illustrate the format; they are not NVIDIA relevance judgments.

Every question has a stable ID, exactly one split, and a leakage group. Supported splits are
`development` and `held_out`. A verified question requires one or more physical PDF page numbers
and explicit manual-source-review provenance. A provisional question may record zero or more
candidate pages, but it cannot declare `relevant_pages` or verification. Provisional retrieval is
included in the report for inspection while its metrics remain `null` and it is excluded from all
official aggregates.

The evaluator validates that required review metadata is present, but it cannot prove that a
manual review occurred or that a judgment is correct. Review provenance remains an accountable
input supplied by the dataset author.

The dataset document ID and source hash must match the selected-evidence export. All verified
relevant pages must exist in selected evidence; otherwise evaluation stops instead of reporting a
misleading zero caused by unavailable input.

## Development and held-out splits

Evaluate exactly one split per run. Development questions may guide tokenizer, parameter, and
retriever changes. Do not inspect held-out results repeatedly while tuning. The loader rejects a
`leakage_group` that occurs in both splits and rejects exact questions that cross splits after
case/punctuation normalization.

Reviewers must assign the same leakage group to paraphrases and near-duplicates that test the same
underlying report fact, table lookup, or calculation. Automatic text checks cannot reliably detect
semantic near-duplicates. Record any reports or examples used for future training or fine-tuning
so training/test leakage can also be audited.

## Run an evaluation

Windows PowerShell:

```powershell
uv run ledgerlens evaluate-retrieval `
  "artifacts/evidence/sha256-<hash>" `
  "configs/evaluation/questions.json" `
  --split development `
  --k 1 3 5 10 `
  --output "artifacts/evaluation/development-bm25.json"
```

Linux:

```bash
uv run ledgerlens evaluate-retrieval \
  "artifacts/evidence/sha256-<hash>" \
  "configs/evaluation/questions.json" \
  --split development \
  --k 1 3 5 10 \
  --output "artifacts/evaluation/development-bm25.json"
```

`uv run python -m ledgerlens evaluate-retrieval ...` is equivalent. Existing reports are protected
unless `--overwrite` is explicit. Keep generated reports under `artifacts/`; that directory is
ignored by Git. The output cannot replace its judgment file or selected-evidence inputs.

The report records:

- the evaluation-set path, schema version, split, and SHA-256 hash;
- selected-evidence document, source, JSONL, and manifest hashes;
- LedgerLens version and exact BM25 parameters;
- ranked physical pages, scores, extraction method, text/source hashes, and limitations;
- per-question metrics and verification provenance; and
- aggregate metrics and counts of verified versus excluded provisional questions.

The creation timestamp is operational metadata, so reruns are not byte-identical. With unchanged
hashed inputs and software, the rankings and metric values are deterministic.

## Metrics

For each verified question and cutoff `K`:

- **Hit@K** is 1 when at least one verified relevant page is among the first K results, otherwise 0.
- **Recall@K** is the number of distinct verified relevant pages retrieved in the first K divided
  by the total number of verified relevant pages. This gives partial credit to questions requiring
  multiple supporting pages.
- **Reciprocal rank** is `1 / rank` for the first verified relevant page, or 0 when none has a
  positive-score rank. **MRR** is its macro-average over verified questions.

Aggregate Hit@K and Recall@K are also macro-averages, so each verified question has equal weight.
MRR uses the full positive-score page ranking rather than only the largest requested K. A split
with no verified questions reports `aggregate_metrics: null`; provisional questions never create
an accuracy number.

These measures evaluate page retrieval only. They do not establish extraction accuracy, table
structure, numerical correctness, citation support, answer quality, or human approval of retrieved
text.

## Future evaluation stages

The same judgment set can compare later embedding or hybrid retrievers if they produce ranked
physical pages from the same selected evidence. Future milestones will separately evaluate
numerical accuracy (including periods, currencies, units, signs, and formulas), citation support,
missing-information behavior, and latency. Those measures are not implemented here.
