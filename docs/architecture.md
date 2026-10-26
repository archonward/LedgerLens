# Architecture

PDF text ingestion, explicit reviewed-evidence selection, page-level lexical, semantic, and hybrid
retrieval, a validated financial-fact representation, deterministic calculations, and narrow
evidence-grounded answer generation are implemented as CPU-only foundations. Later components
described here remain **planned and unimplemented**.

## Pipeline

1. **Ingestion (partially implemented):** the current command extracts page-linked PDF text and
   records document identity plus a user-supplied report year. Targeted single-page OCR recovery is
   available, while structured table extraction, printed page-label detection, and interpretation
   of periods, currency, and units remain planned. Future normalization must retain links to the
   original representation.
2. **Evidence selection (implemented for the prototype):** select automatically eligible
   canonical pages plus only those reviewed recovery derivatives named by an explicit,
   hash-validated manifest. Flagged pages without a valid selection remain excluded. Selection
   does not approve numerical interpretation or establish extraction accuracy.
3. **Retrieval (BM25, semantic, and hybrid baselines implemented):** validate the selected-evidence
   export
   and rank one document per physical page. BM25 builds a deterministic in-memory lexical index.
   The optional semantic baseline stores pinned-model, normalized page embeddings in a reusable,
   hash-validated cache and ranks them with cosine similarity. The chunk-aware semantic variant
   makes overlapping token windows inside each physical page, embeds those exact text slices, and
   aggregates chunk scores back to pages. The initial hybrid retriever combines unchanged BM25 and
   page-semantic ranks using weighted reciprocal rank fusion and reuses the validated page-embedding
   cache. All retain selected-text and source hashes, extraction
   method, review provenance, screening warnings, and limitations. Cross-page passage merging,
   learned fusion, reranking, and persistent vector databases remain planned.
4. **Financial evidence representation (implemented foundation):** bind a reported value and its
   normalized decimal form to an exact selected-evidence excerpt and physical PDF page. Reporting
   period, currency, units/scale, and table-column association carry explicit known, missing, or
   ambiguous states. Verification is separate from extraction quality and requires review
   metadata plus resolved critical context. Automatic extraction and table reconstruction are not
   implemented.
5. **Calculations (implemented foundation):** compute year-over-year growth, profit margins, and
   compatible two-input ratios using a fixed Decimal context. Every input must be a verified fact
   that still validates against its selected-evidence corpus. Results retain formulas, complete
   input fact snapshots, reporting periods, currencies, units, source provenance, and arithmetic
   settings. Automatic input selection, broader formula systems, and presentation rounding remain
   planned.
   A narrow NVIDIA FY2026 integration demonstrates this boundary with four visually reviewed,
   report-backed facts and three reproducible calculations. Its source-derived artifacts remain
   Git-ignored, and its Codex-assisted review is not independent human verification.
6. **Generation (implemented narrow foundation):** accept an explicitly selected revenue-growth or
   net-profit-margin task, revalidate and recompute its supplied facts and result, then ask a
   configurable local-model backend for a tightly constrained qualitative explanation.
   Deterministic code renders all financial values, formulas, and physical-page citations. Model
   text containing numbers, citations, unsupported vocabulary, or the wrong result direction is
   rejected, and invalid, stale, mismatched, or unsupported inputs abstain. Runtime-specific model
   loading for other runtimes, broader question support, semantic entailment, automatic input
   selection, and a conversational interface remain planned. An optional loopback-only Ollama
   adapter supports the exact locally installed `qwen3:4b-instruct` tag without downloading models;
   its model digest, inference settings, response latency, and grounding outcome are recorded. An
   opt-in acceptance harness repeats the two reviewed NVIDIA tasks and checks controlled negative
   abstentions while keeping live-model execution separate from deterministic offline tests.
7. **Evaluation (retrieval framework implemented):** evaluate BM25, page-semantic, chunk-aware
   semantic, or hybrid ranked physical pages using separate, manually verified relevance judgments
   and development/held-out splits.
   Hit@K, Recall@K, and MRR are implemented; numerical, citation, abstention, generation, and
   latency evaluation remain planned.

## Information types

LedgerLens must keep these categories explicit rather than presenting them as interchangeable:

- **Report statements** are facts or passages extracted from a specific report and supported by a
  document and page reference.
- **Calculated results** are deterministic transformations of identified inputs, with the formula,
  reporting periods, currencies, and units recorded.
- **Model inferences** are interpretations or generated conclusions. They must be labeled as such
  and supported by cited evidence; they are not report statements or calculation outputs.

The initial target is one company and two consecutive annual reports. The design should first be
validated on that narrow scope before introducing additional document types, models, or storage
systems. CPU-only local execution remains the default. Future GPU-backed model work will be
submitted through Slurm rather than run on a cluster login node.
