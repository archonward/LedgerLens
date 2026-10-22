# Architecture

PDF text ingestion, explicit reviewed-evidence selection, page-level lexical, semantic, and hybrid
retrieval, a validated financial-fact representation, and initial deterministic calculations are
implemented as CPU-only foundations. Later components described here are **planned and
unimplemented**.

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
6. **Generation (planned):** compose a direct answer using retrieved evidence and calculated
   results, attach citations to supported claims, and state when available evidence is insufficient.
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
