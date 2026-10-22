# Architecture

PDF text ingestion, explicit reviewed-evidence selection, and page-level BM25 retrieval are
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
3. **Retrieval (BM25 and semantic baselines implemented):** validate the selected-evidence export
   and rank one document per physical page. BM25 builds a deterministic in-memory lexical index.
   The optional semantic baseline stores pinned-model, normalized page embeddings in a reusable,
   hash-validated cache and ranks them with cosine similarity. The chunk-aware semantic variant
   makes overlapping token windows inside each physical page, embeds those exact text slices, and
   aggregates chunk scores back to pages. All retain selected-text and source hashes, extraction
   method, review provenance, screening warnings, and limitations. Cross-page passage merging,
   hybrid retrieval, reranking, and persistent vector databases remain planned.
4. **Calculations (planned):** perform arithmetic through deterministic, tested functions. Inputs,
   period choices, currency, units, formula, and result must be recorded so the calculation can be
   reproduced.
5. **Generation (planned):** compose a direct answer using retrieved evidence and calculated
   results, attach citations to supported claims, and state when available evidence is insufficient.
6. **Evaluation (retrieval framework implemented):** evaluate BM25, page-semantic, or chunk-aware
   semantic ranked physical pages using separate, manually verified relevance judgments and
   development/held-out splits.
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
