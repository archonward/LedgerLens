# Planned architecture

All components described here are **planned and unimplemented**. The current repository contains
only an executable scaffold.

## Pipeline

1. **Ingestion (planned):** extract text and tables without losing the source document identity,
   page reference, reporting period, currency, or units. Normalization must retain links to the
   original representation.
2. **Retrieval (planned):** index the extracted material, retrieve candidates relevant to a
   question, and rank evidence while retaining its complete provenance.
3. **Calculations (planned):** perform arithmetic through deterministic, tested functions. Inputs,
   period choices, currency, units, formula, and result must be recorded so the calculation can be
   reproduced.
4. **Generation (planned):** compose a direct answer using retrieved evidence and calculated
   results, attach citations to supported claims, and state when available evidence is insufficient.
5. **Evaluation (planned):** measure retrieval, numerical, citation, abstention, and latency
   behavior against manually verified examples.

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
