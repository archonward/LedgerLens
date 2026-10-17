# Planned evaluation

No evaluation has been performed and this document reports no benchmark results. Evaluation will
begin only after the corresponding pipeline components are implemented.

## Evaluation set

A human reviewer will create questions for the initial pair of consecutive annual reports and
manually verify each expected answer and its supporting document evidence. Each example will
record document identity, page references, reporting periods, currencies, and units where
applicable.

Development questions used to design prompts, retrieval, or calculations will be separated from
a final held-out evaluation set. Splits will be made before iterative tuning. Exact and
near-duplicate questions—including paraphrases with the same underlying facts or calculations—must
not cross between development and held-out sets. Reports or derived examples used for any future
training or fine-tuning must be recorded so training/test leakage can be checked.

## Measures

- **Retrieval quality:** whether the required passages and tables appear in ranked evidence, using
  measures such as recall at a fixed rank and human relevance judgments.
- **Numerical accuracy:** whether values, formulas, signs, reporting periods, currencies, units,
  rounding, and final results are correct.
- **Citation support:** whether citations identify the right document and page and actually support
  the associated claim.
- **Missing-information behavior:** whether the system clearly declines or qualifies an answer
  when the reports do not provide enough evidence.
- **Latency:** end-to-end and stage-level elapsed time under a recorded execution environment.

Results will be reproducible from versioned evaluation definitions and code, while reports and
generated artifacts remain outside Git. Any future model, prompt, retrieval configuration, and
hardware context used for a run will be recorded alongside its results.
