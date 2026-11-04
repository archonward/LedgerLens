# Reconciliation-to-calculation fact selection

The reconciliation selection adapter in `ledgerlens.reconciliation_selection` is a narrow bridge
between validated `ReconciliationReport` results and the existing calculation `FactOperand`
interface. It does not calculate values, reinterpret metrics, modify reconciliation results, or
orchestrate answers.

## Selection policy

Agreement does not identify a preferred source. For every requested calculation role, the caller
must provide both the exact reconciliation result ID and an exact source-specific fact ID from
that result. The adapter never selects the first observation, newest report, or any source based
on document ordering. The structured decision records the caller's requested fact ID, every
candidate observation, and the complete selected fact snapshot.

Single-source observations use an explicit policy:

- `abstain` is the default and produces an `insufficient_evidence` abstention;
- `allow_explicit` permits the observation only when the caller names its exact fact ID.

Enabling `allow_explicit` does not describe the observation as corroborated. The decision retains
the reconciliation status `insufficient_evidence` and records the policy that permitted its use.

The following conditions always abstain and cannot be enabled by policy:

- an unresolved reconciliation conflict;
- an incompatibility result, including incompatible periods, units, or currencies;
- duplicate fact IDs or multiple candidate records from one source document;
- a missing, duplicated, malformed, or mismatched reconciliation result;
- a requested fact ID that is not a candidate in the selected result; and
- missing, invalid, or stale evidence for any candidate observation in the selected group.

Every candidate is wrapped in the existing `FactOperand` before selection succeeds. This reruns
verification and evidence binding checks for the selected source and its corroborating sources.
The adapter therefore cannot rely on stale corroboration while returning a still-valid selected
fact.

## Result contract

`prepare_calculation_facts` accepts ordered `FactSelectionRequest` values and a mapping from
document ID to validated `EvidenceCorpus`. It returns ordered `FactSelectionDecision` records.
Each decision contains:

- the calculation role and requested reconciliation result and fact IDs;
- the reconciliation status and every full candidate observation;
- either a selected fact with its complete evidence and verification provenance, or a closed
  abstention reason and detail; and
- the applied source-selection and single-source policy.

`CalculationFactSelection.operands` returns `FactOperand` values in request order only when every
request succeeds. If any role abstains, it returns `None`, preventing a partial operand set from
being passed accidentally to a calculation. Existing Decimal arithmetic, formulas, calculation
validation, and citation serialization are unchanged.

## Scope limits

The adapter requires callers to identify reconciliation result IDs and source fact IDs. It does
not map natural-language questions to metrics or periods, choose sources, publish selection
artifacts, persist user preferences, or perform multi-document orchestration. Those decisions
require a separate, explicit integration layer.

The [narrow multi-document orchestration pathway](multi-document-orchestration.md) is that
explicit calculation-only integration. It retains this adapter's decisions and abstentions and
does not weaken the per-source evidence boundary.
