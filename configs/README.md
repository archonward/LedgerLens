# Configuration

Version-controlled pipeline and experiment configuration belongs here. Retrieval evaluation-set
definitions may be stored under `configs/evaluation/` after their relevance judgments and split
assignments have been reviewed. Generated evaluation results belong under ignored `artifacts/`,
not here. Secrets and machine-specific paths must stay outside the repository.

The two `nvidia-*-multi.json` files are explicit Phase 3C smoke requests for existing ignored,
reviewed NVIDIA artifacts. They contain exact source identities, reconciliation IDs, fact IDs,
roles, and selection policy; they do not contain or generate financial facts. See
[`docs/multi-document-cli.md`](../docs/multi-document-cli.md).
