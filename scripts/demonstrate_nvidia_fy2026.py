"""Reproduce reviewed NVIDIA FY2026 calculations from ignored local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ledgerlens.calculations import FactOperand, profit_margin, year_over_year_growth
from ledgerlens.financial_facts import FinancialFact, load_financial_facts
from ledgerlens.retrieval.bm25 import RetrievalError, load_evidence_export

DOCUMENT_ID = "sha256-0e725ba048221539dca3eb1a4e70febfcbb785e9afb96cd3ff0b035d7d734e5c"
SOURCE_SHA256 = DOCUMENT_ID.removeprefix("sha256-")
FACT_IDS = {
    "revenue_current": "nvidia-fy2026-revenue",
    "revenue_prior": "nvidia-fy2025-revenue",
    "net_income_current": "nvidia-fy2026-net-income",
    "net_income_prior": "nvidia-fy2025-net-income",
}


class DemonstrationError(ValueError):
    """The reviewed local artifacts cannot reproduce the demonstration safely."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DemonstrationError(f"could not read {path}: {exc}") from exc


def _read_review(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DemonstrationError(f"could not read review record at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DemonstrationError("review record must be a JSON object")
    if value.get("schema_version") != "1.0":
        raise DemonstrationError("review record has an unsupported schema_version")
    if value.get("document_id") != DOCUMENT_ID:
        raise DemonstrationError("review record has the wrong document_id")
    review = value.get("review")
    if not isinstance(review, dict):
        raise DemonstrationError("review record is missing review metadata")
    if review.get("reviewer_identity") != "OpenAI Codex automated coding agent":
        raise DemonstrationError("review record has an unexpected reviewer identity")
    if review.get("independent_human_verification_claimed") is not False:
        raise DemonstrationError(
            "review record must explicitly disclaim independent human verification"
        )
    if review.get("physical_pages_reviewed") != [22, 69, 139, 145]:
        raise DemonstrationError("review record does not cover the required physical pages")
    if not isinstance(review.get("method"), str) or not review["method"].strip():
        raise DemonstrationError("review record is missing its review method")
    limitations = value.get("limitations")
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) or not item.strip() for item in limitations
    ):
        raise DemonstrationError("review record limitations must be non-empty strings")
    return value


def _index_facts(facts: tuple[FinancialFact, ...]) -> dict[str, FinancialFact]:
    by_id = {fact.fact_id: fact for fact in facts}
    missing = set(FACT_IDS.values()) - set(by_id)
    unexpected = set(by_id) - set(FACT_IDS.values())
    if missing or unexpected:
        raise DemonstrationError(
            f"fact set mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return by_id


def build_demonstration(
    source_pdf: Path, evidence_directory: Path, facts_path: Path, review_path: Path
) -> dict[str, object]:
    """Validate reviewed artifacts and calculate the fixed demonstration outputs."""
    if _sha256(source_pdf) != SOURCE_SHA256:
        raise DemonstrationError("source PDF hash does not match the reviewed document")
    review = _read_review(review_path)
    review_sha256 = _sha256(review_path)
    corpus = load_evidence_export(evidence_directory)
    if corpus.document_id != DOCUMENT_ID:
        raise DemonstrationError("selected evidence is not the reviewed NVIDIA FY2026 document")
    facts = load_financial_facts(facts_path, corpus)
    by_id = _index_facts(facts)
    review_marker = f"review_record_sha256={review_sha256}"
    for fact in facts:
        if fact.verification.note is None or review_marker not in fact.verification.note:
            raise DemonstrationError(
                f"fact {fact.fact_id!r} is not bound to the supplied review record"
            )

    operands = {name: FactOperand(by_id[fact_id], corpus) for name, fact_id in FACT_IDS.items()}
    calculations = {
        "revenue_growth_fy2026_vs_fy2025": year_over_year_growth(
            operands["revenue_current"], operands["revenue_prior"]
        ).to_dict(),
        "net_profit_margin_fy2026": profit_margin(
            operands["net_income_current"], operands["revenue_current"]
        ).to_dict(),
        "net_profit_margin_fy2025": profit_margin(
            operands["net_income_prior"], operands["revenue_prior"]
        ).to_dict(),
    }
    return {
        "schema_version": "1.0",
        "demonstration": "nvidia_fy2026_reviewed_financial_calculations",
        "source": {
            "document_id": corpus.document_id,
            "source_sha256": corpus.source_sha256,
            "evidence_jsonl_sha256": corpus.evidence_jsonl_sha256,
            "evidence_manifest_sha256": corpus.manifest_sha256,
        },
        "review": {
            "review_record_sha256": review_sha256,
            "reviewer_identity": review["review"]["reviewer_identity"],
            "method": review["review"]["method"],
            "independent_human_verification_claimed": False,
            "physical_pages_reviewed": review["review"]["physical_pages_reviewed"],
        },
        "facts": [fact.to_dict() for fact in facts],
        "calculations": calculations,
        "limitations": review["limitations"],
    }


def _build_parser() -> argparse.ArgumentParser:
    root = Path("artifacts")
    evidence = root / "evidence" / DOCUMENT_ID
    facts_root = root / "financial-facts" / DOCUMENT_ID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pdf", type=Path, default=Path("data/raw/nvidia-2026-annual-report.pdf")
    )
    parser.add_argument("--evidence-directory", type=Path, default=evidence)
    parser.add_argument("--facts", type=Path, default=facts_root / "facts.jsonl")
    parser.add_argument("--review-record", type=Path, default=facts_root / "review.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reviewed demonstration and emit deterministic JSON."""
    args = _build_parser().parse_args(argv)
    try:
        payload = build_demonstration(
            args.source_pdf, args.evidence_directory, args.facts, args.review_record
        )
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if args.output is None:
            sys.stdout.write(serialized)
        else:
            if args.output.exists() and not args.overwrite:
                raise DemonstrationError(
                    f"output already exists at {args.output}; use --overwrite to replace it"
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8", newline="\n")
            print(f"Wrote reviewed demonstration to {args.output}")
    except (DemonstrationError, OSError, RetrievalError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
