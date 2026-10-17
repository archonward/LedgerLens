"""Command-line interface for LedgerLens development utilities."""

from __future__ import annotations

import argparse
import platform
import shutil
from collections.abc import Sequence

from ledgerlens import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerlens",
        description="Evidence-based analysis tools for financial reports.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report local environment information without running GPU workloads.",
    )
    doctor_parser.set_defaults(handler=_run_doctor)
    return parser


def _run_doctor(_: argparse.Namespace) -> int:
    nvidia_smi = shutil.which("nvidia-smi")
    print(f"Python: {platform.python_version()}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"LedgerLens: {__version__}")
    print(f"nvidia-smi on PATH: {'yes' if nvidia_smi else 'no'}")
    print(
        "GPU availability: unverified; finding nvidia-smi alone does not establish "
        "a usable GPU allocation."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the LedgerLens command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)
