"""Behavior tests for the LedgerLens command-line interface."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from ledgerlens import __version__


def run_command(
    command: Sequence[str], *, path: str | None = None
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if path is not None:
        env["PATH"] = path
    return subprocess.run(command, capture_output=True, check=False, env=env, text=True)


@pytest.mark.parametrize("entry_point", [[sys.executable, "-m", "ledgerlens"], ["ledgerlens"]])
def test_help_works_for_both_entry_points(entry_point: list[str]) -> None:
    result = run_command([*entry_point, "--help"])

    assert result.returncode == 0
    assert "doctor" in result.stdout
    assert "--version" in result.stdout


@pytest.mark.parametrize("entry_point", [[sys.executable, "-m", "ledgerlens"], ["ledgerlens"]])
def test_version_works_for_both_entry_points(entry_point: list[str]) -> None:
    result = run_command([*entry_point, "--version"])

    assert result.returncode == 0
    assert result.stdout.strip() == f"ledgerlens {__version__}"


def test_doctor_completes_when_nvidia_smi_is_unavailable(tmp_path: Path) -> None:
    result = run_command(
        [sys.executable, "-m", "ledgerlens", "doctor"],
        path=str(tmp_path),
    )

    assert result.returncode == 0
    assert "Python:" in result.stdout
    assert "Operating system:" in result.stdout
    assert f"LedgerLens: {__version__}" in result.stdout
    assert "nvidia-smi on PATH: no" in result.stdout
    assert "GPU availability: unverified" in result.stdout


def test_doctor_detects_nvidia_smi_on_path(tmp_path: Path) -> None:
    executable = tmp_path / ("nvidia-smi.exe" if os.name == "nt" else "nvidia-smi")
    executable.touch()
    if os.name != "nt":
        executable.chmod(0o755)

    result = run_command(
        [sys.executable, "-m", "ledgerlens", "doctor"],
        path=str(tmp_path),
    )

    assert result.returncode == 0
    assert "nvidia-smi on PATH: yes" in result.stdout
    assert "GPU availability: unverified" in result.stdout
