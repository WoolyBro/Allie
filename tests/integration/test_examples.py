"""Every example in `examples/` actually runs.

Added in Phase 9 after finding `hello_world.ql` had been broken since
Phase 0 -- it called `load_csv`/`sma`, which the stdlib registers as
`load`/`SMA`. It was written as a target to build toward and then never
revisited, so nothing caught it. A broken file named "hello world" is
the worst one to leave lying around, since it's the first thing a reader
tries.

Examples run from the repo root (their `load(...)` paths are
repo-relative), so this chdirs rather than assuming pytest's cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from allie.cli import run_pipeline

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = _REPO_ROOT / "examples"

# Examples that exist to demonstrate a *rejection*, not a successful run --
# excluded from "every example runs clean" and checked by their own test
# below instead. Named explicitly (not glob-excluded by a naming
# convention) so a new negative example has to be a deliberate addition
# to both lists, not something that silently slips through either test.
_EXPECTED_TO_FAIL = {"cheat_demo": "look-ahead bias"}


def _example_paths() -> list[Path]:
    return sorted(_EXAMPLES.glob("*.qs"))


def _runnable_example_paths() -> list[Path]:
    return [p for p in _example_paths() if p.stem not in _EXPECTED_TO_FAIL]


def test_examples_directory_is_not_empty() -> None:
    assert _example_paths(), "no examples found -- did the glob pattern change?"


@pytest.mark.parametrize("path", _runnable_example_paths(), ids=lambda p: p.stem)
def test_example_runs_clean(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    previous = Path.cwd()
    os.chdir(_REPO_ROOT)
    try:
        exit_code = run_pipeline(path, backend="tree")
    finally:
        os.chdir(previous)

    captured = capsys.readouterr()
    assert exit_code == 0, f"{path.name} exited {exit_code}\n{captured.err}"
    assert captured.out.strip(), f"{path.name} produced no output"


@pytest.mark.parametrize("path", _runnable_example_paths(), ids=lambda p: p.stem)
def test_example_agrees_on_every_backend(path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """What each example's header promises: `--verify` runs it on the
    tree interpreter, the IR interpreter and the LLVM JIT, and reports a
    result only if all three agree."""
    previous = Path.cwd()
    os.chdir(_REPO_ROOT)
    try:
        exit_code = run_pipeline(path, verify=True)
    finally:
        os.chdir(previous)

    captured = capsys.readouterr()
    assert exit_code == 0, f"{path.name} exited {exit_code}\n{captured.err}"
    assert "[verified: tree, ir, llvm agree]" in captured.err


def test_no_stale_ql_examples_remain() -> None:
    """The project settled on `.qs`. A leftover `.ql` file would be
    skipped by the glob above and could rot unnoticed -- exactly how
    hello_world.ql survived broken for six phases."""
    stale = sorted(p.name for p in _EXAMPLES.glob("*.ql"))
    assert not stale, f"examples still using the old .ql extension: {stale}"


@pytest.mark.parametrize("stem,expected_reason", sorted(_EXPECTED_TO_FAIL.items()))
def test_negative_example_fails_for_its_stated_reason(
    stem: str, expected_reason: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guards the other half of `_EXPECTED_TO_FAIL`: a negative example
    must still fail, and for the reason it's there to demonstrate --
    not pass by accident after an unrelated change, and not fail with
    an unrelated error that would make the demo confusing to run."""
    path = _EXAMPLES / f"{stem}.qs"
    previous = Path.cwd()
    os.chdir(_REPO_ROOT)
    try:
        exit_code = run_pipeline(path, backend="tree")
    finally:
        os.chdir(previous)

    assert exit_code != 0, f"{stem}.qs was expected to fail but exited 0"
    captured = capsys.readouterr()
    # Every compile and runtime error goes to stderr, and nothing ran.
    assert expected_reason in captured.err
    assert captured.out == ""
