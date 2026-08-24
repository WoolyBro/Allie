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


def _example_paths() -> list[Path]:
    return sorted(_EXAMPLES.glob("*.qs"))


def test_examples_directory_is_not_empty() -> None:
    assert _example_paths(), "no examples found -- did the glob pattern change?"


@pytest.mark.parametrize("path", _example_paths(), ids=lambda p: p.stem)
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


def test_no_stale_ql_examples_remain() -> None:
    """The project settled on `.qs`. A leftover `.ql` file would be
    skipped by the glob above and could rot unnoticed -- exactly how
    hello_world.ql survived broken for six phases."""
    stale = sorted(p.name for p in _EXAMPLES.glob("*.ql"))
    assert not stale, f"examples still using the old .ql extension: {stale}"
