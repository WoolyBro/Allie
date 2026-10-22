"""Comments explain the code, not the history of writing it.

A comment like "Phase 6.4 specified this" or "per the spec" tells a
reader when something was asked for, not why it is the way it is -- and
it rots: several such comments described designs that had since
changed. The history belongs in docs/ (ROADMAP.md, the phase notes in
docs/architecture.md), where it is dated and expected. This keeps it out
of the code, the tests and the tooling.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_BUILD_LOG = re.compile(r"\bPhase \d|\bper the spec\b|\bthe spec (asked|said|called)|merge notes")


def _sources() -> list[Path]:
    paths = [_ROOT / "main.py"]
    for folder in ("allie", "tests", "benchmarks", "scripts", "examples"):
        paths += (_ROOT / folder).rglob("*.py")
        paths += (_ROOT / folder).rglob("*.qs")
    return sorted(path for path in paths if path != Path(__file__).resolve())


def test_comments_explain_the_code_not_its_build_log() -> None:
    found = [
        f"{path.relative_to(_ROOT)}:{number}: {line.strip()}"
        for path in _sources()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _BUILD_LOG.search(line)
    ]
    assert not found, "say why, not when it was asked for:\n" + "\n".join(found)


def test_the_check_would_catch_one() -> None:
    assert _BUILD_LOG.search("# Phase 6.4 asked for this")
    assert _BUILD_LOG.search("chosen per the spec")
    assert not _BUILD_LOG.search("# the phase of the moon")
