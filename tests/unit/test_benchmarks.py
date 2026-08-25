"""Tests for the benchmark harness -- Phase 8.2.

A benchmark suite nobody runs in CI rots silently: a workload whose
Allie source stops compiling, or a report generator that crashes on an
edge case, would go unnoticed until someone tried to regenerate the
numbers for the write-up. These tests keep the harness honest without
paying for real timing runs -- every workload is *executed* here, but at
a single trial, and no output files are written.
"""

from __future__ import annotations

import pytest

from benchmarks.run import (
    Timing,
    WorkloadResult,
    _front_end,
    measure,
    render_markdown,
)
from benchmarks.workloads import (
    Workload,
    all_workloads,
    ensure_benchmark_data,
    significant_lines,
)


def test_every_family_is_represented() -> None:
    """Phase 8.2 asked for three distinct families. Without this, losing
    the pure-CS or blended workload would quietly narrow what the
    benchmark actually proves."""
    families = {w.family for w in all_workloads()}
    assert families == {"quant", "pure-cs", "blended"}


def test_benchmark_data_is_generated_and_reproducible() -> None:
    first = ensure_benchmark_data()
    assert first.exists()
    digest = first.read_bytes()
    second = ensure_benchmark_data()
    assert second.read_bytes() == digest  # same seed, same file


@pytest.mark.parametrize(
    "workload", [w for w in all_workloads() if w.allie_source], ids=lambda w: w.name
)
def test_every_allie_workload_compiles(workload: Workload) -> None:
    """The front end (lex + parse + resolve + type-check) must accept
    every workload. This is the check that catches a workload silently
    rotting after a language change."""
    ensure_benchmark_data()
    assert workload.allie_source is not None
    _front_end(workload.allie_source)


def test_python_only_workload_declares_a_reason() -> None:
    """A workload with no Allie side must say why, so a language gap is
    reported rather than looking like an oversight."""
    for workload in all_workloads():
        if workload.allie_source is None:
            assert workload.allie_unsupported_reason, (
                f"{workload.name} has no Allie source and no stated reason"
            )
            assert workload.allie_loc is None


def test_significant_lines_ignores_blanks_and_comments() -> None:
    source = """
// a comment
let x = 1;

# another comment style
let y = 2;
"""
    assert significant_lines(source) == 2


def test_measure_runs_a_workload_end_to_end() -> None:
    """One real (single-trial) measurement of the cheapest workload --
    proves the timing path itself works, not just that sources compile."""
    ensure_benchmark_data()
    workload = next(w for w in all_workloads() if w.name == "backtest_ma")
    result = measure(workload, trials=1)

    assert not result.error, result.error
    for timing in (result.python, result.interpreted, result.jit_unopt, result.jit_opt):
        assert timing is not None
        assert timing.total_s >= 0.0

    # The Allie side has a real compile phase; the hand-written Python
    # does not, and is reported as 0.0 rather than folded into run time.
    assert result.jit_opt is not None and result.jit_opt.compile_s > 0.0
    assert result.python is not None and result.python.compile_s == 0.0


def test_render_markdown_handles_a_workload_with_no_allie_side() -> None:
    """The report generator must not crash (or silently omit) when a
    workload has no Allie implementation."""
    workload = Workload(
        name="python_only",
        family="quant",
        description="d",
        allie_source=None,
        allie_unsupported_reason="no Matrix literal syntax",
        python_impl=lambda: None,
        python_source="pass",
    )
    result = WorkloadResult(workload, Timing(0.0, 0.01, 1.0), None, None, None)
    report = render_markdown([result], trials=3)

    assert "python_only" in report
    assert "n/a" in report
    assert "no Matrix literal syntax" in report


def test_render_markdown_survives_all_measurements_failing() -> None:
    workload = next(w for w in all_workloads() if w.allie_source)
    result = WorkloadResult(workload, None, None, None, None, error="boom")
    report = render_markdown([result], trials=3)
    assert "boom" in report
    assert "No workload produced a complete set of timings." in report
