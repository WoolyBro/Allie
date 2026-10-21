"""Tests for the benchmark harness (benchmarks/).

A benchmark suite nobody runs rots silently: a workload that stops
compiling, a Python twin that drifts from its Allie program, or a
statistic computed wrong would go unnoticed until someone regenerated
the numbers. These keep the harness honest without real timing runs:
every workload is executed and checked here, but once, and no files are
written.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import stats
from benchmarks.run import (
    IMPLEMENTATIONS,
    Measured,
    Sample,
    WorkloadResult,
    _front_end,
    _run_allie,
    _run_python,
    _same_result,
    compare,
    environment,
    main,
    measure,
    measure_scaling,
    render_comparison,
    render_markdown,
    to_json,
    write_csv,
)
from benchmarks.workloads import (
    ScalingStudy,
    Workload,
    all_workloads,
    ensure_benchmark_data,
    loop_allie,
    loop_python,
    significant_lines,
)

_ENV = {"machine": "test machine", "commit": "abc123"}


def _workload(allie: str | None = "print(1 + 2);", python: object = 3) -> Workload:
    return Workload(
        name="tiny",
        family="pure-cs",
        description="d",
        allie_source=allie,
        python_impl=lambda: python,
        python_source="x = 1 + 2",
    )


# ---- the workloads ----


def test_every_family_is_represented() -> None:
    assert {w.family for w in all_workloads()} == {"quant", "pure-cs", "blended"}


def test_benchmark_data_is_generated_and_reproducible() -> None:
    first = ensure_benchmark_data()
    digest = first.read_bytes()
    assert ensure_benchmark_data().read_bytes() == digest  # same seed, same file


@pytest.mark.parametrize("workload", all_workloads(), ids=lambda w: w.name)
def test_every_implementation_of_every_workload_computes_the_same_answer(
    workload: Workload,
) -> None:
    """The guarantee the timing tables rest on: an implementation that
    computes something else isn't being benchmarked, it's being
    compared with a different program."""
    ensure_benchmark_data()
    assert workload.allie_source is not None, "every workload has an Allie side now"
    _front_end(workload.allie_source)
    expected = _run_python(workload.python_impl).output.strip()
    for implementation in IMPLEMENTATIONS[1:]:
        output = _run_allie(workload.allie_source, implementation).output
        assert _same_result(expected, output), (implementation, output, expected)


def test_the_scaling_loop_agrees_with_its_python_twin() -> None:
    for n in (0, 1, 7, 100):
        output = _run_allie(loop_allie(n), "jit_O2").output
        assert int(output) == loop_python(n)


def test_significant_lines_ignores_blanks_and_comments() -> None:
    assert significant_lines("\n// a comment\nlet x = 1;\n\n# another\nlet y = 2;\n") == 2


# ---- statistics ----


def test_summarize() -> None:
    s = stats.summarize([5.0, 1.0, 3.0, 2.0, 4.0])
    assert (s.median, s.q1, s.q3, s.minimum, s.n) == (3.0, 2.0, 4.0, 1.0, 5)
    assert s.spread == pytest.approx(2.0 / 3.0)
    assert stats.summarize([2.0]).spread == 0.0
    with pytest.raises(ValueError, match="no samples"):
        stats.summarize([])


def test_a_clear_difference_is_significant_and_noise_is_not() -> None:
    slow = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0]
    fast = [1.0, 1.05, 0.95, 1.0, 1.02, 0.98]
    clear = stats.ratio(slow, fast)
    assert clear.point == pytest.approx(2.0, rel=0.05)
    assert clear.low <= clear.point <= clear.high
    assert clear.significant
    noisy = stats.ratio([1.0, 1.2, 0.9, 1.1], [1.1, 0.95, 1.05, 1.0])
    assert not noisy.significant
    assert stats.ratio(slow, fast) == clear  # seeded: the same samples, the same interval


def test_ratio_formatting() -> None:
    assert stats.Ratio(12.34, 11.8, 12.9).times() == "12.3x [11.8, 12.9]"
    assert stats.Ratio(0.5, 0.45, 0.55).times() == "0.50x [0.45, 0.55]"


def test_linear_fit_recovers_slope_and_intercept() -> None:
    fit = stats.linear_fit([1.0, 2.0, 3.0, 4.0], [3.0, 5.0, 7.0, 9.0])
    assert (fit.slope, fit.intercept, fit.r_squared) == pytest.approx((2.0, 1.0, 1.0))
    assert stats.geometric_mean([1.0, 4.0]) == pytest.approx(2.0)


# ---- measuring ----


def test_measure_verifies_then_times_every_implementation() -> None:
    result = measure(_workload(), trials=2)
    assert not result.error, result.error
    assert result.expected == "3"
    assert set(result.measured) == set(IMPLEMENTATIONS)
    jit = result.measured["jit_O2"]
    assert len(jit.samples) == 2  # the warm-up round is discarded
    assert set(jit.samples[0].phases) == {"front", "lower", "codegen", "run"}
    assert all(phase > 0 for phase in jit.samples[0].phases.values())
    assert set(result.measured["python"].samples[0].phases) == {"run"}
    assert jit.peak_kb > 0


def test_a_twin_that_computes_something_else_fails_the_workload() -> None:
    result = measure(_workload(python=4), trials=1)
    assert result.error == "Allie tree printed '3', but Python returned '4'"
    assert not result.measured


def test_an_allie_program_that_fails_is_reported_not_raised() -> None:
    result = measure(_workload(allie="print(1 / 0);"), trials=1)
    assert "division by zero" in result.error


def test_the_last_printed_number_is_compared_to_relative_precision() -> None:
    assert _same_result("0.1", "noise\n0.10000000000000001\n")
    assert not _same_result("0.1", "0.1001\n")
    assert not _same_result("1", "")


def test_measure_scaling_fits_every_implementation() -> None:
    study = ScalingStudy(sizes={"python": (10, 20, 40), "jit_O0": (10, 20, 40)})
    result = measure_scaling(study, trials=1)
    assert set(result.medians) == {"python", "jit_O0"}
    assert result.fit("python").r_squared <= 1.0


# ---- the report ----


def _result(name: str = "w", scale: float = 1.0) -> WorkloadResult:
    def samples(base: float) -> list[Sample]:
        return [
            Sample({"front": 0.001, "lower": 0.001, "codegen": 0.002, "run": base * scale * k})
            for k in (1.0, 1.02, 0.98, 1.01, 0.99)
        ]

    workload = _workload()
    workload.name = name
    measured = {
        "python": Measured([Sample({"run": 0.01 * k}) for k in (1.0, 1.01, 0.99)], 10.0),
        "tree": Measured(samples(1.0), 20.0),
        "ir": Measured(samples(0.8), 20.0),
        "jit_O0": Measured(samples(0.02), 30.0),
        "jit_O2": Measured(samples(0.01), 30.0),
    }
    return WorkloadResult(workload, measured, expected="3")


def test_render_markdown_states_the_environment_and_every_table() -> None:
    report = render_markdown([_result()], trials=5, env=_ENV)
    for heading in (
        "| machine | test machine |",
        "## Headline",
        "## Time per workload",
        "## Ratios, with 95% confidence intervals",
        "## Where the JIT's time goes",
        "## Optimizer effect",
        "## Peak Python heap (KB)",
    ):
        assert heading in report


def test_render_markdown_reports_failures_and_gaps() -> None:
    failed = WorkloadResult(_workload(), error="boom")
    gap = WorkloadResult(_workload(allie=None))
    gap.workload.allie_unsupported_reason = "no such feature"
    report = render_markdown([failed, gap], trials=3, env=_ENV)
    assert "failed: boom" in report
    assert "has no Allie side: no such feature" in report
    assert "No workload produced a complete set of timings." in report


def test_csv_and_json_carry_every_sample(tmp_path: Path) -> None:
    result = _result()
    write_csv([result], tmp_path / "results.csv")
    rows = (tmp_path / "results.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("workload,family,implementation,trials,median_ms")
    assert len(rows) == 1 + len(IMPLEMENTATIONS)
    data = to_json([result], _ENV, None)
    samples = data["workloads"]["w"]["implementations"]["tree"]["samples"]  # type: ignore[index]
    assert len(samples) == 5
    json.dumps(data)  # serializable


def test_compare_flags_a_regression_and_ignores_noise() -> None:
    old = to_json([_result()], _ENV, None)
    same = compare(old, to_json([_result()], _ENV, None))
    assert {c.verdict for c in same} == {"unchanged"}
    slower = compare(old, to_json([_result(scale=1.5)], _ENV, None))
    verdicts = {c.implementation: c.verdict for c in slower}
    assert verdicts["tree"] == "regression"
    assert verdicts["python"] == "unchanged"  # its samples didn't change
    assert "regression" in render_comparison(slower)


def test_environment_names_the_machine_and_versions() -> None:
    env = environment()
    for key in ("machine", "os", "python", "numpy", "llvmlite", "commit", "date"):
        assert env[key]
    assert math.isfinite(len(env["llvmlite"]))


def test_an_unknown_workload_is_a_usage_error() -> None:
    with pytest.raises(SystemExit):
        main(["--quick", "--only", "no_such_workload"])
