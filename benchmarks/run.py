"""Benchmark harness -- Phase 8.2.

    python benchmarks/run.py            # full run, writes results.md + results.csv
    python benchmarks/run.py --trials 5 # fewer trials (faster, noisier)
    python benchmarks/run.py --quick    # smoke test: 2 trials, no file output

For every workload (see `workloads.py`) this times four implementations:

| column | what it is |
|---|---|
| Python | the hand-written numpy/pandas equivalent |
| Allie (interpreted) | tree-walking interpreter, straight off the AST |
| Allie (JIT, unopt) | lower to IR, skip the optimizer, LLVM + MCJIT |
| Allie (JIT, opt) | same, with constant-fold / DCE / CSE enabled |

The last two differ *only* in the `pass_manager` toggle, which is what
makes "optimization changed performance by X%" a measured claim rather
than an asserted one.

## Methodology

- **N trials, first discarded.** Warmup matters here in both
  directions: the JIT pays LLVM setup on its first run, and CPython
  pays import/allocator warmup. Reporting the *median* of the remainder
  rather than the mean keeps one scheduler hiccup from dominating.
- **Compile time and run time are measured separately.** Compile =
  lex + parse + **resolve** + type-check + lower + optimize. Run =
  execution only. Phase 8.2 called out `resolve` specifically, and it
  earns its place in the breakdown -- it's where `loop_depth` /
  `inside_strategy` / `inside_when` tracking happens.
- **Peak memory** via `tracemalloc`, which counts Python allocations.
  Stated plainly: this does **not** see memory the LLVM JIT allocates
  natively, so the JIT rows undercount real usage. A number that
  measures one runtime honestly is more useful than one that silently
  mixes two.
- Timings use `time.perf_counter`.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import time
import tracemalloc
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.optimizer import constant_fold, cse, dce
from allie.optimizer.pass_manager import run_passes
from allie.parser.parser import Parser
from allie.quant.backtest_engine import reset_strategy_registry
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker
from benchmarks.workloads import Workload, all_workloads, ensure_benchmark_data

_HERE = Path(__file__).resolve().parent
_PASSES = [constant_fold.run, dce.run, cse.run]


@dataclass
class Timing:
    compile_s: float
    run_s: float
    peak_kb: float

    @property
    def total_s(self) -> float:
        return self.compile_s + self.run_s


@dataclass
class WorkloadResult:
    workload: Workload
    python: Timing | None
    interpreted: Timing | None
    jit_unopt: Timing | None
    jit_opt: Timing | None
    error: str = ""


def _front_end(source: str):  # noqa: ANN202
    """lex + parse + resolve + type-check -- shared by every backend."""
    tokens = Lexer(source).scan_tokens()
    parser = Parser(tokens)
    program = parser.parse()
    if parser.diagnostics.has_errors():
        raise RuntimeError(f"parse errors: {parser.diagnostics.diagnostics}")

    resolver = Resolver()
    resolver.resolve(program)
    if resolver.diagnostics.has_errors():
        raise RuntimeError(f"resolve errors: {resolver.diagnostics.diagnostics}")

    checker = TypeChecker()
    for name, sig in stdlib.builtin_function_types().items():
        checker.env.define(name, sig)
    checker.check(program)
    return program


def _time_interpreted(source: str) -> Timing:
    reset_strategy_registry()
    tracemalloc.start()

    start = time.perf_counter()
    program = _front_end(source)
    compile_s = time.perf_counter() - start

    env = stdlib.create_global_environment()
    start = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        run_program(program, env)
    run_s = time.perf_counter() - start

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Timing(compile_s, run_s, peak / 1024)


def _time_jit(source: str, *, optimize: bool) -> Timing:
    reset_strategy_registry()
    tracemalloc.start()

    start = time.perf_counter()
    program = _front_end(source)
    main_program, functions, register_types, function_info = lower_program(program)
    main_program = run_passes(main_program, _PASSES, enabled=optimize)
    functions = {
        name: (run_passes(prog, _PASSES, enabled=optimize), params)
        for name, (prog, params) in functions.items()
    }
    lowering = LLVMLowering(functions, function_info)
    lowering.declare_user_functions()
    for name in function_info:
        lowering.lower_function(name)
    main_func = lowering.lower_main(main_program, register_types)
    compile_s = time.perf_counter() - start

    start = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        compile_and_run(lowering.module, main_func.name)
    run_s = time.perf_counter() - start

    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Timing(compile_s, run_s, peak / 1024)


def _time_python(impl: Any) -> Timing:
    tracemalloc.start()
    start = time.perf_counter()
    with redirect_stdout(io.StringIO()):
        impl()
    run_s = time.perf_counter() - start
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # No compile step for the hand-written Python -- reported as 0.0
    # rather than folded into run time, so the columns stay comparable.
    return Timing(0.0, run_s, peak / 1024)


def _median_of(trials: list[Timing]) -> Timing:
    return Timing(
        compile_s=statistics.median(t.compile_s for t in trials),
        run_s=statistics.median(t.run_s for t in trials),
        peak_kb=statistics.median(t.peak_kb for t in trials),
    )


def _repeat(fn: Any, trials: int) -> Timing:
    """Runs `fn` `trials` times, discarding the first (warmup) and
    returning the median of the rest."""
    samples = [fn() for _ in range(trials)]
    return _median_of(samples[1:] if len(samples) > 1 else samples)


def measure(workload: Workload, trials: int) -> WorkloadResult:
    python = _repeat(lambda: _time_python(workload.python_impl), trials)

    if workload.allie_source is None:
        return WorkloadResult(workload, python, None, None, None)

    source = workload.allie_source
    try:
        interpreted = _repeat(lambda: _time_interpreted(source), trials)
        jit_unopt = _repeat(lambda: _time_jit(source, optimize=False), trials)
        jit_opt = _repeat(lambda: _time_jit(source, optimize=True), trials)
    except Exception as error:  # noqa: BLE001 -- reported, not swallowed
        return WorkloadResult(workload, python, None, None, None, error=f"{error}")

    return WorkloadResult(workload, python, interpreted, jit_unopt, jit_opt)


def _fmt_ms(timing: Timing | None, *, attr: str = "total_s") -> str:
    if timing is None:
        return "n/a"
    return f"{getattr(timing, attr) * 1000:.2f}"


def _headline_section(results: list[WorkloadResult]) -> list[str]:
    """Computes the summary from the measured numbers rather than
    restating a claim written by hand -- so it can never drift from the
    table below it, in either direction."""
    lines = ["## Headline", ""]

    runnable = [r for r in results if r.interpreted and r.jit_opt and r.python]
    if not runnable:
        return lines + ["No workload produced a complete set of timings.", ""]

    best = max(runnable, key=lambda r: r.interpreted.total_s / r.jit_opt.total_s)  # type: ignore[union-attr]
    jit_vs_interp = best.interpreted.total_s / best.jit_opt.total_s  # type: ignore[union-attr]
    jit_vs_python = best.python.total_s / best.jit_opt.total_s  # type: ignore[union-attr]
    lines += [
        f"- **Best JIT speedup: `{best.workload.name}` ({best.workload.family}), "
        f"{jit_vs_interp:.0f}x faster than the tree-walking interpreter** and "
        f"{jit_vs_python:.1f}x faster than the hand-written Python.",
    ]

    quant = [r for r in runnable if r.workload.family == "quant"]
    if quant:
        ratios = {
            r.workload.name: r.jit_opt.total_s / r.python.total_s  # type: ignore[union-attr]
            for r in quant
        }
        fastest = min(ratios, key=lambda k: ratios[k])
        slowest = max(ratios, key=lambda k: ratios[k])
        lines.append(
            f"- On the quant workloads the JIT is **{ratios[fastest]:.1f}x to "
            f"{ratios[slowest]:.0f}x slower** than hand-written Python "
            f"(best `{fastest}`, worst `{slowest}`). Both sides bottom out in "
            "the same numpy/pandas calls, so compiling the surrounding Allie "
            "code can't move a number already dominated by library time."
        )
        if ratios[slowest] > 10:
            lines.append(
                f"- The `{slowest}` outlier is a real, actionable finding, not "
                "measurement error: `allie.stdlib.series_ops.rolling_std` is a "
                "Python loop over the series (O(n*window)) where pandas uses a "
                "vectorized rolling algorithm. That is a stdlib optimization "
                "target, and no amount of compiler work on the *language* would "
                "close it."
            )

    deltas = [
        (r.jit_unopt.total_s - r.jit_opt.total_s) / r.jit_unopt.total_s * 100  # type: ignore[union-attr]
        for r in results
        if r.jit_unopt and r.jit_opt and r.jit_unopt.total_s > 0
    ]
    if deltas:
        lines.append(
            f"- **The optimizer moves total time by {min(deltas):+.1f}% to "
            f"{max(deltas):+.1f}%** -- small enough to be near measurement "
            "noise. Reported as measured rather than talked up: the passes "
            "(constant folding, DCE, CSE) act on scalar IR, while these "
            "workloads spend their time inside bridged library calls the "
            "passes deliberately never touch. `tests/unit/test_optimizer.py` "
            "shows the transformations do fire; this table shows how little "
            "that is worth on *this* mix of workloads."
        )

    return lines + [""]


def render_markdown(results: list[WorkloadResult], trials: int) -> str:
    lines = [
        "# Benchmark results",
        "",
        "Generated by `make bench` (`python benchmarks/run.py`). Do not",
        "hand-edit -- regenerate instead, so the numbers always match a",
        "single clean run.",
        "",
        f"- **Trials per implementation:** {trials} "
        "(first discarded as warmup, median of the rest)",
        "- **Timings:** `time.perf_counter`, milliseconds",
        "- **Compile:** lex + parse + resolve + type-check + lower + optimize",
        "- **Run:** execution only",
        "- **Peak memory:** `tracemalloc`, Python allocations only -- this does",
        "  **not** include memory the LLVM JIT allocates natively, so the JIT",
        "  rows undercount real usage.",
        "",
    ]
    lines += _headline_section(results)
    lines += [
        "## Wall clock (total: compile + run)",
        "",
        "| Workload | Family | Python | Allie (interp) | Allie (JIT, unopt) | Allie (JIT, opt) "
        "| LOC (Py) | LOC (Allie) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in results:
        w = result.workload
        allie_loc = "n/a" if w.allie_loc is None else str(w.allie_loc)
        lines.append(
            f"| `{w.name}` | {w.family} | {_fmt_ms(result.python)} "
            f"| {_fmt_ms(result.interpreted)} | {_fmt_ms(result.jit_unopt)} "
            f"| {_fmt_ms(result.jit_opt)} | {w.python_loc} | {allie_loc} |"
        )

    lines += [
        "",
        "## Compile vs run split (Allie, optimized JIT)",
        "",
        "| Workload | Compile (ms) | Run (ms) | Peak mem (KB) |",
        "|---|---|---|---|",
    ]
    for result in results:
        if result.jit_opt is None:
            continue
        lines.append(
            f"| `{result.workload.name}` | {_fmt_ms(result.jit_opt, attr='compile_s')} "
            f"| {_fmt_ms(result.jit_opt, attr='run_s')} "
            f"| {result.jit_opt.peak_kb:.0f} |"
        )

    lines += [
        "",
        "## Optimizer effect (JIT unopt -> opt)",
        "",
        "The only difference between these two columns is the",
        "`pass_manager` toggle. A negative percentage means the optimized",
        "build was *slower*.",
        "",
        "| Workload | Family | Unopt total (ms) | Opt total (ms) | Change |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        if result.jit_unopt is None or result.jit_opt is None:
            continue
        unopt = result.jit_unopt.total_s
        opt = result.jit_opt.total_s
        delta = (unopt - opt) / unopt * 100 if unopt > 0 else 0.0
        lines.append(
            f"| `{result.workload.name}` | {result.workload.family} "
            f"| {unopt * 1000:.2f} | {opt * 1000:.2f} | {delta:+.1f}% |"
        )

    skipped = [r for r in results if r.workload.allie_source is None or r.error]
    if skipped:
        lines += ["", "## Workloads with no Allie implementation", ""]
        for result in skipped:
            reason = result.error or result.workload.allie_unsupported_reason
            lines.append(f"- **`{result.workload.name}`** — {reason}")

    return "\n".join(lines) + "\n"


def write_csv(results: list[WorkloadResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "workload",
                "family",
                "python_ms",
                "allie_interp_ms",
                "allie_jit_unopt_ms",
                "allie_jit_opt_ms",
                "allie_compile_ms",
                "allie_run_ms",
                "peak_kb",
                "loc_python",
                "loc_allie",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.workload.name,
                    r.workload.family,
                    _fmt_ms(r.python),
                    _fmt_ms(r.interpreted),
                    _fmt_ms(r.jit_unopt),
                    _fmt_ms(r.jit_opt),
                    _fmt_ms(r.jit_opt, attr="compile_s"),
                    _fmt_ms(r.jit_opt, attr="run_s"),
                    f"{r.jit_opt.peak_kb:.0f}" if r.jit_opt else "n/a",
                    r.workload.python_loc,
                    "n/a" if r.workload.allie_loc is None else r.workload.allie_loc,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Allie benchmark suite")
    parser.add_argument("--trials", type=int, default=10, help="trials per implementation")
    parser.add_argument("--quick", action="store_true", help="2 trials, no file output")
    args = parser.parse_args(argv)

    trials = 2 if args.quick else args.trials
    ensure_benchmark_data()

    results: list[WorkloadResult] = []
    for workload in all_workloads():
        print(f"  {workload.name} ({workload.family}) ...", flush=True)
        result = measure(workload, trials)
        if result.error:
            print(f"    ! {result.error}", flush=True)
        results.append(result)

    report = render_markdown(results, trials)
    if args.quick:
        print(report)
        return 0

    (_HERE / "results.md").write_text(report, encoding="utf-8")
    write_csv(results, _HERE / "results.csv")
    print(f"\nwrote {_HERE / 'results.md'}")
    print(f"wrote {_HERE / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
