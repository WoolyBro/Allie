"""Benchmark harness.

    python -m benchmarks.run                    # full run: results.md, .csv, .json
    python -m benchmarks.run --trials 5         # fewer trials (faster, noisier)
    python -m benchmarks.run --quick            # smoke test: 3 trials, prints, writes nothing
    python -m benchmarks.run --only fib_20      # some workloads
    python -m benchmarks.run --compare old.json # flag regressions against an earlier run

For every workload (benchmarks/workloads.py) it times five
implementations:

| key | what it is |
|---|---|
| `python` | the hand-written Python/numpy/pandas equivalent |
| `tree` | the tree-walking interpreter, straight off the AST |
| `ir` | the IR interpreter, on optimized IR |
| `jit_O0` | LLVM + MCJIT with every optimization off: the Allie passes and LLVM's |
| `jit_O2` | LLVM + MCJIT as `allie run --backend llvm` runs it: the passes and LLVM -O2 |

## Methodology

- **Correctness first.** Every workload prints one number; each
  implementation's output is checked against the Python twin's before
  any timing counts. A mismatch fails the workload rather than timing
  two programs that compute different things.
- **Interleaved trials.** Trial k runs every implementation once, in an
  order rotated each trial, so drift (thermal throttling, a background
  job) lands on all of them rather than on whichever ran last. The
  first trial is a warm-up and is discarded.
- **No instrumentation while timing.** Memory is measured in a separate
  run under `tracemalloc`, never during a timed one: tracing every
  allocation slows allocation-heavy Python code several-fold, which
  once inflated the "JIT vs Python" ratio on the loop workload.
  Garbage collection is collected before and disabled during each
  timed run, as `timeit` does.
- **Phases.** The time is split into front end (lex, parse, resolve,
  type-check), lowering (IR, optimizer passes, LLVM IR), code
  generation (LLVM parsing, optimizing and emitting machine code) and
  run -- the JIT's code generation is compilation, not execution, and
  is reported as such.
- **Uncertainty.** Each figure is a median with its interquartile range;
  each ratio comes with a bootstrap 95% confidence interval, and one
  whose interval contains 1 is reported as within noise.
- **Scaling.** One loop at three sizes per backend separates fixed cost
  from cost per iteration, which a single size can't.
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import json
import math
import platform
import subprocess
import sys
import time
import tracemalloc
from collections.abc import Callable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.optimizer import DEFAULT_PASSES
from allie.optimizer import optimize as optimize_ir
from allie.parser.parser import Parser
from allie.quant.backtest_engine import reset_strategy_registry
from allie.runtime.interpreter import run_program
from allie.semantic.lookahead import check_lookahead
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker
from benchmarks.stats import Ratio, Summary, geometric_mean, linear_fit, ratio, summarize
from benchmarks.workloads import ScalingStudy, Workload, all_workloads, ensure_benchmark_data

_HERE = Path(__file__).resolve().parent

IMPLEMENTATIONS = ("python", "tree", "ir", "jit_O0", "jit_O2")
LABELS = {
    "python": "Python",
    "tree": "Allie tree",
    "ir": "Allie IR",
    "jit_O0": "Allie JIT -O0",
    "jit_O2": "Allie JIT -O2",
}
PHASES = ("front", "lower", "codegen", "run")


# ---- one timed run ----


@dataclass
class Sample:
    """One run of one implementation, in seconds per phase."""

    phases: dict[str, float]
    output: str = ""

    @property
    def total(self) -> float:
        return sum(self.phases.values())


class _Clock:
    """Phase timer: `lap(name)` charges the time since the last lap."""

    def __init__(self) -> None:
        self.phases: dict[str, float] = {}
        self._last = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.phases[name] = self.phases.get(name, 0.0) + now - self._last
        self._last = now


def _front_end(source: str):  # noqa: ANN202
    """lex + parse + resolve + type-check + look-ahead: what every Allie
    backend runs before it does anything else."""
    parser = Parser(Lexer(source).scan_tokens())
    program = parser.parse()
    if parser.diagnostics.has_errors():
        raise RuntimeError(f"parse errors: {parser.diagnostics.diagnostics}")
    resolver = Resolver()
    resolver.resolve(program)
    if resolver.diagnostics.has_errors():
        raise RuntimeError(f"resolve errors: {resolver.diagnostics.diagnostics}")
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)
    lookahead = check_lookahead(program)
    if lookahead.has_errors():
        raise RuntimeError(f"look-ahead errors: {lookahead.diagnostics}")
    return program


def _run_allie(source: str, implementation: str) -> Sample:
    reset_strategy_registry()
    out = io.StringIO()
    clock = _Clock()
    program = _front_end(source)
    clock.lap("front")
    with redirect_stdout(out):
        if implementation == "tree":
            run_program(program, stdlib.create_global_environment())
            clock.lap("run")
        else:
            optimized = implementation != "jit_O0"
            main, functions, register_types, function_info = lower_program(program)
            main, functions = optimize_ir(main, functions, DEFAULT_PASSES, optimized)
            if implementation == "ir":
                clock.lap("lower")
                run_ir_program(main, stdlib.create_global_environment(), functions, stdlib.BUILTINS)
                clock.lap("run")
            else:
                lowering = LLVMLowering(functions, function_info)
                lowering.declare_user_functions()
                for name in function_info:
                    lowering.lower_function(name)
                entry = lowering.lower_main(main, register_types)
                clock.lap("lower")
                compile_and_run(
                    lowering.module,
                    entry.name,
                    opt_level=2 if optimized else 0,
                    on_ready=lambda: clock.lap("codegen"),
                )
                clock.lap("run")
    return Sample(clock.phases, out.getvalue())


def _run_python(impl: Callable[[], object]) -> Sample:
    clock = _Clock()
    with redirect_stdout(io.StringIO()):
        value = impl()
    clock.lap("run")
    return Sample(clock.phases, f"{value}\n")


def _timed(run: Callable[[], Sample]) -> Sample:
    gc.collect()
    gc.disable()
    try:
        return run()
    finally:
        gc.enable()


def _peak_kb(run: Callable[[], Sample]) -> float:
    """Peak Python heap during one untimed run."""
    gc.collect()
    tracemalloc.start()
    try:
        run()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / 1024


# ---- a workload ----


@dataclass
class Measured:
    samples: list[Sample]
    peak_kb: float

    def totals(self) -> list[float]:
        return [s.total for s in self.samples]

    def phase(self, name: str) -> list[float]:
        return [s.phases.get(name, 0.0) for s in self.samples]


@dataclass
class WorkloadResult:
    workload: Workload
    measured: dict[str, Measured] = field(default_factory=dict)
    expected: str = ""
    error: str = ""

    def has(self, *implementations: str) -> bool:
        return all(i in self.measured for i in implementations)

    def ratio(self, slower: str, faster: str) -> Ratio:
        return ratio(self.measured[slower].totals(), self.measured[faster].totals())


def _runner(workload: Workload, implementation: str) -> Callable[[], Sample]:
    if implementation == "python":
        return lambda: _run_python(workload.python_impl)
    source = workload.allie_source
    assert source is not None
    return lambda: _run_allie(source, implementation)


def _same_result(expected: str, actual: str) -> bool:
    """The last printed number, compared to 1e-9 relative."""
    lines = [line for line in actual.strip().splitlines() if line.strip()]
    if not lines:
        return False
    try:
        a, b = float(expected), float(lines[-1])
    except ValueError:
        return expected.strip() == lines[-1].strip()
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def measure(workload: Workload, trials: int) -> WorkloadResult:
    """Verifies, then times every implementation `trials` times,
    interleaved, after one discarded warm-up round."""
    result = WorkloadResult(workload)
    implementations = [
        i
        for i in IMPLEMENTATIONS
        if (i == "python" or workload.allie_source is not None) and i not in workload.skip
    ]
    runners = {i: _runner(workload, i) for i in implementations}
    samples: dict[str, list[Sample]] = {i: [] for i in implementations}
    try:
        result.expected = runners["python"]().output.strip()
        for round_ in range(trials + 1):  # round 0 is the warm-up
            shift = round_ % len(implementations)
            for implementation in implementations[shift:] + implementations[:shift]:
                sample = _timed(runners[implementation])
                if round_ == 0:
                    if not _same_result(result.expected, sample.output):
                        result.error = (
                            f"{LABELS[implementation]} printed {sample.output.strip()!r}, "
                            f"but Python returned {result.expected!r}"
                        )
                        return result
                else:
                    samples[implementation].append(sample)
        for implementation in implementations:
            peak = _peak_kb(runners[implementation])
            result.measured[implementation] = Measured(samples[implementation], peak)
    except Exception as error:  # noqa: BLE001 -- reported, not swallowed
        result.error = f"{type(error).__name__}: {error}"
    return result


# ---- the scaling study ----


@dataclass
class ScalingResult:
    sizes: dict[str, tuple[int, ...]]
    medians: dict[str, list[float]]  # seconds, one per size

    def fit(self, implementation: str):  # noqa: ANN201
        return linear_fit(
            list(map(float, self.sizes[implementation])), self.medians[implementation]
        )


def _scaling_runner(study: ScalingStudy, implementation: str, n: int) -> Callable[[], Sample]:
    if implementation == "python":
        return lambda: _run_python(lambda: study.python_impl(n))
    source = study.allie_source(n)
    return lambda: _run_allie(source, implementation)


def measure_scaling(study: ScalingStudy, trials: int) -> ScalingResult:
    medians: dict[str, list[float]] = {}
    for implementation, sizes in study.sizes.items():
        medians[implementation] = []
        for n in sizes:
            run = _scaling_runner(study, implementation, n)
            totals = [_timed(run).total for _ in range(trials + 1)][1:]
            medians[implementation].append(summarize(totals).median)
    return ScalingResult(study.sizes, medians)


# ---- the environment ----


def environment() -> dict[str, str]:
    import llvmlite
    import llvmlite.binding as llvm
    import numpy
    import pandas

    return {
        "machine": f"{_cpu()} ({platform.machine()})",
        "os": platform.platform(terse=True),
        "python": f"{platform.python_implementation()} {platform.python_version()}",
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "llvmlite": f"{llvmlite.__version__} (LLVM {'.'.join(map(str, llvm.llvm_version_info))})",
        "commit": _git_commit(),
        "date": time.strftime("%Y-%m-%d"),
    }


def _cpu() -> str:
    if sys.platform == "darwin":
        brand = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if brand:
            return brand
    return platform.processor() or "unknown CPU"


def _git_commit() -> str:
    commit = _command(["git", "rev-parse", "--short", "HEAD"])
    if not commit:
        return "unknown"
    dirty = _command(["git", "status", "--porcelain", "--untracked-files=no"])
    return f"{commit} (with uncommitted changes)" if dirty else commit


def _command(argv: list[str]) -> str:
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


# ---- the report ----


def _ms(seconds: float) -> str:
    milliseconds = seconds * 1000
    if milliseconds >= 100:
        return f"{milliseconds:.0f}"
    if milliseconds >= 10:
        return f"{milliseconds:.1f}"
    return f"{milliseconds:.2f}"


def _cell(summary: Summary) -> str:
    return f"{_ms(summary.median)} ±{summary.spread * 50:.0f}%"


def _headline(results: list[WorkloadResult], scaling: ScalingResult | None) -> list[str]:
    """Computed from the measurements -- never restated by hand -- and
    over every workload, never only the best one."""
    lines = ["## Headline", ""]
    complete = [r for r in results if not r.error and r.has("python", "tree", "jit_O2")]
    if not complete:
        return lines + ["No workload produced a complete set of timings.", ""]

    if scaling is not None and all(i in scaling.medians for i in ("python", "tree", "jit_O2")):
        per = {
            i: scaling.fit(i).slope * 1e9
            for i in ("python", "tree", "ir", "jit_O2")
            if i in scaling.medians
        }
        lines.append(
            f"- **Per loop iteration, the JIT takes {per['jit_O2']:.1f} ns** against "
            f"CPython's {per['python']:.0f} ns ({per['python'] / per['jit_O2']:.0f}x) and the "
            f"tree interpreter's {per['tree']:.0f} ns ({per['tree'] / per['jit_O2']:.0f}x) "
            "-- the slopes of the scaling study below, which leave out fixed costs."
        )

    for family in ("pure-cs", "quant", "blended"):
        members = [r for r in complete if r.workload.family == family]
        if not members:
            continue
        vs_tree = geometric_mean([r.ratio("tree", "jit_O2").point for r in members])
        vs_python = geometric_mean([r.ratio("jit_O2", "python").point for r in members])
        run_vs_python = geometric_mean(
            [
                ratio(r.measured["jit_O2"].phase("run"), r.measured["python"].totals()).point
                for r in members
            ]
        )
        count = f"{len(members)} workload{'s' if len(members) != 1 else ''}"
        lines.append(
            f"- **{family}** ({count}, geometric mean): the JIT is {_speed(vs_tree)} than the "
            f"tree interpreter, and {_against_python(vs_python)} with compilation included "
            f"-- {_against_python(run_vs_python)} counting only the native run."
        )

    quant = [r for r in complete if r.workload.family == "quant"]
    if quant:
        load = next((r for r in quant if r.workload.name == "load_10k"), None)
        loader = ""
        if load is not None:
            allie = summarize(load.measured["tree"].totals()).median
            pandas = summarize(load.measured["python"].totals()).median
            loader = (
                f" Reading the 10k-row file alone takes Allie {_ms(allie)} ms against "
                f"pandas' {_ms(pandas)} ms (`load_10k`): pandas parses CSV in C, Allie's "
                "`load` in Python with numpy for the numbers -- most of the gap on the "
                "workloads that start by loading it."
            )
        lines.append(
            "- On quant workloads both sides spend their time in the same numpy work, so "
            "compiling the Allie around it changes little, and the JIT's code generation "
            f"is pure overhead there.{loader} The case for the language on this family is "
            "the LOC column and what it checks, not speed."
        )

    if scaling is not None and all(i in scaling.medians for i in ("jit_O0", "jit_O2")):
        o0, o2 = scaling.fit("jit_O0"), scaling.fit("jit_O2")
        if o0.slope > o2.slope and o2.intercept > o0.intercept:
            break_even = (o2.intercept - o0.intercept) / (o0.slope - o2.slope)
            lines.append(
                f"- **-O2 pays for itself after about {_two_figures(break_even)} loop "
                "iterations**: it "
                f"costs {_ms(o2.intercept - o0.intercept)} ms more to compile and saves "
                f"{(o0.slope - o2.slope) * 1e9:.1f} ns per iteration (the scaling fit). "
                "Every workload here is shorter than that."
            )

    verdicts = [(r.workload.name, r.ratio("jit_O0", "jit_O2")) for r in complete if r.has("jit_O0")]
    faster = [name for name, v in verdicts if v.significant and v.point > 1]
    slower = [name for name, v in verdicts if v.significant and v.point < 1]
    noise = [name for name, v in verdicts if not v.significant]
    parts = []
    if faster:
        parts.append(f"faster on {', '.join(f'`{n}`' for n in faster)}")
    if slower:
        parts.append(f"slower on {', '.join(f'`{n}`' for n in slower)}")
    if noise:
        parts.append(f"within noise on {', '.join(f'`{n}`' for n in noise)}")
    if parts:
        lines.append(
            "- **Turning optimization on (the Allie passes and LLVM -O2)** makes the total "
            f"{'; '.join(parts)} -- by a 95% confidence interval on the ratio of medians. "
            "-O2 spends more time generating code to save time running it, so it pays off "
            "only where the program runs long enough; see the optimizer table."
        )
    return lines + [""]


def _two_figures(x: float) -> str:
    """1,456,139 -> 1,500,000: a fitted estimate, so no false precision."""
    if x <= 0:
        return "0"
    digits = max(0, int(math.floor(math.log10(x))) - 1)
    return f"{round(x, -digits):,.0f}"


def _speed(speedup: float) -> str:
    """`speedup` is how many times longer the other takes."""
    return f"{speedup:.1f}x faster" if speedup >= 1 else f"{1 / speedup:.1f}x slower"


def _against_python(time_ratio: float) -> str:
    """`time_ratio` is Allie's time over Python's."""
    return f"{_speed(1 / time_ratio)} than hand-written Python"


def render_markdown(
    results: list[WorkloadResult],
    trials: int,
    env: dict[str, str],
    scaling: ScalingResult | None = None,
) -> str:
    lines = [
        "# Benchmark results",
        "",
        "Generated by `make bench` (`python -m benchmarks.run`) -- regenerate rather than",
        "hand-edit, so the numbers always come from one clean run. Methodology:",
        "[`README.md`](README.md) and `run.py`'s docstring.",
        "",
        "| | |",
        "|---|---|",
        *(f"| {key} | {value} |" for key, value in env.items()),
        f"| trials | {trials} per implementation, interleaved, after a discarded warm-up |",
        "",
    ]
    lines += _headline(results, scaling)

    lines += [
        "## Time per workload",
        "",
        "Median milliseconds, compile and run together, ± half the interquartile range.",
        "Every implementation's result was checked against the Python twin's first.",
        "",
        "| Workload | Family | "
        + " | ".join(LABELS[i] for i in IMPLEMENTATIONS)
        + " | LOC Py | LOC Allie |",
        "|---|---|" + "---|" * len(IMPLEMENTATIONS) + "---|---|",
    ]
    for r in results:
        w = r.workload
        cells = [
            _cell(summarize(r.measured[i].totals())) if i in r.measured else "—"
            for i in IMPLEMENTATIONS
        ]
        lines.append(
            f"| `{w.name}` | {w.family} | "
            + " | ".join(cells)
            + f" | {w.python_loc} | {w.allie_loc if w.allie_loc is not None else '—'} |"
        )

    ratio_rows = [r for r in results if not r.error and r.has("python", "tree", "jit_O2")]
    if ratio_rows:
        lines += [
            "",
            "## Ratios, with 95% confidence intervals",
            "",
            "How many times longer the first takes than the second (above 1: the second "
            "is faster).",
            "",
            "| Workload | tree / JIT -O2 | Python / JIT -O2 | IR / tree |",
            "|---|---|---|---|",
        ]
        for r in ratio_rows:
            ir_vs_tree = r.ratio("ir", "tree").times() if r.has("ir") else "—"
            lines.append(
                f"| `{r.workload.name}` | {r.ratio('tree', 'jit_O2').times()} "
                f"| {r.ratio('python', 'jit_O2').times()} | {ir_vs_tree} |"
            )

    jit_rows = [r for r in results if not r.error and r.has("jit_O0", "jit_O2")]
    if jit_rows:
        lines += [
            "",
            "## Where the JIT's time goes",
            "",
            "Median milliseconds per phase. *Codegen* is LLVM parsing, optimizing and",
            "emitting machine code -- compilation, though it happens at run time.",
            "",
            "| Workload | Build | Front end | Lowering | Codegen | Run |",
            "|---|---|---|---|---|---|",
        ]
        for r in jit_rows:
            for build in ("jit_O0", "jit_O2"):
                m = r.measured[build]
                phases = " | ".join(_ms(summarize(m.phase(p)).median) for p in PHASES)
                lines.append(f"| `{r.workload.name}` | {LABELS[build][10:]} | {phases} |")
        lines += [
            "",
            "## Optimizer effect",
            "",
            "-O0 against -O2, the only difference being `--no-optimize`: the Allie passes",
            "(constant folding, DCE, CSE) and LLVM's -O2 pipeline. Above 1, -O2 is faster.",
            "",
            "| Workload | Run time -O0 / -O2 | Total -O0 / -O2 | Verdict |",
            "|---|---|---|---|",
        ]
        for r in jit_rows:
            run_ratio = ratio(r.measured["jit_O0"].phase("run"), r.measured["jit_O2"].phase("run"))
            total = r.ratio("jit_O0", "jit_O2")
            if not total.significant:
                verdict = "within noise"
            else:
                verdict = "-O2 faster" if total.point > 1 else "-O2 slower overall"
            lines.append(
                f"| `{r.workload.name}` | {run_ratio.times()} | {total.times()} | {verdict} |"
            )

    if scaling is not None:
        lines += [
            "",
            "## Scaling: fixed cost and cost per iteration",
            "",
            "`loop_50k`'s loop at three sizes per implementation, least-squares fit of",
            "median total time against iterations. The slope is what one more iteration",
            "costs; the intercept is what running it at all costs (for the JIT, mostly",
            "compiling).",
            "",
            "| Implementation | Sizes | ns per iteration | Fixed cost (ms) | R² |",
            "|---|---|---|---|---|",
        ]
        for i in IMPLEMENTATIONS:
            if i not in scaling.medians:
                continue
            fit = scaling.fit(i)
            sizes = ", ".join(f"{n:,}" for n in scaling.sizes[i])
            lines.append(
                f"| {LABELS[i]} | {sizes} | {fit.slope * 1e9:.1f} | "
                f"{max(fit.intercept, 0.0) * 1000:.2f} | {fit.r_squared:.4f} |"
            )

    memory_rows = [r for r in results if not r.error and r.measured]
    if memory_rows:
        lines += [
            "",
            "## Peak Python heap (KB)",
            "",
            "From a separate, untimed run under `tracemalloc`. It counts Python allocations",
            "only: numpy buffers are included, but memory LLVM allocates natively for the",
            "JIT is not.",
            "",
            "| Workload | " + " | ".join(LABELS[i] for i in IMPLEMENTATIONS) + " |",
            "|---|" + "---|" * len(IMPLEMENTATIONS),
        ]
        for r in memory_rows:
            cells = [
                f"{r.measured[i].peak_kb:,.0f}" if i in r.measured else "—" for i in IMPLEMENTATIONS
            ]
            lines.append(f"| `{r.workload.name}` | " + " | ".join(cells) + " |")

    problems = [r for r in results if r.error or r.workload.allie_source is None or r.workload.skip]
    if problems:
        lines += ["", "## Not measured", ""]
        for r in problems:
            if r.error:
                lines.append(f"- **`{r.workload.name}`** failed: {r.error}")
            if r.workload.allie_source is None:
                reason = r.workload.allie_unsupported_reason
                lines.append(f"- **`{r.workload.name}`** has no Allie side: {reason}")
            for implementation, reason in r.workload.skip.items():
                lines.append(f"- **`{r.workload.name}`**, {LABELS[implementation]}: {reason}")
    return "\n".join(lines) + "\n"


def write_csv(results: list[WorkloadResult], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "workload", "family", "implementation", "trials", "median_ms", "q1_ms",
                "q3_ms", "min_ms", *(f"{p}_ms" for p in PHASES), "peak_kb", "loc",
            ]
        )  # fmt: skip
        for r in results:
            for implementation, m in r.measured.items():
                s = summarize(m.totals())
                loc = r.workload.python_loc if implementation == "python" else r.workload.allie_loc
                writer.writerow(
                    [
                        r.workload.name, r.workload.family, implementation, s.n,
                        *(f"{x * 1000:.4f}" for x in (s.median, s.q1, s.q3, s.minimum)),
                        *(f"{summarize(m.phase(p)).median * 1000:.4f}" for p in PHASES),
                        f"{m.peak_kb:.0f}", loc,
                    ]
                )  # fmt: skip


def to_json(
    results: list[WorkloadResult], env: dict[str, str], scaling: ScalingResult | None
) -> dict[str, object]:
    """Every raw sample, so a later run can be compared against this one."""
    return {
        "environment": env,
        "workloads": {
            r.workload.name: {
                "error": r.error,
                "expected": r.expected,
                "implementations": {
                    i: {"peak_kb": m.peak_kb, "samples": [s.phases for s in m.samples]}
                    for i, m in r.measured.items()
                },
            }
            for r in results
        },
        "scaling": None
        if scaling is None
        else {"sizes": scaling.sizes, "medians": scaling.medians},
    }


# ---- comparing two runs ----


@dataclass
class Change:
    workload: str
    implementation: str
    ratio: Ratio  # new / old

    @property
    def verdict(self) -> str:
        if self.ratio.low > 1 + REGRESSION_THRESHOLD:
            return "regression"
        if self.ratio.high < 1 - REGRESSION_THRESHOLD:
            return "improvement"
        return "unchanged"


# Smaller than this, a change is not worth failing a build over even when
# it is statistically real.
REGRESSION_THRESHOLD = 0.05


def compare(old: dict[str, object], new: dict[str, object]) -> list[Change]:
    """New against old, for every workload and implementation both have."""
    changes: list[Change] = []
    old_workloads, new_workloads = old["workloads"], new["workloads"]
    assert isinstance(old_workloads, dict) and isinstance(new_workloads, dict)
    for name, entry in new_workloads.items():
        if name not in old_workloads:
            continue
        before, after = old_workloads[name]["implementations"], entry["implementations"]
        for implementation in IMPLEMENTATIONS:
            if implementation in before and implementation in after:
                totals = [
                    [sum(sample.values()) for sample in side[implementation]["samples"]]
                    for side in (after, before)
                ]
                if totals[0] and totals[1]:
                    changes.append(Change(name, implementation, ratio(totals[0], totals[1])))
    return changes


def render_comparison(changes: list[Change]) -> str:
    lines = [
        "| Workload | Implementation | New / old | Verdict |",
        "|---|---|---|---|",
        *(
            f"| `{c.workload}` | {LABELS[c.implementation]} | {c.ratio.times()} | {c.verdict} |"
            for c in changes
        ),
    ]
    return "\n".join(lines) + "\n"


# ---- main ----


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Allie benchmark suite")
    parser.add_argument("--trials", type=int, default=10, help="timed trials per implementation")
    parser.add_argument("--quick", action="store_true", help="3 trials, print, write nothing")
    parser.add_argument("--only", help="comma-separated workload names")
    parser.add_argument("--no-scaling", action="store_true", help="skip the scaling study")
    parser.add_argument("--compare", type=Path, help="an earlier results.json to compare against")
    parser.add_argument(
        "--out", type=Path, default=_HERE, help="where to write results.{md,csv,json}"
    )
    args = parser.parse_args(argv)

    trials = 3 if args.quick else args.trials
    ensure_benchmark_data()
    workloads = all_workloads()
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {w.name for w in workloads}
        if unknown:
            parser.error(f"unknown workload(s): {', '.join(sorted(unknown))}")
        workloads = [w for w in workloads if w.name in wanted]

    results: list[WorkloadResult] = []
    for workload in workloads:
        print(f"  {workload.name} ({workload.family}) ...", file=sys.stderr, flush=True)
        result = measure(workload, trials)
        if result.error:
            print(f"    ! {result.error}", file=sys.stderr, flush=True)
        results.append(result)
    scaling = None
    if not args.no_scaling:
        print("  scaling study ...", file=sys.stderr, flush=True)
        scaling = measure_scaling(ScalingStudy(), trials=min(trials, 5))

    env = environment()
    report = render_markdown(results, trials, env, scaling)
    data = to_json(results, env, scaling)
    status = 1 if any(r.error for r in results) else 0
    if args.compare is not None:
        changes = compare(json.loads(args.compare.read_text(encoding="utf-8")), data)
        print(render_comparison(changes))
        if any(c.verdict == "regression" for c in changes):
            status = 1
    if args.quick:
        print(report)
        return status

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.md").write_text(report, encoding="utf-8")
    write_csv(results, args.out / "results.csv")
    (args.out / "results.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    for name in ("results.md", "results.csv", "results.json"):
        print(f"wrote {args.out / name}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
