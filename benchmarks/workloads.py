"""Benchmark workloads: each defined twice, as an Allie program and as the
equivalent hand-written Python, and checked to compute the same answer.

Three families, deliberately kept distinct:

- **quant** -- loading a 10k-row price file, SMA and rolling std over
  it, a 200x200 matrix multiply, and a moving-average backtest over 10k
  bars. What the language is ostensibly for. The load is also measured
  alone, since every other quant workload starts with it.
- **pure-cs** -- a `for` loop and a recursive function, no quant content
  at all. These show whether the compiler does anything for
  control-flow-heavy code; without them every speedup would be a
  statement about numpy call overhead.
- **blended** -- a `for` loop sweeping SMA windows and summarizing each:
  both halves of the language together rather than side by side.

**Every workload ends by printing one number, and the Python twin
returns the same number.** The harness checks that every implementation
agrees before it times any of them (benchmarks/run.py), so a benchmark
can't quietly compare two programs that compute different things.

Line counts (the LOC columns) are computed from the source with blank
lines and comment-only lines stripped, so they measure code rather than
commentary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The 60-row test fixture is for correctness tests; benchmarks need a
# realistically sized series. `_BIG_CSV` is generated on demand from a
# fixed seed (see `ensure_benchmark_data`) rather than checked in, so the
# repo stays small and the data reproducible.
_BIG_CSV = Path(__file__).resolve().parent / "data" / "synthetic_10k.csv"

# Deterministic inputs -- every run benchmarks the same numbers, so
# timings are comparable across runs (docs/spec.md "Determinism").
_RNG_SEED = 20260825
_SERIES_ROWS = 10_000
_MATRIX_SIZE = 200


def _synthetic_series(n: int = _SERIES_ROWS) -> np.ndarray:
    rng = np.random.default_rng(_RNG_SEED)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, size=n))


def ensure_benchmark_data() -> Path:
    """Writes the 10k-row OHLCV benchmark CSV if absent, and returns its
    path. Deterministic: same seed, same file, every run."""
    if _BIG_CSV.exists():
        return _BIG_CSV
    _BIG_CSV.parent.mkdir(parents=True, exist_ok=True)
    closes = _synthetic_series()
    opens = np.concatenate([[closes[0]], closes[:-1]])
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-03", periods=len(closes), freq="B"),
            "open": opens,
            "high": np.maximum(opens, closes) * 1.002,
            "low": np.minimum(opens, closes) * 0.998,
            "close": closes,
            "volume": np.full(len(closes), 1_000_000),
        }
    )
    frame.to_csv(_BIG_CSV, index=False)
    return _BIG_CSV


def significant_lines(source: str) -> int:
    """Non-blank, non-comment-only lines. Both languages use `//`- or
    `#`-style leading comments, so this is a fair like-for-like count."""
    count = 0
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith(("//", "#")):
            continue
        count += 1
    return count


@dataclass
class Workload:
    name: str
    family: str  # "quant" | "pure-cs" | "blended"
    description: str
    allie_source: str | None
    python_impl: Callable[[], object]
    python_source: str
    # Set when `allie_source is None`: why this workload has no Allie
    # side. Reported in the results rather than silently omitted, so a
    # gap in the language shows up as a gap instead of vanishing.
    allie_unsupported_reason: str = ""
    # Implementations too slow to be worth timing at this size, with why.
    skip: dict[str, str] = field(default_factory=dict)

    @property
    def allie_loc(self) -> int | None:
        return significant_lines(self.allie_source) if self.allie_source else None

    @property
    def python_loc(self) -> int:
        return significant_lines(self.python_source)


# ---- quant family ----

_LOAD_ALLIE = f"""
let prices = load("{_BIG_CSV.as_posix()}");
print(prices.close[len(prices.close) - 1]);
"""

_LOAD_PYTHON = """
import pandas as pd
df = pd.read_csv(CSV)
last = float(df["close"].iloc[-1])
"""


def _load_python() -> object:
    return float(pd.read_csv(_BIG_CSV)["close"].iloc[-1])


_SMA_ALLIE = f"""
let prices = load("{_BIG_CSV.as_posix()}");
let base: Series[Float] = prices.close;
let m: Float = 0.0;
for let rep: Int = 0; rep < 200; rep = rep + 1 {{
    let s: Series[Float] = SMA(base, 20);
    m = mean(s);
}}
print(m);
"""

_SMA_PYTHON = """
import pandas as pd
df = pd.read_csv(CSV)
base = df["close"].to_numpy()
for _ in range(200):
    s = pd.Series(base).rolling(20).mean()
    m = float(s.mean())
"""


def _sma_python() -> object:
    base = pd.read_csv(_BIG_CSV)["close"].to_numpy()
    m = 0.0
    for _ in range(200):
        s = pd.Series(base).rolling(20).mean()
        m = float(s.mean())
    return m


_ROLLING_STD_ALLIE = f"""
let prices = load("{_BIG_CSV.as_posix()}");
let base: Series[Float] = prices.close;
let m: Float = 0.0;
for let rep: Int = 0; rep < 20; rep = rep + 1 {{
    let s: Series[Float] = rolling_std(base, 20);
    m = mean(s);
}}
print(m);
"""

_ROLLING_STD_PYTHON = """
import pandas as pd
df = pd.read_csv(CSV)
base = df["close"].to_numpy()
for _ in range(20):
    s = pd.Series(base).rolling(20).std()
    m = float(s.mean())
"""


def _rolling_std_python() -> object:
    base = pd.read_csv(_BIG_CSV)["close"].to_numpy()
    m = 0.0
    for _ in range(20):
        s = pd.Series(base).rolling(20).std()
        m = float(s.mean())
    return m


_BACKTEST_ALLIE = f"""
let prices = load("{_BIG_CSV.as_posix()}");
let fast: Series[Float] = SMA(prices.close, 5);
let slow: Series[Float] = SMA(prices.close, 20);
strategy MovingAverage {{
    when fast crosses_above slow: buy();
    when fast crosses_below slow: sell();
}}
// The model the Python twin implements: fill at the signal bar's close,
// no costs -- so both sides compute the same Sharpe ratio.
let r = backtest MovingAverage on prices from 0 to len(prices.close)
    with fill: "close", slippage: 0.0;
print(r.sharpe);
"""

_BACKTEST_PYTHON = """
import numpy as np, pandas as pd
df = pd.read_csv(CSV)
close = df["close"].to_numpy()
fast = pd.Series(close).rolling(5).mean().to_numpy()
slow = pd.Series(close).rolling(20).mean().to_numpy()
up = (fast > slow) & (np.roll(fast, 1) <= np.roll(slow, 1))
down = (fast < slow) & (np.roll(fast, 1) >= np.roll(slow, 1))
up[0] = down[0] = False
cash, shares, equity = 10000.0, 0.0, []
for i in range(len(close)):
    if up[i] and shares == 0.0:
        shares, cash = cash / close[i], 0.0
    elif down[i] and shares > 0.0:
        cash, shares = shares * close[i], 0.0
    equity.append(cash + shares * close[i])
eq = np.asarray(equity)
returns = np.diff(eq) / eq[:-1]
sharpe = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
"""


def _backtest_python() -> object:
    close = pd.read_csv(_BIG_CSV)["close"].to_numpy()
    fast = pd.Series(close).rolling(5).mean().to_numpy()
    slow = pd.Series(close).rolling(20).mean().to_numpy()
    up = (fast > slow) & (np.roll(fast, 1) <= np.roll(slow, 1))
    down = (fast < slow) & (np.roll(fast, 1) >= np.roll(slow, 1))
    up[0] = down[0] = False

    cash, shares = 10_000.0, 0.0
    equity: list[float] = []
    for i in range(len(close)):
        if up[i] and shares == 0.0:
            shares, cash = cash / close[i], 0.0
        elif down[i] and shares > 0.0:
            cash, shares = shares * close[i], 0.0
        equity.append(cash + shares * close[i])

    eq = np.asarray(equity)
    returns = np.diff(eq) / eq[:-1]
    return float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))


_MATMUL_ALLIE = f"""
let n: Int = {_MATRIX_SIZE};
let a = matrix(normal(0.0, 1.0, n * n, {_RNG_SEED}), n, n);
let b = matrix(normal(0.0, 1.0, n * n, {_RNG_SEED + 1}), n, n);
let c = dot(a, b);
// Frobenius norm: det() of a random 200x200 matrix overflows to inf.
let total: Float = 0.0;
for let i: Int = 0; i < n; i = i + 1 {{
    let r: Series[Float] = row(c, i);
    total = total + sum(r * r);
}}
print(sqrt(total));
"""

_MATMUL_PYTHON = """
import numpy as np
a = np.random.default_rng(SEED).normal(0.0, 1.0, n * n).reshape(n, n)
b = np.random.default_rng(SEED + 1).normal(0.0, 1.0, n * n).reshape(n, n)
c = a @ b
norm = float(np.sqrt((c * c).sum()))
"""


def _matmul_python() -> object:
    n = _MATRIX_SIZE
    a = np.random.default_rng(_RNG_SEED).normal(0.0, 1.0, n * n).reshape(n, n)
    b = np.random.default_rng(_RNG_SEED + 1).normal(0.0, 1.0, n * n).reshape(n, n)
    c = a @ b
    return float(np.sqrt((c * c).sum()))


# ---- pure-CS family ----
#
# No quant content. `loop_sum` is written with redundant subexpressions
# and a dead store on purpose: it's where constant folding / DCE / CSE
# could show up at all.

_LOOP_ITERATIONS = 50_000


def loop_allie(n: int) -> str:
    return f"""
let total: Int = 0;
let dead: Int = 0;
for let i: Int = 0; i < {n}; i = i + 1 {{
    dead = 7 * 6;
    if i % 3 == 0 {{ continue; }}
    total = total + (i * 2) + (i * 2);
}}
print(total);
"""


_LOOP_PYTHON = """
total = 0
for i in range(n):
    dead = 7 * 6
    if i % 3 == 0:
        continue
    total = total + (i * 2) + (i * 2)
"""


def loop_python(n: int) -> int:
    total = 0
    for i in range(n):
        _dead = 7 * 6
        if i % 3 == 0:
            continue
        total = total + (i * 2) + (i * 2)
    return total


_FIB_N = 20

_FIB_ALLIE = f"""
func fib(n: Int) -> Int {{
    if n < 2 {{ return n; }}
    return fib(n - 1) + fib(n - 2);
}}
print(fib({_FIB_N}));
"""

_FIB_PYTHON = """
def fib(n):
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)
result = fib(20)
"""


def _fib(n: int) -> int:
    return n if n < 2 else _fib(n - 1) + _fib(n - 2)


# ---- blended family ----

_BLENDED_ALLIE = f"""
let prices = load("{_BIG_CSV.as_posix()}");
let slow: Series[Float] = SMA(prices.close, 40);
let spread: Float = 0.0;
for let w: Int = 5; w <= 40; w = w + 5 {{
    let fast: Series[Float] = SMA(prices.close, w);
    spread = mean(fast) - mean(slow);
    let vol: Float = std(fast);
}}
print(spread);
"""

_BLENDED_PYTHON = """
import pandas as pd
df = pd.read_csv(CSV)
close = pd.Series(df["close"].to_numpy())
slow = close.rolling(40).mean()
for w in range(5, 41, 5):
    fast = close.rolling(w).mean()
    spread = fast.mean() - slow.mean()
    vol = fast.std()
"""


def _blended_python() -> object:
    close = pd.Series(pd.read_csv(_BIG_CSV)["close"].to_numpy())
    slow = close.rolling(40).mean()
    spread = 0.0
    for w in range(5, 41, 5):
        fast = close.rolling(w).mean()
        spread = float(fast.mean() - slow.mean())
        _vol = float(fast.std())
    return spread


def all_workloads() -> list[Workload]:
    return [
        Workload(
            name="load_10k",
            family="quant",
            description=(
                "reading the 10k-row OHLCV file -- the cost every other quant workload "
                "starts with, measured alone"
            ),
            allie_source=_LOAD_ALLIE,
            python_impl=_load_python,
            python_source=_LOAD_PYTHON,
        ),
        Workload(
            name="sma_10k",
            family="quant",
            description="SMA(20) + mean over 10k rows, 200 repetitions",
            allie_source=_SMA_ALLIE,
            python_impl=_sma_python,
            python_source=_SMA_PYTHON,
        ),
        Workload(
            name="rolling_std_10k",
            family="quant",
            description="rolling_std(20) + mean over 10k rows, 20 repetitions",
            allie_source=_ROLLING_STD_ALLIE,
            python_impl=_rolling_std_python,
            python_source=_ROLLING_STD_PYTHON,
        ),
        Workload(
            name="backtest_10k",
            family="quant",
            description="moving-average crossover backtest over 10k bars",
            allie_source=_BACKTEST_ALLIE,
            python_impl=_backtest_python,
            python_source=_BACKTEST_PYTHON,
        ),
        Workload(
            name="matmul_200",
            family="quant",
            description="two random 200x200 matrices, their product, its Frobenius norm",
            allie_source=_MATMUL_ALLIE,
            python_impl=_matmul_python,
            python_source=_MATMUL_PYTHON,
        ),
        Workload(
            name="loop_50k",
            family="pure-cs",
            description="50k-iteration for loop with integer arithmetic, no quant content",
            allie_source=loop_allie(_LOOP_ITERATIONS),
            python_impl=lambda: loop_python(_LOOP_ITERATIONS),
            python_source=_LOOP_PYTHON,
        ),
        Workload(
            name="fib_20",
            family="pure-cs",
            description="naive recursive Fibonacci of 20: ~22k function calls",
            allie_source=_FIB_ALLIE,
            python_impl=lambda: _fib(_FIB_N),
            python_source=_FIB_PYTHON,
        ),
        Workload(
            name="blended_sweep",
            family="blended",
            description="for loop sweeping 8 SMA windows over 10k rows, statistics per window",
            allie_source=_BLENDED_ALLIE,
            python_impl=_blended_python,
            python_source=_BLENDED_PYTHON,
        ),
    ]


# ---- the scaling study ----


@dataclass
class ScalingStudy:
    """`loop_50k` at several sizes, so a backend's time can be split into
    a fixed cost (the intercept: compilation, setup) and a cost per
    iteration (the slope). A single size can't tell those apart: at 50k
    iterations the JIT's time is mostly LLVM compiling the loop, not
    running it."""

    name: str = "loop"
    allie_source: Callable[[int], str] = loop_allie
    python_impl: Callable[[int], object] = loop_python
    # Per implementation, since an interpreter at the JIT's sizes would
    # take minutes.
    sizes: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: {
            "python": (100_000, 400_000, 1_600_000),
            "tree": (5_000, 20_000, 80_000),
            "ir": (5_000, 20_000, 80_000),
            # Large enough that the native run, not compilation noise,
            # dominates the fit; 16M iterations still fit an Int's sum.
            "jit_O0": (1_000_000, 4_000_000, 16_000_000),
            "jit_O2": (1_000_000, 4_000_000, 16_000_000),
        }
    )


__all__ = [
    "ScalingStudy",
    "Workload",
    "all_workloads",
    "ensure_benchmark_data",
    "loop_allie",
    "loop_python",
    "significant_lines",
]
