"""Benchmark workload definitions -- Phase 8.2.

Each workload is defined twice: once as an Allie program, once as
equivalent Python/numpy/pandas. `benchmarks/run.py` times both.

Three families, deliberately kept distinct rather than folded together:

- **quant** -- SMA over 10k rows, 200x200 matrix multiply, rolling std
  over 10k rows, and the `MovingAverage` backtest. What the language is
  ostensibly for.
- **pure-CS** -- a `for`-loop-heavy summation with no quant content at
  all. This is the one that shows whether constant folding / DCE / CSE
  matter for control-flow-heavy code, not just series and matrix math.
  Without it, "optimization improved performance by X%" would only ever
  be a claim about numpy call overhead.
- **blended** -- a `for` loop sweeping SMA window sizes and running a
  backtest per window. The only workload that exercises both halves of
  the language *together* rather than side by side.

Line counts (for the results table's LOC columns) are computed from the
source text with blank lines and comment-only lines stripped, so the
comparison measures code rather than commentary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
# The 60-row test fixture is for correctness tests; benchmarks need a
# realistically sized series. `_BIG_CSV` is generated on demand from a
# fixed seed (see `ensure_benchmark_data`) rather than checked in, so
# the repo stays small and the data stays reproducible.
_CSV = _REPO_ROOT / "tests" / "fixtures" / "AAPL_sample.csv"
_BIG_CSV = Path(__file__).resolve().parent / "data" / "synthetic_10k.csv"

# Deterministic synthetic data -- every run benchmarks the same numbers,
# so timings are comparable across runs (docs/spec.md "Determinism").
_RNG_SEED = 20260825
_SERIES_ROWS = 10_000
_MATRIX_SIZE = 200


def _synthetic_series(n: int = _SERIES_ROWS) -> np.ndarray:
    rng = np.random.default_rng(_RNG_SEED)
    return 100.0 * np.cumprod(1.0 + rng.normal(0.0002, 0.01, size=n))


def _synthetic_matrix(n: int = _MATRIX_SIZE) -> np.ndarray:
    rng = np.random.default_rng(_RNG_SEED + 1)
    return rng.normal(size=(n, n))


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
    # side. Reported in the results table rather than silently omitted,
    # so a gap in the language shows up as a gap instead of vanishing.
    allie_unsupported_reason: str = ""

    @property
    def allie_loc(self) -> int | None:
        return significant_lines(self.allie_source) if self.allie_source else None

    @property
    def python_loc(self) -> int:
        return significant_lines(self.python_source)


# ---- quant family ----

_SMA_ALLIE = f"""
let prices = load("{_BIG_CSV.as_posix()}");
let base: Series[Float] = prices.close;
for let rep: Int = 0; rep < 200; rep = rep + 1 {{
    let s: Series[Float] = SMA(base, 20);
    let m: Float = mean(s);
}}
"""

_SMA_PYTHON = """
import pandas as pd
df = pd.read_csv(CSV)
base = df["close"].to_numpy()
for _ in range(200):
    s = pd.Series(base).rolling(20).mean().to_numpy()
    m = float(pd.Series(s).mean())
"""


def _sma_python() -> object:
    df = pd.read_csv(_BIG_CSV)
    base = df["close"].to_numpy()
    result = None
    for _ in range(200):
        s = pd.Series(base).rolling(20).mean().to_numpy()
        result = float(np.nanmean(s))
    return result


_MATMUL_PYTHON = """
import numpy as np
c = a @ b
n = float(np.sqrt((c * c).sum()))
"""


def _matmul_python() -> object:
    a = _synthetic_matrix()
    b = _synthetic_matrix()
    c = a @ b
    # Frobenius norm, not determinant: det() of a 200x200 random matrix
    # overflows float64 and reports a meaningless inf.
    return float(np.sqrt((c * c).sum()))


_ROLLING_STD_ALLIE = f"""
let prices = load("{_BIG_CSV.as_posix()}");
let base: Series[Float] = prices.close;
for let rep: Int = 0; rep < 5; rep = rep + 1 {{
    let s: Series[Float] = rolling_std(base, 20);
    let m: Float = mean(s);
}}
"""

_ROLLING_STD_PYTHON = """
import pandas as pd
df = pd.read_csv(CSV)
base = df["close"].to_numpy()
for _ in range(5):
    s = pd.Series(base).rolling(20).std().to_numpy()
    m = float(pd.Series(s).mean())
"""


def _rolling_std_python() -> object:
    df = pd.read_csv(_BIG_CSV)
    base = df["close"].to_numpy()
    result = None
    for _ in range(5):
        s = pd.Series(base).rolling(20).std().to_numpy()
        result = float(np.nanmean(s))
    return result


_BACKTEST_ALLIE = f"""
let AAPL = load("{_CSV.as_posix()}");
let fast: Series[Float] = SMA(AAPL.close, 5);
let slow: Series[Float] = SMA(AAPL.close, 20);
strategy MovingAverage {{
    when fast crosses_above slow: buy();
    when fast crosses_below slow: sell();
}}
backtest MovingAverage on AAPL from 0 to 100;
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
peak = np.maximum.accumulate(eq)
max_dd = float(np.max((peak - eq) / peak))
"""


def _backtest_python() -> object:
    df = pd.read_csv(_CSV)
    close = df["close"].to_numpy()
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


# ---- pure-CS family ----
#
# No quant content whatsoever. Written with redundant subexpressions and
# a dead store on purpose: this is the workload that shows whether
# constant folding / DCE / CSE do anything for control-flow-heavy code.

_LOOP_ALLIE = """
let total: Int = 0;
let dead: Int = 0;
for let i: Int = 0; i < 50000; i = i + 1 {
    dead = 7 * 6;
    if i % 3 == 0 { continue; }
    total = total + (i * 2) + (i * 2);
}
"""

_LOOP_PYTHON = """
total = 0
for i in range(50000):
    dead = 7 * 6
    if i % 3 == 0:
        continue
    total = total + (i * 2) + (i * 2)
"""


def _loop_python() -> object:
    total = 0
    for i in range(50_000):
        _dead = 7 * 6
        if i % 3 == 0:
            continue
        total = total + (i * 2) + (i * 2)
    return total


# ---- blended family ----

_BLENDED_ALLIE = f"""
let AAPL = load("{_CSV.as_posix()}");
for let w: Int = 5; w <= 40; w = w + 5 {{
    let fast: Series[Float] = SMA(AAPL.close, w);
    let slow: Series[Float] = SMA(AAPL.close, 40);
    let spread: Float = mean(fast) - mean(slow);
    let vol: Float = std(fast);
}}
"""

_BLENDED_PYTHON = """
import numpy as np, pandas as pd
df = pd.read_csv(CSV)
close = df["close"].to_numpy()
for w in range(5, 41, 5):
    fast = pd.Series(close).rolling(w).mean().to_numpy()
    slow = pd.Series(close).rolling(40).mean().to_numpy()
    spread = np.nanmean(fast) - np.nanmean(slow)
    vol = np.nanstd(fast, ddof=1)
"""


def _blended_python() -> object:
    df = pd.read_csv(_CSV)
    close = df["close"].to_numpy()
    result = None
    for w in range(5, 41, 5):
        fast = pd.Series(close).rolling(w).mean().to_numpy()
        slow = pd.Series(close).rolling(40).mean().to_numpy()
        spread = float(np.nanmean(fast) - np.nanmean(slow))
        _vol = float(np.nanstd(fast, ddof=1))
        result = spread
    return result


def all_workloads() -> list[Workload]:
    return [
        Workload(
            name="sma_10k",
            family="quant",
            description="SMA(20) + mean over a price series, 200 repetitions",
            allie_source=_SMA_ALLIE,
            python_impl=_sma_python,
            python_source=_SMA_PYTHON,
        ),
        Workload(
            name="rolling_std",
            family="quant",
            description=(
                "rolling_std(20) + mean, 5 repetitions -- only 5 because "
                "allie.stdlib.series_ops.rolling_std is a Python loop over "
                "the series (O(n*window)) while pandas uses a vectorized "
                "rolling algorithm; 50 repetitions took ~23s. The gap this "
                "workload reports is a real optimization target, not noise."
            ),
            allie_source=_ROLLING_STD_ALLIE,
            python_impl=_rolling_std_python,
            python_source=_ROLLING_STD_PYTHON,
        ),
        Workload(
            name="backtest_ma",
            family="quant",
            description="MovingAverage crossover backtest with full metrics",
            allie_source=_BACKTEST_ALLIE,
            python_impl=_backtest_python,
            python_source=_BACKTEST_PYTHON,
        ),
        Workload(
            name="matmul_200",
            family="quant",
            description="200x200 matrix multiply + Frobenius norm",
            allie_source=None,
            allie_unsupported_reason=(
                "Allie cannot construct a Matrix from source: array literals "
                "produce Series (a nested one types as Series<Series<Float>>), "
                "and every Matrix builtin requires a Matrix argument, so there "
                "is no constructor to start from. Verified, not assumed."
            ),
            python_impl=_matmul_python,
            python_source=_MATMUL_PYTHON,
        ),
        Workload(
            name="loop_sum",
            family="pure-cs",
            description="50k-iteration for loop, no quant content",
            allie_source=_LOOP_ALLIE,
            python_impl=_loop_python,
            python_source=_LOOP_PYTHON,
        ),
        Workload(
            name="blended_sweep",
            family="blended",
            description="for loop sweeping SMA windows, stats per window",
            allie_source=_BLENDED_ALLIE,
            python_impl=_blended_python,
            python_source=_BLENDED_PYTHON,
        ),
    ]


__all__ = [
    "Workload",
    "ensure_benchmark_data",
    "all_workloads",
    "significant_lines",
]
