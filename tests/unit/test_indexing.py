"""`s[i]` and `len(s)` end to end: type rules, runtime errors, and
agreement across all three backends with the optimizer on and off --
run through `run_pipeline`, the same path `allie run` takes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from allie.cli import BACKENDS, run_pipeline

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _repo_cwd() -> object:
    # Programs load the fixture CSV by its repo-relative path.
    previous = Path.cwd()
    os.chdir(_REPO)
    yield
    os.chdir(previous)


def _run(
    tmp_path: Path,
    source: str,
    capsys: pytest.CaptureFixture[str],
    *,
    backend: str = "tree",
    optimize: bool = True,
) -> tuple[int, str]:
    path = tmp_path / "p.qs"
    path.write_text(source, encoding="utf-8")
    code = run_pipeline(path, backend=backend, optimize=optimize)
    captured = capsys.readouterr()
    return code, captured.out + captured.err


_PROGRAM = """
let AAPL = load("tests/fixtures/AAPL_sample.csv");
let close: Series[Float] = AAPL.close;
let n: Int = len(close);
print(n);
print(close[0]);
print(close[n - 1]);
let total: Float = 0.0;
for let i: Int = 0; i < n; i = i + 1 {
    total = total + close[i];
}
print(total / 60.0);
let ints: Series[Int] = [10, 20, 30];
let k: Int = ints[1] + 1;
print(k);
let flags: Series[Bool] = [true, false, true, true];
let count: Int = 0;
for let i: Int = 0; i < 4; i = i + 1 {
    if flags[i] { count = count + 1; }
}
print(count);
print([1.5, 2.5,][1]);
"""

_EXPECTED = ["60", "155.74", "157.68", "159.61166666666668", "21", "3", "2.5"]


@pytest.mark.parametrize("optimize", [True, False], ids=["opt", "no-opt"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_indexing_agrees_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], backend: str, optimize: bool
) -> None:
    code, output = _run(tmp_path, _PROGRAM, capsys, backend=backend, optimize=optimize)
    assert code == 0, output
    # The Int element prints as `21`, not `21.0`: integer literals are
    # stored as float64, and indexing must convert back on every backend.
    assert output.split() == _EXPECTED


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    ("index", "message"),
    [
        ("2", "series index 2 is out of range for a series of length 2"),
        ("0 - 1", "series index -1 is negative"),
    ],
)
def test_bad_indices_fail_the_same_way_on_every_backend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    backend: str,
    index: str,
    message: str,
) -> None:
    source = f"let s: Series[Float] = [1.0, 2.0];\nprint(s[{index}]);\n"
    code, output = _run(tmp_path, source, capsys, backend=backend)
    assert code == 1
    assert message in output
    assert "Traceback" not in output


@pytest.mark.parametrize("backend", BACKENDS)
def test_an_unused_out_of_range_index_still_fails_with_the_optimizer_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], backend: str
) -> None:
    """Dead-code elimination must not delete `s[5];` -- its result is
    unused, but its error is observable. The `__index_*` builtins are
    classified impure for exactly this reason."""
    source = "let s: Series[Float] = [1.0, 2.0];\ns[5];\nprint(1);\n"
    code, output = _run(tmp_path, source, capsys, backend=backend, optimize=True)
    assert code == 1
    assert "out of range" in output


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("let x: Int = 3;\nprint(x[0]);", "cannot index a value of type Int"),
        ('let p = load("tests/fixtures/AAPL_sample.csv");\nprint(p[0]);', "type PriceSeries"),
        ("let s: Series[Float] = [1.0];\nprint(s[1.0]);", "a series index must be Int, got Float"),
        ("let m = [[1.0], [2.0]];\nprint(m[0]);", "a series can only hold numbers or Bools"),
        ("let s: Series[Float] = [1.0];\nprint(__index_float(s, 0));", "undefined name"),
    ],
    ids=["int", "price-series", "float-index", "nested", "internal-builtin"],
)
def test_type_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str, message: str
) -> None:
    code, output = _run(tmp_path, source, capsys)
    assert code == 1
    assert message in output


_STRATEGY = """
let AAPL = load("tests/fixtures/AAPL_sample.csv");
let c: Series[Float] = AAPL.close;
let slow: Series[Float] = SMA(c, 20);
{setup}
strategy S {{
    when signal crosses_above slow: buy();
}}
backtest S on AAPL from 0 to 60;
"""


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        ("let signal: Series[Float] = [c[1], c[0]];", "assembled from individually indexed"),
        (
            "let signal: Series[Float] = [shift(c, -1)[0], c[1]];",
            "derived from a backwards time shift",
        ),
        (
            "func picked(s: Series[Float]) -> Series[Float] { return [s[1], s[0]]; }\n"
            "let signal: Series[Float] = picked(c);",
            "assembled from individually indexed",
        ),
    ],
    ids=["hand-built", "inherits-shift", "via-function"],
)
def test_hand_built_series_cannot_drive_a_trade(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], setup: str, reason: str
) -> None:
    code, output = _run(tmp_path, _STRATEGY.format(setup=setup), capsys)
    assert code == 1
    assert "look-ahead bias" in output
    assert reason in output


def test_indexing_for_analysis_is_allowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The look-ahead sink is only `when`: reading elements to print or
    summarize them is legitimate."""
    source = _STRATEGY.format(setup="let signal: Series[Float] = SMA(c, 5);")
    source += "print(c[0] + c[59]);\nprint([c[1], c[0]]);\n"
    code, output = _run(tmp_path, source, capsys)
    assert code == 0, output
    assert "BACKTEST RESULTS" in output
