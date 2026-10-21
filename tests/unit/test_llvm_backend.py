"""LLVM backend behavior that spans compilation and execution: LLVM's own
optimization pipeline, module-level variables shared with functions,
boxed (Series/PriceSeries) function boundaries, and how runtime errors
and deep recursion end a native run. Instruction-level lowering is in
test_llvm_lowering.py."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from allie.cli import BACKENDS, run_pipeline
from allie.codegen import jit

_REPO = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "p.qs"
    path.write_text(source, encoding="utf-8")
    return path


def _run_everywhere(
    tmp_path: Path, source: str, capsys: pytest.CaptureFixture[str]
) -> dict[str, tuple[int, str, str]]:
    path = _write(tmp_path, source)
    results = {}
    for backend in BACKENDS:
        code = run_pipeline(path, backend=backend)
        captured = capsys.readouterr()
        results[backend] = (code, captured.out, captured.err)
    return results


# ---- LLVM's optimizer ----


def test_llvm_passes_run_and_promote_variables_to_registers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Lowering keeps every variable in a stack slot; LLVM's -O2 turns
    them into SSA registers (mem2reg/SROA) -- the change that makes hot
    loops several times faster."""
    path = _write(
        tmp_path,
        "func total(n: Int) -> Int {\n"
        "    let sum: Int = 0;\n"
        "    for let i: Int = 0; i < n; i = i + 1 { sum = sum + i; }\n"
        "    return sum;\n"
        "}\n"
        "print(total(10));\n",
    )
    assert run_pipeline(path, backend="llvm", debug_ir=True) == 0
    out = capsys.readouterr().out
    before, after = out.split("=== LLVM IR (after LLVM -O2) ===")
    assert "alloca" in before.split("=== LLVM IR ===")[1]
    optimized_total = re.search(r'define i64 @"?total"?\(.*?\n}', after, re.DOTALL)
    assert optimized_total is not None
    assert "alloca" not in optimized_total.group(0)
    assert "phi" in optimized_total.group(0)  # the loop's variables are SSA values now
    assert out.strip().endswith("45")


def test_no_optimize_skips_llvm_passes_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "print(1 + 2);\n")
    assert run_pipeline(path, backend="llvm", debug_ir=True, optimize=False) == 0
    assert "after LLVM -O2" not in capsys.readouterr().out


def test_llvm_rejecting_generated_code_is_an_internal_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(_text: str) -> None:
        raise RuntimeError("simulated verifier failure")

    monkeypatch.setattr(jit.llvm, "parse_assembly", broken)
    assert run_pipeline(_write(tmp_path, "print(1);\n"), backend="llvm") == 70
    err = capsys.readouterr().err
    assert "internal compiler error: LLVM rejected the generated code" in err
    assert "Traceback" not in err


# ---- module-level variables ----


def test_functions_read_and_assign_module_level_variables(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Used to crash LLVM codegen with a raw KeyError, and fail on the IR
    interpreter, which gave each call an environment with no parent."""
    source = (
        "let rate: Float = 0.05;\n"
        "let calls: Int = 0;\n"
        "func grow(p: Float) -> Float {\n"
        "    calls = calls + 1;\n"
        "    let rate: Float = rate * 2.0;\n"  # a local shadowing the global
        "    return p * (1.0 + rate);\n"
        "}\n"
        "print(grow(100.0));\n"
        "rate = 0.5;\n"
        "print(grow(100.0));\n"
        "print(calls);\n"
        "if true {\n"
        "    let k: Int = 5;\n"
        "    func add_k(x: Int) -> Int { return x + k; }\n"
        "    print(add_k(1));\n"
        "}\n"
    )
    results = _run_everywhere(tmp_path, source, capsys)
    for backend, (code, out, err) in results.items():
        assert (code, err) == (0, ""), backend
        assert out.split() == ["110.00000000000001", "200.0", "2", "6"], backend


def test_a_variable_read_before_its_let_runs_fails_the_same_everywhere(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Hoisting lets `f` be called before `rate` is assigned. LLVM keeps a
    # "has been assigned" flag per shared variable rather than reading 0.
    source = "print(f());\nlet rate: Float = 0.05;\nfunc f() -> Float { return rate; }\n"
    results = _run_everywhere(tmp_path, source, capsys)
    for backend, (code, out, err) in results.items():
        assert code == 1, backend
        assert out == "", backend
        assert "AllieNameError: undefined name: 'rate'" in err, backend


# ---- boxed function boundaries ----


@pytest.mark.usefixtures("_in_repo")
def test_functions_take_and_return_series_and_price_series(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = (
        'let AAPL = load("tests/fixtures/AAPL_sample.csv");\n'
        "func last(p: PriceSeries) -> Float { return p.close[len(p.close) - 1]; }\n"
        "func fast(s: Series[Float]) -> Series[Float] { return SMA(s, 5); }\n"
        'func label(x: Float) -> String { return "close="; }\n'
        "print(last(AAPL));\n"
        "print(mean(fast(AAPL.close)));\n"
        "print(label(1.0));\n"
    )
    results = _run_everywhere(tmp_path, source, capsys)
    outputs = {backend: out for backend, (_, out, _) in results.items()}
    assert len(set(outputs.values())) == 1, outputs
    assert outputs["llvm"].split("\n")[0] == "157.68"


@pytest.fixture
def _in_repo() -> object:
    previous = Path.cwd()
    os.chdir(_REPO)
    yield
    os.chdir(previous)


# ---- how a native run ends ----


def test_an_error_inside_a_loop_stops_native_code_immediately(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The bridge used to record the error and hand back a dummy 0.0, and
    native code carried on: this loop ran past the end of `s` until the
    counter overflowed -- effectively forever -- instead of stopping."""
    source = (
        "let s: Series[Float] = [0.0, 0.0];\n"
        "let i: Int = 0;\n"
        "while s[i] == 0.0 { i = i + 1; }\n"
        "print(i);\n"
    )
    results = _run_everywhere(tmp_path, source, capsys)
    for backend, (code, out, err) in results.items():
        assert code == 1, backend
        assert out == "", backend
        assert "series index 2 is out of range" in err, backend


def test_deep_recursion_is_an_error_on_every_backend_not_a_crash(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Native recursion had no limit at all: deep enough, it overflowed
    the C stack and killed the process. Now past MAX_CALL_DEPTH it
    raises, like the interpreters' RecursionError."""
    source = (
        "func depth(n: Int) -> Int { if n == 0 { return 0; } return 1 + depth(n - 1); }\n"
        "print(depth(1000000));\n"
    )
    results = _run_everywhere(tmp_path, source, capsys)
    for backend, (code, _out, err) in results.items():
        assert code == 1, backend
        assert "recursion too deep" in err, backend
        assert "Traceback" not in err, backend


def test_moderate_recursion_runs_natively(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(
        tmp_path,
        "func depth(n: Int) -> Int { if n == 0 { return 0; } return 1 + depth(n - 1); }\n"
        "print(depth(9000));\n",
    )
    assert run_pipeline(path, backend="llvm") == 0
    assert capsys.readouterr().out.strip() == "9000"


def test_a_failed_run_leaves_nothing_behind_for_the_next(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    failing = _write(tmp_path, "let z: Int = 0;\nprint(1 % z);\n")
    assert run_pipeline(failing, backend="llvm") == 1
    capsys.readouterr()
    fine = tmp_path / "fine.qs"
    fine.write_text("print(41 + 1);\n", encoding="utf-8")
    assert run_pipeline(fine, backend="llvm") == 0
    assert capsys.readouterr().out.strip() == "42"
