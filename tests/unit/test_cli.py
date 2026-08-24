"""Tests for allie/cli.py -- the actual user entry point.

Added during the final review pass: `run_pipeline` sat at 39% coverage
despite being the code every user actually hits. The debug flags, the
error paths, and the backend selection had no direct tests at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from allie.cli import BACKENDS, build_parser, run_pipeline


@pytest.fixture
def program(tmp_path: Path) -> Path:
    path = tmp_path / "p.qs"
    path.write_text("let x: Int = 1 + 2;\nprint(x);\n", encoding="utf-8")
    return path


def _write(tmp_path: Path, source: str) -> Path:
    path = tmp_path / "p.qs"
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize("backend", BACKENDS)
def test_runs_on_every_backend(
    program: Path, backend: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_pipeline(program, backend=backend) == 0
    assert capsys.readouterr().out.strip().endswith("3")


def test_missing_file_reports_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_pipeline(tmp_path / "nope.qs") == 1
    assert "no such file" in capsys.readouterr().err


# ---- each pipeline stage's failure path returns 1, not a traceback ----


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        ("let x @ = 1;", "LexError"),
        ("let x = ;", "expected"),
        ("break;", "outside a loop"),
        ('let x: Int = "s";', "TypeError"),
        ("let x: Int = 1 / 0;", "division by zero"),
    ],
    ids=["lex", "parse", "resolve", "typecheck", "runtime"],
)
def test_each_failure_stage_exits_nonzero(
    tmp_path: Path, source: str, marker: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_pipeline(_write(tmp_path, source)) == 1
    captured = capsys.readouterr()
    assert marker in (captured.err + captured.out)


def test_deep_recursion_reports_an_allie_error_not_a_python_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: this used to dump a raw Python RecursionError
    traceback full of interpreter internals, bypassing the whole
    AllieRuntimeError hierarchy."""
    source = (
        "func d(n: Int) -> Int { if n <= 0 { return 0; } else { return d(n - 1); } }\n"
        "print(d(5000));\n"
    )
    assert run_pipeline(_write(tmp_path, source)) == 1
    err = capsys.readouterr().err
    assert "recursion too deep" in err
    assert "Traceback" not in err


# ---- debug flags ----


def test_debug_tokens(program: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_pipeline(program, debug_tokens=True)
    out = capsys.readouterr().out
    assert "=== TOKENS ===" in out
    assert "LET" in out


def test_debug_ast(program: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_pipeline(program, debug_ast=True)
    assert "=== AST ===" in capsys.readouterr().out


def test_debug_ir_shows_both_ir_sections(
    program: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--debug-ir must dump the custom IR *and* the LLVM IR, clearly
    labelled as two separate sections."""
    run_pipeline(program, backend="llvm", debug_ir=True)
    out = capsys.readouterr().out
    assert "=== ALLIE IR ===" in out
    assert "=== LLVM IR ===" in out
    assert out.index("=== ALLIE IR ===") < out.index("=== LLVM IR ===")


def test_debug_ir_works_on_the_tree_backend_too(
    program: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tree-walker skips IR lowering entirely unless --debug-ir asks
    for it, so this path is easy to break without noticing."""
    assert run_pipeline(program, backend="tree", debug_ir=True) == 0
    assert "=== ALLIE IR ===" in capsys.readouterr().out


def test_flags_combine(program: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        run_pipeline(
            program, backend="llvm", debug_tokens=True, debug_ast=True, debug_ir=True
        )
        == 0
    )
    out = capsys.readouterr().out
    for section in ("=== TOKENS ===", "=== AST ===", "=== ALLIE IR ===", "=== LLVM IR ==="):
        assert section in out


def test_no_optimize_leaves_more_ir_than_optimized(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The --no-optimize toggle must actually reach the pass manager --
    a flag that silently does nothing would still exit 0."""
    source = "let dead: Int = 7 * 6;\nlet x: Int = 1 + 2;\nprint(x);\n"
    path = _write(tmp_path, source)

    run_pipeline(path, backend="ir", debug_ir=True, optimize=False)
    unoptimized = capsys.readouterr().out.count("\n")
    run_pipeline(path, backend="ir", debug_ir=True, optimize=True)
    optimized = capsys.readouterr().out.count("\n")

    assert optimized < unoptimized


# ---- argument parser ----


def test_parser_accepts_every_backend_and_defaults_to_tree() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "f.qs"]).backend == "tree"
    for backend in BACKENDS:
        assert parser.parse_args(["run", "f.qs", "--backend", backend]).backend == backend


def test_legacy_dump_flags_still_work() -> None:
    """--dump-tokens/--dump-ast are documented aliases; dropping them
    would break anything written against the older interface."""
    args = build_parser().parse_args(["run", "f.qs", "--dump-tokens", "--dump-ast"])
    assert args.debug_tokens and args.debug_ast
