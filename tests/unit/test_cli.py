"""Tests for allie/cli.py -- the actual user entry point.

Added during the final review pass: `run_pipeline` sat at 39% coverage
despite being the code every user actually hits. The debug flags, the
error paths, and the backend selection had no direct tests at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from allie.cli import BACKENDS, build_parser, main, run_pipeline


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


def test_missing_file_reports_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_pipeline(tmp_path / "nope.qs") == 2
    assert "no such file" in capsys.readouterr().err


# ---- each pipeline stage's failure path returns 1, not a traceback ----


@pytest.mark.parametrize(
    ("source", "marker"),
    [
        ("let x @ = 1;", "LexError"),
        ("let x = ;", "expected"),
        ("break;", "outside a loop"),
        ('let x: Int = "s";', "TypeError"),
        ("let x: Float = 1 / 0;", "division by zero"),
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


def test_debug_ir_shows_both_ir_sections(program: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
        run_pipeline(program, backend="llvm", debug_tokens=True, debug_ast=True, debug_ir=True) == 0
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


# ---- --verify: three-backend agreement as a user-facing guarantee ----


def test_verify_passes_when_backends_agree(
    program: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_pipeline(program, verify=True) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "3"
    assert "verified" in captured.err
    for backend in BACKENDS:
        assert backend in captured.err


def test_verify_reports_output_exactly_once(
    program: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Three backends run, but the user sees one result -- not three
    copies of it."""
    run_pipeline(program, verify=True)
    assert capsys.readouterr().out.count("3") == 1


def test_verify_agrees_on_a_shared_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """All three failing identically is agreement, not disagreement --
    but the exit code must still be nonzero."""
    path = _write(tmp_path, "let x: Float = 1 / 0;\nprint(x);\n")
    assert run_pipeline(path, verify=True) == 1
    assert "verified" in capsys.readouterr().err


def test_verify_catches_a_divergence_and_refuses_to_report(
    program: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The behaviour the flag exists for. A backend is monkeypatched to
    lie, and --verify must refuse rather than pick a winner."""
    import allie.cli as cli_module

    real = cli_module._run_capturing

    def lying(program_arg: object, backend: str, *, optimize: bool):  # noqa: ANN202
        outcome = real(program_arg, backend, optimize=optimize)  # type: ignore[arg-type]
        return cli_module._Outcome(outcome.code, "999\n") if backend == "llvm" else outcome

    monkeypatch.setattr(cli_module, "_run_capturing", lying)

    assert run_pipeline(program, verify=True) == 3
    captured = capsys.readouterr()
    assert "VERIFICATION FAILED" in captured.err
    assert "Refusing to report" in captured.err
    # The report must show each backend's result so the user can see
    # *how* they differ, not merely that they did.
    for backend in BACKENDS:
        assert backend in captured.err
    assert "999" in captured.err
    # And the untrustworthy result must not reach stdout.
    assert captured.out == ""


def test_verify_survives_a_backend_that_cannot_compile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """If one backend can't compile a program, --verify must report a
    disagreement rather than crash. (Nothing that type-checks is
    rejected by LLVM any more, so the failure is forced.)"""
    from allie.codegen.llvm_lowering import LLVMLowering, LLVMLoweringError

    def refuse(*_args: object) -> None:
        raise LLVMLoweringError("not today")

    monkeypatch.setattr(LLVMLowering, "lower_main", refuse)
    path = _write(tmp_path, "print(1);\n")
    assert run_pipeline(path, verify=True) == 3
    assert "VERIFICATION FAILED" in capsys.readouterr().err


def test_a_function_taking_a_series_verifies_on_every_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Used to be rejected by the LLVM backend, which only compiled
    functions with scalar parameters."""
    path = _write(
        tmp_path,
        "func total(xs: Series[Float]) -> Float { return mean(xs); }\n"
        "print(total([1.0, 2.0, 3.0]));\n",
    )
    assert run_pipeline(path, verify=True) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "2.0"
    assert "verified" in captured.err


def test_verify_flag_is_exposed_on_the_cli() -> None:
    args = build_parser().parse_args(["run", "f.qs", "--verify"])
    assert args.verify is True
    assert build_parser().parse_args(["run", "f.qs"]).verify is False


# ---- runtime errors say where ----


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_runtime_error_names_and_shows_its_line_on_every_backend(
    tmp_path: Path, backend: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "let z: Int = 0;\nlet w: Int = 1;\nprint(w % z);\n")
    assert run_pipeline(path, backend=backend) == 1
    err = capsys.readouterr().err
    assert err.splitlines()[:3] == [
        "AllieValueError: modulo by zero. Line 3, Column 9",
        "  print(w % z);",
        "          ^",
    ]


def test_an_error_inside_a_function_points_at_the_failing_expression(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = (
        "func pick(s: Series[Float], i: Int) -> Float {\n"
        "    return s[i];\n"
        "}\n"
        "print(pick([1.0, 2.0], 5));\n"
    )
    for backend in BACKENDS:
        assert run_pipeline(_write(tmp_path, source), backend=backend) == 1
        err = capsys.readouterr().err
        assert "Line 2, Column 13" in err, backend  # at s[i], not the call on line 4


def test_verify_compares_the_errors_too(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _write(tmp_path, "print(1);\nlet z: Int = 0;\nprint(2 / z);\n")
    assert run_pipeline(path, verify=True) == 1
    captured = capsys.readouterr()
    assert captured.out == "1\n"
    assert captured.err.count("division by zero") == 1  # once, not once per backend
    assert "  print(2 / z);" in captured.err


# ---- check, stdin, trace, time, version ----


def test_check_runs_nothing_and_reports_every_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "good.qs"
    good.write_text('print("ran");\n', encoding="utf-8")
    bad = tmp_path / "bad.qs"
    bad.write_text('let x: Int = "s";\n', encoding="utf-8")
    assert main(["check", str(good)]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""  # not run
    assert "ok: 1 file checked" in captured.err
    assert main(["check", str(good), str(bad)]) == 1
    assert "cannot assign String to Int" in capsys.readouterr().err
    assert main(["check", str(tmp_path / "missing.qs")]) == 2


def test_a_program_can_come_from_stdin(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("print(6 * 7);\n"))
    assert main(["run", "-"]) == 0
    assert capsys.readouterr().out == "42\n"


def test_a_directory_is_a_usage_error_not_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_pipeline(tmp_path) == 2
    assert "is a directory" in capsys.readouterr().err


def test_trace_prints_each_statement_as_it_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "let x: Int = 1;\nif x > 0 {\n    print(x);\n}\n")
    assert run_pipeline(path, trace=True) == 0
    captured = capsys.readouterr()
    assert captured.out == "1\n"
    assert captured.err.splitlines() == [
        "[trace]    1 | let x: Int = 1;",
        "[trace]    2 | if x > 0 {",
        "[trace]    3 | print(x);",
    ]


def test_trace_needs_the_tree_backend(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["run", "f.qs", "--trace", "--backend", "llvm"])
    assert exit_info.value.code == 2
    assert "--trace needs the tree backend" in capsys.readouterr().err


def test_time_reports_each_phase(program: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_pipeline(program, backend="llvm", timing=True) == 0
    line = capsys.readouterr().err.strip()
    for phase in ("front end", "lowering", "codegen", "run", "total"):
        assert phase in line


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    import allie

    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == f"allie {allie.__version__}"


def test_debug_ast_is_an_indented_tree_with_types(
    program: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_pipeline(program, debug_ast=True)
    out = capsys.readouterr().out
    assert "LetStmt name='x' @1:1" in out
    assert "  initializer: Binary operator='+' @1:16 : Int" in out


def test_an_unexpected_exception_is_an_internal_error_not_a_traceback(
    program: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import allie.cli as cli_module

    def broken(*_args: object, **_kwargs: object) -> None:
        raise TypeError("simulated bug")

    monkeypatch.setattr(cli_module, "run_program", broken)
    assert run_pipeline(program) == 70
    err = capsys.readouterr().err
    assert "internal compiler error: TypeError: simulated bug" in err
    assert "ALLIE_TRACEBACK=1" in err
    assert "Traceback" not in err


def test_the_llvm_backend_is_not_loaded_to_run_the_interpreters(tmp_path: Path) -> None:
    import subprocess
    import sys

    path = _write(tmp_path, "print(1);\n")
    probe = (
        "import sys; from allie.cli import main; main(['run', sys.argv[1]]); "
        "print('llvmlite' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe, str(path)], capture_output=True, text=True, check=True
    )
    assert done.stdout.split() == ["1", "False"]
