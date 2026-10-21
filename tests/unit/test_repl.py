"""`allie repl` (allie/repl.py): incremental checking, values echoed,
rejected submissions forgotten, and the command loop."""

from __future__ import annotations

import io
from pathlib import Path

from allie.repl import Session, _open_brackets, run_repl

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "AAPL_sample.csv"


def _session() -> tuple[Session, io.StringIO, io.StringIO]:
    out, err = io.StringIO(), io.StringIO()
    return Session(out, err), out, err


def test_statements_build_on_each_other_and_expressions_echo() -> None:
    session, out, err = _session()
    assert session.submit("let x: Int = 20;")
    assert session.submit("func twice(n: Int) -> Int { return n * 2; }")
    assert session.submit("twice(x) + 2")  # no ';' needed for an expression
    assert session.submit('print("side effect");')
    assert out.getvalue() == "42\nside effect\n"
    assert err.getvalue() == ""


def test_a_void_expression_echoes_nothing_extra() -> None:
    session, out, _ = _session()
    assert session.submit("print(1)")
    assert out.getvalue() == "1\n"


def test_errors_point_at_the_line_just_typed() -> None:
    session, _, err = _session()
    assert session.submit("let a: Int = 1;")
    assert not session.submit('let b: Int = "s";')
    first = err.getvalue().splitlines()[0]
    assert first == "TypeError: cannot assign String to Int. Line 1, Column 14"


def test_a_rejected_submission_leaves_nothing_behind() -> None:
    session, _, err = _session()
    assert not session.submit("let x: Int = 1 / 0;")  # fails at run time (a Float, too)
    assert not session.submit("let y: Int = 1 % 0;")  # fails at run time
    assert not session.submit("print(y);")  # so `y` was never defined
    assert "undefined name 'y'" in err.getvalue()
    assert session.submit("let y: Int = 2;")  # and the name is free to define


def test_series_and_strategies_work_across_submissions() -> None:
    session, out, _ = _session()
    assert session.submit(f'let p = load("{_FIXTURE.as_posix()}");')
    assert session.submit("let fast: Series[Float] = SMA(p.close, 5);")
    assert session.submit("let slow: Series[Float] = SMA(p.close, 20);")
    assert session.submit(
        "strategy S { when fast crosses_above slow: buy(); when fast crosses_below slow: sell(); }"
    )
    assert session.submit("(backtest S on p from 0 to 60).trades")
    assert out.getvalue().strip() == "1"


def test_look_ahead_is_caught_in_a_session_too() -> None:
    session, _, err = _session()
    assert session.submit(f'let p = load("{_FIXTURE.as_posix()}");')
    assert session.submit("let peek: Series[Float] = shift(p.close, -1);")
    assert not session.submit("strategy S { when peek > p.close: buy(); }")
    assert "look-ahead bias" in err.getvalue()


def test_a_rejected_submission_cannot_clean_a_variable() -> None:
    session, _, _ = _session()
    assert session.submit(f'let c: Series[Float] = load("{_FIXTURE.as_posix()}").close;')
    assert session.submit("let s: Series[Float] = shift(c, -1);")
    # Rejected before it ran, so `s = c` never happened...
    assert not session.submit("s = c; strategy X { when shift(c, -1) > c: buy(); }")
    # ...and `s` still holds tomorrow's prices.
    assert not session.submit("strategy S { when s > c: buy(); }")


def test_a_submission_that_failed_part_way_keeps_what_it_assigned() -> None:
    session, _, _ = _session()
    assert session.submit(f'let c: Series[Float] = load("{_FIXTURE.as_posix()}").close;')
    assert session.submit("let t: Series[Float] = c;")
    assert not session.submit("t = shift(c, -1); let y: Int = 1 % 0;")  # ran up to the error
    assert not session.submit("strategy T { when t > c: buy(); }")


def test_type_of_checks_without_running_or_keeping() -> None:
    session, out, _ = _session()
    assert session.submit("let c: Series[Float] = [1.0, 2.0];")
    assert session.type_of("c > 1.5") == "Series<Bool>"
    assert session.type_of("len(c) / 2") == "Float"
    assert session.type_of("print(1)") == "Void"
    assert out.getvalue() == ""  # nothing ran
    assert session.type_of("let z = 1;") is None


def test_reset_forgets_everything() -> None:
    session, _, err = _session()
    assert session.submit("let x: Int = 1;")
    session.reset()
    assert not session.submit("x")
    assert "undefined name 'x'" in err.getvalue()


def test_open_brackets_ignore_strings_and_comments() -> None:
    assert _open_brackets("func f() {") == 1
    assert _open_brackets('print("{"); // {') == 0
    assert _open_brackets("let s = [1.0,\n 2.0];") == 0


def test_the_loop_reads_until_brackets_close_and_handles_commands(
    monkeypatch: object, capsys: object
) -> None:
    script = io.StringIO(
        "let n: Int = 3;\n"
        "func f(k: Int) -> Int {\n"
        "    return k * n;\n"
        "}\n"
        "f(2)\n"
        ":type f(1)\n"
        ":help\n"
        ":nonsense\n"
        ":reset\n"
        ":quit\n"
        "print(99);\n"  # after :quit: never read
    )
    assert run_repl(script) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    lines = captured.out.splitlines()
    assert lines[:2] == ["6", "Int"]
    assert ":type EXPR" in captured.out
    assert "(reset)" in captured.out
    assert "99" not in captured.out
    assert "unknown command :nonsense" in captured.err


def test_the_loop_ends_at_end_of_input() -> None:
    assert run_repl(io.StringIO("print(1);\n")) == 0
