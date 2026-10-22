"""Fixture-driven integration tests.

Every `.qs` file in `fixtures/` is paired with a `.expected` file, and
one parametrized test runs each through the full pipeline and diffs the
output. Three families are represented:

- **pure-CS** (`cs_*.qs`): `for`/`while`/`break`/`continue`, recursion,
  goofy keyword spellings -- no quant content at all.
- **quant-only** (`quant_*.qs`): stdlib series calls and the full
  `strategy`/`when`/`backtest` DSL.
- **combined** (`blended_*.qs`): a `for` loop driving quant calls --
  both halves of the language together rather than side by side.

Each fixture runs on **all three backends** (tree-walking, IR, LLVM/JIT)
against the same `.expected` file. That's stricter than diffing one
backend's output: it re-asserts the three-way agreement standard on
every fixture, so a backend that drifts fails here even if its output
happens to be self-consistent.

`@FIXTURE@` in a `.qs` source is replaced with the absolute path to
`tests/fixtures/AAPL_sample.csv`. Fixtures can't hardcode a relative
path because pytest's working directory isn't guaranteed to be the repo
root; the placeholder keeps the checked-in source readable while making
the test location-independent.

Regenerate the `.expected` files after an intentional behavior change:

    python tests/integration/test_fixtures.py --regenerate
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.quant.backtest_engine import reset_strategy_registry
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_HERE = Path(__file__).resolve().parent
_FIXTURE_DIR = _HERE / "fixtures"
_CSV = _HERE.parent / "fixtures" / "AAPL_sample.csv"

BACKENDS = ("tree", "ir", "llvm")


def _fixture_paths() -> list[Path]:
    return sorted(_FIXTURE_DIR.glob("*.qs"))


def _load_source(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("@FIXTURE@", _CSV.as_posix())


def _compile(source: str):  # noqa: ANN202
    tokens = Lexer(source).scan_tokens()
    parser = Parser(tokens)
    program = parser.parse()
    assert not parser.diagnostics.has_errors(), parser.diagnostics.diagnostics

    resolver = Resolver()
    resolver.resolve(program)
    assert not resolver.diagnostics.has_errors(), resolver.diagnostics.diagnostics

    checker = TypeChecker()
    for name, sig in stdlib.builtin_function_types().items():
        checker.env.define(name, sig)
    checker.check(program)
    return program


def run_fixture(source: str, backend: str) -> str:
    """Runs `source` on `backend`, returning everything it printed."""
    reset_strategy_registry()
    program = _compile(source)
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        if backend == "tree":
            run_program(program, stdlib.create_global_environment())
        else:
            main_program, functions, register_types, function_info = lower_program(program)
            if backend == "ir":
                run_ir_program(
                    main_program,
                    stdlib.create_global_environment(),
                    functions,
                    stdlib.BUILTINS,
                )
            else:
                lowering = LLVMLowering(functions, function_info)
                lowering.declare_user_functions()
                for name in function_info:
                    lowering.lower_function(name)
                main_func = lowering.lower_main(main_program, register_types)
                compile_and_run(lowering.module, main_func.name)

    return buffer.getvalue()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("path", _fixture_paths(), ids=lambda p: p.stem)
def test_fixture_output_matches_expected(path: Path, backend: str) -> None:
    expected_path = path.with_suffix(".expected")
    assert expected_path.exists(), (
        f"missing expected-output file for {path.name} -- "
        f"run `python {Path(__file__).name} --regenerate`"
    )
    actual = run_fixture(_load_source(path), backend)
    expected = expected_path.read_text(encoding="utf-8")
    assert actual == expected, f"{path.name} output changed on the {backend!r} backend"


def test_every_family_is_represented() -> None:
    """This suite is only meaningful
    if all three families are actually present. Without this, deleting
    (or never adding) the combined fixture would silently shrink
    coverage while the suite still passed."""
    stems = {p.stem for p in _fixture_paths()}
    assert any(s.startswith("cs_") for s in stems), "no pure-CS fixture"
    assert any(s.startswith("quant_") for s in stems), "no quant-only fixture"
    assert any(s.startswith("blended_") for s in stems), "no combined fixture"


def _regenerate() -> None:
    for path in _fixture_paths():
        output = run_fixture(_load_source(path), "tree")
        path.with_suffix(".expected").write_text(output, encoding="utf-8")
        print(f"wrote {path.with_suffix('.expected').name} ({len(output)} bytes)")


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print(__doc__)
