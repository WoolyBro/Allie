"""Cross-cutting checklist: every control-flow construct works in both
keyword spellings, end to end on every backend -- not just at the lexer.

`tests/unit/test_lexer.py` already proves each alias produces the right
`TokenType`. That's necessary but not sufficient: it says nothing about
whether a program *written* in slang parses, resolves, type-checks,
lowers, and executes identically. The lexer is where alias transparency
is *implemented*, so it's the last place a regression would show up --
an alias that leaked past the lexer into the parser or IR would still
pass every lexer test.

Each pair below is the same program twice, once canonical and once
goofy, asserted to produce byte-identical output on all three backends.
Every alias in `KEYWORD_ALIASES` is covered, and a test fails if a new
alias is added without a pair here.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import pytest

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.keywords import KEYWORD_ALIASES
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.quant.backtest_engine import reset_strategy_registry
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

BACKENDS = ("tree", "ir", "llvm")

# (id, canonical source, goofy source, aliases exercised)
PAIRS: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "let_and_if_else",
        """
        let x: Int = 7;
        if x > 5 { print("big"); } else { print("small"); }
        """,
        """
        ts x: Int = 7;
        lowkey x > 5 { print("big"); } nahhh { print("small"); }
        """,
        ("let", "ts", "if", "lowkey", "else", "nahhh"),
    ),
    (
        "while_with_break",
        """
        let i: Int = 0;
        while true {
            if i == 4 { break; }
            i = i + 1;
        }
        print(i);
        """,
        """
        ts i: Int = 0;
        Ahhh fr {
            lowkey i == 4 { bruh; }
            i = i + 1;
        }
        print(i);
        """,
        ("while", "Ahhh", "break", "bruh", "true", "fr"),
    ),
    (
        "for_with_continue",
        """
        let total: Int = 0;
        for let i: Int = 0; i < 6; i = i + 1 {
            if i == 2 { continue; }
            total = total + i;
        }
        print(total);
        """,
        """
        ts total: Int = 0;
        grindset ts i: Int = 0; i < 6; i = i + 1 {
            lowkey i == 2 { pullup; }
            total = total + i;
        }
        print(total);
        """,
        ("for", "grindset", "continue", "pullup"),
    ),
    (
        "func_with_return",
        """
        func double(n: Int) -> Int { return n * 2; }
        print(double(21));
        """,
        """
        be double(n: Int) -> Int { yeet n * 2; }
        print(double(21));
        """,
        ("func", "be", "return", "yeet"),
    ),
    (
        "false_literal_and_import",
        """
        import "unused";
        let flag: Bool = false;
        if flag { print("yes"); } else { print("no"); }
        """,
        """
        fax "unused";
        ts flag: Bool = pmo;
        lowkey flag { print("yes"); } nahhh { print("no"); }
        """,
        ("import", "fax", "false", "pmo"),
    ),
]


def _run(source: str, backend: str) -> str:
    reset_strategy_registry()
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
@pytest.mark.parametrize(
    ("canonical", "goofy"),
    [(p[1], p[2]) for p in PAIRS],
    ids=[p[0] for p in PAIRS],
)
def test_spellings_are_interchangeable(canonical: str, goofy: str, backend: str) -> None:
    assert _run(canonical, backend) == _run(goofy, backend)


def test_goofy_spelling_still_produces_correct_values() -> None:
    """Interchangeability alone would be satisfied if *both* spellings
    were broken identically. These pin the actual expected answers."""
    assert _run(PAIRS[0][2], "tree").strip() == "big"
    assert _run(PAIRS[1][2], "tree").strip() == "4"
    assert _run(PAIRS[2][2], "tree").strip() == "13"  # 0+1+3+4+5, skipping 2
    assert _run(PAIRS[3][2], "tree").strip() == "42"
    assert _run(PAIRS[4][2], "tree").strip() == "no"


def test_every_alias_is_exercised_end_to_end() -> None:
    """Guards the checklist item itself: adding an alias to
    KEYWORD_ALIASES without adding it to a pair above fails here, rather
    than silently shipping a spelling only the lexer has ever seen."""
    covered = {alias for pair in PAIRS for alias in pair[3]}
    missing = set(KEYWORD_ALIASES) - covered
    # Quant keywords are canonical-only by design (docs/keywords.md), so
    # they have no second spelling to be transparent about.
    quant_only = {
        "strategy",
        "when",
        "crosses_above",
        "crosses_below",
        "backtest",
        "on",
        "from",
        "to",
        "and",
        "or",
    }
    uncovered = sorted(missing - quant_only)
    assert not uncovered, f"aliases never run end to end: {uncovered}"
