"""Golden snapshot tests for lowering output -- Phase 8.1.

For a handful of representative programs, this snapshots both the Allie
IR dump and the LLVM IR dump. A test fails (with a diff) if lowering
output changes unexpectedly, catching silent regressions the
behavioral tests would miss -- a lowering can produce different
instructions and still compute the right answer, right up until the day
it doesn't.

Two programs are here specifically because Phase 8.1 called them out as
the paths most likely to silently drift:

- `for_loop` -- guards the Phase 5.2 desugar-to-`while` lowering. If a
  `for`-specific code path ever creeps into lowering or codegen, the
  Label/Jump shape here changes and this fails.
- `strategy_backtest` -- guards the `Call`-into-backtest-engine
  lowering, which Phase 7 had to rewrite once precisely because it was
  structurally wrong and nothing caught it.

## LLVM IR is not deterministic without normalization

Verified, not assumed: running the same program twice produces
different LLVM IR. Non-scalar constants (strings, and the packed specs
the backtest lowering emits) cross into JIT code as *boxed handles* --
`id()` of a live Python object (see `allie.codegen.runtime_bridge`) --
so their numeric value is a memory address that changes every run.

`_normalize` replaces any integer literal at or above
`_HANDLE_THRESHOLD` with `<handle>`. That threshold is a documented
heuristic, not a guarantee: it assumes a program's own integer literals
stay below 2**32 while CPython object addresses sit far above it. Every
golden program here holds to that by construction. A future golden
program using a genuinely huge integer literal would have it normalized
too -- losing a little precision in the snapshot, never producing a
false pass, since the surrounding instruction structure is still
compared exactly.

Regenerate after an intentional lowering change -- and read the diff
before committing it, since that's the entire point of these files:

    python tests/golden/test_golden.py --regenerate
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import allie.stdlib as stdlib
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.parser.parser import Parser
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_HERE = Path(__file__).resolve().parent
_PROGRAM_DIR = _HERE / "programs"
_SNAPSHOT_DIR = _HERE / "snapshots"
_CSV = _HERE.parent / "fixtures" / "AAPL_sample.csv"

# CPython object addresses sit far above this; program integer literals
# sit far below. See this module's docstring.
_HANDLE_THRESHOLD = 2**32
_INT_LITERAL = re.compile(r"\b(\d{6,})\b")


def _normalize(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return "<handle>" if int(match.group(1)) >= _HANDLE_THRESHOLD else match.group(1)

    # The fixture path is absolute (see `_load_source`), so leaving it in
    # would bake this machine's home directory into the snapshot and
    # fail on CI or any other checkout. Fold it back to the placeholder
    # the source file actually contains.
    text = text.replace(_CSV.as_posix(), "@FIXTURE@")
    return _INT_LITERAL.sub(replace, text)


def _program_paths() -> list[Path]:
    return sorted(_PROGRAM_DIR.glob("*.qs"))


def _load_source(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("@FIXTURE@", _CSV.as_posix())


def _dump_ir_and_llvm(source: str) -> tuple[str, str]:
    """Returns (allie IR dump, LLVM IR dump), both normalized.

    Deliberately dumps *unoptimized* IR: these snapshots exist to guard
    the lowering passes, and running the optimizer first would let a
    lowering regression hide behind a pass that happens to clean it up.
    The optimizer has its own exact before/after tests in
    tests/unit/test_optimizer.py.
    """
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

    main_program, functions, register_types, function_info = lower_program(program)

    ir_lines = ["=== MAIN ==="]
    ir_lines += [str(instr) for instr in main_program.instructions]
    for name in sorted(functions):
        prog, params = functions[name]
        ir_lines.append(f"=== FUNCTION {name}({', '.join(params)}) ===")
        ir_lines += [str(instr) for instr in prog.instructions]

    lowering = LLVMLowering(functions, function_info)
    lowering.declare_user_functions()
    for name in sorted(function_info):
        lowering.lower_function(name)
    lowering.lower_main(main_program, register_types)

    return _normalize("\n".join(ir_lines) + "\n"), _normalize(str(lowering.module))


@pytest.mark.parametrize("path", _program_paths(), ids=lambda p: p.stem)
def test_ir_dump_matches_snapshot(path: Path) -> None:
    snapshot = _SNAPSHOT_DIR / f"{path.stem}.ir.txt"
    assert snapshot.exists(), f"missing snapshot for {path.name} -- run with --regenerate"
    actual, _ = _dump_ir_and_llvm(_load_source(path))
    assert actual == snapshot.read_text(encoding="utf-8"), (
        f"Allie IR for {path.name} changed. If intentional, review the diff "
        f"then run: python tests/golden/test_golden.py --regenerate"
    )


@pytest.mark.parametrize("path", _program_paths(), ids=lambda p: p.stem)
def test_llvm_dump_matches_snapshot(path: Path) -> None:
    snapshot = _SNAPSHOT_DIR / f"{path.stem}.llvm.txt"
    assert snapshot.exists(), f"missing snapshot for {path.name} -- run with --regenerate"
    _, actual = _dump_ir_and_llvm(_load_source(path))
    assert actual == snapshot.read_text(encoding="utf-8"), (
        f"LLVM IR for {path.name} changed. If intentional, review the diff "
        f"then run: python tests/golden/test_golden.py --regenerate"
    )


def test_normalization_actually_makes_dumps_stable() -> None:
    """The golden files are only meaningful if normalization works --
    without it every LLVM snapshot would fail on every run (boxed
    handles are `id()` values). This proves the normalizer earns its
    place rather than being incidental."""
    source = 'let s: String = "hello"; print(s);'
    first_ir, first_llvm = _dump_ir_and_llvm(source)
    second_ir, second_llvm = _dump_ir_and_llvm(source)
    assert first_ir == second_ir
    assert first_llvm == second_llvm
    assert "<handle>" in second_llvm  # normalization did fire


def test_required_lowering_paths_are_covered() -> None:
    """Phase 8.1 named two programs specifically. Guarding their
    presence keeps a later cleanup from quietly removing the only
    coverage of the two most drift-prone lowering paths."""
    stems = {p.stem for p in _program_paths()}
    assert "for_loop" in stems, "no for-loop golden program (guards the Phase 5.2 desugar)"
    assert "strategy_backtest" in stems, "no strategy/backtest golden program"


def _regenerate() -> None:
    _SNAPSHOT_DIR.mkdir(exist_ok=True)
    for path in _program_paths():
        ir_dump, llvm_dump = _dump_ir_and_llvm(_load_source(path))
        (_SNAPSHOT_DIR / f"{path.stem}.ir.txt").write_text(ir_dump, encoding="utf-8")
        (_SNAPSHOT_DIR / f"{path.stem}.llvm.txt").write_text(llvm_dump, encoding="utf-8")
        print(f"wrote {path.stem}.ir.txt and {path.stem}.llvm.txt")


if __name__ == "__main__":
    import sys

    if "--regenerate" in sys.argv:
        _regenerate()
    else:
        print(__doc__)
