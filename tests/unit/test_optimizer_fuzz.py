"""Differential fuzzing of the optimizer and both compiled backends.

Generates random, type-correct programs -- Int/Float/Bool arithmetic,
widening, `if`/`else`, bounded `for` loops with `break`/`continue`,
functions that read and assign module-level variables, and series:
element-wise arithmetic and comparisons with numbers broadcast,
crossings, bar-by-bar `and`/`or`/`!`, indexing, `len` and `count` --
and runs each
five ways: the tree-walking interpreter, and the IR interpreter and the
LLVM JIT each with optimization off and on ("on" for the JIT includes
LLVM's own `-O2` pipeline). All
five must print exactly the same thing, runtime errors (division by
zero, Int overflow, series of different lengths, an index out of range)
included, and the IR must pass the verifier after
every optimizer pass. A miscompilation anywhere in lowering, the passes
or codegen shows up as a disagreement, reported with the program that
triggered it.

Seeded, so a failure reproduces exactly.
"""

from __future__ import annotations

import io
import random
from collections.abc import Callable
from contextlib import redirect_stdout

import allie.stdlib as stdlib
from allie.codegen.jit import compile_and_run
from allie.codegen.llvm_lowering import LLVMLowering
from allie.ir.interpreter import run_ir_program
from allie.ir.lowering import lower_program
from allie.lexer.lexer import Lexer
from allie.optimizer import DEFAULT_PASSES, optimize
from allie.parser.parser import Parser
from allie.runtime.errors import AllieRuntimeError
from allie.runtime.interpreter import run_program
from allie.semantic.resolver import Resolver
from allie.typechecker.checker import TypeChecker

_TYPES = ("Int", "Float", "Bool")
_SERIES = ("Series[Float]", "Series[Bool]")


class _ProgramGenerator:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.counter = 0
        self.functions: list[tuple[str, list[str], str]] = []

    def fresh(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}{self.counter}"

    def program(self) -> str:
        # Module-level variables first, so the functions after them can
        # read and assign them.
        module_scope: list[tuple[str, str, bool]] = []
        lines: list[str] = []
        for _ in range(self.rng.randint(0, 3)):
            lines += self.statement(module_scope, depth=3, in_loop=False, kinds=["let"])
        for _ in range(self.rng.randint(0, 2)):
            lines += self.function(module_scope)
        lines += self.block(module_scope, depth=0, count=self.rng.randint(3, 8), in_loop=False)
        return "\n".join(lines) + "\n"

    def function(self, module_scope: list[tuple[str, str, bool]]) -> list[str]:
        name = self.fresh("f")
        params = [(self.fresh("p"), self.rng.choice(_TYPES)) for _ in range(self.rng.randint(0, 2))]
        ret = self.rng.choice(_TYPES)
        scope = [*module_scope, *((p, t, True) for p, t in params)]
        body = self.block(scope, depth=1, count=self.rng.randint(0, 3), in_loop=False)
        body.append(f"return {self.expr(ret, scope + self._locals(body), 2)};")
        self.functions.append((name, [t for _, t in params], ret))
        signature = ", ".join(f"{p}: {t}" for p, t in params)
        return [f"func {name}({signature}) -> {ret} {{", *_indent(body), "}"]

    def _locals(self, body: list[str]) -> list[tuple[str, str, bool]]:
        found = []
        for line in body:
            if line.startswith("let ") and ":" in line:
                name, rest = line[4:].split(":", 1)
                found.append((name.strip(), rest.split("=", 1)[0].strip(), True))
        return found

    def block(
        self, scope: list[tuple[str, str, bool]], depth: int, count: int, in_loop: bool
    ) -> list[str]:
        local_scope = list(scope)
        lines: list[str] = []
        for _ in range(count):
            lines += self.statement(local_scope, depth, in_loop)
        return lines

    def statement(
        self,
        scope: list[tuple[str, str, bool]],
        depth: int,
        in_loop: bool,
        kinds: list[str] | None = None,
    ) -> list[str]:
        if kinds is None:
            kinds = ["let", "let", "print", "print", "assign"]
            if depth < 3:
                kinds += ["if", "for"]
            if in_loop:
                kinds += ["break", "continue"]
        kind = self.rng.choice(kinds)

        if kind == "let":
            var_type = self.rng.choice(_TYPES + _SERIES)
            name = self.fresh("v")
            value = self.expr(self._source_type(var_type), scope, 3)
            scope.append((name, var_type, True))
            return [f"let {name}: {var_type} = {value};"]
        if kind == "assign":
            targets = [(n, t) for n, t, assignable in scope if assignable]
            if targets:
                name, var_type = self.rng.choice(targets)
                return [f"{name} = {self.expr(self._source_type(var_type), scope, 3)};"]
            kind = "print"
        if kind == "print":
            return [f"print({self.expr(self.rng.choice(_TYPES + _SERIES), scope, 3)});"]
        if kind == "if":
            lines = [f"if {self.expr('Bool', scope, 2)} {{"]
            lines += _indent(self.block(scope, depth + 1, self.rng.randint(0, 3), in_loop))
            if self.rng.random() < 0.5:
                lines.append("} else {")
                lines += _indent(self.block(scope, depth + 1, self.rng.randint(0, 3), in_loop))
            return [*lines, "}"]
        if kind == "for":
            counter = self.fresh("i")
            bound = self.rng.randint(0, 4)
            inner = [*scope, (counter, "Int", False)]  # never reassigned: loops terminate
            body = self.block(inner, depth + 1, self.rng.randint(0, 3), in_loop=True)
            header = (
                f"for let {counter}: Int = 0; {counter} < {bound}; {counter} = {counter} + 1 {{"
            )
            return [header, *_indent(body), "}"]
        jump = "break;" if kind == "break" else "continue;"
        if self.rng.random() < 0.5:
            return [f"if {self.expr('Bool', scope, 1)} {{ {jump} }}"]
        return [jump]

    def _source_type(self, target: str) -> str:
        """Usually the target type; sometimes Int into a Float slot, to
        exercise implicit widening."""
        return "Int" if target == "Float" and self.rng.random() < 0.3 else target

    def _series_length(self) -> int:
        """Mostly 4, so most operations succeed; occasionally 3, so
        some fail on a length mismatch."""
        return 3 if self.rng.random() < 0.06 else 4

    def _series_expr(self, t: str, scope: list[tuple[str, str, bool]], depth: int) -> str:
        rng = self.rng
        variables = [name for name, var_type, _ in scope if var_type == t]
        if depth <= 0 or rng.random() < 0.3:
            if variables and rng.random() < 0.6:
                return rng.choice(variables)
            if t == "Series[Bool]":
                elements = (rng.choice(["true", "false"]) for _ in range(self._series_length()))
            elif rng.random() < 0.2:  # a Series<Int> literal, widened
                elements = (str(rng.randint(-5, 5)) for _ in range(self._series_length()))
            else:
                elements = (self._literal("Float") for _ in range(self._series_length()))
            return f"[{', '.join(elements)}]"
        roll = rng.random()
        if t == "Series[Float]":
            if roll < 0.1:
                return f"(-{self._series_expr(t, scope, depth - 1)})"
            if roll < 0.2:
                return f"SMA({self._series_expr(t, scope, depth - 1)}, {rng.randint(1, 3)})"
            op = rng.choice("+-*/")
            left, right = self._series_operands("Series[Float]", scope, depth)
            return f"({left} {op} {right})"
        if roll < 0.15:
            return f"(!{self._series_expr(t, scope, depth - 1)})"
        if roll < 0.5:
            op = rng.choice(["<", ">", "<=", ">=", "crosses_above", "crosses_below"])
            left, right = self._series_operands("Series[Float]", scope, depth)
            return f"({left} {op} {right})"
        op = rng.choice(["and", "or"])
        left, right = self._series_operands("Series[Bool]", scope, depth)
        return f"({left} {op} {right})"

    def _series_operands(
        self, t: str, scope: list[tuple[str, str, bool]], depth: int
    ) -> tuple[str, str]:
        """Two operands for a binary operator on series of type `t`: both
        series, or one a scalar broadcast to every bar."""
        series = self._series_expr(t, scope, depth - 1)
        roll = self.rng.random()
        if roll < 0.6:
            return series, self._series_expr(t, scope, depth - 1)
        scalar_type = "Bool" if t == "Series[Bool]" else self.rng.choice(_TYPES[:2])
        scalar = self.expr(scalar_type, scope, depth - 2)
        return (series, scalar) if roll < 0.8 else (scalar, series)

    def _series_reduction(self, t: str, scope: list[tuple[str, str, bool]], depth: int) -> str:
        """A scalar computed from a series, so series results flow back
        into the scalar program."""
        if t == "Int":
            if self.rng.random() < 0.5:
                return f"count({self._series_expr('Series[Bool]', scope, depth - 1)})"
            series_type = self.rng.choice(_SERIES)
            return f"len({self._series_expr(series_type, scope, depth - 1)})"
        series_type = "Series[Bool]" if t == "Bool" else "Series[Float]"
        index = self.rng.randint(0, 4)  # 4 is out of range for most series
        return f"{self._series_expr(series_type, scope, depth - 1)}[{index}]"

    def expr(self, t: str, scope: list[tuple[str, str, bool]], depth: int) -> str:
        if t in _SERIES:
            return self._series_expr(t, scope, depth)
        rng = self.rng
        variables = [name for name, var_type, _ in scope if var_type == t]
        calls = [f for f in self.functions if f[2] == t]
        if depth <= 0 or rng.random() < 0.25:
            if variables and rng.random() < 0.6:
                return rng.choice(variables)
            return self._literal(t)
        roll = rng.random()
        if calls and roll < 0.15:
            name, params, _ = rng.choice(calls)
            args = ", ".join(self.expr(self._source_type(p), scope, depth - 1) for p in params)
            return f"{name}({args})"
        if variables and roll < 0.35:
            return rng.choice(variables)
        if roll < 0.45:
            return self._series_reduction(t, scope, depth)
        if t == "Int":
            if rng.random() < 0.15:
                return f"(-{self.expr('Int', scope, depth - 1)})"
            op = rng.choice(["+", "-", "*", "%"])
            left, right = self.expr("Int", scope, depth - 1), self.expr("Int", scope, depth - 1)
            return f"({left} {op} {right})"
        if t == "Float":
            if rng.random() < 0.3:
                op, left_type, right_type = "/", rng.choice(_TYPES[:2]), rng.choice(_TYPES[:2])
            else:
                op, left_type, right_type = rng.choice("+-*%"), "Float", rng.choice(_TYPES[:2])
            left = self.expr(left_type, scope, depth - 1)
            right = self.expr(right_type, scope, depth - 1)
            return f"({left} {op} {right})"
        # Bool
        choice = rng.random()
        if choice < 0.4:
            op = rng.choice(["<", ">", "<=", ">=", "==", "!="])
            left_type, right_type = rng.choice(_TYPES[:2]), rng.choice(_TYPES[:2])
        elif choice < 0.85:
            op = rng.choice(["and", "or", "=="])
            left_type = right_type = "Bool"
        else:
            return f"(!{self.expr('Bool', scope, depth - 1)})"
        left = self.expr(left_type, scope, depth - 1)
        right = self.expr(right_type, scope, depth - 1)
        return f"({left} {op} {right})"

    def _literal(self, t: str) -> str:
        if t == "Int":
            return str(self.rng.randint(-20, 20))
        if t == "Float":
            return f"{self.rng.uniform(-10, 10):.2f}"
        return self.rng.choice(["true", "false"])


def _indent(lines: list[str]) -> list[str]:
    return ["    " + line for line in lines]


def _capture(run: Callable[..., object], *args: object) -> str:
    """Everything `run(*args)` prints, plus the runtime error it ended
    with, if any."""
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            run(*args)
    except AllieRuntimeError as error:
        return buffer.getvalue() + f"!{error}"
    return buffer.getvalue()


def _run_ir(main, functions) -> None:  # noqa: ANN001
    run_ir_program(main, stdlib.create_global_environment(), functions, stdlib.BUILTINS)


def _run_llvm(main, functions, register_types, function_info, opt_level) -> None:  # noqa: ANN001
    lowering = LLVMLowering(functions, function_info)
    lowering.declare_user_functions()
    for name in function_info:
        lowering.lower_function(name)
    entry = lowering.lower_main(main, register_types)
    compile_and_run(lowering.module, entry.name, opt_level=opt_level)


def _check(source: str):  # noqa: ANN202
    program = Parser(Lexer(source).scan_tokens()).parse()
    Resolver().resolve(program)
    checker = TypeChecker()
    for name, signature in stdlib.builtin_function_types().items():
        checker.env.define(name, signature)
    checker.check(program)  # a generator bug if this raises
    return program


def _all_outputs(source: str) -> dict[str, str]:
    program = _check(source)
    outputs = {"tree": _capture(run_program, program, stdlib.create_global_environment())}
    for enabled in (False, True):
        main, functions, register_types, function_info = lower_program(program)
        main, functions = optimize(main, functions, DEFAULT_PASSES, enabled, verify=True)
        suffix = "-opt" if enabled else ""
        outputs[f"ir{suffix}"] = _capture(_run_ir, main, functions)
        outputs[f"llvm{suffix}"] = _capture(
            _run_llvm, main, functions, register_types, function_info, 2 if enabled else 0
        )
    return outputs


def test_every_backend_and_optimization_level_agrees_on_random_programs() -> None:
    rng = random.Random(20261021)
    errors = 0
    for case in range(150):
        source = _ProgramGenerator(rng).program()
        outputs = _all_outputs(source)
        reference = outputs["tree"]
        disagreeing = {k: v for k, v in outputs.items() if v != reference}
        assert not disagreeing, (
            f"case {case}: backends disagree\n--- program ---\n{source}"
            f"--- tree ---\n{reference}\n"
            + "\n".join(f"--- {k} ---\n{v}" for k, v in disagreeing.items())
        )
        errors += reference.count("!")
    # The generator must actually reach the runtime-error paths, or the
    # "errors match too" half of this test is vacuous.
    assert errors > 0
