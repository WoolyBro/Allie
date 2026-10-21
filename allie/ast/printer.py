"""An indented, readable rendering of the AST -- what `--debug-ast` prints.

The dataclass `repr` of a node is complete but unreadable: one line per
statement, every span spelled out in full. This prints one node per
line, children indented under their parent, each with its position and
-- once the type checker has run -- its type:

    LetStmt name='fast' @3:1
      type_annotation: TypeAnnotation name='Series' @3:11
        type_arg: TypeAnnotation name='Float' @3:18
      initializer: Call @3:27 : Series<Float>
        callee: Identifier name='SMA' @3:27
        arguments:
          FieldAccess name='close' @3:31 : Series<Float>
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass

from allie.ast.nodes import Program


def format_ast(program: Program) -> str:
    lines: list[str] = []
    for statement in program.statements:
        _node(statement, 0, "", lines)
    return "\n".join(lines)


def _is_node(value: object) -> bool:
    return is_dataclass(value) and hasattr(value, "span")


def _node(node: object, depth: int, label: str, lines: list[str]) -> None:
    pad = "  " * depth
    scalars: list[str] = []
    children: list[tuple[str, object]] = []
    for f in fields(node):  # type: ignore[arg-type]
        if f.name in ("span", "inferred_type"):
            continue
        value = getattr(node, f.name)
        if _is_node(value) or (isinstance(value, list) and value and _is_node(value[0])):
            children.append((f.name, value))
        elif isinstance(value, list) and not value:
            continue
        elif value is not None:
            scalars.append(f"{f.name}={value!r}")
    span = getattr(node, "span", None)
    where = f" @{span.line}:{span.column}" if span is not None else ""
    inferred = getattr(node, "inferred_type", None)
    typed = f" : {inferred}" if inferred is not None else ""
    head = " ".join([type(node).__name__, *scalars])
    lines.append(f"{pad}{label}{head}{where}{typed}")
    for name, value in children:
        if isinstance(value, list):
            lines.append(f"{pad}  {name}:")
            for item in value:
                _node(item, depth + 2, "", lines)
        else:
            _node(value, depth + 1, f"{name}: ", lines)
