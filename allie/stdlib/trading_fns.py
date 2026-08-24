"""Trading-action builtins -- Phase 7.2. `buy()`/`sell()` are ordinary
zero-argument registered builtins, NOT lexer keywords: the merged
grammar treats them as `Call` expressions resolved through
`allie.stdlib.registry`, exactly like `print`/`load`. (The QuantLang-only
draft listed them as keywords; confirmed against
`allie/lexer/keywords.py` that this repo does not, and kept it that way
-- it means they need no parser, lowering, or codegen support of their
own, and they reach the LLVM backend through the same bridge as every
other builtin.)

**Valid only inside a strategy body**, enforced at two levels for two
different reasons:

- Statically, `allie.semantic.resolver` already rejects a `when` clause
  outside a strategy, so a `buy()` in a `when` body is structurally
  confined.
- Dynamically, these check for an active engine
  (`allie.quant.backtest_engine.current_engine`) and raise otherwise.
  This is the check that catches a bare top-level `buy();` -- the
  resolver deliberately does not existence-check call targets (no
  builtin registry existed when it was written), so nothing earlier in
  the pipeline knows a given `Call` is specifically `buy`.
"""

from __future__ import annotations

from allie.quant.backtest_engine import current_engine
from allie.runtime.errors import AllieValueError
from allie.stdlib.registry import register
from allie.typechecker.types import VOID, FunctionType


def _require_engine(name: str) -> object:
    engine = current_engine()
    if engine is None:
        raise AllieValueError(
            f"{name}() is only valid inside a strategy body running under a backtest"
        )
    return engine


def buy() -> None:
    engine = _require_engine("buy")
    engine.buy()  # type: ignore[attr-defined]


def sell() -> None:
    engine = _require_engine("sell")
    engine.sell()  # type: ignore[attr-defined]


register("buy", FunctionType(params=(), ret=VOID), buy)
register("sell", FunctionType(params=(), ret=VOID), sell)
