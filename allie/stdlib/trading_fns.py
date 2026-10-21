"""Order builtins: `buy()`, `sell()`, `short()`, `cover()` and
`target(fraction)`. Ordinary registered builtins, not keywords -- calls
resolved through `allie.stdlib.registry` exactly like `print`/`load`, so
they need no parser, lowering or codegen support of their own and reach
the LLVM backend through the same bridge as every other builtin. What
each one does to the position is in allie/quant/backtest_engine.py
("Positions").

**Valid only while a strategy runs**, enforced at two levels:

- Statically, the type checker rejects an order outside a `when` action
  or a function body (a function may be called from an action).
- Dynamically, each checks for an active engine
  (`allie.quant.backtest_engine.current_engine`) and raises otherwise --
  which catches a function that places an order being called from
  outside a backtest.
"""

from __future__ import annotations

from allie.quant.backtest_engine import BacktestEngine, current_engine
from allie.runtime.errors import AllieValueError
from allie.stdlib.registry import register
from allie.typechecker.types import FLOAT, VOID, FunctionType


def _require_engine(name: str) -> BacktestEngine:
    engine = current_engine()
    if engine is None:
        raise AllieValueError(
            f"{name}() is only valid inside a strategy body running under a backtest"
        )
    return engine


def buy() -> None:
    _require_engine("buy").buy()


def sell() -> None:
    _require_engine("sell").sell()


def short() -> None:
    _require_engine("short").short()


def cover() -> None:
    _require_engine("cover").cover()


def target(fraction: float) -> None:
    _require_engine("target").target(fraction)


register("buy", FunctionType(params=(), ret=VOID), buy)
register("sell", FunctionType(params=(), ret=VOID), sell)
register("short", FunctionType(params=(), ret=VOID), short)
register("cover", FunctionType(params=(), ret=VOID), cover)
register("target", FunctionType(params=(FLOAT,), ret=VOID), target)
