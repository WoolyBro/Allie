"""Allie standard library, resolved by name in import statements
(docs/spec.md "Module / import system"). Trading-calendar/holiday
utilities live here in the future; CSV loading is allie.stdlib.io.load.

Importing this package (not just allie.stdlib.registry) registers every
builtin as a side effect -- import allie.stdlib once before constructing
a TypeChecker or Environment that need builtins visible (see
allie.cli.run_command).
"""

from allie.stdlib import io, math_fns, matrix_ops, series_ops, stats_fns, trading_fns
from allie.stdlib.registry import BUILTINS, builtin_function_types, create_global_environment

__all__ = [
    "BUILTINS",
    "builtin_function_types",
    "create_global_environment",
    "io",
    "math_fns",
    "matrix_ops",
    "series_ops",
    "stats_fns",
    "trading_fns",
]
