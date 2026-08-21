"""Name resolution: the Resolver pass that runs after parsing and before
type-checking (allie.typechecker.checker). Validates break/continue/
return/when placement and variable existence via a scope stack -- see
allie.semantic.resolver for the full rationale.

(Originally slated for gradual-typing checks per an earlier plan; that
work now lives in allie.typechecker, which needs its own TypeEnvironment
distinct from this module's Scope -- see allie/typechecker/checker.py.)
"""

from allie.semantic.resolver import Resolver, Scope

__all__ = ["Resolver", "Scope"]
