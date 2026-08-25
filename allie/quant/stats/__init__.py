"""Statistical rigor tooling (docs/spec.md "Statistical rigor").

The multiple-comparisons correction and permutation testing this module
was reserved for now live in `allie.stdlib.significance`, alongside the
rest of the registered builtins -- `bonferroni`, `benjamini_hochberg`,
`sharpe`, and `permutation_pvalue`. They are registered there rather
than here because every Allie-callable function goes through
`allie.stdlib.registry`, and splitting the implementation from its
registration would mean two places to look.

Walk-forward / out-of-sample splitting is still unbuilt -- it needs a
way to express a date range in source, which the language does not have
yet (`Date`/`DateRange` parse but nothing produces one).
"""
