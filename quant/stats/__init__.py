"""Statistical rigor tooling: walk-forward/out-of-sample splitting and
multiple-comparisons correction (Bonferroni, permutation/Monte Carlo null).

This exists so evaluating the 5 target indicators doesn't p-hack the
paper's central claim by construction (docs/spec.md "Statistical rigor").
All stochastic functions here take an explicit seed (docs/spec.md
"Determinism") — no hidden global RNG state.
"""
