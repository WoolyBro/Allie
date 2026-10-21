"""Placeholder from Phase 0; the backtest engine is
allie/quant/backtest_engine.py.

What this package was reserved for exists there: the engine runs bar by
bar, fills orders at the next bar's open by default, and charges slippage
and commission unless the source says `with slippage: 0.0` -- the
"first-class cost model" docs/spec.md asked for, as a `with` clause on
`backtest` rather than a `slippage_model:` block.
"""
