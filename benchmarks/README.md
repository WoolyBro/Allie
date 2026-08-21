# Benchmarks

Compares an equivalent backtest implemented in QuantLang against a
hand-written pandas backtest, to give the paper a concrete performance
claim (docs/spec.md "Benchmarking").

Not runnable yet — depends on the interpreter (Phase 1+) and the backtest
engine (`quant.backtest`). Structure once those land:

```
benchmarks/
  bench_backtest.py       # entry point, runs both and reports wall time
  strategies/
    sma_crossover.ql       # QuantLang version
    sma_crossover_pandas.py  # hand-written equivalent
```
