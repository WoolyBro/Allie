# Benchmarks

Compares an equivalent backtest implemented in Allie against a
hand-written pandas backtest, to give the paper a concrete performance
claim (docs/spec.md "Benchmarking").

Not runnable yet — depends on the interpreter having a real parser feeding
it (Phase 2+) and the backtest engine (`allie.quant.backtest`). Structure
once those land:

```
benchmarks/
  bench_backtest.py       # entry point, runs both and reports wall time
  strategies/
    sma_crossover.ql       # Allie version
    sma_crossover_pandas.py  # hand-written equivalent
```
