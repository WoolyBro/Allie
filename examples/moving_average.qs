// The canonical Allie strategy: a fast/slow moving-average crossover,
// backtested over a price history.
//
// Runs identically on all three execution backends -- see
// tests/unit/test_backtest_cross_validation.py:
//   allie run examples/moving_average.qs                  (tree-walking)
//   allie run examples/moving_average.qs --backend ir     (IR interpreter)
//   python main.py examples/moving_average.qs             (LLVM/JIT)

let AAPL = load("tests/fixtures/AAPL_sample.csv");

let fast: Series[Float] = SMA(AAPL.close, 5);
let slow: Series[Float] = SMA(AAPL.close, 20);

strategy MovingAverage {
    when fast crosses_above slow: buy();
    when fast crosses_below slow: sell();
}

backtest MovingAverage on AAPL from 0 to 100;
