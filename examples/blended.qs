// blended.qs -- both halves of Allie in one program.
//
// General-purpose control flow (a `for` loop, a function, `continue`)
// driving a quant-specific DSL (`Series`, `SMA`, `strategy`/`when`,
// `backtest`). Neither half knows the other exists; they compose
// because they lower to the same IR.
//
//   allie run examples/blended.qs                 # tree-walking
//   allie run examples/blended.qs --backend ir    # IR interpreter
//   python main.py examples/blended.qs --debug-ir # LLVM/JIT, showing IR
//
// All three print identical output -- see
// tests/unit/test_backtest_cross_validation.py.

let AAPL = load("tests/fixtures/AAPL_sample.csv");

// A plain function over scalars -- compiled to native LLVM, no
// quant types involved.
func spread_bps(fast_avg: Float, slow_avg: Float) -> Float {
    return (fast_avg - slow_avg) / slow_avg * 10000.0;
}

// Sweep fast-MA windows. The loop is ordinary control flow; the body
// calls into the numpy-backed stdlib.
//
// EMA rather than SMA for this part on purpose: SMA NaN-pads its
// first window-1 bars (pandas convention), and mean() over a series
// containing NaN is NaN -- correct, but it would print `nan` here.
// EMA seeds from the first observation, so there is no warmup gap.
let slow: Series[Float] = EMA(AAPL.close, 20);

for let w: Int = 4; w <= 16; w = w + 4 {
    let fast: Series[Float] = EMA(AAPL.close, w);

    // Skip windows too close to the slow MA to cross meaningfully.
    if w == 12 { continue; }

    print(w);
    print(spread_bps(mean(fast), mean(slow)));
}

// The quant DSL. `crosses_above`/`crosses_below` produce Series<Bool>
// signals; the engine walks bars and runs the matching action.
let fast_signal: Series[Float] = SMA(AAPL.close, 5);
let slow_signal: Series[Float] = SMA(AAPL.close, 20);

strategy Crossover {
    when fast_signal crosses_above slow_signal: buy();
    when fast_signal crosses_below slow_signal: sell();
}

backtest Crossover on AAPL from 0 to 100;
