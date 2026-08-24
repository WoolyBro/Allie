// The smallest complete Allie program that touches both halves of the
// language. Start here, then read examples/blended.qs.
//
// Renamed from hello_world.ql in Phase 9. The original was written in
// Phase 0 as a target to build toward and never actually ran -- it
// called `load_csv`/`sma`, which are named `load`/`SMA` in the stdlib
// registry, and backtested a symbol it never loaded. Fixed rather than
// left as a broken file named "hello world" in the examples directory.

let greeting: String = "hello, " + "allie";
print(greeting);

let AAPL = load("tests/fixtures/AAPL_sample.csv");
let fast: Series[Float] = SMA(AAPL.close, 5);
let slow: Series[Float] = SMA(AAPL.close, 20);

strategy MyFirstStrategy {
    when fast crosses_above slow: buy();
    when fast crosses_below slow: sell();
}

backtest MyFirstStrategy on AAPL from 0 to 100;
