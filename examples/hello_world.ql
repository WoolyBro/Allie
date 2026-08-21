// The north star: every Phase should get incrementally closer to running
// this file end-to-end (see docs/spec.md). Brace-based syntax, gradual
// typing, event-driven backtest block.
//
// Updated for the Phase 2.2 grammar (docs/grammar.md): capitalized base
// types (String/Float, not str/float), no keyword arguments or
// subscript indexing (neither is in the grammar yet), and the actual
// strategy/when/backtest constructs the grammar defines -- not the
// earlier ad-hoc `backtest NAME { on_bar { ... } }` sketch, which never
// matched what got built.

let greeting: String = "hello, allie";
print(greeting);

let prices = load_csv("examples/data/sample_prices.csv");
let fast_ma = sma(prices, 10);
let slow_ma = sma(prices, 50);

strategy MyFirstStrategy {
    when fast_ma crosses_above slow_ma: buy();
    when fast_ma crosses_below slow_ma: sell();
}

backtest MyFirstStrategy on AAPL from 0 to 100;
