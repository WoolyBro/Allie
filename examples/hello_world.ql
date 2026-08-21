// The north star: every Phase should get incrementally closer to running
// this file end-to-end (see docs/spec.md). Brace-based syntax, gradual
// typing, event-driven backtest block.

let greeting: str = "hello, allie";
print(greeting);

let prices: Series[float] = load_csv("examples/data/sample_prices.csv", column: "close");
let fast_ma = sma(prices, window: 10);
let slow_ma = sma(prices, window: 50);

backtest MyFirstStrategy {
    on_bar(bar) {
        if fast_ma[bar.index] > slow_ma[bar.index] {
            buy(size: 100);
        } else {
            sell(size: 100);
        }
    }

    slippage_model: fixed_bps(5);
}
