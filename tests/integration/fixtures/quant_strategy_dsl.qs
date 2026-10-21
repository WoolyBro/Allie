// Quant-only family: the strategy DSL as examples/strategies.qs uses it --
// compound and level-crossing conditions, results as values, computed
// bounds, and strategies declared in a loop.
let AAPL = load("@FIXTURE@");
let c: Series[Float] = AAPL.close;

// ---- Indicators are one line each ----
// Operators work on whole series, bar by bar, and a number on either side
// applies to every bar. Warm-up bars are NaN, and a comparison with NaN is
// false, so nothing below can signal before its indicator has data.
let macd: Series[Float] = EMA(c, 12) - EMA(c, 26);
let macd_signal: Series[Float] = EMA(macd, 9);
let mid: Series[Float] = SMA(c, 20);
let lower: Series[Float] = mid - 2.0 * rolling_std(c, 20);
let rsi: Series[Float] = RSI(c, 14);
let range_pct: Series[Float] = ATR(AAPL, 14) / c * 100.0;

print("MACD crossovers, RSI dips below 40:");
print(count(macd crosses_above macd_signal));
print(count(rsi crosses_below 40.0));

// ---- A `when` condition is any Series<Bool> ----
strategy TrendWithFilter {
    when macd crosses_above macd_signal and rsi < 70.0: buy();
    when macd crosses_below macd_signal or c < lower: sell();
}

// Crossing a number is a level crossing.
strategy MeanReversion {
    when rsi crosses_below 40.0 and range_pct < 3.0: buy();
    when rsi crosses_above 55.0: sell();
}

// The statement form prints the results table.
backtest TrendWithFilter on AAPL from 0 to len(c);

// ---- Results are values ----
let r = backtest MeanReversion on AAPL from 0 to len(c);
print("mean reversion: trades, Sharpe, max drawdown:");
print(r.trades);
print(r.sharpe);
print(r.max_drawdown);
// r.returns are the same per-bar returns the Sharpe above was computed from.
print(sharpe(r.returns, 252) == r.sharpe);

// ---- In-sample / out-of-sample ----
// The bounds are Int expressions, so a split is two backtests of one strategy.
let split: Int = 40;
let train = backtest TrendWithFilter on AAPL from 0 to split;
let test = backtest TrendWithFilter on AAPL from split to len(c);
print("trend: in-sample Sharpe, out-of-sample Sharpe:");
print(train.sharpe);
print(test.sharpe);
// On this 60-bar sample the strategy's one good trade lands in-sample, and
// out-of-sample it does nothing: the in-sample Sharpe alone would mislead.

// ---- A parameter sweep ----
// A strategy captures its condition when it is declared, so declaring one
// inside a loop gives each iteration its own -- and trying several windows
// is several tests, which the threshold below corrects for.
let slow: Series[Float] = SMA(c, 20);
let windows: Int = 0;
let best_sharpe: Float = -1000.0;
let best_window: Int = 0;
for let w: Int = 3; w <= 12; w = w + 3 {
    let fast: Series[Float] = SMA(c, w);
    strategy Sweep {
        when fast crosses_above slow: buy();
        when fast crosses_below slow: sell();
    }
    let result = backtest Sweep on AAPL from 0 to len(c);
    windows = windows + 1;
    if result.sharpe > best_sharpe {
        best_sharpe = result.sharpe;
        best_window = w;
    }
}
print("best window, its Sharpe, and the p-value 0.05 becomes after the sweep:");
print(best_window);
print(best_sharpe);
print(bonferroni(0.05, windows));
