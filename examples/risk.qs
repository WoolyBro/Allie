// risk.qs -- a risk report, a regression, and a strategy built on trailing
// statistics, from one price file.
//
//   allie run examples/risk.qs
//   allie run examples/risk.qs --verify    # all three backends must agree

let AAPL = load("tests/fixtures/AAPL_sample.csv");
let c: Series[Float] = AAPL.close;
let r: Series[Float] = pct_change(c);

// ---- A risk report ----
// Statistics skip missing bars: r has no return at its first bar, and that
// doesn't turn the volatility into NaN.
print("annualized volatility, Sharpe, Sortino:");
print(volatility(r, 252));
print(sharpe(r, 252));
print(sortino(r, 252));
print("one-day 95% value at risk and expected shortfall:");
print(value_at_risk(r, 0.95));
print(expected_shortfall(r, 0.95));
print("max drawdown, and how far the last bar sits below its high:");
print(max_drawdown(c));
print(drawdown(c)[len(c) - 1]);

// ---- Does yesterday's return predict today's? ----
// ols(y, X) fits y on X's columns; a column of ones is the intercept. Rows
// with a missing value -- the first two, here -- are left out.
let ones: Series[Float] = c * 0.0 + 1.0;
let X = with_column(matrix(ones, len(c), 1), shift(r, 1));
let fit: Series[Float] = ols(r, X);
print("intercept and slope on the previous return:");
print(fit[0]);
print(fit[1]);

// ---- Statistics in a strategy must be trailing ----
// zscore(c) measures every bar against the mean of ALL the bars, later ones
// included, so `when zscore(c) < -1.0` would trade on the future -- the
// compiler rejects it and suggests rolling_zscore, which uses the last 20
// bars only.
let z: Series[Float] = rolling_zscore(c, 20);
strategy Revert {
    when z < -1.0: buy();
    when z > 0.0: sell();
}

// A 70/30 split: fit on the first part, judge on the rest.
let split: Int = floor(len(c) * 0.7);
let train = backtest Revert on AAPL from 0 to split;
let test = backtest Revert on AAPL from split to len(c);
print("mean reversion, in-sample and out-of-sample Sharpe:");
print(train.sharpe);
print(test.sharpe);
// Good in-sample, bad out-of-sample: the reason to split at all.
