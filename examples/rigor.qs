// rigor.qs -- is a backtest's result real, or the best of many tries?
//
// A window sweep is a multiple-comparisons problem. Try five moving-average
// windows and the best one WILL look good, even if all five are noise --
// that is what "best of five" means. This program runs the sweep, then asks
// the three questions a backtesting library leaves to you:
//
//   1. Is each strategy's timing better than its own positions at random times?
//   2. Which strategies survive having tried all five?
//   3. Is the best one's Sharpe ratio real, given that it is the best of five?
//
//   allie run examples/rigor.qs
//   allie run examples/rigor.qs --verify    # all three backends must agree

let AAPL = load("tests/fixtures/AAPL_sample.csv");
let c: Series[Float] = AAPL.close;

let sharpes: Series[Float] = [];
let pvalues: Series[Float] = [];
let best_sharpe: Float = 0.0 - 1000.0;
let best_returns: Series[Float] = [];

for let w: Int = 4; w <= 20; w = w + 4 {
    let trend: Series[Float] = SMA(c, w);
    strategy Cross {
        when c crosses_above trend: buy();
        when c crosses_below trend: sell();
    }
    let r = backtest Cross on AAPL from 0 to len(c);

    // 1. The same positions, shifted to every other point in time: how
    // often do they earn as much? Shifting keeps how long the strategy
    // holds, which shuffling bars one by one would not.
    let p: Float = backtest_pvalue(r);
    print("window / Sharpe / timing p-value:");
    print(w);
    print(r.sharpe);
    print(p);

    sharpes = append(sharpes, r.sharpe);
    pvalues = append(pvalues, p);
    if r.sharpe > best_sharpe {
        best_sharpe = r.sharpe;
        best_returns = r.returns;
    }
}

// 2. Which survive having tried five? Holm bounds the chance of even one
// false positive; Benjamini-Hochberg the share of false ones.
print("significant after correcting for five tries (Holm, then BH):");
print(holm(pvalues, 0.05));
print(benjamini_hochberg(pvalues, 0.05));

// 3. The best Sharpe, judged as one strategy and then as the best of five.
// The first ignores the sweep; the second is the honest number.
print("best Sharpe, and the chance its true Sharpe is above 0 -- alone, then deflated:");
print(best_sharpe);
print(probabilistic_sharpe(best_returns, 0.0, 252));
print(deflated_sharpe(best_returns, sharpes, 252));

// A signal can be tested before it becomes a strategy: does yesterday's
// distance from the 10-bar average predict today's return? Test returns
// and spreads like this one, not price levels -- a series that trends
// looks unlike its own shifted copies whatever it is compared with.
let stretch: Series[Float] = shift(c / SMA(c, 10) - 1.0, 1);
print("does the stretch from SMA(10) predict the next return? p-value:");
print(permutation_pvalue(stretch, pct_change(c)));

// What the compiler won't let through: choosing the window by its Sharpe
// and then trading it on the same bars. best_returns is fine to report,
// but a strategy built on a window picked by r.sharpe would be trading on
// a statistic of every bar -- a look-ahead error.
