// rigor.qs -- what the language does that a backtesting library doesn't.
//
// A window sweep is a multiple-comparisons problem. Try five moving-average
// windows and the best one WILL look good, even if all five are noise --
// that is what "best of five" means. This program measures each window and
// then corrects for having tried them all.
//
//   allie run examples/rigor.qs
//   allie run examples/rigor.qs --verify    # all three backends must agree

let AAPL = load("tests/fixtures/AAPL_sample.csv");

// Next-period returns: what a signal would have to predict to be useful.
let returns: Series[Float] = pct_change(AAPL.close);

// Five candidate signals. `permutation_pvalue` shuffles each signal a few
// hundred times and asks how often chance alone matches the association it
// actually shows -- the seed is explicit, so this is reproducible.
let trials: Int = 499;
let seed: Int = 20260825;
let tested: Int = 5;

for let w: Int = 5; w <= 25; w = w + 5 {
    // shift(.., 1) lags the signal by one bar: yesterday's average against
    // today's return. Without the lag the test is contemporaneous, not
    // predictive -- SMA at bar i already contains close at bar i.
    // This is the SAFE direction of shift; see the note at the bottom.
    let signal: Series[Float] = shift(SMA(AAPL.close, w), 1);
    let raw: Float = permutation_pvalue(signal, returns, trials, seed);

    print("window / raw p / p corrected for 5 tests:");
    print(w);
    print(raw);
    print(bonferroni(raw, tested));
}

// Note what is NOT possible here. Swapping the signal for
//     shift(AAPL.close, -1)
// -- tomorrow's price, known today -- would make the association look
// spectacular, and the compiler rejects it before the program runs.
// See docs/how-it-works.md for that check.
