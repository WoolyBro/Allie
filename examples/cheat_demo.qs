// A strategy that CHEATS: it peeks one bar into the future.
// shift(x, -1)[i] == x[i+1] -- tomorrow's price, available today. Were it
// allowed to run, it would report Sharpe 3.31, against 1.14 for the same
// rule on today's close. `allie check --audit` shows where it cheats.
let AAPL = load("tests/fixtures/AAPL_sample.csv");
let slow: Series[Float] = SMA(AAPL.close, 20);
let tomorrow: Series[Float] = shift(AAPL.close, -1);

strategy Clairvoyant {
    when tomorrow crosses_above slow: buy();
    when tomorrow crosses_below slow: sell();
}
backtest Clairvoyant on AAPL from 0 to 100;
