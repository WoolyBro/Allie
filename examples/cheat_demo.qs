// A strategy that CHEATS: it peeks one bar into the future.
// shift(x, -1)[i] == x[i+1] -- tomorrow's price, available today.
let AAPL = load("tests/fixtures/AAPL_sample.csv");
let slow: Series[Float] = SMA(AAPL.close, 20);
let tomorrow: Series[Float] = shift(AAPL.close, -1);

strategy Clairvoyant {
    when tomorrow crosses_above slow: buy();
    when tomorrow crosses_below slow: sell();
}
backtest Clairvoyant on AAPL from 0 to 100;
