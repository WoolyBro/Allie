// The smallest complete Allie program that touches both halves of the
// language: a string, and a strategy backtested on a price file. Start
// here, then read examples/blended.qs.

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
