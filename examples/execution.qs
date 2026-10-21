// execution.qs -- what a backtest assumes about trading, written in the source.
//
//   allie run examples/execution.qs
//   allie run examples/execution.qs --verify    # all three backends must agree

let AAPL = load("tests/fixtures/AAPL_sample.csv");
let c: Series[Float] = AAPL.close;
let fast: Series[Float] = SMA(c, 5);
let slow: Series[Float] = SMA(c, 20);

strategy Trend {
    when fast crosses_above slow: buy();
    when fast crosses_below slow: sell();
}

// ---- Costs are on unless the source turns them off ----
// Every fill pays 5 bps of slippage by default, and an order signalled at a
// bar's close fills at the NEXT bar's open -- the first price anyone could
// actually have traded at. (`with fill: "close"` assumes you traded at the
// very close the signal was computed from.) How much of a result survives
// friction is a question worth asking of every strategy:
print("final equity at 0, 25 and 50 bps of slippage per fill:");
for let bps: Int = 0; bps <= 50; bps = bps + 25 {
    let r = backtest Trend on AAPL from 0 to len(c) with slippage: bps / 10000.0;
    print(r.final_equity);
}

// ---- Long, short, and sized ----
// buy()/sell() go long and flat; short()/cover() go short and flat; buy()
// from a short (or short() from a long) reverses. target(f) holds f times
// equity: 0.5 is half in, -1.0 all-in short.
strategy LongShort {
    when fast crosses_above slow: buy();
    when fast crosses_below slow: short();
}
strategy HalfIn {
    when fast crosses_above slow: target(0.5);
    when fast crosses_below slow: target(0.0);
}
let ls = backtest LongShort on AAPL from 0 to len(c);
let half = backtest HalfIn on AAPL from 0 to len(c);
print("long/short exposure, half-in exposure:");
print(ls.exposure);
print(half.exposure);

// ---- Stops ----
// Checked against each bar's high and low; a bar that gaps past a level
// exits at the open, and a bar that reaches both is read as the stop.
let rsi: Series[Float] = RSI(c, 5);
strategy Dip {
    when rsi < 45.0: buy();
}
let dip = backtest Dip on AAPL from 0 to len(c)
    with stop_loss: 0.01, take_profit: 0.02, commission: 0.0005;
print(dip);
// Stops turn one long hold into many short trades, and every one pays its
// costs twice: compare Costs Paid with what the strategy made.

// ---- Trades are data ----
print("trades, win rate, mean and spread of trade returns:");
print(dip.trades);
print(dip.win_rate);
print(mean(dip.trade_returns));
print(std(dip.trade_returns));
