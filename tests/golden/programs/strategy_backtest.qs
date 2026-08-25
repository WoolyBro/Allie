// Guards the Call-into-backtest-engine lowering. Phase 7 had to rewrite
// this path entirely because the Phase 5.2 version was structurally
// wrong (nameless declare, `when` bodies lowered inline and
// unconditionally, raw strings where register names were expected) and
// nothing caught it. This snapshot is that missing guard.
let AAPL = load("@FIXTURE@");
let fast: Series[Float] = SMA(AAPL.close, 5);
let slow: Series[Float] = SMA(AAPL.close, 20);

strategy MovingAverage {
    when fast crosses_above slow: buy();
    when fast crosses_below slow: sell();
}

backtest MovingAverage on AAPL from 0 to 100;
