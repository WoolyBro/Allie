// Combined family: a `for` loop driving quant calls -- the one fixture
// that exercises both halves of the language together rather than side
// by side.
let AAPL = load("@FIXTURE@");

for let w: Int = 5; w <= 20; w = w + 5 {
    let s: Series[Float] = SMA(AAPL.close, w);
    print(w);
    print(mean(AAPL.close));
    print(s);
}
