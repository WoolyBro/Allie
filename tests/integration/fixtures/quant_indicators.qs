// Quant-only family: stdlib series/stats calls, no control flow.
let prices = load("@FIXTURE@");
let sma5: Series[Float] = SMA(prices.close, 5);
print(sma5);
print(mean(prices.close));
print(median(prices.close));
print(std(prices.close));
