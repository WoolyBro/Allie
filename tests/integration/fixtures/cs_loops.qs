// Pure-CS family: for/while/break/continue, no quant content at all.
let total: Int = 0;
for let i: Int = 0; i < 10; i = i + 1 {
    if i == 8 { break; }
    if i == 3 { continue; }
    total = total + i;
}
print(total);

let countdown: Int = 5;
let steps: Int = 0;
while countdown > 0 {
    countdown = countdown - 1;
    steps = steps + 1;
}
print(steps);
