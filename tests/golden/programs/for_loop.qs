// Guards the Phase 5.2 desugar-to-`while` lowering: a `for` loop must
// lower to the same Label/Jump/JumpIfFalse primitives `while` uses,
// with a dedicated increment label for `continue` to target. If a
// `for`-specific path ever appears in lowering or codegen, this
// snapshot changes.
let total: Int = 0;
for let i: Int = 0; i < 5; i = i + 1 {
    if i == 2 { continue; }
    total = total + i;
}
print(total);
