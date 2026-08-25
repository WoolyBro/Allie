// A representative scalar program: recursion, if/else, arithmetic --
// the native-LLVM path with no boxing involved.
func factorial(n: Int) -> Int {
    if n <= 1 { return 1; } else { return n * factorial(n - 1); }
}
print(factorial(5));
