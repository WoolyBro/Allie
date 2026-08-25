// Pure-CS family: recursion and scalar arithmetic.
func factorial(n: Int) -> Int {
    lowkey n <= 1 { yeet 1; } nahhh { yeet n * factorial(n - 1); }
}
func fib(n: Int) -> Int {
    if n < 2 { return n; } else { return fib(n - 1) + fib(n - 2); }
}
print(factorial(10));
print(fib(15));
print(17 % 5);
print(-3 + 2 * 4);
