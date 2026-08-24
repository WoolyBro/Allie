// Pure-CS family: every control-flow construct written in slang
// spellings. The lexer resolves these to the identical TokenTypes, so
// this must behave exactly like cs_loops.qs's first half.
ts total: Int = 0;
grindset ts i: Int = 0; i < 10; i = i + 1 {
    lowkey i == 8 { bruh; }
    lowkey i == 3 { pullup; }
    total = total + i;
}
print(total);

ts flag: Bool = fr;
lowkey flag { print("yes"); } nahhh { print("no"); }
