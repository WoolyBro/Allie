// Regression: block shadowing must behave identically on all three
// backends. It did not before the alpha-renaming fix in IR lowering --
// lowering flattens blocks away, and both the IR interpreter and LLVM
// keep one flat name->value map, so a shadowing `let` clobbered the
// outer binding and never restored it. This printed 2 on those two
// backends and 1 on the tree-walker.
//
// This fixture runs on all three backends against one .expected file,
// so any future divergence fails here.

let x: Int = 1;
{
    let x: Int = 2;
    print(x);
}
print(x);

// Nested shadowing, and reading the outer value in the initializer.
let y: Int = 10;
{
    let y: Int = y + 5;
    {
        let y: Int = y + 5;
        print(y);
    }
    print(y);
}
print(y);

// A loop variable must not leak out or clobber an outer name.
let i: Int = 99;
for let i: Int = 0; i < 2; i = i + 1 {
    print(i);
}
print(i);

// Assigning (not declaring) inside a block still hits the outer binding.
let z: Int = 1;
{
    z = 7;
}
print(z);
