# Array Tools

[![Crate](https://img.shields.io/crates/v/array-tools.svg)](https://crates.io/crates/array-tools)
[![Documentation](https://docs.rs/array-tools/badge.svg)](https://docs.rs/array-tools)
[![Build Status](https://travis-ci.com/L117/array-tools.svg?branch=master)](https://travis-ci.com/L117/array-tools)
[![Build status](https://ci.appveyor.com/api/projects/status/9f4ctfoat9i9h86w?svg=true)](https://ci.appveyor.com/project/L117/array-tools)

A collection of tools that make it easier to work with arrays.

All currently supported features are `no_std` compatible.

```rust
use array_tools as at;

let left: [usize; 4] = at::init_with_mapped_idx(|idx| idx + 1);
let right: [usize; 3] = at::init_with_iter(5..8).unwrap();

let joined: [usize; 7] = at::join(left, right);

let (left, right): ([usize; 2], [usize; 5]) = at::split(joined);

assert_eq!(left, [1, 2]);
assert_eq!(right, [3, 4, 5, 6, 7]);
```

### License

Licensed under the terms of either Apache 2.0 or MIT license.


