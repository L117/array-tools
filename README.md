# Array Tools

[![Crate](https://img.shields.io/crates/v/array-tools.svg)](https://crates.io/crates/array-tools)
[![Documentation](https://docs.rs/array-tools/badge.svg)](https://docs.rs/array-tools)
[![Build Status](https://travis-ci.com/L117/array-tools.svg?branch=master)](https://travis-ci.com/L117/array-tools)
[![Build status](https://ci.appveyor.com/api/projects/status/9f4ctfoat9i9h86w?svg=true)](https://ci.appveyor.com/project/L117/array-tools)

A collection of tools to help dealing with our beloved ❤️ fixed size arrays (Including generic contexts). 

## Stability notice

Requires nightly.

This crate heavely uses `FixedSizeArray` trait, which is currently experimental.
Because of this, crate is experimental as well.
No other sources of severe breakage should be expected.

## Features

- **Metafeature**: all features below should work for arrays of **any** size.
- Initialization with iterator.
- Initialization with function (with or without index as argument).
- Consuming iterator.
- Consuming chunks iterator.
- Consuming split.
- Consuming join.
- No dependency on `std` and no heap allocations, thanks to underlaying fixed-capacity stack-allocated deque-like structure.

## Examples

See [documentation](https://docs.rs/array-tools) for examples, it covers most if not all use cases.

## Contributing

Contributions of any shape and form are welcome.