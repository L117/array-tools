# Array Tools

A little collection of array-related utils aiming to make life easier. 


## Stability warning

Requires nightly.

Consider this crate experimental. Some (all?) of currently provided features 
are most likely will be integrated into `rust`'s core/std library sooner or 
later, and with arrival of const generics public interfaces are most likely 
will be changed.

## Features

- **Metafeature:** all features below should work for arrays of **any** size.
- Array initialization with iterator.
- Array initizlization with function (without or without index passed as 
  argument).
- Array by-value "into" iterator.
- No dependency on `std` and no heap allocations.

## Examples

```rust
use array_tools::{self, ArrayIntoIterator};


// Array initialization with iterator.
let array1: [u64; 7] = 
    array_tools::try_init_from_iterator(0u64..17).unwrap();

assert_eq!(array1, [0, 1, 2, 3, 4, 5, 6]);


// Array initialization with function (w/o index).
let mut value = 0u64;

let array2: [u64; 7] = array_tools::init_with(|| {
    let tmp = value;
    value += 1;
    tmp
});

assert_eq!(array2, [0, 1, 2, 3, 4, 5, 6]);


// Array initialization with function (w/ index).
let array3: [u64; 7] = array_tools::indexed_init_with(|idx| {
    idx as u64
});

assert_eq!(array3, [0, 1, 2, 3, 4, 5, 6]);


// Array by-value iterator.
#[derive(Debug, PartialEq, Eq)]
struct NonCopyable(u64);

let array4: [NonCopyable; 7] = 
    array_tools::indexed_init_with(|idx| NonCopyable(idx as u64));

let iter = ArrayIntoIterator::new(array4);

let array5: [NonCopyable; 7] = 
    array_tools::try_init_from_iterator(iter).unwrap();

assert_eq!(array5, [
    NonCopyable(0),
    NonCopyable(1),
    NonCopyable(2),
    NonCopyable(3),
    NonCopyable(4),
    NonCopyable(5),
    NonCopyable(6),
]);
```