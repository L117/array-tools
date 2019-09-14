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
- Initialization with iterator.
- Initizlization with function (with or without index passed as
  argument).
- By-value into iterator.
- By-value chunks iterator.
- By-value split.
- By-value join.
- No dependency on `std` and no heap allocations.

## Examples

```rust
use array_tools::{self, ArrayIntoIterator};


// Initialization with iterator.
let array1: [u64; 7] = 
    array_tools::try_init_from_iterator(0u64..17).unwrap();

assert_eq!(array1, [0, 1, 2, 3, 4, 5, 6]);


// Initialization with function (w/o index).
let mut value = 0u64;

let array2: [u64; 7] = array_tools::init_with(|| {
    let tmp = value;
    value += 1;
    tmp
});

assert_eq!(array2, [0, 1, 2, 3, 4, 5, 6]);


// Initialization with function (w/ index).
let array3: [u64; 7] = array_tools::indexed_init_with(|idx| {
    idx as u64
});

assert_eq!(array3, [0, 1, 2, 3, 4, 5, 6]);


// By-value iterator.
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

// Split
let array6: [u64; 7] = [1, 2, 3, 4, 5, 6, 7];
let (array7, array8): ([u64; 3], [u64; 4]) = array_tools::split(array6);

assert_eq!(array7, [1, 2, 3]);
assert_eq!(array8, [4, 5, 6, 7]);

// Join
let array9: [u64; 3] = [1, 2, 3];
let array10: [u64; 4] = [4, 5, 6, 7];
let array11: [u64; 7] = array_tools::join(array9, array10);

assert_eq!(array11, [1, 2, 3, 4, 5, 6, 7]);

// Chunks
use array_tools::{ArrayChunk, ArrayChunks};
use core::marker::PhantomData;

let array12: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];

let mut chunks: ArrayChunks<u64, [u64; 8], [u64; 3], [u64; 2]> = ArrayChunks::new(array12);

assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([1, 2, 3], PhantomData)));
assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([4, 5, 6], PhantomData)));
assert_eq!(chunks.next(), Some(ArrayChunk::Stump([7, 8], PhantomData)));
assert_eq!(chunks.next(), None);
```
