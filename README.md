# Array Tools

A little collection of array-related utils aiming to make life easier. 


## Warning

Requires nightly.

Consider this crate experimental. Some (all?) of currently provided features are most likely will be 
integrated into `rust`'s core/std library sooner or later, and with arrival of const generics public 
interfaces are most likely will be changed.

## Features

- Fixed-size array (of arbitrary size) initialization with iterator.
```rust
use array_tools;

fn main() {
    
    let array: [u64; 10] = array_tools::try_init_from_iterator(0..10).unwrap();
    
    println!("{:?}", array);
    // Prints [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

- Fixed-size array (of arbitrary size) by-value "into" iterator.
```rust
use array_tools::ArrayIntoIterator;

#[derive(Debug)]
struct NonCopyable(u64);

fn main() {
    
    let array = [NonCopyable(0), NonCopyable(1), NonCopyable(2), NonCopyable(3), NonCopyable(4)];

    let vec: Vec<NonCopyable> = ArrayIntoIterator::new(array).rev().collect();

    println!("{:?}", vec);
    // Prints [NonCopyable(4), NonCopyable(3), NonCopyable(2), NonCopyable(1), NonCopyable(0)]
}
```