//! # Array Tools
//!
//! [![Crate](https://img.shields.io/crates/v/array-tools.svg)](https://crates.io/crates/array-tools)
//! [![Documentation](https://docs.rs/array-tools/badge.svg)](https://docs.rs/array-tools)
//! [![Build Status](https://travis-ci.com/L117/array-tools.svg?branch=master)](https://travis-ci.com/L117/array-tools)
//! [![Build status](https://ci.appveyor.com/api/projects/status/9f4ctfoat9i9h86w?svg=true)](https://ci.appveyor.com/project/L117/array-tools)
//!
//! A collection of tools to help dealing with our beloved ❤️ fixed size arrays (Including generic contexts).
//!
//! ## Stability notice
//!
//! Requires nightly.
//!
//! This crate depends on [FixedSizeArray](core::array::FixedSizeArray) trait, which is currently experimental.
//! Because of this, crate is experimental as well.
//! No other sources of severe breakage should be expected.
//!
//! ## Features
//!
//! - **Metafeature**: all features below should work for arrays of **any** size.
//! - Initialization with iterator.
//! - Initialization with function (with or without index as argument).
//! - Consuming iterator.
//! - Consuming chunks iterator.
//! - Consuming split.
//! - Consuming join.
//! - No dependency on `std` and no heap allocations, thanks to underlaying fixed-capacity stack-allocated deque-like structure.
//!
//! ## Examples
//!
//! See [documentation](https://docs.rs/array-tools) for examples, it covers most if not all use cases.
//!
//! ## Contributing
//!
//! Contributions of any shape and form are welcome.

#![no_std]
#![feature(fixed_size_array)]
use core::array::FixedSizeArray;
use core::fmt::{self, Debug};
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::{mem, ptr};

struct FixedCapacityDequeLike<T, A>
where
    A: FixedSizeArray<T>,
{
    array: MaybeUninit<A>,
    begining: usize,
    end: usize,
    _element: PhantomData<T>,
}

impl<T, A> FixedCapacityDequeLike<T, A>
where
    A: FixedSizeArray<T>,
{
    fn new() -> FixedCapacityDequeLike<T, A> {
        FixedCapacityDequeLike {
            array: MaybeUninit::uninit(),
            begining: 0,
            end: 0,
            _element: PhantomData,
        }
    }

    fn from_array(array: A) -> FixedCapacityDequeLike<T, A> {
        let length = array.as_slice().len();
        FixedCapacityDequeLike {
            array: MaybeUninit::new(array),
            begining: 0,
            end: length,
            _element: PhantomData,
        }
    }

    fn capacity(&self) -> usize {
        unsafe { (*self.array.as_ptr()).as_slice().len() }
    }

    fn length(&self) -> usize {
        self.end - self.begining
    }

    fn is_empty(&self) -> bool {
        self.length() == 0
    }

    fn is_full(&self) -> bool {
        self.length() == self.capacity()
    }

    fn push_back(&mut self, element: T) {
        if self.end < self.capacity() {
            let item_index = self.end;
            let slice = unsafe { (*self.array.as_mut_ptr()).as_mut_slice() };
            unsafe {
                ptr::write(slice.get_unchecked_mut(item_index), element);
            }
            self.end += 1;
        } else {
            panic!("No capacity left at the end.");
        }
    }

    #[allow(dead_code)]
    fn push_front(&mut self, element: T) {
        if self.begining != 0 {
            let item_index = self.begining - 1;
            let slice = unsafe { (*self.array.as_mut_ptr()).as_mut_slice() };
            unsafe {
                ptr::write(slice.get_unchecked_mut(item_index), element);
            }
            self.begining -= 1;
        } else {
            panic!("No capacity left at the begining.")
        }
    }

    fn pop_back(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let item_index = self.end - 1;
            let item = unsafe {
                let slice = (*self.array.as_ptr()).as_slice();
                ptr::read(slice.get_unchecked(item_index))
            };
            self.end -= 1;
            Some(item)
        }
    }

    fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let item_index = self.begining;
            let item = unsafe {
                let slice = (*self.array.as_ptr()).as_slice();
                ptr::read(slice.get_unchecked(item_index))
            };
            self.begining += 1;
            Some(item)
        }
    }

    fn try_extract_array(&mut self) -> Option<A> {
        if self.length() == self.capacity() {
            let array_shallow_copy = unsafe { ptr::read(self.array.as_ptr()) };
            self.begining = 0;
            self.end = 0;
            Some(array_shallow_copy)
        } else {
            None
        }
    }

    fn as_slice(&self) -> &[T] {
        unsafe { &(*self.array.as_ptr()).as_slice()[self.begining..self.end] }
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { &mut (*self.array.as_mut_ptr()).as_mut_slice()[self.begining..self.end] }
    }
}

impl<T, A> Drop for FixedCapacityDequeLike<T, A>
where
    A: FixedSizeArray<T>,
{
    fn drop(&mut self) {
        while let Some(item) = self.pop_back() {
            mem::drop(item);
        }
    }
}

impl<T, A> Debug for FixedCapacityDequeLike<T, A>
where
    A: FixedSizeArray<T>,
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedCapacityDequeLike")
            .field("array", &self.as_slice())
            .field("begining", &self.begining)
            .field("end", &self.end)
            .field("_element", &self._element)
            .finish()
    }
}

impl<T, A> PartialEq for FixedCapacityDequeLike<T, A>
where
    A: FixedSizeArray<T>,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.capacity() == other.capacity()
            && self.begining == other.begining
            && self.end == other.end
            && self.as_slice() == other.as_slice()
    }
}

impl<T, A> Eq for FixedCapacityDequeLike<T, A>
where
    A: FixedSizeArray<T>,
    T: Eq,
{
}

impl<T, A> Clone for FixedCapacityDequeLike<T, A>
where
    A: FixedSizeArray<T>,
    T: Clone,
{
    fn clone(&self) -> Self {
        let mut clone = FixedCapacityDequeLike {
            array: MaybeUninit::uninit(),
            begining: self.begining,
            end: self.end,
            _element: PhantomData,
        };

        unsafe {
            for (src, dst) in self.as_slice().iter().zip(clone.as_mut_slice()) {
                ptr::write(dst, src.clone());
            }
        }

        clone
    }
}

/// Attempts to create instance of array from iterator.
///
/// - If iterator yields not enough items to fill array, this function returns [`None`](core::option::Option::None).
/// - If iterator yields enough items, this function returns [`Some(array)`](core::option::Option::Some).
/// - If iterator yields excessive items, this function only takes number of items
///   enough to fill array.
///
/// # Panics
///
/// - Only if iterator does.
///
/// # Examples
///
/// Not enough items case.
/// ```rust
/// use array_tools::try_init_from_iterator;
///
/// let mut iter = (0..4);
///
/// let maybe_array: Option<[u64; 5]> = try_init_from_iterator(iter.by_ref());
///
/// assert_eq!(maybe_array, None);
/// assert_eq!(iter.next(), None);
/// ```
///
/// Enough items case.
/// ```rust
/// use array_tools::try_init_from_iterator;
///
/// let mut iter = (0..5);
///
/// let maybe_array: Option<[u64; 5]> = try_init_from_iterator(iter.by_ref());
///
/// assert_eq!(maybe_array, Some([0, 1, 2, 3, 4]));
/// assert_eq!(iter.next(), None);
/// ```
///
/// Excessive items case.
/// ```rust
/// use array_tools::try_init_from_iterator;
///
/// let mut iter = (0..7);
///
/// let maybe_array: Option<[u32; 5]> = try_init_from_iterator(iter.by_ref());
///
/// assert_eq!(maybe_array, Some([0, 1, 2, 3, 4]));
/// assert_eq!(iter.next(), Some(5));
/// assert_eq!(iter.next(), Some(6));
/// assert_eq!(iter.next(), None);
/// ```
pub fn try_init_from_iterator<T, A, I>(mut iter: I) -> Option<A>
where
    A: FixedSizeArray<T>,
    I: Iterator<Item = T>,
{
    let mut deque = FixedCapacityDequeLike::new();
    loop {
        if deque.is_full() {
            break deque.try_extract_array();
        } else if let Some(item) = iter.next() {
            deque.push_back(item)
        } else {
            break None;
        }
    }
}

/// Attempts to create instance of array from slice.
///
/// - If slice length is less than array length, this function returns `None`.
/// - If slice length is greater or equal to array length, this function returns `Some(array)`.
///
/// This function *clones* slice elements.
///
/// # Examples
///
/// Not enough elements case.
/// ```rust
/// use array_tools::try_init_from_slice;
///
/// let source = [1, 2, 3, 4, 5, 6];
///
/// let maybe_array: Option<[u64; 8]> = try_init_from_slice(&source);
///
/// assert_eq!(maybe_array, None);
/// ```
///
/// Enough or excessive elements case.
/// ```rust
/// use array_tools::try_init_from_slice;
///
/// let source = [1, 2, 3, 4, 5, 6];
///
/// let maybe_array: Option<[u64; 4]> = try_init_from_slice(&source);
///
/// assert_eq!(maybe_array, Some([1, 2, 3, 4]));
/// ```
pub fn try_init_from_slice<T, A>(slice: &[T]) -> Option<A>
where
    A: FixedSizeArray<T>,
    T: Clone,
{
    if length_of::<T, A>() <= slice.len() {
        try_init_from_iterator(slice.iter().cloned())
    } else {
        None
    }
}

/// Creates a new array instance filled with values generated by a given function.
/// This variant expects function without arguments.
///
/// # Panics
///
/// - Only panics if provided function does.
///
/// # Examples
///
/// ```rust
/// use array_tools;
///
/// let mut value: u64 = 0;
/// let array: [u64; 7] = array_tools::init_with(|| {
///     let return_value = value;
///     value += 1;
///     return_value
/// });
///
/// assert_eq!(array, [0, 1, 2, 3, 4, 5, 6]);
/// ```
pub fn init_with<T, A, F>(mut initializer_fn: F) -> A
where
    A: FixedSizeArray<T>,
    F: FnMut() -> T,
{
    let mut deque = FixedCapacityDequeLike::new();
    while !deque.is_full() {
        deque.push_back(initializer_fn());
    }
    deque.try_extract_array().unwrap()
}

/// Creates a new array instance filled with values generated by a given function.
/// This variant expects a function with single argument - an index of element
/// to initialize.
///
/// # Panics
///
/// - Only panics if provided function does.
///
/// # Examples
///
/// ```rust
/// use array_tools;
///
/// let array: [u64; 7] = array_tools::indexed_init_with(|idx| {
///     idx as u64 * 2
/// });
///
/// assert_eq!(array, [0, 2, 4, 6, 8, 10, 12]);
/// ```
pub fn indexed_init_with<T, A, F>(mut initializer_fn: F) -> A
where
    A: FixedSizeArray<T>,
    F: FnMut(usize) -> T,
{
    let mut deque = FixedCapacityDequeLike::new();
    let mut idx = 0;
    while !deque.is_full() {
        deque.push_back(initializer_fn(idx));
        idx += 1
    }
    deque.try_extract_array().unwrap()
}

/// Returns the length of array.
///
/// Though it is not much useful when concrete array type is given and it's
/// size is known, it may come in handy when dealing with arrays hidden behind
/// `FixedSizeArray` in generic contexts.
///
/// # Examples
///
/// ```rust
/// use array_tools::length_of;
///
/// let length = length_of::<u64, [u64; 7]>();
/// assert_eq!(length, 7);
/// ```
///
/// # Note
///
/// Though currently it seems impossible to covert this one into `const fn`,
/// this sure will be done when this will become possible.
pub fn length_of<T, A>() -> usize
where
    A: FixedSizeArray<T>,
{
    let array: MaybeUninit<A> = MaybeUninit::uninit();
    unsafe { (*array.as_ptr()).as_slice().len() }
}

/// Splits array into two subarrays.
///
/// Because this function deals with arrays of constant size, it does not
/// expect any arguments, instead it expects two generic parameters - a two
/// array types that represent output subarrays. These must have the
/// same element type that input array does, sum of their lengths must
/// match exactly that of input array.
///
/// Does not perform any cloning operations, only moves values.
///
/// # Panics
///
/// Panics if sum of output subarrays' length does not match exactly
/// the length of input array.
///
/// # Note
///
/// Though currently it panics if output arrays have incompatible lengths,
/// this behavior will be changed to perform this check at compile time,
/// when this will become possible.
///
/// # Examples
///
/// ```rust
/// use array_tools::split;
///
/// let array = [1u64, 2, 3, 4, 5, 6, 7, 8];
/// let (left, right): ([u64; 2], [u64; 6]) = array_tools::split(array);
///
/// assert_eq!(left, [1u64, 2]);
/// assert_eq!(right, [3u64, 4, 5, 6, 7, 8]);
/// ```
pub fn split<T, A, L, R>(array: A) -> (L, R)
where
    A: FixedSizeArray<T>,
    L: FixedSizeArray<T>,
    R: FixedSizeArray<T>,
{
    assert_eq!(
        array.as_slice().len(),
        length_of::<T, L>() + length_of::<T, R>(),
        "Sum of outputs' lengths is not equal to length of input."
    );

    let mut iter = ArrayIntoIterator::new(array);
    let left: L = try_init_from_iterator(iter.by_ref()).unwrap();
    let right: R = try_init_from_iterator(iter.by_ref()).unwrap();
    (left, right)
}

/// Joins two arrays.
///
/// Creates a new array instance of length equal to sum of input arrays' lengths
/// and containing elements of `right` array appended to elements of the `left`.
///
/// Element types of first, second and output arrays must match. Sum of input
/// arrays' lengths must match exactly length of output array.
///
/// Does not perform any cloning operations, only moves values.
///
/// # Panics
///
/// - Panics if output array length is not equal to sum of input arrays' lengths.
///
/// # Note
///
/// Though currently it panics if output array has incompatible length,
/// this behavior will be changed to perform this check at compile time,
/// when this will become possible.
///
/// # Examples
///
/// ```rust
/// use array_tools;
///
/// let left = [1u64, 2];
/// let right = [3u64, 4, 5, 6, 7, 8];
/// let joined: [u64; 8] = array_tools::join(left, right);
/// assert_eq!(joined, [1u64, 2, 3, 4, 5, 6, 7, 8]);
/// ```
pub fn join<T, A, L, R>(left: L, right: R) -> A
where
    A: FixedSizeArray<T>,
    L: FixedSizeArray<T>,
    R: FixedSizeArray<T>,
{
    assert_eq!(
        length_of::<T, A>(),
        left.as_slice().len() + right.as_slice().len(),
        "Sum of inputs' lengths is not equal to output length."
    );
    let left_iter = ArrayIntoIterator::new(left);
    let right_iter = ArrayIntoIterator::new(right);
    try_init_from_iterator(left_iter.chain(right_iter)).unwrap()
}

/// A consuming iterator over array elements.
///
/// Consumes array upon creation and yields it's elements one-by-one.
///
/// Does not perform any cloning operations, only moves values.
///
/// # Examples
///
/// ```rust
/// use array_tools;
///
/// // Notice - this struct is neither Copy nor Clone.
/// #[derive(Debug, PartialEq, Eq)]
/// struct Foo(u32);
///
/// let array = [Foo(1), Foo(2), Foo(3)];
/// let mut iter = array_tools::ArrayIntoIterator::new(array);
/// assert_eq!(iter.next(), Some(Foo(1)));
/// assert_eq!(iter.next(), Some(Foo(2)));
/// assert_eq!(iter.next(), Some(Foo(3)));
/// assert_eq!(iter.next(), None);
/// ```
pub struct ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
{
    deque: FixedCapacityDequeLike<T, A>,
}

impl<T, A> PartialEq for ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.deque.as_slice() == other.deque.as_slice()
    }
}

impl<T, A> Eq for ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
    T: Eq,
{
}

impl<T, A> Clone for ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
    T: Clone,
{
    fn clone(&self) -> Self {
        ArrayIntoIterator {
            deque: self.deque.clone(),
        }
    }
}

impl<T, A> Debug for ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayIntoIterator")
            .field("deque", &self.deque)
            .finish()
    }
}

impl<T, A> ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
{
    pub fn new(array: A) -> ArrayIntoIterator<T, A> {
        ArrayIntoIterator {
            deque: FixedCapacityDequeLike::from_array(array),
        }
    }
}

impl<T, A> Iterator for ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        self.deque.pop_front()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.deque.length(), Some(self.deque.length()))
    }

    fn count(self) -> usize {
        self.deque.length()
    }

    fn last(mut self) -> Option<T> {
        self.deque.pop_back()
    }

    fn nth(&mut self, mut nth: usize) -> Option<T> {
        while nth > 0 {
            mem::drop(self.deque.pop_front());
            nth -= 1;
        }
        self.deque.pop_front()
    }
}

impl<T, A> DoubleEndedIterator for ArrayIntoIterator<T, A>
where
    A: FixedSizeArray<T>,
{
    fn next_back(&mut self) -> Option<T> {
        self.deque.pop_back()
    }
}

/// An item of [`ArrayChunks`](crate::ArrayChunks) iterator.
///
/// See [`ArrayChunks`](crate::ArrayChunks) documentation.
///
/// Each variant contains `PhantomData`, but it should be ignored completely.
pub enum ArrayChunk<T, C, S>
where
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
{
    /// A normal chunk.
    Chunk(C, PhantomData<T>),
    /// A reaminder that has insufficient length to be a chunk.
    Stump(S, PhantomData<T>),
}

impl<T, C, S> Debug for ArrayChunk<T, C, S>
where
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ArrayChunk::Chunk(chunk, pd) => f
                .debug_tuple("Chunk")
                .field(&chunk.as_slice())
                .field(&pd)
                .finish(),
            ArrayChunk::Stump(stump, pd) => f
                .debug_tuple("Stump")
                .field(&stump.as_slice())
                .field(&pd)
                .finish(),
        }
    }
}

impl<T, C, S> PartialEq for ArrayChunk<T, C, S>
where
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ArrayChunk::Chunk(chunk, _), ArrayChunk::Chunk(other_chunk, _))
                if chunk.as_slice() == other_chunk.as_slice() =>
            {
                true
            }
            (ArrayChunk::Stump(stump, _), ArrayChunk::Stump(other_stump, _))
                if stump.as_slice() == other_stump.as_slice() =>
            {
                true
            }
            _ => false,
        }
    }
}

impl<T, C, S> Eq for ArrayChunk<T, C, S>
where
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: Eq,
{
}

impl<T, C, S> Clone for ArrayChunk<T, C, S>
where
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: Clone,
{
    fn clone(&self) -> Self {
        match self {
            ArrayChunk::Chunk(chunk, _) => {
                let chunk_clone: C =
                    try_init_from_iterator(chunk.as_slice().iter().cloned()).unwrap();
                ArrayChunk::Chunk(chunk_clone, PhantomData)
            }
            ArrayChunk::Stump(stump, _) => {
                let stump_clone: S =
                    try_init_from_iterator(stump.as_slice().iter().cloned()).unwrap();
                ArrayChunk::Stump(stump_clone, PhantomData)
            }
        }
    }
}

/// A consuming iterator over non-overlaping subarrays of equal size.
///
/// Consumes array upon creation and yields subarrays "chunks" of requested size.
///
/// If array can't be divided evenly into chunks, the last yielded item will be
/// a "stump" - an array of length `input_array_length % chunk_size_length`,
/// containing what remains at the end.
///
/// If array can be divided evenly into "chunks", there will be no "stump".
///
/// Because this iterator deals with arrays of constant size, it does not
/// expect chunk size argument, instead it expects four generic parameters:
/// - Element type.
/// - Consumed array type.
/// - Chunk array type.
/// - Stump array type.
///
/// Element type of consumed, chunk and stump array types must match.
/// Consumed array length could be anything, including zero.
/// Chunk array length must be non-zero.
/// In case input array can't be divided evenly into chunks, stump array length
/// must be `consumed_array_length % chunk_array_length`.
/// In case input array can be divided evenly, stump array length must be 0.
///
/// Does not perform any cloning operations, only moves values.
///
/// # Examples
///
/// Case without "stump".
/// ```rust
/// use array_tools::{ArrayChunk, ArrayChunks};
/// use core::marker::PhantomData;
///
/// let array = [1u64, 2, 3, 4, 5, 6, 7, 8];
///
/// // Divide array `[u64; 8]` into `[u64; 2]` chunks. It can be divided evenly,
/// // so there will be no stump.
/// let mut chunks: ArrayChunks<u64, [u64; 8], [u64; 2], [u64; 0]> = ArrayChunks::new(array);
///
/// assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([1u64, 2], PhantomData)));
/// assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([3u64, 4], PhantomData)));
/// assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([5u64, 6], PhantomData)));
/// assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([7u64, 8], PhantomData)));
/// assert_eq!(chunks.next(), None);
/// ```
///
/// Case with "stump".
/// ```rust
/// use array_tools::{ArrayChunk, ArrayChunks};
/// use core::marker::PhantomData;
///
/// let array = [1u64, 2, 3, 4, 5, 6, 7, 8];
///
/// // Divide array `[u64; 8]` into `[u64; 3]` chunks. It can't be divided evenly, so last item
/// // will be stump of type `[u64; 2]`.
/// let mut chunks: ArrayChunks<u64, [u64; 8], [u64; 3], [u64; 2]> = ArrayChunks::new(array);
///
/// assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([1u64, 2, 3], PhantomData)));
/// assert_eq!(chunks.next(), Some(ArrayChunk::Chunk([4u64, 5, 6], PhantomData)));
/// assert_eq!(chunks.next(), Some(ArrayChunk::Stump([7u64, 8], PhantomData)));
/// assert_eq!(chunks.next(), None);
/// ```
///
/// Actually, most generic parameters may be ommited:
/// ```rust
/// use array_tools::ArrayChunks;
///
/// let array = [1, 2, 3, 4, 5, 6, 7, 8];
/// let _chunks: ArrayChunks<_, _, [_; 3], [_; 2]> = ArrayChunks::new(array);
/// ```
pub struct ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
{
    iter: ArrayIntoIterator<T, A>,
    has_stump: bool,
    _chunk_pd: PhantomData<C>,
    _stump_pd: PhantomData<S>,
}

impl<T, A, C, S> Debug for ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArrayChunks")
            .field("iter", &self.iter)
            .field("has_stump", &self.has_stump)
            .field("_chunk_pd", &self._chunk_pd)
            .field("_stump_pd", &self._stump_pd)
            .finish()
    }
}

impl<T, A, C, S> PartialEq for ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter == other.iter && self.has_stump == other.has_stump
    }
}

impl<T, A, C, S> Eq for ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: Eq,
{
}

impl<T, A, C, S> Clone for ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
    T: Clone,
{
    fn clone(&self) -> Self {
        ArrayChunks {
            iter: self.iter.clone(),
            has_stump: self.has_stump,
            _chunk_pd: PhantomData,
            _stump_pd: PhantomData,
        }
    }
}

impl<T, A, C, S> ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
{
    /// Creates a new instance of [`ArrayChunks`](crate::ArrayChunks) iterator.
    ///
    /// # Panics
    ///
    /// - If chunk size is 0.
    /// - If stump size is not equal to `consumed_array_length % chunk_array_length`.
    ///
    /// # Note
    ///
    /// Though currently this function panics if chunk size is 0 and/or stump size
    /// is not valid, this behavior will be changed to perform these checks
    /// at compile time, when this will become possible.
    pub fn new(array: A) -> ArrayChunks<T, A, C, S> {
        let chunk_length = length_of::<T, C>();
        assert_ne!(chunk_length, 0);
        let array_length = length_of::<T, A>();
        let stump_length = length_of::<T, S>();
        assert_eq!(
            array_length % chunk_length,
            stump_length,
            "Invalid stump length, expected {}.",
            stump_length
        );

        let iter = ArrayIntoIterator::new(array);

        let (elements_remain, _) = iter.size_hint();
        let has_stump = elements_remain % length_of::<T, C>() > 0;

        ArrayChunks {
            iter,
            has_stump,
            _chunk_pd: PhantomData,
            _stump_pd: PhantomData,
        }
    }

    fn items_remain(&self) -> usize {
        let (elements_remain, _) = self.iter.size_hint();
        if self.has_stump {
            elements_remain / length_of::<T, C>() + 1
        } else {
            elements_remain / length_of::<T, C>()
        }
    }

    fn has_chunks(&self) -> bool {
        let (elements_remain, _) = self.iter.size_hint();
        elements_remain / length_of::<T, C>() > 0
    }
}

impl<T, A, C, S> Iterator for ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
{
    type Item = ArrayChunk<T, C, S>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.has_chunks() {
            let chunk: C = try_init_from_iterator(self.iter.by_ref()).unwrap();
            Some(ArrayChunk::Chunk(chunk, PhantomData))
        } else if self.has_stump {
            let stump: S = try_init_from_iterator(self.iter.by_ref()).unwrap();
            self.has_stump = false;
            Some(ArrayChunk::Stump(stump, PhantomData))
        } else {
            None
        }
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.items_remain(), Some(self.items_remain()))
    }
    fn count(self) -> usize {
        self.items_remain()
    }

    fn last(mut self) -> Option<Self::Item> {
        self.next_back()
    }
    fn nth(&mut self, mut nth: usize) -> Option<Self::Item> {
        while nth > 0 {
            mem::drop(self.next());
            nth -= 1;
        }
        self.next()
    }
}

impl<T, A, C, S> DoubleEndedIterator for ArrayChunks<T, A, C, S>
where
    A: FixedSizeArray<T>,
    C: FixedSizeArray<T>,
    S: FixedSizeArray<T>,
{
    fn next_back(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.has_stump {
            let mut stump: S = try_init_from_iterator(self.iter.by_ref().rev()).unwrap();
            stump.as_mut_slice().reverse();
            self.has_stump = false;
            Some(ArrayChunk::Stump(stump, PhantomData))
        } else if self.has_chunks() {
            let mut chunk: C = try_init_from_iterator(self.iter.by_ref().rev()).unwrap();
            chunk.as_mut_slice().reverse();
            Some(ArrayChunk::Chunk(chunk, PhantomData))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn not_enough_items() {
        let maybe_array: Option<[u64; 5]> = super::try_init_from_iterator(1..=4);
        assert_eq!(maybe_array, None);
    }

    #[test]
    fn exact_item_count() {
        let maybe_array: Option<[u64; 5]> = super::try_init_from_iterator(1..=5);
        assert_eq!(maybe_array, Some([1, 2, 3, 4, 5]));
    }

    #[test]
    fn too_many_items() {
        let maybe_array: Option<[u32; 5]> = super::try_init_from_iterator(1..=100);
        assert_eq!(maybe_array, Some([1, 2, 3, 4, 5]));
    }

    #[test]
    fn array_into_iterator() {
        let mut iter = super::ArrayIntoIterator::new([1, 2, 3, 4, 5]);
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn array_into_iterator_reverse() {
        let mut iter = super::ArrayIntoIterator::new([1, 2, 3, 4, 5]).rev();
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn array_into_iterator_take_from_two_sides() {
        let mut iter = super::ArrayIntoIterator::new([1, 2, 3, 4, 5]);
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(5));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next_back(), Some(4));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn array_into_iterator_next() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let mut iter = super::ArrayIntoIterator::new(array);
        assert_eq!(iter.next(), Some(1));
    }

    #[test]
    fn array_into_iterator_size_hint() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let mut iter = super::ArrayIntoIterator::new(array);
        assert_eq!(iter.size_hint(), (7, Some(7)));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.size_hint(), (6, Some(6)));
    }

    #[test]
    fn array_into_iterator_count() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let iter = super::ArrayIntoIterator::new(array);
        assert_eq!(iter.count(), 7);
    }

    #[test]
    fn array_into_iterator_last() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let iter = super::ArrayIntoIterator::new(array);
        assert_eq!(iter.last(), Some(7));
    }

    #[test]
    fn array_into_iterator_nth() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let mut iter = super::ArrayIntoIterator::new(array);
        assert_eq!(iter.nth(5), Some(6));
    }

    #[test]
    fn init_with_fn() {
        fn initializer() -> u64 {
            7
        }

        let array: [u64; 7] = super::init_with(initializer);

        assert_eq!(array, [7, 7, 7, 7, 7, 7, 7]);
    }

    #[test]
    fn init_with_closure() {
        let mut value = 0;

        let array: [u64; 7] = super::init_with(|| {
            let return_value = value;
            value += 1;
            return return_value;
        });

        assert_eq!(array, [0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn indexed_init_with_fn() {
        fn initializer(idx: usize) -> u64 {
            (idx % 2) as u64
        }

        let array: [u64; 7] = super::indexed_init_with(initializer);

        assert_eq!(array, [0, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn indexed_init_with_closure() {
        let array: [u64; 7] = super::indexed_init_with(|idx| (idx % 2) as u64);

        assert_eq!(array, [0, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn length_of_empty() {
        let length = super::length_of::<Option<u64>, [Option<u64>; 0]>();
        assert_eq!(length, 0);
    }

    #[test]
    fn length_of_non_empty() {
        let length = super::length_of::<Option<u64>, [Option<u64>; 7]>();
        assert_eq!(length, 7);
    }

    #[test]
    fn split_okay() {
        let array: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let (left, right): ([u64; 2], [u64; 6]) = super::split(array);
        assert_eq!(left, [1, 2]);
        assert_eq!(right, [3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn split_okay_empty_left() {
        let array: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let (left, right): ([u64; 0], [u64; 8]) = super::split(array);
        assert_eq!(left, []);
        assert_eq!(right, [1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn split_okay_empty_right() {
        let array: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let (left, right): ([u64; 8], [u64; 0]) = super::split(array);
        assert_eq!(left, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(right, []);
    }
    #[test]
    #[should_panic]
    fn split_split_invalid_lengths_sum() {
        let array: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let (_left, _right): ([u64; 2], [u64; 4]) = super::split(array);
    }

    #[test]
    fn split_empty_into_empty_and_empty() {
        let array: [u64; 0] = [];
        let (_left, _right): ([u64; 0], [u64; 0]) = super::split(array);
    }

    #[test]
    fn join_okay() {
        let left: [u64; 2] = [1, 2];
        let right: [u64; 6] = [3, 4, 5, 6, 7, 8];
        let joined: [u64; 8] = super::join(left, right);
        assert_eq!(joined, [1u64, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn join_empty_and_empty() {
        let left: [u64; 0] = [];
        let right: [u64; 0] = [];
        let joined: [u64; 0] = super::join(left, right);
        assert_eq!(joined, []);
    }

    #[test]
    fn join_okay_empty_left() {
        let left: [u64; 0] = [];
        let right: [u64; 6] = [1, 2, 3, 4, 5, 6];
        let joined: [u64; 6] = super::join(left, right);
        assert_eq!(joined, [1u64, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn join_okay_empty_right() {
        let left: [u64; 6] = [1, 2, 3, 4, 5, 6];
        let right: [u64; 0] = [];
        let joined: [u64; 6] = super::join(left, right);
        assert_eq!(joined, [1u64, 2, 3, 4, 5, 6]);
    }

    #[test]
    #[should_panic]
    fn join_insufficient_capacity() {
        let left: [u64; 2] = [1, 2];
        let right: [u64; 5] = [3, 4, 5, 6, 7];
        let _joined: [u64; 6] = super::join(left, right);
    }

    #[test]
    #[should_panic]
    fn join_excessive_capacity() {
        let left: [u64; 2] = [1, 2];
        let right: [u64; 5] = [3, 4, 5, 6, 7];
        let _joined: [u64; 10] = super::join(left, right);
    }

    #[test]
    fn array_chunks_with_stump() {
        use super::{ArrayChunk, ArrayChunks};
        use core::marker::PhantomData;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut chunks: ArrayChunks<u64, [u64; 8], [u64; 3], [u64; 2]> = ArrayChunks::new(array);
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([1u64, 2, 3], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([4u64, 5, 6], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Stump([7u64, 8], PhantomData))
        );
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn array_chunks_without_stump() {
        use super::{ArrayChunk, ArrayChunks};
        use core::marker::PhantomData;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: ArrayChunks<u64, [u64; 9], [u64; 3], [u64; 0]> = ArrayChunks::new(array);
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([1u64, 2, 3], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([4u64, 5, 6], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([7u64, 8, 9], PhantomData))
        );
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn array_chunks_with_stump_rev() {
        use super::{ArrayChunk, ArrayChunks};
        use core::iter::Rev;
        use core::marker::PhantomData;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut chunks: Rev<ArrayChunks<u64, [u64; 8], [u64; 3], [u64; 2]>> =
            ArrayChunks::new(array).rev();
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Stump([7u64, 8], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([4u64, 5, 6], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([1u64, 2, 3], PhantomData))
        );
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn array_chunks_withot_stump_rev() {
        use super::{ArrayChunk, ArrayChunks};
        use core::iter::Rev;
        use core::marker::PhantomData;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: Rev<ArrayChunks<u64, [u64; 9], [u64; 3], [u64; 0]>> =
            ArrayChunks::new(array).rev();
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([7u64, 8, 9], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([4u64, 5, 6], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([1u64, 2, 3], PhantomData))
        );
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn array_chunks_with_stump_take_from_two_sides_front_first() {
        use super::{ArrayChunk, ArrayChunks};
        use core::marker::PhantomData;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: ArrayChunks<u64, [u64; 9], [u64; 2], [u64; 1]> = ArrayChunks::new(array);
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([1u64, 2], PhantomData))
        );
        assert_eq!(
            chunks.next_back(),
            Some(ArrayChunk::Stump([9u64], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([3u64, 4], PhantomData))
        );
        assert_eq!(
            chunks.next_back(),
            Some(ArrayChunk::Chunk([7u64, 8], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([5u64, 6], PhantomData))
        );
        assert_eq!(chunks.next_back(), None);
        assert_eq!(chunks.next(), None);
    }
    #[test]
    fn array_chunks_with_stump_take_from_two_sides_back_first() {
        use super::{ArrayChunk, ArrayChunks};
        use core::marker::PhantomData;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: ArrayChunks<u64, [u64; 9], [u64; 2], [u64; 1]> = ArrayChunks::new(array);
        assert_eq!(
            chunks.next_back(),
            Some(ArrayChunk::Stump([9u64], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([1u64, 2], PhantomData))
        );
        assert_eq!(
            chunks.next_back(),
            Some(ArrayChunk::Chunk([7u64, 8], PhantomData))
        );
        assert_eq!(
            chunks.next(),
            Some(ArrayChunk::Chunk([3u64, 4], PhantomData))
        );
        assert_eq!(
            chunks.next_back(),
            Some(ArrayChunk::Chunk([5u64, 6], PhantomData))
        );
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.next_back(), None);
    }

    #[test]
    fn array_chunks_with_stump_size_hint() {
        use super::ArrayChunks;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let chunks: ArrayChunks<u64, [u64; 8], [u64; 3], [u64; 2]> = ArrayChunks::new(array);
        assert_eq!(chunks.size_hint(), (3, Some(3)));
    }

    #[test]
    fn array_chunks_without_stump_size_hint() {
        use super::ArrayChunks;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let chunks: ArrayChunks<u64, [u64; 9], [u64; 3], [u64; 0]> = ArrayChunks::new(array);
        assert_eq!(chunks.size_hint(), (3, Some(3)));
    }

    #[test]
    fn array_chunks_with_stump_count() {
        use super::ArrayChunks;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let chunks: ArrayChunks<u64, [u64; 8], [u64; 3], [u64; 2]> = ArrayChunks::new(array);
        assert_eq!(chunks.count(), 3);
    }

    #[test]
    fn array_chunks_without_stump_count() {
        use super::ArrayChunks;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let chunks: ArrayChunks<u64, [u64; 9], [u64; 3], [u64; 0]> = ArrayChunks::new(array);
        assert_eq!(chunks.count(), 3);
    }

    #[test]
    fn array_chunks_with_stump_last() {
        use super::{ArrayChunk, ArrayChunks};
        use core::marker::PhantomData;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let chunks: ArrayChunks<u64, [u64; 8], [u64; 3], [u64; 2]> = ArrayChunks::new(array);
        assert_eq!(
            chunks.last(),
            Some(ArrayChunk::Stump([7u64, 8], PhantomData))
        );
    }

    #[test]
    fn array_chunks_without_stump_last() {
        use super::{ArrayChunk, ArrayChunks};
        use core::marker::PhantomData;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let chunks: ArrayChunks<u64, [u64; 9], [u64; 3], [u64; 0]> = ArrayChunks::new(array);
        assert_eq!(
            chunks.last(),
            Some(ArrayChunk::Chunk([7u64, 8, 9], PhantomData))
        );
    }

    #[test]
    fn array_chunks_nth() {
        use super::{ArrayChunk, ArrayChunks};
        use core::marker::PhantomData;
        let array: [u64; 17] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
        let mut chunks: ArrayChunks<u64, [u64; 17], [u64; 2], [u64; 1]> = ArrayChunks::new(array);
        assert_eq!(
            chunks.nth(5),
            Some(ArrayChunk::Chunk([11u64, 12], PhantomData))
        );
    }

    #[test]
    fn fixed_capacity_deque_like_eq() {
        use super::FixedCapacityDequeLike;
        let mut deque1: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque1.push_back(1);
        deque1.push_back(2);
        deque1.push_back(3);
        deque1.push_back(4);
        deque1.push_back(5);
        deque1.pop_front();
        deque1.pop_front();

        let mut deque2: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque2.push_back(1);
        deque2.push_back(2);
        deque2.push_back(3);
        deque2.push_back(4);
        deque2.push_back(5);
        deque2.pop_front();
        deque2.pop_front();

        assert_eq!(deque1, deque2);
    }

    #[test]
    fn fixed_capacity_deque_like_eq_empty() {
        use super::FixedCapacityDequeLike;

        let deque1: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        let deque2: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();

        assert_eq!(deque1, deque2);
    }

    #[test]
    fn fixed_capacity_deque_like_not_eq_different_items() {
        use super::FixedCapacityDequeLike;

        let mut deque1: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque1.push_back(1);
        deque1.push_back(2);

        let mut deque2: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque2.push_back(2);
        deque1.push_back(3);

        assert_ne!(deque1, deque2);
    }

    #[test]
    fn fixed_capacity_deque_like_not_eq_different_offsets() {
        use super::FixedCapacityDequeLike;

        let mut deque1: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque1.push_back(0);
        deque1.push_back(1);
        deque1.push_back(2);
        deque1.pop_back();

        let mut deque2: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque2.push_back(1);
        deque1.push_back(2);

        assert_ne!(deque1, deque2);
    }

    #[test]
    fn fixed_capacity_deque_like_clone_no_offset() {
        use super::FixedCapacityDequeLike;

        let mut deque: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        let clone = deque.clone();

        assert_eq!(deque, clone);
    }

    #[test]
    fn fixed_capacity_deque_like_clone_with_offset() {
        use super::FixedCapacityDequeLike;

        let mut deque: FixedCapacityDequeLike<u64, [u64; 10]> = FixedCapacityDequeLike::new();
        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);
        deque.push_back(4);
        deque.push_back(5);
        deque.pop_back();
        deque.pop_front();

        let clone = deque.clone();

        assert_eq!(deque, clone);
    }

    #[test]
    fn array_into_iterator_eq() {
        use super::ArrayIntoIterator;

        let a = [1, 2, 3, 4, 5];
        let b = [1, 2, 3, 4, 5];

        let iter_a = ArrayIntoIterator::new(a);
        let iter_b = ArrayIntoIterator::new(b);

        assert_eq!(iter_a, iter_b);
    }

    #[test]
    fn array_into_iterator_eq_if_items_eq() {
        use super::ArrayIntoIterator;
        use core::mem;

        let a = [1, 2, 3, 4, 5, 6];
        let b = [0, 0, 0, 3, 4, 5];

        let mut iter_a = ArrayIntoIterator::new(a);
        let mut iter_b = ArrayIntoIterator::new(b);

        mem::drop(iter_a.next());
        mem::drop(iter_a.next());
        mem::drop(iter_a.next_back());

        mem::drop(iter_b.next());
        mem::drop(iter_b.next());
        mem::drop(iter_b.next());

        assert_eq!(iter_a, iter_b);
    }

    #[test]
    fn array_into_iterator_clone() {
        use super::ArrayIntoIterator;
        let a = [1, 2, 3, 4, 5];
        let a_iter = ArrayIntoIterator::new(a);
        let a_iter_clone = a_iter.clone();
        assert_eq!(a_iter, a_iter_clone);
    }

    #[test]
    fn array_chunk_eq_chunk() {
        use super::ArrayChunk;
        use core::marker::PhantomData;

        let chunk1: ArrayChunk<u64, [u64; 4], [u64; 3]> =
            ArrayChunk::Chunk([1, 2, 3, 4], PhantomData);
        let chunk2: ArrayChunk<u64, [u64; 4], [u64; 3]> =
            ArrayChunk::Chunk([1, 2, 3, 4], PhantomData);

        assert_eq!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_eq_stump() {
        use super::ArrayChunk;
        use core::marker::PhantomData;

        let chunk1: ArrayChunk<u64, [u64; 4], [u64; 3]> = ArrayChunk::Stump([1, 2, 3], PhantomData);
        let chunk2: ArrayChunk<u64, [u64; 4], [u64; 3]> = ArrayChunk::Stump([1, 2, 3], PhantomData);

        assert_eq!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_ne_stump_and_chunk() {
        use super::ArrayChunk;
        use core::marker::PhantomData;

        let chunk1: ArrayChunk<u64, [u64; 4], [u64; 3]> =
            ArrayChunk::Chunk([1, 2, 3, 4], PhantomData);
        let chunk2: ArrayChunk<u64, [u64; 4], [u64; 3]> = ArrayChunk::Stump([1, 2, 3], PhantomData);

        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_ne_different_chunk_items() {
        use super::ArrayChunk;
        use core::marker::PhantomData;

        let chunk1: ArrayChunk<u64, [u64; 4], [u64; 3]> =
            ArrayChunk::Chunk([1, 2, 3, 4], PhantomData);
        let chunk2: ArrayChunk<u64, [u64; 4], [u64; 3]> =
            ArrayChunk::Chunk([4, 3, 2, 1], PhantomData);

        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_ne_different_stump_items() {
        use super::ArrayChunk;
        use core::marker::PhantomData;

        let chunk1: ArrayChunk<u64, [u64; 4], [u64; 3]> = ArrayChunk::Stump([1, 2, 3], PhantomData);
        let chunk2: ArrayChunk<u64, [u64; 4], [u64; 3]> = ArrayChunk::Stump([4, 3, 2], PhantomData);

        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_clone_chunk() {
        use super::ArrayChunk;
        use core::marker::PhantomData;

        let chunk: ArrayChunk<u64, [u64; 4], [u64; 3]> =
            ArrayChunk::Chunk([1, 2, 3, 4], PhantomData);
        let chunk_clone = chunk.clone();

        assert_eq!(chunk, chunk_clone);
    }

    #[test]
    fn array_chunk_clone_stump() {
        use super::ArrayChunk;
        use core::marker::PhantomData;

        let chunk: ArrayChunk<u64, [u64; 4], [u64; 3]> = ArrayChunk::Stump([1, 2, 3], PhantomData);
        let chunk_clone = chunk.clone();

        assert_eq!(chunk, chunk_clone);
    }

    #[test]
    fn array_chunks_eq_case_1() {
        use super::ArrayChunks;
        use core::mem;

        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut a_chunks = ArrayChunks::<u64, [u64; 8], [u64; 3], [u64; 2]>::new(a);

        let b = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut b_chunks = ArrayChunks::<u64, [u64; 8], [u64; 3], [u64; 2]>::new(b);

        assert_eq!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_eq!(a_chunks, b_chunks);

        mem::drop(a_chunks.next_back());
        mem::drop(b_chunks.next_back());

        assert_eq!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_eq!(a_chunks, b_chunks);
    }

    #[test]
    fn array_chunks_eq_case_2() {
        use super::ArrayChunks;
        use core::mem;

        let a = [0, 1, 2, 4, 5, 6, 8, 9];
        let mut a_chunks = ArrayChunks::<u64, [u64; 8], [u64; 3], [u64; 2]>::new(a);

        let b = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut b_chunks = ArrayChunks::<u64, [u64; 8], [u64; 3], [u64; 2]>::new(b);

        assert_ne!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_ne!(a_chunks, b_chunks);

        mem::drop(a_chunks.next_back());
        mem::drop(b_chunks.next_back());

        assert_eq!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_eq!(a_chunks, b_chunks);
    }

    #[test]
    fn array_chunks_eq_case_3() {
        use super::ArrayChunks;
        use core::mem;

        let a = [0, 1, 2, 4, 5, 6, 8, 9];
        let mut a_chunks = ArrayChunks::<u64, [u64; 8], [u64; 3], [u64; 2]>::new(a);

        let b = [10, 11, 12, 13, 14, 15, 16, 17];
        let mut b_chunks = ArrayChunks::<u64, [u64; 8], [u64; 3], [u64; 2]>::new(b);

        assert_ne!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_ne!(a_chunks, b_chunks);

        mem::drop(a_chunks.next_back());
        mem::drop(b_chunks.next_back());

        assert_ne!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_eq!(a_chunks, b_chunks);
    }

    #[test]
    fn array_chunks_clone() {
        use super::ArrayChunks;
        use core::mem;

        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut a_chunks = ArrayChunks::<u64, [u64; 8], [u64; 3], [u64; 2]>::new(a);
        let mut b_chunks = a_chunks.clone();

        assert_eq!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_eq!(a_chunks, b_chunks);

        mem::drop(a_chunks.next_back());
        mem::drop(b_chunks.next_back());

        assert_eq!(a_chunks, b_chunks);

        mem::drop(a_chunks.next());
        mem::drop(b_chunks.next());

        assert_eq!(a_chunks, b_chunks);
    }

    #[test]
    fn array_chunks_ommit_generic_parameters() {
        use super::ArrayChunks;

        let array = [1, 2, 3, 4, 5, 6, 7, 8];
        let _chunks: ArrayChunks<_, _, [_; 3], [_; 2]> = ArrayChunks::new(array);
    }
}
