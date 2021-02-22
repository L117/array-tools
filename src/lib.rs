//! # Array Tools
//!
//! [![Crate](https://img.shields.io/crates/v/array-tools.svg)](https://crates.io/crates/array-tools)
//! [![Documentation](https://docs.rs/array-tools/badge.svg)](https://docs.rs/array-tools)
//! [![Build Status](https://travis-ci.com/L117/array-tools.svg?branch=master)](https://travis-ci.com/L117/array-tools)
//! [![Build status](https://ci.appveyor.com/api/projects/status/9f4ctfoat9i9h86w?svg=true)](https://ci.appveyor.com/project/L117/array-tools)
//!
//! A collection of tools that make it easier to work with arrays.
//!
//! All currently supported features are `no_std` compatible.
//!
//! ```rust
//! use array_tools as at;
//!
//! let left: [usize; 4] = at::init_with_mapped_idx(|idx| idx + 1);
//! let right: [usize; 3] = at::init_with_iter(5..8).unwrap();
//!
//! let joined: [usize; 7] = at::join(left, right);
//!
//! let (left, right): ([usize; 2], [usize; 5]) = at::split(joined);
//!
//! assert_eq!(left, [1, 2]);
//! assert_eq!(right, [3, 4, 5, 6, 7]);
//! ```
//!
//! ### License
//!
//! Licensed under the terms of either Apache 2.0 or MIT license.

#![no_std]
use core::fmt::{self, Debug};
use core::mem::MaybeUninit;
use core::{mem, ptr};

struct DequeLike<T, const N: usize> {
    array: [MaybeUninit<T>; N],
    begining: usize,
    end: usize,
}

impl<T, const N: usize> DequeLike<T, N> {
    fn new() -> DequeLike<T, N> {
        DequeLike {
            array: unsafe { MaybeUninit::uninit().assume_init() },
            begining: 0,
            end: 0,
        }
    }

    fn from_array(array: [T; N]) -> DequeLike<T, N> {
        unsafe {
            let transmuted_array =
                ptr::read(mem::transmute::<&[T; N], &[MaybeUninit<T>; N]>(&array));
            mem::forget(array);

            DequeLike {
                array: transmuted_array,
                begining: 0,
                end: N,
            }
        }
    }

    fn capacity(&self) -> usize {
        N
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

    fn push_back(&mut self, item: T) {
        if self.end < self.capacity() {
            let item_index = self.end;
            self.array[item_index] = MaybeUninit::new(item);
            self.end += 1;
        } else {
            panic!("No capacity left at the end.");
        }
    }

    #[allow(dead_code)]
    fn push_front(&mut self, item: T) {
        if self.begining != 0 {
            let item_index = self.begining - 1;
            self.array[item_index] = MaybeUninit::new(item);
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
                mem::replace(&mut self.array[item_index], MaybeUninit::uninit()).assume_init()
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
                mem::replace(&mut self.array[item_index], MaybeUninit::uninit()).assume_init()
            };
            self.begining += 1;
            Some(item)
        }
    }

    fn try_extract_array(&mut self) -> Option<[T; N]> {
        if self.length() == self.capacity() {
            self.begining = 0;
            self.end = 0;
            unsafe {
                let uninit_array =
                    mem::replace(&mut self.array, MaybeUninit::uninit().assume_init());
                let array = ptr::read(mem::transmute::<&[MaybeUninit<T>; N], &[T; N]>(
                    &uninit_array,
                ));
                mem::forget(uninit_array);
                Some(array)
            }
        } else {
            None
        }
    }

    fn as_slice(&self) -> &[T] {
        unsafe {
            &mem::transmute::<&[MaybeUninit<T>; N], &[T; N]>(&self.array)[self.begining..self.end]
        }
    }
}

impl<T, const N: usize> Drop for DequeLike<T, N> {
    fn drop(&mut self) {
        while let Some(item) = self.pop_back() {
            mem::drop(item);
        }
    }
}

impl<T, const N: usize> Debug for DequeLike<T, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DequeLike")
            .field("array", &self.as_slice())
            .field("begining", &self.begining)
            .field("end", &self.end)
            .finish()
    }
}

impl<T, const N: usize> PartialEq for DequeLike<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.capacity() == other.capacity()
            && self.begining == other.begining
            && self.end == other.end
            && self.as_slice() == other.as_slice()
    }
}

impl<T, const N: usize> Eq for DequeLike<T, N> where T: Eq {}

impl<T, const N: usize> Clone for DequeLike<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        let mut clone = DequeLike {
            array: unsafe { MaybeUninit::uninit().assume_init() },
            begining: self.begining,
            end: self.begining,
        };

        while clone.end != self.end {
            let end = clone.end;
            unsafe {
                let src: &T = &(*self.array[end].as_ptr());
                clone.array[end] = MaybeUninit::new(src.clone());
            }
            clone.end += 1;
        }

        clone
    }
}

/// Initializes array `[T; N]` with items obtained form iterator.
///
/// Collects first `N` items from iterator and builds the array. Item order is preserved.
///
/// If iterator yields `None` before `N` items are collected, all previously collected items are dropped and `None` is returned.
///
/// # Examples
///
/// ```rust
/// use array_tools as art;
///
/// let mut iter = [3i32, 2, 1, 0].iter().copied();
/// let result: Option<[i32; 5]> = art::init_with_iter(iter.by_ref());
///
/// assert_eq!(result, None);
/// assert_eq!(iter.next(), None);
///
/// let mut iter = [4i32, 3, 2, 1, 0].iter().copied();
/// let result: Option<[i32; 5]> = art::init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Some([4i32, 3, 2, 1, 0]));
/// assert_eq!(iter.next(), None);
///
/// let mut iter = [5i32, 4, 3, 2, 1, 0].iter().copied();
/// let result: Option<[i32; 5]> = art::init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Some([5i32, 4, 3, 2, 1]));
/// assert_eq!(iter.next(), Some(0));
/// assert_eq!(iter.next(), None);
/// ```
pub fn init_with_iter<I, T, const N: usize>(mut iter: I) -> Option<[T; N]>
where
    I: Iterator<Item = T>,
{
    let mut deque = DequeLike::new();
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

/// Initializes array `[T; N]` with items form iterator over `Result`s.
///
/// Collects first `N` items from iterator and builds the array. Item order is preserved.
///
/// If iterator yields `None` before `N` items are collected, all previously collected items are dropped and `Ok(None)` is returned.
///
/// If iterator yields `Some(Errr(_))`, all previously collected items are dropped and `Err(_)` is returned.
///
/// # Examples
///
/// ```rust
/// use array_tools as art;
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// struct MyErr;
///
/// let mut iter = [Ok(3i32), Ok(2), Ok(1), Ok(0)].iter().copied();
/// let result: Result<Option<[i32; 5]>, MyErr> = art::try_init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Ok(None));
/// assert_eq!(iter.next(), None);
///
/// let mut iter = [Ok(4i32), Ok(3), Ok(2), Ok(1), Ok(0)].iter().copied();
/// let result: Result<Option<[i32; 5]>, MyErr> = art::try_init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Ok(Some([4i32, 3, 2, 1, 0])));
/// assert_eq!(iter.next(), None);
///
/// let mut iter = [Ok(5i32), Ok(4), Ok(3), Ok(2), Ok(1), Ok(0)].iter().copied();
/// let result: Result<Option<[i32; 5]>, MyErr> = art::try_init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Ok(Some([5i32, 4, 3, 2, 1])));
/// assert_eq!(iter.next(), Some(Ok(0)));
/// assert_eq!(iter.next(), None);
///
///
/// let mut iter = [Ok(3i32), Ok(2), Ok(1), Err(MyErr)].iter().copied();
/// let result: Result<Option<[i32; 5]>, MyErr> = art::try_init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Err(MyErr));
/// assert_eq!(iter.next(), None);
///
/// let mut iter = [Ok(4i32), Ok(3), Ok(2), Ok(1), Err(MyErr)].iter().copied();
/// let result: Result<Option<[i32; 5]>, MyErr> = art::try_init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Err(MyErr));
/// assert_eq!(iter.next(), None);
///
/// let mut iter = [Ok(5i32), Ok(4), Ok(3), Ok(2), Ok(1), Err(MyErr)].iter().copied();
/// let result: Result<Option<[i32; 5]>, MyErr> = art::try_init_with_iter(iter.by_ref());
///
/// assert_eq!(result, Ok(Some([5i32, 4, 3, 2, 1])));
/// assert_eq!(iter.next(), Some(Err(MyErr)));
/// assert_eq!(iter.next(), None);
/// ```
pub fn try_init_with_iter<I, T, E, const N: usize>(mut iter: I) -> Result<Option<[T; N]>, E>
where
    I: Iterator<Item = Result<T, E>>,
{
    let mut deque = DequeLike::new();
    loop {
        if deque.is_full() {
            break Ok(deque.try_extract_array());
        } else if let Some(item) = iter.next() {
            deque.push_back(item?)
        } else {
            break Ok(None);
        }
    }
}

/// Initializes array `[T; N]` with values cloned form slice.
///
/// Clones first `N` items from slice and builds the array.
///
/// # Examples
///
/// ```rust
/// use array_tools as art;
///
/// let slice = &[1i32, 2, 3, 4, 5, 6];
/// let result: Option<[i32; 8]> = art::init_with_slice(slice);
///
/// assert_eq!(result, None);
///
///
/// let slice = &[1i32, 2, 3, 4, 5, 6];
/// let result: Option<[i32; 4]> = art::init_with_slice(slice);
///
/// assert_eq!(result, Some([1i32, 2, 3, 4]));
/// ```
pub fn init_with_slice<T, const N: usize>(slice: &[T]) -> Option<[T; N]>
where
    T: Clone,
{
    if N <= slice.len() {
        init_with_iter(slice.iter().cloned())
    } else {
        None
    }
}

/// Initializes array `[T; N]` with values obtained by repeatedly calling a function.
///
/// Repeatedly calls provided function `N` times and collects items to build the array. Item order is preserved.
///
/// # Examples
///
/// ```rust
/// use array_tools as art;
///
/// let mut value: u64 = 0;
/// let array: [u64; 7] = art::init_with(|| {
///     let return_value = value;
///     value += 1;
///     return_value
/// });
///
/// assert_eq!(array, [0, 1, 2, 3, 4, 5, 6]);
/// ```
pub fn init_with<F, T, const N: usize>(mut initializer_fn: F) -> [T; N]
where
    F: FnMut() -> T,
{
    let mut deque = DequeLike::new();
    while !deque.is_full() {
        deque.push_back(initializer_fn());
    }
    deque.try_extract_array().unwrap()
}

/// Initializes array `[T; N]` with values obtained by repeatedly calling a function.
///
/// Repeatedly calls provided function `N` times and collects items to build the array. Item order is preserved.
///
/// If provided function returns `Err(_)`, all previously collected items are dropped and `Err(_)` is returned.
///
/// # Examples
///
/// ```rust
/// use array_tools as art;
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// struct MyErr;
///
/// let mut value: u64 = 0;
/// let result: Result<[u64; 7], MyErr> = art::try_init_with(|| {
///     let return_value = value;
///     value += 1;
///     Ok(return_value)
/// });
///
/// assert_eq!(result, Ok([0, 1, 2, 3, 4, 5, 6]));
///
///
/// let mut value: u64 = 0;
/// let result: Result<[u64; 7], MyErr> = art::try_init_with(|| {
///     let return_value = value;
///     value += 1;
///     if value != 3 {
///         Ok(return_value)
///     } else {
///         Err(MyErr)
///     }
/// });
///
/// assert_eq!(result, Err(MyErr));
/// ```
pub fn try_init_with<F, E, T, const N: usize>(mut initializer_fn: F) -> Result<[T; N], E>
where
    F: FnMut() -> Result<T, E>,
{
    let mut deque = DequeLike::new();
    while !deque.is_full() {
        deque.push_back(initializer_fn()?);
    }
    Ok(deque.try_extract_array().unwrap())
}

/// Initializes array `[T; N]` with values obtained by repeatedly calling a function with index.
///
/// Repeatedly calls provided function `N` times with item index as argument and collects items to build the array.
///
/// # Examples
///
/// ```rust
/// use array_tools as art;
///
/// let array: [u64; 7] = art::init_with_mapped_idx(|idx| {
///     idx as u64 * 2
/// });
///
/// assert_eq!(array, [0, 2, 4, 6, 8, 10, 12]);
/// ```
pub fn init_with_mapped_idx<F, T, const N: usize>(mut initializer_fn: F) -> [T; N]
where
    F: FnMut(usize) -> T,
{
    let mut deque = DequeLike::new();
    let mut idx = 0;
    while !deque.is_full() {
        deque.push_back(initializer_fn(idx));
        idx += 1
    }
    deque.try_extract_array().unwrap()
}

/// Initializes array `[T; N]` with values obtained by repeatedly calling a function with index.
///
/// Repeatedly calls provided function `N` times with item index as argument and collects items to build the array.
///
/// If provided function returns `Err(_)`, all previously collected items are dropped and `Err(_)` is returned.
///
/// # Examples
///
/// ```rust
/// use array_tools as art;
///
/// #[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// struct MyErr;
///
/// let result: Result<[u64; 7], MyErr> = art::try_init_with_mapped_idx(|idx| {
///     Ok(idx as u64 * 2)
/// });
///
/// assert_eq!(result, Ok([0, 2, 4, 6, 8, 10, 12]));
///
///
/// let result: Result<[u64; 7], MyErr> = art::try_init_with_mapped_idx(|idx| {
///     if idx != 3 {
///         Ok(idx as u64 * 2)
///     } else {
///         Err(MyErr)
///     }
/// });
///
/// assert_eq!(result, Err(MyErr));
/// ```
pub fn try_init_with_mapped_idx<F, E, T, const N: usize>(mut initializer_fn: F) -> Result<[T; N], E>
where
    F: FnMut(usize) -> Result<T, E>,
{
    let mut deque = DequeLike::new();
    let mut idx = 0;
    while !deque.is_full() {
        deque.push_back(initializer_fn(idx)?);
        idx += 1
    }
    Ok(deque.try_extract_array().unwrap())
}

/// Splits array `[T; N]` into two subarrays: `[T; L]` and `[T; R]`.
///
/// # Panics
///
/// If `N != L + R`.
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
pub fn split<T, const N: usize, const L: usize, const R: usize>(array: [T; N]) -> ([T; L], [T; R]) {
    assert_eq!(
        N,
        L + R,
        "Sum of outputs' lengths is not equal to length of input."
    );

    let mut iter = IntoIter::new(array);
    let left = init_with_iter(iter.by_ref()).unwrap();
    let right = init_with_iter(iter.by_ref()).unwrap();
    (left, right)
}

/// Joins two arrays.
///
/// # Panics
///
/// If `N != L + R`.
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
pub fn join<T, const N: usize, const L: usize, const R: usize>(
    left: [T; L],
    right: [T; R],
) -> [T; N] {
    assert_eq!(
        N,
        L + R,
        "Sum of inputs' lengths is not equal to output length."
    );
    let left_iter = IntoIter::new(left);
    let right_iter = IntoIter::new(right);
    init_with_iter(left_iter.chain(right_iter)).unwrap()
}

/// Consuming iterator over array items.
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
/// let mut iter = array_tools::IntoIter::new(array);
/// assert_eq!(iter.next(), Some(Foo(1)));
/// assert_eq!(iter.next(), Some(Foo(2)));
/// assert_eq!(iter.next(), Some(Foo(3)));
/// assert_eq!(iter.next(), None);
/// ```
pub struct IntoIter<T, const N: usize> {
    deque: DequeLike<T, N>,
}

impl<T, const N: usize> PartialEq for IntoIter<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.deque.as_slice() == other.deque.as_slice()
    }
}

impl<T, const N: usize> Eq for IntoIter<T, N> where T: Eq {}

impl<T, const N: usize> Clone for IntoIter<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        IntoIter {
            deque: self.deque.clone(),
        }
    }
}

impl<T, const N: usize> Debug for IntoIter<T, N>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntoIter")
            .field("deque", &self.deque)
            .finish()
    }
}

impl<T, const N: usize> IntoIter<T, N> {
    pub fn new(array: [T; N]) -> IntoIter<T, N> {
        IntoIter {
            deque: DequeLike::from_array(array),
        }
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
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

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    fn next_back(&mut self) -> Option<T> {
        self.deque.pop_back()
    }
}

/// An item of [`Chunks`](crate::Chunks) iterator.
///
/// See [`Chunks`](crate::Chunks) documentation.
pub enum Chunk<T, const CHUNK_LEN: usize, const TAIL_LEN: usize> {
    /// A normal chunk.
    Chunk([T; CHUNK_LEN]),
    /// A reaminder that has insufficient length to be a chunk.
    Tail([T; TAIL_LEN]),
}

impl<T, const CHUNK_LEN: usize, const TAIL_LEN: usize> Debug for Chunk<T, CHUNK_LEN, TAIL_LEN>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Chunk::Chunk(chunk) => f.debug_tuple("Chunk").field(&chunk).finish(),
            Chunk::Tail(tail) => f.debug_tuple("Tail").field(&tail).finish(),
        }
    }
}

impl<T, const CHUNK_LEN: usize, const TAIL_LEN: usize> PartialEq for Chunk<T, CHUNK_LEN, TAIL_LEN>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Chunk::Chunk(chunk), Chunk::Chunk(other_chunk)) => chunk == other_chunk,
            (Chunk::Tail(tail), Chunk::Tail(other_tail)) => tail == other_tail,
            _ => false,
        }
    }
}

impl<T, const CHUNK_LEN: usize, const TAIL_LEN: usize> Eq for Chunk<T, CHUNK_LEN, TAIL_LEN> where
    T: Eq
{
}

impl<T, const CHUNK_LEN: usize, const TAIL_LEN: usize> Clone for Chunk<T, CHUNK_LEN, TAIL_LEN>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        match &self {
            &Chunk::Chunk(chunk) => {
                let chunk_clone = init_with_slice(chunk).unwrap();
                Chunk::Chunk(chunk_clone)
            }
            &Chunk::Tail(tail) => {
                let tail_clone = init_with_slice(tail).unwrap();
                Chunk::Tail(tail_clone)
            }
        }
    }
}

/// By-value iterator over non-overlaping subarrays of equal size.
///
/// Consumes array upon creation and yields subarrays "chunks" of requested size.
///
/// If array can't be divided evenly into chunks, the last yielded item will be
/// a "tail" - an array of length `N % CHUNK_LEN`, containing what remains at the end.
///
/// If array can be divided evenly into "chunks", there will be no "tail".
///
/// # Examples
///
/// ```rust
/// use array_tools::{Chunk, Chunks};
///
/// let array = [1u64, 2, 3, 4, 5, 6, 7, 8];
///
/// // Divide array `[u64; 8]` into `[u64; 2]` chunks. It can be divided evenly,
/// // so there will be no tail.
/// let mut chunks: Chunks<u64, 8, 2, 0> = Chunks::new(array);
///
/// assert_eq!(chunks.next(), Some(Chunk::Chunk([1, 2])));
/// assert_eq!(chunks.next(), Some(Chunk::Chunk([3, 4])));
/// assert_eq!(chunks.next(), Some(Chunk::Chunk([5, 6])));
/// assert_eq!(chunks.next(), Some(Chunk::Chunk([7, 8])));
/// assert_eq!(chunks.next(), None);
///
///
/// let array = [1u64, 2, 3, 4, 5, 6, 7, 8];
///
/// // Divide array `[u64; 8]` into `[u64; 3]` chunks. It can't be divided evenly, so last item
/// // will be tail of type `[u64; 2]`.
/// let mut chunks: Chunks<u64, 8, 3, 2> = Chunks::new(array);
///
/// assert_eq!(chunks.next(), Some(Chunk::Chunk([1u64, 2, 3])));
/// assert_eq!(chunks.next(), Some(Chunk::Chunk([4u64, 5, 6])));
/// assert_eq!(chunks.next(), Some(Chunk::Tail([7u64, 8])));
/// assert_eq!(chunks.next(), None);
/// ```
pub struct Chunks<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize> {
    iter: IntoIter<T, N>,
    has_tail: bool,
}

impl<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize> Debug
    for Chunks<T, N, CHUNK_LEN, TAIL_LEN>
where
    T: Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chunks")
            .field("iter", &self.iter)
            .field("has_tail", &self.has_tail)
            .finish()
    }
}

impl<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize> PartialEq
    for Chunks<T, N, CHUNK_LEN, TAIL_LEN>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.iter == other.iter && self.has_tail == other.has_tail
    }
}

impl<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize> Eq
    for Chunks<T, N, CHUNK_LEN, TAIL_LEN>
where
    T: Eq,
{
}

impl<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize> Clone
    for Chunks<T, N, CHUNK_LEN, TAIL_LEN>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Chunks {
            iter: self.iter.clone(),
            has_tail: self.has_tail,
        }
    }
}

impl<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize>
    Chunks<T, N, CHUNK_LEN, TAIL_LEN>
{
    /// Creates a new instance of [`Chunks`](crate::Chunks) iterator.
    ///
    /// # Panics
    ///
    /// If `CHUNK_LEN` is `0`.
    ///
    /// If `N % CHUNK_LEN != TAIL_LEN`.
    pub fn new(array: [T; N]) -> Chunks<T, N, CHUNK_LEN, TAIL_LEN> {
        assert_ne!(CHUNK_LEN, 0);
        assert_eq!(
            N % CHUNK_LEN,
            TAIL_LEN,
            "Invalid tail length, expected {}.",
            TAIL_LEN
        );

        let iter = IntoIter::new(array);
        let has_tail = N % CHUNK_LEN > 0;

        Chunks { iter, has_tail }
    }

    fn items_remain(&self) -> usize {
        let (elements_remain, _) = self.iter.size_hint();
        if self.has_tail {
            elements_remain / CHUNK_LEN + 1
        } else {
            elements_remain / CHUNK_LEN
        }
    }

    fn has_chunks(&self) -> bool {
        let (elements_remain, _) = self.iter.size_hint();
        elements_remain / CHUNK_LEN > 0
    }
}

impl<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize> Iterator
    for Chunks<T, N, CHUNK_LEN, TAIL_LEN>
{
    type Item = Chunk<T, CHUNK_LEN, TAIL_LEN>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.has_chunks() {
            let chunk = init_with_iter(self.iter.by_ref()).unwrap();
            Some(Chunk::Chunk(chunk))
        } else if self.has_tail {
            let tail = init_with_iter(self.iter.by_ref()).unwrap();
            self.has_tail = false;
            Some(Chunk::Tail(tail))
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

impl<T, const N: usize, const CHUNK_LEN: usize, const TAIL_LEN: usize> DoubleEndedIterator
    for Chunks<T, N, CHUNK_LEN, TAIL_LEN>
{
    fn next_back(&mut self) -> Option<<Self as Iterator>::Item> {
        if self.has_tail {
            let mut tail = init_with_iter(self.iter.by_ref().rev()).unwrap();
            tail.reverse();
            self.has_tail = false;
            Some(Chunk::Tail(tail))
        } else if self.has_chunks() {
            let mut chunk = init_with_iter(self.iter.by_ref().rev()).unwrap();
            chunk.reverse();
            Some(Chunk::Chunk(chunk))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn not_enough_items() {
        let maybe_array: Option<[u64; 5]> = super::init_with_iter(1..=4);
        assert_eq!(maybe_array, None);
    }

    #[test]
    fn exact_item_count() {
        let maybe_array: Option<[u64; 5]> = super::init_with_iter(1..=5);
        assert_eq!(maybe_array, Some([1, 2, 3, 4, 5]));
    }

    #[test]
    fn too_many_items() {
        let maybe_array: Option<[u32; 5]> = super::init_with_iter(1..=100);
        assert_eq!(maybe_array, Some([1, 2, 3, 4, 5]));
    }

    #[test]
    fn array_into_iterator() {
        let mut iter = super::IntoIter::new([1, 2, 3, 4, 5]);
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn array_into_iterator_reverse() {
        let mut iter = super::IntoIter::new([1, 2, 3, 4, 5]).rev();
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn array_into_iterator_take_from_two_sides() {
        let mut iter = super::IntoIter::new([1, 2, 3, 4, 5]);
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
        let mut iter = super::IntoIter::new(array);
        assert_eq!(iter.next(), Some(1));
    }

    #[test]
    fn array_into_iterator_size_hint() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let mut iter = super::IntoIter::new(array);
        assert_eq!(iter.size_hint(), (7, Some(7)));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.size_hint(), (6, Some(6)));
    }

    #[test]
    fn array_into_iterator_count() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let iter = super::IntoIter::new(array);
        assert_eq!(iter.count(), 7);
    }

    #[test]
    fn array_into_iterator_last() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let iter = super::IntoIter::new(array);
        assert_eq!(iter.last(), Some(7));
    }

    #[test]
    fn array_into_iterator_nth() {
        let array = [1, 2, 3, 4, 5, 6, 7];
        let mut iter = super::IntoIter::new(array);
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
    fn init_by_index_with_fn() {
        fn initializer(idx: usize) -> u64 {
            (idx % 2) as u64
        }

        let array: [u64; 7] = super::init_with_mapped_idx(initializer);

        assert_eq!(array, [0, 1, 0, 1, 0, 1, 0]);
    }

    #[test]
    fn init_by_index_with_closure() {
        let array: [u64; 7] = super::init_with_mapped_idx(|idx| (idx % 2) as u64);

        assert_eq!(array, [0, 1, 0, 1, 0, 1, 0]);
    }

    /*
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
    */

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
    fn chunks_with_tail() {
        use super::{Chunk, Chunks};
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut chunks: Chunks<u64, 8, 3, 2> = Chunks::new(array);
        assert_eq!(chunks.next(), Some(Chunk::Chunk([1u64, 2, 3])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([4u64, 5, 6])));
        assert_eq!(chunks.next(), Some(Chunk::Tail([7u64, 8])));
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn chunks_without_tail() {
        use super::{Chunk, Chunks};
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: Chunks<u64, 9, 3, 0> = Chunks::new(array);
        assert_eq!(chunks.next(), Some(Chunk::Chunk([1u64, 2, 3])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([4u64, 5, 6])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([7u64, 8, 9])));
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn chunks_with_tail_rev() {
        use super::{Chunk, Chunks};
        use core::iter::Rev;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut chunks: Rev<Chunks<u64, 8, 3, 2>> = Chunks::new(array).rev();
        assert_eq!(chunks.next(), Some(Chunk::Tail([7u64, 8])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([4u64, 5, 6])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([1u64, 2, 3])));
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn chunks_withot_tail_rev() {
        use super::{Chunk, Chunks};
        use core::iter::Rev;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: Rev<Chunks<u64, 9, 3, 0>> = Chunks::new(array).rev();
        assert_eq!(chunks.next(), Some(Chunk::Chunk([7u64, 8, 9])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([4u64, 5, 6])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([1u64, 2, 3])));
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn chunks_with_tail_take_from_two_sides_front_first() {
        use super::{Chunk, Chunks};
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: Chunks<u64, 9, 2, 1> = Chunks::new(array);
        assert_eq!(chunks.next(), Some(Chunk::Chunk([1u64, 2])));
        assert_eq!(chunks.next_back(), Some(Chunk::Tail([9u64])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([3u64, 4])));
        assert_eq!(chunks.next_back(), Some(Chunk::Chunk([7u64, 8])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([5u64, 6])));
        assert_eq!(chunks.next_back(), None);
        assert_eq!(chunks.next(), None);
    }

    #[test]
    fn chunks_with_tail_take_from_two_sides_back_first() {
        use super::{Chunk, Chunks};
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut chunks: Chunks<u64, 9, 2, 1> = Chunks::new(array);
        assert_eq!(chunks.next_back(), Some(Chunk::Tail([9u64])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([1u64, 2])));
        assert_eq!(chunks.next_back(), Some(Chunk::Chunk([7u64, 8])));
        assert_eq!(chunks.next(), Some(Chunk::Chunk([3u64, 4])));
        assert_eq!(chunks.next_back(), Some(Chunk::Chunk([5u64, 6])));
        assert_eq!(chunks.next(), None);
        assert_eq!(chunks.next_back(), None);
    }

    #[test]
    fn chunks_with_tail_size_hint() {
        use super::Chunks;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let chunks: Chunks<u64, 8, 3, 2> = Chunks::new(array);
        assert_eq!(chunks.size_hint(), (3, Some(3)));
    }

    #[test]
    fn chunks_without_tail_size_hint() {
        use super::Chunks;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let chunks: Chunks<u64, 9, 3, 0> = Chunks::new(array);
        assert_eq!(chunks.size_hint(), (3, Some(3)));
    }

    #[test]
    fn chunks_with_tail_count() {
        use super::Chunks;
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let chunks: Chunks<u64, 8, 3, 2> = Chunks::new(array);
        assert_eq!(chunks.count(), 3);
    }

    #[test]
    fn chunks_without_tail_count() {
        use super::Chunks;
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let chunks: Chunks<u64, 9, 3, 0> = Chunks::new(array);
        assert_eq!(chunks.count(), 3);
    }

    #[test]
    fn chunks_with_tail_last() {
        use super::{Chunk, Chunks};
        let array: [u64; 8] = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let chunks: Chunks<u64, 8, 3, 2> = Chunks::new(array);
        assert_eq!(chunks.last(), Some(Chunk::Tail([7u64, 8])));
    }

    #[test]
    fn chunks_without_tail_last() {
        use super::{Chunk, Chunks};
        let array: [u64; 9] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9];
        let chunks: Chunks<u64, 9, 3, 0> = Chunks::new(array);
        assert_eq!(chunks.last(), Some(Chunk::Chunk([7u64, 8, 9])));
    }

    #[test]
    fn chunks_nth() {
        use super::{Chunk, Chunks};
        let array: [u64; 17] = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17];
        let mut chunks: Chunks<u64, 17, 2, 1> = Chunks::new(array);
        assert_eq!(chunks.nth(5), Some(Chunk::Chunk([11u64, 12])));
    }

    #[test]
    fn fixed_capacity_deque_like_eq() {
        use super::DequeLike;
        let mut deque1: DequeLike<u64, 10> = DequeLike::new();
        deque1.push_back(1);
        deque1.push_back(2);
        deque1.push_back(3);
        deque1.push_back(4);
        deque1.push_back(5);
        deque1.pop_front();
        deque1.pop_front();

        let mut deque2: DequeLike<u64, 10> = DequeLike::new();
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
        use super::DequeLike;

        let deque1: DequeLike<u64, 10> = DequeLike::new();
        let deque2: DequeLike<u64, 10> = DequeLike::new();

        assert_eq!(deque1, deque2);
    }

    #[test]
    fn fixed_capacity_deque_like_not_eq_different_items() {
        use super::DequeLike;

        let mut deque1: DequeLike<u64, 10> = DequeLike::new();
        deque1.push_back(1);
        deque1.push_back(2);

        let mut deque2: DequeLike<u64, 10> = DequeLike::new();
        deque2.push_back(2);
        deque1.push_back(3);

        assert_ne!(deque1, deque2);
    }

    #[test]
    fn fixed_capacity_deque_like_not_eq_different_offsets() {
        use super::DequeLike;

        let mut deque1: DequeLike<u64, 10> = DequeLike::new();
        deque1.push_back(0);
        deque1.push_back(1);
        deque1.push_back(2);
        deque1.pop_back();

        let mut deque2: DequeLike<u64, 10> = DequeLike::new();
        deque2.push_back(1);
        deque1.push_back(2);

        assert_ne!(deque1, deque2);
    }

    #[test]
    fn fixed_capacity_deque_like_clone_no_offset() {
        use super::DequeLike;

        let mut deque: DequeLike<u64, 10> = DequeLike::new();
        deque.push_back(1);
        deque.push_back(2);
        deque.push_back(3);

        let clone = deque.clone();

        assert_eq!(deque, clone);
    }

    #[test]
    fn fixed_capacity_deque_like_clone_with_offset() {
        use super::DequeLike;

        let mut deque: DequeLike<u64, 10> = DequeLike::new();
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
        use super::IntoIter;

        let a = [1, 2, 3, 4, 5];
        let b = [1, 2, 3, 4, 5];

        let iter_a = IntoIter::new(a);
        let iter_b = IntoIter::new(b);

        assert_eq!(iter_a, iter_b);
    }

    #[test]
    fn array_into_iterator_eq_if_items_eq() {
        use super::IntoIter;
        use core::mem;

        let a = [1, 2, 3, 4, 5, 6];
        let b = [0, 0, 0, 3, 4, 5];

        let mut iter_a = IntoIter::new(a);
        let mut iter_b = IntoIter::new(b);

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
        use super::IntoIter;
        let a = [1, 2, 3, 4, 5];
        let a_iter = IntoIter::new(a);
        let a_iter_clone = a_iter.clone();
        assert_eq!(a_iter, a_iter_clone);
    }

    #[test]
    fn array_chunk_eq_chunk() {
        use super::Chunk;

        let chunk1: Chunk<u64, 4, 3> = Chunk::Chunk([1, 2, 3, 4]);
        let chunk2: Chunk<u64, 4, 3> = Chunk::Chunk([1, 2, 3, 4]);

        assert_eq!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_eq_tail() {
        use super::Chunk;

        let chunk1: Chunk<u64, 4, 3> = Chunk::Tail([1, 2, 3]);
        let chunk2: Chunk<u64, 4, 3> = Chunk::Tail([1, 2, 3]);

        assert_eq!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_ne_tail_and_chunk() {
        use super::Chunk;

        let chunk1: Chunk<u64, 4, 3> = Chunk::Chunk([1, 2, 3, 4]);
        let chunk2: Chunk<u64, 4, 3> = Chunk::Tail([1, 2, 3]);

        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_ne_different_chunk_items() {
        use super::Chunk;

        let chunk1: Chunk<u64, 4, 3> = Chunk::Chunk([1, 2, 3, 4]);
        let chunk2: Chunk<u64, 4, 3> = Chunk::Chunk([4, 3, 2, 1]);

        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_ne_different_tail_items() {
        use super::Chunk;

        let chunk1: Chunk<u64, 4, 3> = Chunk::Tail([1, 2, 3]);
        let chunk2: Chunk<u64, 4, 3> = Chunk::Tail([4, 3, 2]);

        assert_ne!(chunk1, chunk2);
    }

    #[test]
    fn array_chunk_clone_chunk() {
        use super::Chunk;

        let chunk: Chunk<u64, 4, 3> = Chunk::Chunk([1, 2, 3, 4]);
        let chunk_clone = chunk.clone();

        assert_eq!(chunk, chunk_clone);
    }

    #[test]
    fn array_chunk_clone_stump() {
        use super::Chunk;

        let chunk: Chunk<u64, 4, 3> = Chunk::Tail([1, 2, 3]);
        let chunk_clone = chunk.clone();

        assert_eq!(chunk, chunk_clone);
    }

    #[test]
    fn chunks_eq_case_1() {
        use super::Chunks;
        use core::mem;

        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut a_chunks = Chunks::<u64, 8, 3, 2>::new(a);

        let b = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut b_chunks = Chunks::<u64, 8, 3, 2>::new(b);

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
    fn chunks_eq_case_2() {
        use super::Chunks;
        use core::mem;

        let a = [0, 1, 2, 4, 5, 6, 8, 9];
        let mut a_chunks = Chunks::<u64, 8, 3, 2>::new(a);

        let b = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut b_chunks = Chunks::<u64, 8, 3, 2>::new(b);

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
    fn chunks_eq_case_3() {
        use super::Chunks;
        use core::mem;

        let a = [0, 1, 2, 4, 5, 6, 8, 9];
        let mut a_chunks = Chunks::<u64, 8, 3, 2>::new(a);

        let b = [10, 11, 12, 13, 14, 15, 16, 17];
        let mut b_chunks = Chunks::<u64, 8, 3, 2>::new(b);

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
    fn chunks_clone() {
        use super::Chunks;
        use core::mem;

        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        let mut a_chunks = Chunks::<u64, 8, 3, 2>::new(a);
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
    fn chunks_ommit_generic_parameters() {
        use super::Chunks;

        let array = [1, 2, 3, 4, 5, 6, 7, 8];
        let _chunks: Chunks<_, 8, 3, 2> = Chunks::new(array);
    }

    #[test]
    #[should_panic]
    fn panicking_clone() {
        use super::DequeLike;

        #[derive(Debug, PartialEq, Eq)]
        struct PanickingThing(usize);

        impl Clone for PanickingThing {
            fn clone(&self) -> Self {
                if self.0 == 2 {
                    panic!("Oh no, my number is 2!");
                } else {
                    Self(self.0)
                }
            }
        }

        impl Drop for PanickingThing {
            fn drop(&mut self) {
                // Touch potentially uninitialized memory.
                assert_eq!(self.0, self.0);
            }
        }

        let panicking_things = [
            PanickingThing(0),
            PanickingThing(1),
            PanickingThing(2),
            PanickingThing(3),
        ];
        let deque = DequeLike::from_array(panicking_things);
        let deque_clone = deque.clone();

        // This is never reached.
        assert_eq!(deque, deque_clone);
    }
}
