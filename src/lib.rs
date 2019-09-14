//! # Array Tools
//!
//! A little collection of array-related utils aiming to make life easier.
//!
//!
//! ## Stability warning
//!
//! Requires nightly.
//!
//! Consider this crate experimental. Some (all?) of currently provided features
//! are most likely will be integrated into `rust`'s core/std library sooner or
//! later, and with arrival of const generics public interfaces are most likely
//! will be changed.
//!
//! ## Features
//!
//! - **Metafeature:** all features below should work for arrays of **any** size.
//! - Array initialization with iterator.
//! - Array initizlization with function (with or without index passed as
//!   argument).
//! - Array by-value "into" iterator.
//! - No dependency on `std` and no heap allocations.
//!
//! ## Examples
//!
//! ```rust
//! use array_tools::{self, ArrayIntoIterator};
//!
//!
//! // Array initialization with iterator.
//! let array1: [u64; 7] =
//!     array_tools::try_init_from_iterator(0u64..17).unwrap();
//!
//! assert_eq!(array1, [0, 1, 2, 3, 4, 5, 6]);
//!
//!
//! // Array initialization with function (w/o index).
//! let mut value = 0u64;
//!
//! let array2: [u64; 7] = array_tools::init_with(|| {
//!     let tmp = value;
//!     value += 1;
//!     tmp
//! });
//!
//! assert_eq!(array2, [0, 1, 2, 3, 4, 5, 6]);
//!
//!
//! // Array initialization with function (w/ index).
//! let array3: [u64; 7] = array_tools::indexed_init_with(|idx| {
//!     idx as u64
//! });
//!
//! assert_eq!(array3, [0, 1, 2, 3, 4, 5, 6]);
//!
//!
//! // Array by-value iterator.
//! #[derive(Debug, PartialEq, Eq)]
//! struct NonCopyable(u64);
//!
//! let array4: [NonCopyable; 7] =
//!     array_tools::indexed_init_with(|idx| NonCopyable(idx as u64));
//!
//! let iter = ArrayIntoIterator::new(array4);
//!
//! let array5: [NonCopyable; 7] =
//!     array_tools::try_init_from_iterator(iter).unwrap();
//!
//! assert_eq!(array5, [
//!     NonCopyable(0),
//!     NonCopyable(1),
//!     NonCopyable(2),
//!     NonCopyable(3),
//!     NonCopyable(4),
//!     NonCopyable(5),
//!     NonCopyable(6),
//! ]);
//! ```

#![no_std]
#![feature(fixed_size_array, const_fn, const_generics)]
use core::array::FixedSizeArray;
use core::marker::PhantomData;
use core::mem::MaybeUninit;
use core::{mem, ptr};

struct FixedCapacityDequeLike<T, A: FixedSizeArray<T>> {
    array: MaybeUninit<A>,
    begining: usize,
    end: usize,
    _element: PhantomData<T>,
}

impl<T, A: FixedSizeArray<T>> FixedCapacityDequeLike<T, A> {
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
}

impl<T, A: FixedSizeArray<T>> Drop for FixedCapacityDequeLike<T, A> {
    fn drop(&mut self) {
        while let Some(item) = self.pop_back() {
            mem::drop(item);
        }
    }
}

/// Attempts to initialize array with iterator.
///
/// # Examples
/// ```rust
/// use array_tools;
///
/// // If iterator yields less items than array capacity, this function will return `None`.
/// let maybe_array: Option<[u64; 5]> = array_tools::try_init_from_iterator(0..4);
/// assert_eq!(maybe_array, None);
///
/// // If iterator yields just enough items to fill array, this function will `Some(array)`.
/// let maybe_array: Option<[u64; 5]> = array_tools::try_init_from_iterator(0..5);
/// assert_eq!(maybe_array, Some([0, 1, 2, 3, 4]));
///
/// // If iterator yields more items than array capacity, only required amount of items will be
/// // taken, function will return `Some(array)`.
/// let maybe_array: Option<[u32; 5]> = array_tools::try_init_from_iterator(0..100);
/// assert_eq!(maybe_array, Some([0, 1, 2, 3, 4]));
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
        } else {
            if let Some(item) = iter.next() {
                deque.push_back(item)
            } else {
                break None;
            }
        }
    }
}

/// Initializes array with values provided by function.
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

/// Initializes array with values provided by function (version with index as argument).
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

/// A functiona akin to `size_of`, but computes length of array.
///
/// It is not much useful if array length is known, but in generic contexts where
/// only `FixedSizeArray` trait is given, it may come in handy.
///
/// ```rust
/// use array_tools;
///
/// let length = array_tools::length_of::<u64, [u64; 7]>();
/// assert_eq!(length, 7);
/// ```
///
/// # Note
///
/// Currently it is not a `const fn`, but this should be fixed in future.
pub fn length_of<T, A: FixedSizeArray<T>>() -> usize {
    let array: MaybeUninit<A> = MaybeUninit::uninit();
    unsafe { (*array.as_ptr()).as_slice().len() }
}

/// A function to split arrays.
///
/// It is akin to `slice`'s `split_at`, except it does not take split index as argument
/// and infers it from output arrays' lengths.
/// ```rust
/// use array_tools;
///
/// let array = [1u64, 2, 3, 4, 5, 6, 7, 8];
/// let (left, right): ([u64; 2], [u64; 6]) = array_tools::split(array);
/// assert_eq!(left, [1u64, 2]);
/// assert_eq!(right, [3u64, 4, 5, 6, 7, 8]);
/// ```
///
/// # Panics
///
/// Panics if sum of outputs' lengths is not equal to length of input.
///
/// # Note
///
/// Currently it panics if output arrays have incompatible lengths,
/// in future this behavior most certainly will be changed to perform
/// this check at compile time.
pub fn split<T, A: FixedSizeArray<T>, LEFT: FixedSizeArray<T>, RIGHT: FixedSizeArray<T>>(
    array: A,
) -> (LEFT, RIGHT) {
    assert_eq!(
        array.as_slice().len(),
        length_of::<T, LEFT>() + length_of::<T, RIGHT>(),
        "Sum of outputs' lengths is not equal to length of input."
    );

    let mut iter = ArrayIntoIterator::new(array);
    let left: LEFT = try_init_from_iterator(iter.by_ref()).unwrap();
    let right: RIGHT = try_init_from_iterator(iter.by_ref()).unwrap();
    (left, right)
}

/// A function to join arrays.
///
/// Takes two arrays as arguments and returns array containing elements of both.
///
/// # Panics
///
/// Panics if output array length is not equal to sum of input arrays' lengths.
///
/// # Note
///
/// Currently it panics if output array has incompatible length.
/// In future this behavior most certainly will be changed to perform this check
/// at compile time.
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
pub fn join<T, A: FixedSizeArray<T>, LEFT: FixedSizeArray<T>, RIGHT: FixedSizeArray<T>>(
    left: LEFT,
    right: RIGHT,
) -> A {
    assert_eq!(
        length_of::<T, A>(),
        left.as_slice().len() + right.as_slice().len(),
        "Sum of inputs' lengths is not equal to output length."
    );
    let left_iter = ArrayIntoIterator::new(left);
    let right_iter = ArrayIntoIterator::new(right);
    try_init_from_iterator(left_iter.chain(right_iter)).unwrap()
}

/// A by-value iterator over array.
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
pub struct ArrayIntoIterator<T, A: FixedSizeArray<T>> {
    deque: FixedCapacityDequeLike<T, A>,
}

impl<T, A: FixedSizeArray<T>> ArrayIntoIterator<T, A> {
    pub fn new(array: A) -> ArrayIntoIterator<T, A> {
        ArrayIntoIterator {
            deque: FixedCapacityDequeLike::from_array(array),
        }
    }
}

impl<T, A: FixedSizeArray<T>> Iterator for ArrayIntoIterator<T, A> {
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

impl<T, A: FixedSizeArray<T>> DoubleEndedIterator for ArrayIntoIterator<T, A> {
    fn next_back(&mut self) -> Option<T> {
        self.deque.pop_back()
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
}
