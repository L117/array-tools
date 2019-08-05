#![feature(fixed_size_array)]
use std::array::FixedSizeArray;
use std::mem::MaybeUninit;
use std::marker::PhantomData;
use std::{ptr, mem};

struct FixedCapacityDequeLike<T, A: FixedSizeArray<T>> {
    array: MaybeUninit<A>,
    begining: usize,
    end: usize,
    _element: PhantomData<T>
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
        unsafe {
            (*self.array.as_ptr()).as_slice().len()
        }
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
            let slice = unsafe {
                (*self.array.as_mut_ptr()).as_mut_slice()
            };
            unsafe {
                ptr::write(&mut slice[item_index], element);
            }
            self.end += 1;
        } else {
            panic!("No capacity left at the end.");
        }
    }

    fn push_front(&mut self, element: T) {
        if self.begining != 0 {
            let item_index = self.begining - 1;
            let slice = unsafe {
                (*self.array.as_mut_ptr()).as_mut_slice()
            };
            unsafe {
                ptr::write(&mut slice[item_index], element);
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
                ptr::read(&slice[item_index])
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
                ptr::read(&slice[item_index])
            };
            self.begining += 1;
            Some(item)
        }
    }

    fn try_extract_array(&mut self) -> Option<A> {
        if self.length() == self.capacity() {
            let array_shallow_copy = unsafe {
                ptr::read(self.array.as_ptr())
            };
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

/// Attempts to initialize fixed-length array from iterator.
/// 
/// # Examples
/// ```rust
/// use array_tools;
/// 
/// // Attempt to initialize array with iterator yielding not enough items will result in returning
/// // `None`. All taken items will be dropped.
/// let maybe_array: Option<[u64; 5]> = array_tools::try_init_from_iterator(1..=4);
/// assert_eq!(maybe_array, None);
/// 
/// // Attempt to initialize array with iterator yielding number of items equal to array length
/// // will result in returning `Some` containing array.
/// let maybe_array: Option<[u64; 5]> = array_tools::try_init_from_iterator(1..=5);
/// assert_eq!(maybe_array, Some([1, 2, 3, 4, 5]));
/// 
/// // Attempt to inititalize array with iterator yielding too many items (Mare than array can 
/// // contain) will result in taking length-of-array items from iterator and returning `Some` 
/// // containing array. Iterator with all remaining items will be dropped.
/// let maybe_array: Option<[u32; 5]> = array_tools::try_init_from_iterator(1..=100);
/// assert_eq!(maybe_array, Some([1, 2, 3, 4, 5])); 
/// ```
pub fn try_init_from_iterator<T, A, I>(mut iter: I) -> Option<A>
where
    A: FixedSizeArray<T>,
    I: Iterator<Item = T>,
{
    let mut vec = FixedCapacityDequeLike::new();
    loop {
        if vec.is_full() {
            break vec.try_extract_array();
        } else {
            if let Some(item) = iter.next() {
                vec.push_back(item)
            } else {
                break None;
            }
        }
    }
}

/// A by-value iterator over array.
/// 
/// # Examples
/// 
/// ```rust
/// use array_tools;
/// 
/// // Notice - this struct is neither Copy not Clone.
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
    deque: FixedCapacityDequeLike<T, A>
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
mod test {
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
}