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
}