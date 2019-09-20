use array_tools::{indexed_init_with, ArrayChunk, ArrayChunks, ArrayIntoIterator};
use core::marker::PhantomData;

#[test]
fn into_iterator_test() {
    let array = [1u64, 2, 3, 4];
    let mut into_iter = ArrayIntoIterator::new(array);
    assert_eq!(into_iter.next(), Some(1));
    assert_eq!(into_iter.next(), Some(2));
    assert_eq!(into_iter.next(), Some(3));
    assert_eq!(into_iter.next(), Some(4));
    assert_eq!(into_iter.next(), None);
}

#[test]
fn indexed_init_with_test() {
    let array: [usize; 4] = indexed_init_with(|idx| idx * 2);
    assert_eq!(array, [0usize, 2, 4, 6]);
}

#[test]
fn array_chunks_test() {
    let array = [0u64, 1, 2, 3, 4, 5, 6, 7];
    let mut chunks: ArrayChunks<_, _, [_; 3], [_; 2]> = ArrayChunks::new(array);
    assert_eq!(
        chunks.next(),
        Some(ArrayChunk::Chunk([0u64, 1, 2], PhantomData))
    );
    assert_eq!(
        chunks.next(),
        Some(ArrayChunk::Chunk([3u64, 4, 5], PhantomData))
    );
    assert_eq!(
        chunks.next(),
        Some(ArrayChunk::Stump([6u64, 7], PhantomData))
    );
    assert_eq!(chunks.next(), None);
}
