use array_tools::ArrayIntoIterator;

#[derive(Debug)]
struct NonCopyable(u64);

fn main() {
    
    let array = [NonCopyable(0), NonCopyable(1), NonCopyable(2), NonCopyable(3), NonCopyable(4)];

    let vec: Vec<NonCopyable> = ArrayIntoIterator::new(array).rev().collect();

    println!("{:?}", vec);
    // Prints [NonCopyable(4), NonCopyable(3), NonCopyable(2), NonCopyable(1), NonCopyable(0)]
}
