use array_tools;

fn main() {
    
    let array: [u64; 10] = array_tools::try_init_from_iterator(0..10).unwrap();
    
    println!("{:?}", array);
    // Prints [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
