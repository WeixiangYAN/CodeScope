use std::io;

fn main() {
    let mut line = String::new();
    io::stdin().read_line(&mut line).expect("Failed to read line");

    let inputs: Vec<i64> = line.split_whitespace()
    .map(|x| x.parse().expect("Not an integer!"))
    .collect();
    println!("{}", inputs[0]+inputs[1]);
}
