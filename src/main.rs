extern crate failure;
extern crate tinn;
extern crate rand;

use tinn::*;
use rand::{Rng, SmallRng, SeedableRng, thread_rng};


fn main() {
    let mut rate = 1.0;
    let mut data = match Data::build("semeion.data") {
        Ok(d) => d,
        Err(e) => {
            println!("{}", e);
            std::process::exit(1);
        }
    };
    let mut tinn = Tinn::new(NIPS, NHID, NOPS);
    let train_start = std::time::Instant::now();
    for _ in 0..100 {
        data.shuffle();
        let mut error = 0.0;
        for j in 0..data.inp.len() {
            error += tinn.train(&data.inp[j], &data.tg[j], rate);
        };
        println!("error {:.12} :: learning rate {:.6}", error / data.inp.len() as f64, rate);
        rate *= ANNEAL;
    };
    let train_stop= std::time::Instant::now();
    println!("Done training in {:?}", train_stop.duration_since(train_start));
    let prediction= tinn.predict(&data.inp[5]);
    println!("Expected:  {:?}", data.tg[5]);
    println!("Predicted: {:?}", prediction);
}


mod tests {
    #[test]
    fn blabla() {}
}
