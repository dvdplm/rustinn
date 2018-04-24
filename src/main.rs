extern crate failure;
extern crate tinn;
extern crate rand;

use tinn::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use failure::Error;
use rand::{Rng, SmallRng, SeedableRng, thread_rng};

const NIPS : usize = 256;
const NOPS : usize = 10;
const NHID : usize = 28;
const ANNEAL : f64 = 0.99;

#[derive(Debug)]
struct Data {
    inp: Vec<Vec<f64>>,
    tg: Vec<Vec<f64>>,
    nips: usize,
    nops: usize,
}


impl Data {
    pub fn shuffle(&mut self) {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
        let mut i = self.inp.len();
        while i >= 2 {
            i -= 1;
            let idx = rng.gen_range(0,i+1);
            self.inp.swap(i, idx);  // Swap input
            self.tg.swap(i, idx);   // Swap output
        }
    }
}

fn build(path: &str) -> Result<Data, Error> {
    let f = File::open(path)?;
    let f = BufReader::new(f);
    let mut data = Data{ inp: Vec::new(), tg: Vec::new(), nips: NIPS, nops: NOPS };
    for line in f.lines() {
        let mut inps = Vec::with_capacity(NIPS);
        let mut tgs = Vec::with_capacity(NOPS);
        for (idx, value) in line.unwrap().split_whitespace().enumerate()
            .map(|(idx, str_val)| (idx, str_val.parse::<f64>().unwrap())) {
            if idx < NIPS {
                inps.push(value);
            } else {
                tgs.push(value);
            }
        }
        data.inp.push(inps);
        data.tg.push(tgs);
    }
    Ok(data)
}

fn main() {
    let mut rate = 1.0;
    let mut data = match build("semeion.data") {
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
//        for j in 0..200 {
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
