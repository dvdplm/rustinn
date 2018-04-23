extern crate failure;
extern crate tinn;
extern crate rand;

use tinn::*;

use std::fs::File;
use std::io::{BufRead, BufReader};
use failure::Error;
use rand::Rng;

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
    rows: usize,
}


impl Data {
    pub fn shuffle(&mut self) {
        let mut rng = rand::thread_rng();
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
//    println!("[build] looking for data in '{}'", path);
    let f = File::open(path)?;
    let f = BufReader::new(f);
    let mut rows = 0; // TODO: needed?
    let mut data = Data{
        inp: Vec::new(), tg: Vec::new(), nips: NIPS, nops: NOPS, rows: 0
    };
    for line in f.lines() {
        rows += 1; // TODO: needed?
        let mut inps = Vec::with_capacity(NIPS);
        let mut tgs = Vec::with_capacity(NOPS);
        for (idx, value) in line.unwrap().split_whitespace().enumerate()
            .map(|(idx, str_val)| (idx, str_val.parse::<f64>().unwrap())) {
//            println!("idx: {}, column: {:?}", idx, value);
            if idx < NIPS {
                inps.push(value);
            } else {
                tgs.push(value);
            }
        }
//        println!("[build, row {}] pushed {} inps and {} tgs", rows, inps.len(), tgs.len());
        data.inp.push(inps);
        data.tg.push(tgs);
    }
    data.rows = rows; // TODO: what do I need this for?
//    println!("\n\nLine count: {}", rows);
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
    for _ in 0..100 {
//    for _ in 0..100 {
        data.shuffle();
        let mut error = 0.0;
        for j in 0..data.inp.len() {
//        for j in 0..200 {
            error += tinn.train(&data.inp[j], &data.tg[j], rate);
        };
        println!("error {:.12} :: learning rate {:.6}", error / data.inp.len() as f64, rate);
        rate *= ANNEAL;
    };
//    println!("Read data.inp[0] {:?}, data.tg: {}, data.rows: {}", data.inp[0], data.tg.len(), data.rows );
    println!("Done training.");
    let prediction= tinn.predict(&data.inp[5]);
    println!("Expected:  {:?}", data.tg[5]);
    println!("Predicted: {:?}", prediction);
}


mod tests {
    use super::*;

    #[test]
    fn blabla() {
        let mut rng = rand::thread_rng();
        for i in 0..100 {
            let n = rng.gen_range(0, 10);
            println!("n: {}", n);
    }
    }

}
