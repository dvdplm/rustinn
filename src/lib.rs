extern crate rand;
use rand::{SmallRng, SeedableRng, thread_rng};
use rand::distributions::{IndependentSample, Range};

pub struct Tinn {
    w: Vec<f64>, // All the weights.
    x: Vec<f64>, // Hidden to output layer weights.
    b: Vec<f64>, // Biases.
    h: Vec<f64>, // Hidden layer.
    o: Vec<f64>, // Output layer.

    nips: usize, // Number of inputs.
    nhid: usize, // Number of hidden neurons.
    nops: usize, // Number of outputs.
}

impl Tinn {
    pub fn new(nips: usize, nhid: usize, nops: usize) -> Self {
        let nw = nhid * (nips + nops);
        let mut t  = Tinn{
            w: vec![0.0;nw],
            x: vec![0.0; nw + nhid * nips],
            b: vec![0.0; 2],
            h: vec![0.0; nhid],
            o: vec![0.0; nops],
            nips,
            nhid,
            nops
        };
        t.randomize_weights_and_biases();
//        println!("[new] x: {}, {:?}", t.x.len(), t.x);
        t
    }

    fn randomize_weights_and_biases(&mut self) {
        let mut rng = SmallRng::from_rng(thread_rng()).unwrap();
        let range = Range::new(-0.5f64, 0.5f64);
        for i in 0..self.w.len() {
            self.w[i] = range.ind_sample(&mut rng);
            self.x[i] = self.w[i];
        }
        for i in 0..self.b.len() {
            self.b[i] = range.ind_sample(&mut rng);
        }
    }

    pub fn train(&mut self, inp: &Vec<f64>, tg: &Vec<f64>, rate: f64) -> f64 {
//        println!("Tinn hidden layer neurons BEFORE fprop: {:?}", self.h[0]);
//        println!("Tinn output neurons BEFORE fprop: {:?}", self.o);

        self.forward_prop(inp);

//        println!("Tinn hidden layer neurons AFTER fprop: {:?}", self.h[0]);
//        println!("Tinn output neurons AFTER fprop: {:?}", self.o);

        self.backward_prop(inp, tg, rate);
        self.total_error(tg)
    }

    fn forward_prop(&mut self, inp: &Vec<f64>) {
        // Calculate hidden layer neuron values.
        for i in 0..self.nhid {
            let mut sum = 0.0;
            for j in 0..self.nips {
                sum += inp[j] * self.w[i*self.nips + j];
            }
            self.h[i] = Self::act(sum + self.b[0]);
        }
        // Calculate output layer neuron values.
//        println!("[fprop] nhid: {}, nops: {}; biases: {:?}", self.nhid, self.nops, self.b);
        for i in 0..self.nops {
            let mut sum = 0.0;
            for j in 0..self.nhid {
//                println!("[fprop] h[{}] * x[{} * {} + {}] = {} * {} = {} (curr sum: {})",j, i, self.nhid, j, self.h[j], self.x[i*self.nhid + j], self.h[j] * self.x[i*self.nhid + j], sum);
                sum += self.h[j] * self.x[i*self.nhid + j];
            }
//            println!("[fprop] setting o[{}] to {} (sum: {}, bias: {})", i, Self::act(sum + self.b[1]), sum, self.b[1]);
            self.o[i] = Self::act(sum + self.b[1]);
        }
    }

    // Activation function
    fn act(a:f64) -> f64 {
        let r = 1.0 / (1.0 + std::f64::consts::E.powf(-a));
//        println!("  act for {} == {}", a, r);
        r
    }

    // Partial derivative of activation function
    fn pdact(a:f64) -> f64 {
        a * (1.0 - a)
    }

    // Partial derivative of error function
    fn pderr(a:f64, b:f64) -> f64 {
        a - b
    }

    fn backward_prop(&mut self, inp: &Vec<f64>, tg: &Vec<f64>, rate: f64) {
        for i in 0..self.nhid {
            let mut sum = 0.0;
            // Calculate total error change with respect to output
            for j in 0..self.nops {
                let a = Self::pderr(self.o[j], tg[j]);
                let b = Self::pdact(self.o[j]);
                sum += a * b * self.x[j * self.nhid + i];
                // Correct weights in hidden to output layer
                self.x[j * self.nhid + i] -= rate * a * b * self.h[i];
            }
            // Correct weights in input to hidden layer.
            for j in 0..self.nips {
                self.w[i * self.nips + j] -= rate * sum * Self::pdact(self.h[i]) * inp[j];
            }
        }
    }

    fn total_error(&self, tg: &Vec<f64>) -> f64 {
//        println!("[total_error] tg: {:?} over {} nops", tg, self.nops);
        let mut sum = 0.0_f64;
        for i in 0..self.nops {

            sum += 0.5 * (tg[i] - self.o[i]).powf(2.0_f64);
//            println!("[total_error] tg[i]: {:.1}, o[i]: {}, err: {}", tg[i], self.o[i], self.err(tg[i], self.o[i]));
        }
//        println!("[total_error] Error for tg: {:?} sum: {}", tg, sum);
        sum
    }

    pub fn predict(&mut self, inp: &Vec<f64>) -> Vec<f64> {
        self.forward_prop(inp);
        self.o.clone()
    }

}