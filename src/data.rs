use candle_core::{Device, IndexOp, Tensor};
use itertools::Itertools;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use std::collections::BTreeMap;

type Batch = (Tensor, Tensor);

pub(crate) struct Dataset {
    tokens: Tensor,
    device: Device,
}

impl Dataset {
    pub(crate) fn new(tokens: Tensor, device: Device) -> Result<Self, candle_core::Error> {
        Ok(Self { tokens, device })
    }

    pub(crate) fn train_test_split(
        self,
        split: f32,
    ) -> Result<(Dataset, Dataset), candle_core::Error> {
        let n = (self.tokens.dims()[0] as f32 * split).round() as usize;
        let train = Dataset {
            tokens: self.tokens.i(..n)?,
            device: self.device.clone(),
        };
        let test = Dataset {
            tokens: self.tokens.i(n..)?,
            device: self.device,
        };
        Ok((train, test))
    }

    pub(crate) fn batch(
        &self,
        rng: &mut impl Rng,
        batch_size: usize,
        block_size: usize,
    ) -> Result<Batch, candle_core::Error> {
        let between = Uniform::from(0..(self.tokens.dims()[0] - block_size));
        let idxs: Vec<usize> = (0..batch_size).map(|_| between.sample(rng)).collect();
        Ok((
            Tensor::stack(
                &idxs
                    .iter()
                    .map(|idx| self.tokens.i(*idx..(idx + block_size)))
                    .collect::<Result<Vec<_>, _>>()?,
                0,
            )?,
            Tensor::stack(
                &idxs
                    .iter()
                    .map(|idx| self.tokens.i(*idx + 1..(idx + block_size + 1)))
                    .collect::<Result<Vec<_>, _>>()?,
                0,
            )?,
        ))
    }
}

pub(crate) struct Tokenizer {
    c_to_i: BTreeMap<char, i64>,
    i_to_c: BTreeMap<i64, char>,
}

impl From<&str> for Tokenizer {
    fn from(value: &str) -> Self {
        let c_to_i: BTreeMap<char, i64> = value
            .chars()
            .unique()
            .sorted()
            .enumerate()
            .map(|(i, c)| (c, i as i64))
            .collect();
        let i_to_c = c_to_i.iter().map(|(c, i)| (*i, *c)).collect();
        Self { c_to_i, i_to_c }
    }
}

impl Tokenizer {
    pub(crate) fn vocab_size(&self) -> usize {
        self.c_to_i.len()
    }

    pub(crate) fn encode(&self, c: &char) -> i64 {
        *self
            .c_to_i
            .get(c)
            .unwrap_or_else(|| panic!("Unable to encode {}", c))
    }

    pub(crate) fn decode(&self, i: &i64) -> char {
        *self
            .i_to_c
            .get(i)
            .unwrap_or_else(|| panic!("Unable to decode {}", i))
    }
}
