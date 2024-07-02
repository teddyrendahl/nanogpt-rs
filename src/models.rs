use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{embedding, loss::cross_entropy, ops::softmax, Embedding, Module, VarBuilder};
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};

use crate::data::Dataset;

pub(crate) struct BigramLanguageModel {
    /// The simplest version of Bigram model
    ///
    /// The model consists of a single VOCAB SIZE x VOCAB SIZE matrix. Each value (i, j) of the matrix
    /// can be thought of as the probability of token j occuring, given that token i preceded it.
    embedding_table: Embedding,
}

impl BigramLanguageModel {
    pub(crate) fn new(vocab_size: usize, vs: VarBuilder) -> Result<Self, candle_core::Error> {
        Ok(Self {
            embedding_table: embedding(vocab_size, vocab_size, vs)?,
        })
    }

    pub(crate) fn train(&self, xs: &Tensor, ys: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(xs)?;
        let (b, t, c) = logits.dims3()?;
        let loss = cross_entropy(&logits.reshape((b * t, c))?, &ys.reshape(b * t)?)?;
        Ok(loss)
    }

    pub(crate) fn generate(
        &self,
        mut idx: Tensor,
        max_new_tokens: usize,
        device: &Device,
        rng: &mut impl Rng,
    ) -> candle_core::Result<Tensor> {
        for _ in 0..max_new_tokens {
            let logits = self.forward(&idx)?;
            let (_, t, _) = logits.dims3()?;
            // Get logit from last time step and find the probabilities
            let probs = softmax(&logits.i((.., t - 1, ..))?, 1)?;
            // Sample a new value from the probability distribution
            let distr = WeightedIndex::new(&probs.to_vec2::<f64>()?[0])
                .expect("Failed to make weighted distributed");
            idx = Tensor::cat(
                &[
                    &idx,
                    &Tensor::from_slice(&[distr.sample(rng) as i64], (1, 1), device)?,
                ],
                1,
            )?;
            // Append it to idx
        }
        Ok(idx)
    }
}

impl candle_core::Module for BigramLanguageModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.embedding_table.forward(xs)
    }
}

pub(crate) fn estimate_loss(
    model: &BigramLanguageModel,
    data: &Dataset,
    rng: &mut impl Rng,
    batch_size: usize,
    block_size: usize,
    eval_iters: usize,
) -> candle_core::Result<f64> {
    let mut v = 0.0;
    for _ in 0..eval_iters {
        let (x, y) = data.batch(rng, batch_size, block_size)?;
        v += model.train(&x, &y)?.to_scalar::<f64>()?;
    }
    Ok(v / (eval_iters as f64))
}
