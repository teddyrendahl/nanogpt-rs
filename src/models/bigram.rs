use candle_core::Tensor;
use candle_nn::{embedding, loss::cross_entropy, Embedding, Module, VarBuilder};

use super::model::Model;

pub(crate) struct BigramLanguageModel {
    /// The simplest version of Bigram model
    ///
    /// The model consists of a single VOCAB SIZE x VOCAB SIZE matrix. Each value (i, j) of the matrix
    /// can be thought of as the probability of token j occuring, given that token i preceded it.
    embedding_table: Embedding,
}
impl Model for BigramLanguageModel {
    fn block_size(&self) -> usize {
        1
    }
    fn train(&self, xs: &Tensor, ys: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(xs)?;
        let (b, t, c) = logits.dims3()?;
        let loss = cross_entropy(&logits.reshape((b * t, c))?, &ys.reshape(b * t)?)?;
        Ok(loss)
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.embedding_table.forward(xs)
    }
}

impl BigramLanguageModel {
    pub(crate) fn new(
        vocab_size: usize,
        vs: VarBuilder,
    ) -> candle_core::Result<BigramLanguageModel> {
        Ok(BigramLanguageModel {
            embedding_table: embedding(vocab_size, vocab_size, vs)?,
        })
    }
}
