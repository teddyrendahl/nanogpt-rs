use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder, VarMap};
use clap::Subcommand;
use itertools::Itertools;
use rand::Rng;
use rand_distr::{Distribution, WeightedIndex};

use crate::data::{Dataset, Tokenizer};

use super::{
    attention::{AttentionModelConfiguration, SelfAttentionModel},
    bigram::BigramLanguageModel,
};

pub(crate) trait Model {
    fn block_size(&self) -> usize;
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor>;
    fn train(&self, xs: &Tensor, ys: &Tensor) -> candle_core::Result<Tensor>;
}

/// Estimate the loss of the model over a number of randomly selected batches
pub(crate) fn estimate_loss(
    model: &impl Model,
    data: &Dataset,
    rng: &mut impl Rng,
    batch_size: usize,
    eval_iters: usize,
) -> candle_core::Result<f64> {
    let mut v = 0.0;
    for _ in 0..eval_iters {
        let (x, y) = data.batch(rng, batch_size, model.block_size())?;
        v += model.train(&x, &y)?.to_scalar::<f64>()?;
    }
    Ok(v / (eval_iters as f64))
}

impl<M: Model + ?Sized> Model for Box<M> {
    fn block_size(&self) -> usize {
        (**self).block_size()
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        (**self).forward(xs)
    }

    fn train(&self, xs: &Tensor, ys: &Tensor) -> candle_core::Result<Tensor> {
        (**self).train(xs, ys)
    }
}

/// Generate a new value by repeatedly sampling the provided Model
pub(crate) fn generate<M: Model>(
    model: &M,
    tokenizer: &Tokenizer,
    max_new_tokens: usize,
    device: &Device,
    rng: &mut impl Rng,
) -> candle_core::Result<String> {
    // TODO: Allow for seeding with custom chars
    let mut idx = Tensor::zeros((1, 1), DType::I64, device)?;
    for _ in 0..max_new_tokens {
        let xs = idx.i((.., idx.dims2()?.1.saturating_sub(model.block_size())..))?;
        let logits = model.forward(&xs)?;
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
    }
    // Decode our data into a String
    Ok(idx.to_vec2::<i64>()?[0]
        .iter()
        .map(|i| tokenizer.decode(i))
        .join(""))
}

#[derive(Clone, Debug, Subcommand)]
pub(crate) enum ModelVariants {
    Bigram,
    Attention(AttentionModelConfiguration),
}

impl ModelVariants {
    pub(crate) fn to_model(
        &self,
        device: &Device,
        vm: &VarMap,
        vocab_size: usize,
    ) -> candle_core::Result<Box<dyn Model>> {
        let vs = VarBuilder::from_varmap(vm, DType::F64, device);
        Ok(match &self {
            ModelVariants::Bigram => Box::new(BigramLanguageModel::new(vocab_size, vs)?),
            ModelVariants::Attention(c) => {
                Box::new(SelfAttentionModel::new(c, device, vs, vocab_size)?)
            }
        })
    }
}
