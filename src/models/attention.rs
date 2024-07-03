use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{
    embedding, linear, linear_no_bias, loss::cross_entropy, ops::softmax, Embedding, Linear,
    Module, VarBuilder,
};
use clap::Args;

use super::model::Model;

struct AttentionHead {
    key: Linear,
    query: Linear,
    value: Linear,
    tril: Tensor,
    neg_inf: Tensor,
}

impl AttentionHead {
    pub(crate) fn new(
        embedding_size: usize,
        block_size: usize,
        head_size: usize,
        vb: VarBuilder,
        device: &Device,
    ) -> candle_core::Result<Self> {
        Ok(Self {
            key: linear_no_bias(embedding_size, head_size, vb.pp("key"))?,
            query: linear_no_bias(embedding_size, head_size, vb.pp("query"))?,
            value: linear_no_bias(embedding_size, head_size, vb.pp("value"))?,
            tril: Tensor::tril2(block_size, DType::U32, device)?,
            neg_inf: Tensor::try_from(f64::NEG_INFINITY)?.to_device(device)?,
        })
    }
}
impl Module for AttentionHead {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (_, t, c) = xs.dims3()?;
        let k = self.key.forward(xs)?;
        let q = self.query.forward(xs)?;
        let mut wei = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * (c as f64).powf(-0.5))?;
        wei = softmax(
            &self
                .tril
                .i((..t, ..t))?
                .broadcast_as(wei.shape())?
                .where_cond(&wei, &self.neg_inf.broadcast_as(wei.shape())?)?,
            D::Minus1,
        )?; // (B, T

        let v = self.value.forward(xs)?;
        wei.matmul(&v)
    }
}
#[derive(Debug, Clone, Args)]
pub(crate) struct AttentionModelConfiguration {
    #[clap(short, long, default_value_t = 8)]
    block_size: usize,
    #[clap(short, long, default_value_t = 32)]
    embedding_size: usize,
}

pub(crate) struct SelfAttentionModel {
    token_embedding: Embedding,
    position_embedding: Embedding,
    attention: AttentionHead,
    head: Linear,
    block_size: usize,
}

impl SelfAttentionModel {
    pub(crate) fn new(
        config: &AttentionModelConfiguration,
        device: &Device,
        vs: VarBuilder,
        vocab_size: usize,
    ) -> candle_core::Result<SelfAttentionModel> {
        Ok(SelfAttentionModel {
            token_embedding: embedding(vocab_size, config.embedding_size, vs.pp("token_embed"))?,
            position_embedding: embedding(
                config.block_size,
                config.embedding_size,
                vs.pp("position_embed"),
            )?,
            attention: AttentionHead::new(
                config.embedding_size,
                config.block_size,
                config.embedding_size,
                vs.clone(),
                device,
            )?,
            head: linear(config.embedding_size, vocab_size, vs.pp("head"))?,
            block_size: config.block_size,
        })
    }
}

impl Model for SelfAttentionModel {
    fn train(&self, xs: &Tensor, ys: &Tensor) -> candle_core::Result<Tensor> {
        let logits = self.forward(xs)?;
        let (b, t, c) = logits.dims3()?;
        let loss = cross_entropy(&logits.reshape((b * t, c))?, &ys.reshape(b * t)?)?;
        Ok(loss)
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (_, t) = xs.dims2()?;
        let tok_emb = self.token_embedding.forward(xs)?;
        let pos_emb =
            self.position_embedding
                .forward(&Tensor::arange(0, t as i64, &Device::Cpu)?)?;
        self.head.forward(
            &self
                .attention
                .forward(&(tok_emb.broadcast_add(&pos_emb))?)?,
        )
    }

    fn block_size(&self) -> usize {
        self.block_size
    }
}
