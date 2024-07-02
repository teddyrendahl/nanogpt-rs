mod data;
mod models;

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use data::{Dataset, Tokenizer};
use itertools::Itertools;
use models::{estimate_loss, BigramLanguageModel};

const DATA_PATH: &str = "data/tinyshakespeare.txt";
const BATCH_SIZE: usize = 32;
const BLOCK_SIZE: usize = 8;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    // Load data
    let source = std::fs::read_to_string(DATA_PATH)?;
    let tokenizer = Tokenizer::from(source.as_str());
    let dataset = Dataset::new(
        Tensor::from_iter(source.chars().map(|c| tokenizer.encode(&c)), &device)?,
        device.clone(),
    )?;
    let (train, val) = dataset.train_test_split(0.9)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let model = BigramLanguageModel::new(tokenizer.vocab_size(), vs)?;

    let mut optimizer = AdamW::new_lr(varmap.all_vars(), 1e-2)?;
    let mut rng = rand::thread_rng();

    for epoch in 0..3000 {
        let (xb, yb) = train.batch(&mut rng, BATCH_SIZE, BLOCK_SIZE)?;
        let loss = model.train(&xb, &yb)?;
        optimizer.backward_step(&loss)?;

        if epoch % 300 == 0 {
            println!(
                "Epoch: {}, Training Loss: {:.4}, Validation Loss: {:.4}",
                epoch,
                estimate_loss(&model, &train, &mut rng, BATCH_SIZE, BLOCK_SIZE, 200)?,
                estimate_loss(&model, &val, &mut rng, BATCH_SIZE, BLOCK_SIZE, 200)?
            )
        }
    }

    let output = model.generate(
        Tensor::zeros((1, 1), DType::I64, &device)?,
        500,
        &device,
        &mut rng,
    )?;
    println!(
        "{}",
        output.to_vec2::<i64>()?[0]
            .iter()
            .map(|i| tokenizer.decode(i))
            .join("")
    );
    Ok(())
}
