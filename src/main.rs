mod data;
mod models;

use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, VarBuilder, VarMap};
use clap::Parser;
use data::{Dataset, Tokenizer};
use itertools::Itertools;
use models::model::{estimate_loss, generate, Model, ModelVariants};

const TRAIN_TEST_SPLIT: f32 = 0.9;
const LOSS_EVAL_ITERATIONS: usize = 200;

#[derive(Parser, Debug)]
struct Args {
    // Path to the input file to use as training data
    #[arg(long)]
    input_file: String,
    #[arg(long, default_value_t = 32)]
    batch_size: usize,
    #[arg(long, default_value_t = 5000)]
    training_iterations: usize,
    #[arg(long, default_value_t = 1e-3)]
    learning_rate: f64,
    #[arg(long, default_value_t = 500)]
    generate_text_token_length: usize,
    // Choice of model variant to use
    #[clap(subcommand)]
    model_type: ModelVariants,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    // Load data
    let source = std::fs::read_to_string(args.input_file)?;
    let tokenizer = Tokenizer::from(source.as_str());
    let dataset = Dataset::new(
        Tensor::from_iter(source.chars().map(|c| tokenizer.encode(&c)), &device)?,
        device.clone(),
    )?;
    let (train, val) = dataset.train_test_split(TRAIN_TEST_SPLIT)?;
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F64, &device);
    let model = args
        .model_type
        .to_model(&device, vs, tokenizer.vocab_size())?;
    let mut optimizer = AdamW::new_lr(varmap.all_vars(), args.learning_rate)?;
    let mut rng = rand::thread_rng();

    for epoch in 0..args.training_iterations {
        let (xb, yb) = train.batch(&mut rng, args.batch_size, model.block_size())?;
        let loss = model.train(&xb, &yb)?;
        optimizer.backward_step(&loss)?;

        if epoch % 500 == 0 {
            println!(
                "Epoch: {}, Training Loss: {:.4}, Validation Loss: {:.4}",
                epoch,
                estimate_loss(
                    &model,
                    &train,
                    &mut rng,
                    args.batch_size,
                    LOSS_EVAL_ITERATIONS
                )?,
                estimate_loss(
                    &model,
                    &val,
                    &mut rng,
                    args.batch_size,
                    LOSS_EVAL_ITERATIONS
                )?
            )
        }
    }

    let output = generate(
        &model,
        Tensor::zeros((1, 1), DType::I64, &device)?,
        args.generate_text_token_length,
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
