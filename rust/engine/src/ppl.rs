//! Teacher-forced perplexity over a plain text file (llama.cpp-style), plus an
//! optional KL divergence between two models.

use crate::rust_decoder::RustDecoder;
use crate::tokenizer::TokenizerBundle;
use crate::GgufModel;
use fllm_gguf_probe::MappedGguf;
use std::fs;
use std::path::PathBuf;

pub fn run(args: &[String]) -> Result<(), String> {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("usage: fllm-rs ppl --model M --file text.txt [--ctx 2048] [--compare-to OTHER]");
        return Ok(());
    }
    let mut model_arg: Option<String> = None;
    let mut file: Option<String> = None;
    let mut ctx = 2048usize;
    let mut compare: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                model_arg = Some(args.get(i).cloned().ok_or("missing --model value")?);
            }
            "--file" => {
                i += 1;
                file = Some(args.get(i).cloned().ok_or("missing --file value")?);
            }
            "--ctx" => {
                i += 1;
                ctx = args.get(i).cloned().ok_or("missing --ctx value")?.parse().map_err(|_| "invalid --ctx")?;
            }
            "--compare-to" => {
                i += 1;
                compare = Some(args.get(i).cloned().ok_or("missing --compare-to value")?);
            }
            other if other.starts_with('-') => return Err(format!("unknown option {other}")),
            other => {
                if model_arg.is_none() {
                    model_arg = Some(other.to_owned());
                } else if file.is_none() {
                    file = Some(other.to_owned());
                } else {
                    return Err(format!("unexpected argument {other}"));
                }
            }
        }
        i += 1;
    }
    let model_arg = model_arg.ok_or("usage: fllm-rs ppl --model M --file text.txt")?;
    let file = file.ok_or("usage: fllm-rs ppl --model M --file text.txt")?;
    let text = fs::read_to_string(&file).map_err(|e| format!("cannot read {file}: {e}"))?;

    let path = resolve_model(&model_arg)?;
    let scores_main = score_model(&path, ctx, 4, &text)?;
    let report = |label: &str, scores: &[TokenScore]| {
        let n = scores.len().max(1) as f64;
        let nll = scores.iter().map(|s| s.nll as f64).sum::<f64>() / n;
        println!("{label}: tokens={} avg_nll={:.6} ppl={:.4}", scores.len(), nll, nll.exp());
    };
    report("model", &scores_main);

    if let Some(other) = compare {
        let path2 = resolve_model(&other)?;
        let scores_other = score_model(&path2, ctx, 4, &text)?;
        // KL(main || other), aligned by token index.
        let mut kl = 0.0f64;
        for (a, b) in scores_main.iter().zip(&scores_other) {
            for (pa, pb) in a.dist.iter().zip(&b.dist) {
                if *pa > 0.0 {
                    kl += (*pa as f64) * ((*pa as f64).ln() - (*pb as f64 + 1e-12).ln());
                }
            }
        }
        let n = scores_main.len().max(1) as f64;
        println!("kl_divergence(main||compare)={:.6} per_token={:.6}", kl, kl / n);
        report("compare", &scores_other);
    }
    Ok(())
}

struct TokenScore {
    nll: f32,
    dist: Vec<f32>,
}

fn resolve_model(arg: &str) -> Result<PathBuf, String> {
    let reference = crate::hub::normalize_reference(arg);
    if reference.starts_with("hf://") {
        crate::hub::resolve(&reference, None, false)
    } else {
        Ok(PathBuf::from(arg))
    }
}

fn score_model(model_path: &PathBuf, ctx: usize, threads: usize, text: &str) -> Result<Vec<TokenScore>, String> {
    let mapping = MappedGguf::open(model_path).map_err(|e| format!("{}", e))?;
    let tokenizer = TokenizerBundle::from_gguf(&mapping)?;
    let model = GgufModel::load(model_path.to_str().ok_or("invalid path")?, ctx, 1, threads as i32)?;
    let decoder = RustDecoder::load(&model, "cpu")?;
    let mut key = vec![0u16; model.key_buffer_len()?];
    let mut value = vec![0u16; model.value_buffer_len()?];
    let ids = tokenizer.encode(text)?;
    let all_tokens: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    let vocab = tokenizer.vocab_size();
    let bs = model.block_size();
    let stride = model.table_stride();
    let usable_ctx = ctx.min(model.max_context()).max(1);
    let n = all_tokens.len().min(usable_ctx);
    let tokens = &all_tokens[..n];
    if tokens.len() < 2 { return Ok(Vec::new()); }
    let mut tables = vec![0i64; tokens.len() * stride];
    let block_count = tokens.len().div_ceil(bs);
    let blocks = block_count.min(model.pool_blocks());
    let mut row_table = vec![0i64; stride];
    for (b, slot) in row_table[..blocks].iter_mut().enumerate() { *slot = b as i64; }
    for r in 0..tokens.len() {
        tables[r*stride..(r+1)*stride].copy_from_slice(&row_table);
    }
    let lengths: Vec<i64> = (1..=tokens.len()).map(|x| x as i64).collect();
    let ids: Vec<i64> = tokens.to_vec();
    let logit_rows: Vec<usize> = (0..tokens.len()).collect();
    let logits = decoder.forward(&ids, &lengths, &tables, &mut key, &mut value, &logit_rows, false)?;
    let mut out = Vec::new();
    for t in 0..(tokens.len() - 1) {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let target = tokens[t + 1] as usize;
        if target >= vocab { return Err("target token out of vocabulary".into()); }
        let mut maxv = f32::NEG_INFINITY;
        for &v in row { if v > maxv { maxv = v; } }
        let mut sum = 0.0f32;
        let mut dist = vec![0.0f32; vocab];
        for (i, &v) in row.iter().enumerate() {
            let e = (v - maxv).exp();
            dist[i] = e;
            sum += e;
        }
        for v in &mut dist { *v /= sum; }
        let p = dist[target].max(1e-12);
        let nll = -p.ln();
        out.push(TokenScore { nll, dist });
    }
    Ok(out)
}

