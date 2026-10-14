//! One validated sampling contract shared by OpenAI, Ollama and SGLang routes.
use crate::sampling::SamplingParams;
use serde_json::Value;

pub fn parse_sampling(
    body: &Value,
    defaults: SamplingParams,
    vocab: usize,
) -> Result<SamplingParams, String> {
    if !body.is_object() {
        return Err("sampling parameters must be an object".into());
    }
    let mut p = defaults;
    let number = |key: &str, default: f64, min: f64, max: f64| -> Result<f64, String> {
        match body.get(key) {
            None => Ok(default),
            Some(value) => value
                .as_f64()
                .filter(|n| n.is_finite() && *n >= min && *n <= max)
                .ok_or_else(|| format!("{key} must be a finite number between {min} and {max}")),
        }
    };
    let integer = |key: &str, default: i64, min: i64, max: i64| -> Result<i64, String> {
        match body.get(key) {
            None => Ok(default),
            Some(value) => value
                .as_i64()
                .filter(|n| *n >= min && *n <= max)
                .ok_or_else(|| format!("{key} must be an integer between {min} and {max}")),
        }
    };
    p.temperature = number("temperature", p.temperature, 0.0, 100.0)?;
    p.top_p = number("top_p", p.top_p, 0.0, 1.0)?;
    p.top_k = integer("top_k", p.top_k as i64, 0, i32::MAX as i64)? as usize;
    p.min_p = number("min_p", p.min_p, 0.0, 1.0)?;
    p.typical_p = number("typical_p", p.typical_p, 0.0, 1.0)?;
    let repeat_key = if body.get("repeat_penalty").is_some() {
        "repeat_penalty"
    } else {
        "repetition_penalty"
    };
    p.repetition_penalty = number(repeat_key, p.repetition_penalty as f64, 0.0001, 100.0)? as f32;
    p.presence_penalty = number("presence_penalty", p.presence_penalty as f64, -2.0, 2.0)? as f32;
    p.frequency_penalty =
        number("frequency_penalty", p.frequency_penalty as f64, -2.0, 2.0)? as f32;
    p.repeat_last_n = integer("repeat_last_n", p.repeat_last_n, -1, i32::MAX as i64)?;
    p.dry_multiplier = number("dry_multiplier", p.dry_multiplier as f64, 0.0, 100.0)? as f32;
    p.dry_base = number("dry_base", p.dry_base as f64, 1.0, 10.0)? as f32;
    p.dry_allowed_length =
        integer("dry_allowed_length", p.dry_allowed_length as i64, 0, 1024)? as usize;
    p.dry_penalty_last_n = integer(
        "dry_penalty_last_n",
        p.dry_penalty_last_n,
        -1,
        i32::MAX as i64,
    )?;
    p.mirostat = integer("mirostat", p.mirostat as i64, 0, 2)? as u8;
    p.mirostat_tau = number("mirostat_tau", p.mirostat_tau as f64, 0.0001, 100.0)? as f32;
    p.mirostat_eta = number("mirostat_eta", p.mirostat_eta as f64, 0.0, 1.0)? as f32;
    if let Some(seed) = body.get("seed") {
        p.seed = if seed.as_i64() == Some(-1) {
            None
        } else {
            Some(
                seed.as_u64()
                    .ok_or("seed must be a non-negative integer or -1")?,
            )
        };
    }
    if let Some(bias) = body.get("logit_bias") {
        let bias = bias
            .as_object()
            .ok_or("logit_bias must be an object keyed by token ID")?;
        for (id, value) in bias {
            let id: usize = id
                .parse()
                .map_err(|_| "logit_bias keys must be token IDs")?;
            if id >= vocab {
                return Err("logit_bias token ID is outside the vocabulary".into());
            }
            let value = value
                .as_f64()
                .filter(|n| n.is_finite() && n.abs() <= 100.0)
                .ok_or("logit_bias values must be finite and between -100 and 100")?;
            p.logit_bias.push((id, value as f32));
        }
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn reject_invalid_parameters_instead_of_ignoring_them() {
        for value in [
            json!({"min_p":1.01}),
            json!({"temperature":"hot"}),
            json!({"repeat_penalty":0}),
            json!({"top_k":-1}),
            json!({"mirostat":3}),
            json!({"logit_bias":{"9":1}}),
        ] {
            assert!(
                parse_sampling(&value, SamplingParams::default(), 8).is_err(),
                "{value}"
            );
        }
    }
    #[test]
    fn ollama_aliases_seed_and_bias_are_preserved() {
        let p = parse_sampling(
            &json!({"repeat_penalty":1.2,"min_p":0.05,"seed":17,"logit_bias":{"2":-9}}),
            SamplingParams::default(),
            8,
        )
        .unwrap();
        assert_eq!(p.seed, Some(17));
        assert_eq!(p.min_p, 0.05);
        assert_eq!(p.logit_bias, vec![(2, -9.0)]);
        assert!(!p.is_greedy());
    }
}
