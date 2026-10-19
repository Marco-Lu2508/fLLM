//! Type of the activations on the device (`FLLM_ACT_DTYPE`).
//!
//! The CUDA backend can keep the residual stream, the norm outputs, the attention result and the feed-forward intermediates in FP16 or BF16 instead of
//! FP32 (docs/activation-dtype.md): no conversion kernels around the Marlin GEMMs, half the bytes for every elementwise kernel. Sums, residual adds,
//! rotations and the gated activation still run in FP32 registers; only the stored values are narrow.
//!
//! `FLLM_ACT_DTYPE=f32|f16|bf16|auto` (default `f32`, the original pipeline, bit-identical to earlier versions). `auto` picks BF16 for models whose
//! float tensors are BF16 in the file and for families whose residual stream is known to exceed the FP16 range (Gemma), FP16 otherwise. A model that
//! cannot run in a narrow type (see [`eligibility`]) stays FP32 whatever was asked, with a line in the log.

use std::fmt;

/// The activation type of a device pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ActDtype {
    #[default]
    F32,
    F16,
    Bf16,
}

impl ActDtype {
    /// Code used by the C API of the CUDA library: 0 FP32, 1 FP16, 2 BF16.
    pub fn code(self) -> i32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Bf16 => 2,
        }
    }

    /// Bytes of one value.
    pub fn bytes(self) -> usize {
        if self == Self::F32 {
            4
        } else {
            2
        }
    }

    pub fn is_narrow(self) -> bool {
        self != Self::F32
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

impl fmt::Display for ActDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What `FLLM_ACT_DTYPE` asked for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActRequest {
    Fixed(ActDtype),
    Auto,
}

/// Parse the value of `FLLM_ACT_DTYPE` (`None` = unset = FP32).
pub fn parse_request(value: Option<&str>) -> Result<ActRequest, String> {
    match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
        None | Some("") | Some("f32") | Some("fp32") | Some("float32") | Some("float") => Ok(ActRequest::Fixed(ActDtype::F32)),
        Some("f16") | Some("fp16") | Some("float16") | Some("half") => Ok(ActRequest::Fixed(ActDtype::F16)),
        Some("bf16") | Some("bfloat16") => Ok(ActRequest::Fixed(ActDtype::Bf16)),
        Some("auto") => Ok(ActRequest::Auto),
        Some(other) => Err(format!("FLLM_ACT_DTYPE={other}: expected f32, f16, bf16 or auto")),
    }
}

/// The request of the environment; an invalid value is reported once and counts as unset.
pub fn request_from_env() -> ActRequest {
    static REQUEST: std::sync::OnceLock<ActRequest> = std::sync::OnceLock::new();
    *REQUEST.get_or_init(|| match parse_request(std::env::var("FLLM_ACT_DTYPE").ok().as_deref()) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[fllm-rs] {e}; using f32");
            ActRequest::Fixed(ActDtype::F32)
        }
    })
}

/// Model families whose residual stream is known to leave the FP16 range (Gemma 2/3 scale the embeddings by sqrt(hidden) and keep large outliers in
/// the residual stream; their reference implementations run in BF16). `auto` never picks FP16 for them.
pub fn overflow_prone(arch: &str) -> bool {
    let a = arch.to_ascii_lowercase();
    a.starts_with("gemma") || a == "vaultgemma"
}

/// The type `auto` picks: BF16 when the file's float tensors are BF16 (the config dtype of the checkpoint was bfloat16) or the family is
/// overflow-prone, FP16 otherwise.
pub fn auto_dtype(arch: &str, source_bf16: bool) -> ActDtype {
    if source_bf16 || overflow_prone(arch) {
        ActDtype::Bf16
    } else {
        ActDtype::F16
    }
}

/// What the decoder tells the policy about the model.
#[derive(Clone, Debug, Default)]
pub struct ModelFacts {
    pub arch: String,
    /// The embedding / output tensors of the file are BF16 (stands in for `torch_dtype: bfloat16`, which the GGUF does not record).
    pub source_bf16: bool,
    /// Why this model cannot keep narrow activations (first reason), if it cannot.
    pub blocker: Option<&'static str>,
    /// The projections are 4-bit AWQ / GPTQ (Marlin) weights: the only format whose GEMMs read and write narrow activations without conversions.
    pub marlin_weights: bool,
}

/// Outcome of the policy: the type to use and, when the request could not be honoured, the reason.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decision {
    pub dtype: ActDtype,
    pub note: Option<String>,
}

/// Apply a request to a model. FP16 forced on an overflow-prone family is honoured but noted (the run-time check `FLLM_ACT_CHECK=1` shows whether it
/// overflowed).
pub fn decide(request: ActRequest, facts: &ModelFacts) -> Decision {
    let wanted = match request {
        ActRequest::Fixed(d) => d,
        ActRequest::Auto => auto_dtype(&facts.arch, facts.source_bf16),
    };
    if !wanted.is_narrow() {
        return Decision { dtype: ActDtype::F32, note: None };
    }
    // `auto` only switches where it pays: the other weight formats widen and narrow around every GEMM
    if request == ActRequest::Auto && !facts.marlin_weights && facts.blocker.is_none() {
        return Decision { dtype: ActDtype::F32, note: Some("FLLM_ACT_DTYPE=auto: the model has no 4-bit Marlin weights, narrow activations would only add conversions; activations stay f32".into()) };
    }
    if let Some(why) = facts.blocker {
        return Decision { dtype: ActDtype::F32, note: Some(format!("FLLM_ACT_DTYPE={wanted} is not used for this model ({why}); activations stay f32")) };
    }
    let note = if wanted == ActDtype::F16 && (overflow_prone(&facts.arch) || facts.source_bf16) {
        Some(format!(
            "FLLM_ACT_DTYPE=f16 on a model that is BF16 or from a family with large residual values ({}): values above 65504 become inf; FLLM_ACT_CHECK=1 reports it, bf16 is the safe choice",
            facts.arch
        ))
    } else {
        None
    };
    Decision { dtype: wanted, note }
}

/// Features of a model the narrow pipeline does not cover (each is a path that has only FP32 kernels). The first one found is the reason.
#[derive(Clone, Copy, Debug, Default)]
pub struct ModelShape {
    pub all_layers_standard_attention: bool,
    pub host_layers: bool,
    pub moe: bool,
    pub mla: bool,
    pub gated_attention: bool,
    pub qkv_clip: bool,
    pub parallel_residual: bool,
    pub layer_norm: bool,
    pub mtp: bool,
    pub recurrent: bool,
}

pub fn eligibility(s: &ModelShape) -> Option<&'static str> {
    if s.host_layers {
        Some("layers that run on the host")
    } else if s.recurrent || !s.all_layers_standard_attention {
        Some("recurrent or non-standard attention layers")
    } else if s.moe {
        Some("mixture of experts")
    } else if s.mla {
        Some("multi-head latent attention")
    } else if s.gated_attention {
        Some("gated attention")
    } else if s.qkv_clip {
        Some("q/k/v clipping")
    } else if s.parallel_residual {
        Some("parallel residual")
    } else if s.layer_norm {
        Some("LayerNorm")
    } else if s.mtp {
        Some("multi-token prediction")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_parse() {
        assert_eq!(parse_request(None).unwrap(), ActRequest::Fixed(ActDtype::F32));
        assert_eq!(parse_request(Some("")).unwrap(), ActRequest::Fixed(ActDtype::F32));
        assert_eq!(parse_request(Some(" F16 ")).unwrap(), ActRequest::Fixed(ActDtype::F16));
        assert_eq!(parse_request(Some("bfloat16")).unwrap(), ActRequest::Fixed(ActDtype::Bf16));
        assert_eq!(parse_request(Some("AUTO")).unwrap(), ActRequest::Auto);
        assert!(parse_request(Some("int8")).is_err());
    }

    #[test]
    fn auto_prefers_bf16_for_bf16_sources_and_gemma() {
        assert_eq!(auto_dtype("llama", false), ActDtype::F16);
        assert_eq!(auto_dtype("qwen2", true), ActDtype::Bf16);
        assert_eq!(auto_dtype("gemma3", false), ActDtype::Bf16);
        assert_eq!(auto_dtype("Gemma2", false), ActDtype::Bf16);
    }

    #[test]
    fn blockers_keep_f32_and_say_why() {
        let facts = ModelFacts { arch: "mixtral".into(), source_bf16: false, blocker: Some("mixture of experts"), marlin_weights: true };
        let d = decide(ActRequest::Fixed(ActDtype::F16), &facts);
        assert_eq!(d.dtype, ActDtype::F32);
        assert!(d.note.unwrap().contains("mixture of experts"));
        // nothing asked, nothing said
        let d = decide(ActRequest::Fixed(ActDtype::F32), &facts);
        assert_eq!((d.dtype, d.note), (ActDtype::F32, None));
    }

    #[test]
    fn forced_f16_on_a_bf16_model_is_honoured_with_a_warning() {
        let facts = ModelFacts { arch: "qwen2".into(), source_bf16: true, blocker: None, marlin_weights: true };
        let d = decide(ActRequest::Fixed(ActDtype::F16), &facts);
        assert_eq!(d.dtype, ActDtype::F16);
        assert!(d.note.is_some());
        let d = decide(ActRequest::Auto, &facts);
        assert_eq!((d.dtype, d.note), (ActDtype::Bf16, None));
    }

    #[test]
    fn auto_stays_f32_without_marlin_weights_but_explicit_requests_are_honoured() {
        let facts = ModelFacts { arch: "llama".into(), source_bf16: false, blocker: None, marlin_weights: false };
        let d = decide(ActRequest::Auto, &facts);
        assert_eq!(d.dtype, ActDtype::F32);
        assert!(d.note.unwrap().contains("Marlin"));
        assert_eq!(decide(ActRequest::Fixed(ActDtype::F16), &facts).dtype, ActDtype::F16);
        let with = ModelFacts { marlin_weights: true, ..facts };
        assert_eq!(decide(ActRequest::Auto, &with).dtype, ActDtype::F16);
    }

    #[test]
    fn eligibility_lists_the_unsupported_shapes() {
        let ok = ModelShape { all_layers_standard_attention: true, ..ModelShape::default() };
        assert_eq!(eligibility(&ok), None);
        assert_eq!(eligibility(&ModelShape { moe: true, ..ok }), Some("mixture of experts"));
        assert_eq!(eligibility(&ModelShape { all_layers_standard_attention: false, ..ok }), Some("recurrent or non-standard attention layers"));
        assert_eq!(eligibility(&ModelShape { layer_norm: true, ..ok }), Some("LayerNorm"));
    }

    #[test]
    fn codes_and_sizes() {
        assert_eq!((ActDtype::F32.code(), ActDtype::F16.code(), ActDtype::Bf16.code()), (0, 1, 2));
        assert_eq!((ActDtype::F32.bytes(), ActDtype::F16.bytes(), ActDtype::Bf16.bytes()), (4, 2, 2));
    }
}
