//! Dispatch planning of the element-wise operations of the less common architectures: value clamping, the softplus attention output gate and the
//! data movement of multi-head latent attention (kernels `arch_clamp`, `arch_softplus_mul`, `arch_mla_latent`, `arch_mla_assemble` in
//! `rust/engine/metal/arch_ops.metal`). The conventions are those of [`super::moe_plan`]: constants struct at buffer 0, buffer slots from index 1,
//! the accesses declared for the hazard tracking; the unit tests compare every plan with the `// FLLM-KERNEL:` headers of the source.
//!
//! Level of verification: **L0** for the plans; the kernels are executed on the CPU by `metal/emu_arch` (see docs/changes/metal.md).

use super::moe_plan::{groups_1d, invalid, plan, Consts, Slots, ELEMENTWISE_THREADS};
use super::OpPlan;
use crate::backend::{BackendError, MlaShape};

fn grid(items: usize) -> [u32; 3] {
    groups_1d(items, ELEMENTWISE_THREADS as usize)
}

/// `Stream::clamp` (`arch_clamp`): `x = clamp(x, -limit, limit)` in place over `count` floats. Slot: `x` (read-write).
pub fn plan_clamp(count: usize, limit: f32) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("clamp of an empty tensor"));
    }
    plan(
        "arch_clamp",
        Consts::new().u("count", count).f("limit", limit),
        grid(count),
        [ELEMENTWISE_THREADS, 1, 1],
        Slots::new().rw("x"),
    )
}

/// `Stream::softplus_gate_mul` (`arch_softplus_mul`): `x *= softplus(beta * gate) / beta` in place over `count` floats. Slots: `x` (read-write), `gate`.
pub fn plan_softplus_gate_mul(count: usize, beta: f32) -> Result<OpPlan, BackendError> {
    if count == 0 {
        return Err(invalid("softplus_gate_mul of an empty tensor"));
    }
    if !beta.is_finite() || beta == 0.0 {
        return Err(invalid(format!("softplus_gate_mul with beta = {beta}")));
    }
    plan(
        "arch_softplus_mul",
        Consts::new().u("count", count).f("beta", beta),
        grid(count),
        [ELEMENTWISE_THREADS, 1, 1],
        Slots::new().rw("x").read("gate"),
    )
}

/// `Stream::mla_latent` (`arch_mla_latent`): the first `kv_lora` floats of each of `rows` rows of `kv_a` (rows of `width` floats). Slots: `kv_a`, `latent`.
pub fn plan_mla_latent(rows: usize, kv_lora: usize, width: usize) -> Result<OpPlan, BackendError> {
    if rows == 0 || kv_lora == 0 || width < kv_lora {
        return Err(invalid(format!("mla_latent of {rows} rows, kv_lora {kv_lora}, width {width}")));
    }
    let total = rows
        .checked_mul(kv_lora)
        .ok_or_else(|| invalid("mla_latent size overflows"))?;
    plan(
        "arch_mla_latent",
        Consts::new().u("total", total).u("kv_lora", kv_lora).u("width", width),
        grid(total),
        [ELEMENTWISE_THREADS, 1, 1],
        Slots::new().read("kv_a").write("latent"),
    )
}

/// `Stream::mla_assemble` (`arch_mla_assemble`): per-head queries `[rotary | content]`, keys `[shared rotary | content]` and values of the expanded
/// cache from the projections. Slots: `q_full`, `kv_a`, `kv`, `q`, `k`, `v`.
pub fn plan_mla_assemble(rows: usize, shape: &MlaShape) -> Result<OpPlan, BackendError> {
    let MlaShape { heads, nope, rope, value, kv_lora } = *shape;
    if rows == 0 || heads == 0 || nope + rope == 0 || value == 0 {
        return Err(invalid(format!("mla_assemble of {rows} rows with {shape:?}")));
    }
    let pairs = rows
        .checked_mul(heads)
        .ok_or_else(|| invalid("mla_assemble size overflows"))?;
    let total_qk = pairs
        .checked_mul(nope + rope)
        .ok_or_else(|| invalid("mla_assemble size overflows"))?;
    let total_v = pairs
        .checked_mul(value)
        .ok_or_else(|| invalid("mla_assemble size overflows"))?;
    plan(
        "arch_mla_assemble",
        Consts::new()
            .u("rows", rows)
            .u("heads", heads)
            .u("nope", nope)
            .u("rope", rope)
            .u("value", value)
            .u("kv_lora", kv_lora)
            .u("total_qk", total_qk)
            .u("total_v", total_v),
        grid(total_qk.max(total_v)),
        [ELEMENTWISE_THREADS, 1, 1],
        Slots::new()
            .read("q_full")
            .read("kv_a")
            .read("kv")
            .write("q")
            .write("k")
            .write("v"),
    )
}

#[cfg(test)]
mod tests {
    use super::super::moe_plan::testkit::check_plan;
    use super::*;

    fn word(p: &OpPlan, i: usize) -> u32 {
        u32::from_le_bytes(p.constants[4 * i..4 * i + 4].try_into().unwrap())
    }

    #[test]
    fn plans_match_their_headers() {
        let p = plan_clamp(1000, 6.0).unwrap();
        check_plan(&p, &[]);
        assert_eq!((word(&p, 0), f32::from_bits(word(&p, 1))), (1000, 6.0));
        assert_eq!(p.threadgroups, [4, 1, 1]);

        let p = plan_softplus_gate_mul(4096, 2.0).unwrap();
        check_plan(&p, &[]);
        assert_eq!(f32::from_bits(word(&p, 1)), 2.0);

        let p = plan_mla_latent(3, 512, 576).unwrap();
        check_plan(&p, &[]);
        assert_eq!((word(&p, 0), word(&p, 1), word(&p, 2)), (3 * 512, 512, 576));

        let shape = MlaShape { heads: 16, nope: 128, rope: 64, value: 128, kv_lora: 512 };
        let p = plan_mla_assemble(5, &shape).unwrap();
        check_plan(&p, &[]);
        assert_eq!(word(&p, 6), 5 * 16 * 192);
        assert_eq!(word(&p, 7), 5 * 16 * 128);
        assert_eq!(p.threadgroups[0] as usize, (5 * 16 * 192usize).div_ceil(256));
    }

    #[test]
    fn invalid_arguments_are_refused() {
        assert!(plan_clamp(0, 1.0).is_err());
        assert!(plan_softplus_gate_mul(8, 0.0).is_err());
        assert!(plan_softplus_gate_mul(8, f32::NAN).is_err());
        assert!(plan_mla_latent(2, 64, 32).is_err());
        let shape = MlaShape { heads: 0, nope: 1, rope: 1, value: 1, kv_lora: 1 };
        assert!(plan_mla_assemble(1, &shape).is_err());
    }

    #[test]
    fn very_large_batches_are_capped_at_the_grid_limit() {
        let p = plan_clamp(1 << 28, 1.0).unwrap();
        assert_eq!(p.threadgroups, [65535, 1, 1]);
    }
}
