//! Rotary embedding tables: plain, linearly scaled, YaRN and per-frequency factors.
use crate::config::{RopeConfig, RopeScaling};

pub struct RopeTables {
    /// Rotating pairs per head (`dims / 2`).
    pub half: usize,
    /// `positions x half`
    pub cos: Vec<f32>,
    /// `positions x half`
    pub sin: Vec<f32>,
}

/// YaRN correction dimension: the frequency index whose wavelength fits `rotations` times into the
/// original context.
fn correction_dim(dims: usize, original: usize, rotations: f32, base: f32) -> f32 {
    dims as f32 * (original as f32 / (rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn ramp(low: f32, high: f32, lane: usize) -> f32 {
    let y = (lane as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// Tables for `positions` token positions using `base` (the global or the sliding-window base).
/// Without scaling the phase is formed in f32 and the trigonometry runs in f64, matching the
/// reference implementation bit for bit on the models checked against it.
pub fn build(rope: &RopeConfig, base: f32, positions: usize) -> RopeTables {
    let half = rope.dims / 2;
    let long = match &rope.scaling {
        RopeScaling::LongRope {
            original_context,
            long,
            ..
        } if positions > *original_context => Some(long),
        _ => None,
    };
    let inverse: Vec<f32> = (0..half)
        .map(|lane| {
            let frequency = 1.0f32 / base.powf((lane * 2) as f32 / rope.dims as f32);
            match long.or(rope.factors.as_ref()) {
                Some(factors) => frequency / factors[lane],
                None => frequency,
            }
        })
        .collect();
    let (mut cos, mut sin) = (
        Vec::with_capacity(positions * half),
        Vec::with_capacity(positions * half),
    );
    match &rope.scaling {
        RopeScaling::None | RopeScaling::Linear { .. } | RopeScaling::LongRope { .. } => {
            let scale = match rope.scaling {
                RopeScaling::Linear { factor } => 1.0 / factor,
                _ => 1.0,
            };
            let magnitude = match (&rope.scaling, long) {
                (RopeScaling::LongRope { attention, .. }, _) => *attention,
                _ => 1.0,
            };
            for position in 0..positions {
                let position = position as f32 * scale;
                for &inverse in &inverse {
                    let phase = (position * inverse) as f64;
                    cos.push(phase.cos() as f32 * magnitude);
                    sin.push(phase.sin() as f32 * magnitude);
                }
            }
        }
        RopeScaling::Yarn {
            factor,
            original_context,
            extrapolation,
            attention,
            beta_fast,
            beta_slow,
            truncate,
        } => {
            let scale = 1.0 / factor;
            let (mut low, mut high) = (
                correction_dim(rope.dims, *original_context, *beta_fast, base),
                correction_dim(rope.dims, *original_context, *beta_slow, base),
            );
            if *truncate {
                low = low.floor();
                high = high.ceil();
            }
            let (low, high) = (low.max(0.0), high.min(rope.dims as f32 - 1.0));
            let magnitude = if *extrapolation != 0.0 {
                attention * (1.0 + 0.1 * (1.0 / scale).ln())
            } else {
                *attention
            };
            for position in 0..positions {
                for (lane, &inverse) in inverse.iter().enumerate() {
                    let extrapolated = position as f32 * inverse;
                    let interpolated = scale * extrapolated;
                    let mix = ramp(low, high, lane) * extrapolation;
                    let theta = if *extrapolation != 0.0 {
                        interpolated * (1.0 - mix) + extrapolated * mix
                    } else {
                        interpolated
                    };
                    let theta = theta as f64;
                    cos.push((theta.cos() as f32) * magnitude);
                    sin.push((theta.sin() as f32) * magnitude);
                }
            }
        }
    }
    if rope.flipped {
        sin.iter_mut().for_each(|v| *v = -*v);
    }
    RopeTables { half, cos, sin }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(scaling: RopeScaling) -> RopeConfig {
        RopeConfig {
            dims: 8,
            base: 10_000.0,
            local_base: None,
            local_dims: None,
            neox: true,
            scaling,
            factors: None,
            flipped: false,
        }
    }

    #[test]
    fn plain_tables_start_at_the_identity_and_follow_the_base() {
        let t = build(&config(RopeScaling::None), 10_000.0, 3);
        assert_eq!((t.half, t.cos.len()), (4, 12));
        assert_eq!(&t.cos[..4], &[1.0; 4]);
        assert!((t.sin[4] - 1.0f32.sin()).abs() < 1e-7);
        // lane 1 uses 10000^(-2/8) = 0.1
        assert!((t.sin[5] - 0.1f32.sin()).abs() < 1e-7);
    }

    #[test]
    fn linear_scaling_divides_the_position() {
        let plain = build(&config(RopeScaling::None), 10_000.0, 9);
        let linear = build(&config(RopeScaling::Linear { factor: 4.0 }), 10_000.0, 9);
        for lane in 0..4 {
            assert!((linear.sin[8 * 4 + lane] - plain.sin[2 * 4 + lane]).abs() < 1e-6);
        }
    }

    #[test]
    fn yarn_keeps_high_frequencies_and_interpolates_low_ones() {
        let yarn = RopeScaling::Yarn {
            factor: 4.0,
            original_context: 64,
            extrapolation: 1.0,
            attention: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            truncate: true,
        };
        let plain = build(&config(RopeScaling::None), 10_000.0, 40);
        let scaled = build(&config(yarn), 10_000.0, 40);
        let magnitude = 1.0 + 0.1 * 4.0f32.ln();
        // the fastest rotating lane is extrapolated: same angle, only the magnitude changes
        let angle = |t: &RopeTables, p: usize, l: usize| t.sin[p * 4 + l].atan2(t.cos[p * 4 + l]);
        assert!((angle(&scaled, 30, 0) - angle(&plain, 30, 0)).abs() < 1e-5);
        let norm = scaled.cos[30 * 4].hypot(scaled.sin[30 * 4]);
        assert!((norm - magnitude).abs() < 1e-5);
        // the slowest lane is interpolated: a quarter of the angle
        assert!((angle(&scaled, 30, 3) - angle(&plain, 30, 3) / 4.0).abs() < 1e-5);
    }
}
