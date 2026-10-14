//! Row quantizers for the GGML block formats, following the reference algorithms of ggml
//! (`quantize_row_*_ref`) so that files produced here are interchangeable with llama.cpp's.
use half::f16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl QType {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name.to_ascii_lowercase().as_str() {
            "q4_0" => Self::Q4_0,
            "q4_1" => Self::Q4_1,
            "q5_0" => Self::Q5_0,
            "q5_1" => Self::Q5_1,
            "q8_0" => Self::Q8_0,
            "q4_k" => Self::Q4K,
            "q5_k" => Self::Q5K,
            "q6_k" => Self::Q6K,
            _ => return None,
        })
    }
    /// GGML type id as stored in GGUF tensor infos.
    pub fn ggml_id(self) -> u32 {
        match self {
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
        }
    }
    pub fn block_elements(self) -> usize {
        match self {
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
            _ => 32,
        }
    }
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q4_0 => 18,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_0 => 34,
            Self::Q4K => 144,
            Self::Q5K => 176,
            Self::Q6K => 210,
        }
    }
    /// Bytes of a row with `elements` values.
    pub fn row_bytes(self, elements: usize) -> usize {
        elements / self.block_elements() * self.block_bytes()
    }
}

fn nearest_int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

fn push_f16(out: &mut Vec<u8>, v: f32) {
    out.extend_from_slice(&f16::from_f32(v).to_bits().to_le_bytes());
}

/// Quantize `x` (a multiple of the block size) and append the blocks to `out`.
pub fn quantize_row(kind: QType, x: &[f32], out: &mut Vec<u8>) {
    assert!(
        x.len().is_multiple_of(kind.block_elements()),
        "row length is not a multiple of the block size"
    );
    for block in x.chunks_exact(kind.block_elements()) {
        match kind {
            QType::Q4_0 => q4_0(block, out),
            QType::Q4_1 => q4_1(block, out),
            QType::Q5_0 => q5_0(block, out),
            QType::Q5_1 => q5_1(block, out),
            QType::Q8_0 => q8_0(block, out),
            QType::Q4K => q45_k(block, out, false),
            QType::Q5K => q45_k(block, out, true),
            QType::Q6K => q6_k(block, out),
        }
    }
}

fn q8_0(x: &[f32], out: &mut Vec<u8>) {
    let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    out.extend(x.iter().map(|v| (v * id).round() as i8 as u8));
}

/// Value with the largest magnitude, sign kept.
fn signed_max(x: &[f32]) -> f32 {
    let mut amax = 0.0f32;
    let mut max = 0.0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    max
}

fn q4_0(x: &[f32], out: &mut Vec<u8>) {
    let max = signed_max(x);
    let d = max / -8.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    let q = |v: f32| ((v * id + 8.5) as i8).min(15) as u8;
    out.extend((0..16).map(|j| q(x[j]) | (q(x[j + 16]) << 4)));
}

fn min_max(x: &[f32]) -> (f32, f32) {
    x.iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

fn q4_1(x: &[f32], out: &mut Vec<u8>) {
    let (min, max) = min_max(x);
    let d = (max - min) / 15.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    push_f16(out, min);
    let q = |v: f32| (((v - min) * id + 0.5) as i8).min(15) as u8;
    out.extend((0..16).map(|j| q(x[j]) | (q(x[j + 16]) << 4)));
}

fn q5_0(x: &[f32], out: &mut Vec<u8>) {
    let max = signed_max(x);
    let d = max / -16.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    let q = |v: f32| ((v * id + 16.5) as i8).min(31) as u8;
    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
        let (a, b) = (q(x[j]), q(x[j + 16]));
        qs[j] = (a & 0xF) | ((b & 0xF) << 4);
        qh |= u32::from((a & 0x10) >> 4) << j;
        qh |= u32::from((b & 0x10) >> 4) << (j + 16);
    }
    out.extend_from_slice(&qh.to_le_bytes());
    out.extend_from_slice(&qs);
}

fn q5_1(x: &[f32], out: &mut Vec<u8>) {
    let (min, max) = min_max(x);
    let d = (max - min) / 31.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    push_f16(out, d);
    push_f16(out, min);
    let q = |v: f32| ((v - min) * id + 0.5) as u8;
    let mut qh = 0u32;
    let mut qs = [0u8; 16];
    for j in 0..16 {
        let (a, b) = (q(x[j]), q(x[j + 16]));
        qs[j] = (a & 0xF) | ((b & 0xF) << 4);
        qh |= u32::from((a & 0x10) >> 4) << j;
        qh |= u32::from((b & 0x10) >> 4) << (j + 16);
    }
    out.extend_from_slice(&qh.to_le_bytes());
    out.extend_from_slice(&qs);
}

/// Symmetric search for the best scale (ggml `make_qx_quants`, RMSE type 1). Returns the scale and
/// fills `l` with values in `0..2*nmax`.
fn make_qx_quants(nmax: i32, x: &[f32], l: &mut [u8]) -> f32 {
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < 1e-15 {
        l.fill(0);
        return 0.0;
    }
    let nm = nmax as f32;
    let clamp = |v: i32| v.clamp(-nmax, nmax - 1);
    let mut iscale = -nm / max;
    let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
    for (i, &v) in x.iter().enumerate() {
        let q = clamp(nearest_int(iscale * v));
        l[i] = (q + nmax) as u8;
        let w = v * v;
        sumlx += w * v * q as f32;
        suml2 += w * (q * q) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nm + 0.1 * is as f32) / max;
        let (mut sumlx, mut suml2) = (0.0f32, 0.0f32);
        for &v in x {
            let q = clamp(nearest_int(iscale * v));
            let w = v * v;
            sumlx += w * v * q as f32;
            suml2 += w * (q * q) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for (i, &v) in x.iter().enumerate() {
                l[i] = (nmax + clamp(nearest_int(iscale * v))) as u8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Affine search `x ~ scale * l + min` (ggml `make_qkx2_quants`, squared error). Returns
/// `(scale, -min)`.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        l[i] = q as u8;
        let diff = scale * q as f32 + min - x[i];
        best_error += weights[i] * diff * diff;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let (mut sum_l, mut sum_l2, mut sum_xl) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..n {
            let q = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = q as u8;
            let (w, qf) = (weights[i], q as f32);
            sum_l += w * qf;
            sum_l2 += w * qf * qf;
            sum_xl += w * qf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_error = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                cur_error += weights[i] * diff * diff;
            }
            if cur_error < best_error {
                l.copy_from_slice(laux);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

fn q45_k(x: &[f32], out: &mut Vec<u8>, five: bool) {
    let nmax = if five { 31 } else { 15 };
    let (rmin, nstep) = if five { (-0.5, 15) } else { (-1.0, 20) };
    let mut l = [0u8; 256];
    let mut laux = [0u8; 32];
    let mut weights = [0.0f32; 32];
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    for j in 0..8 {
        let part = &x[32 * j..32 * j + 32];
        let av = (part.iter().map(|v| v * v).sum::<f32>() / 32.0).sqrt();
        for (w, v) in weights.iter_mut().zip(part) {
            *w = av + v.abs();
        }
        let (s, m) = make_qkx2_quants(
            nmax,
            part,
            &weights,
            &mut l[32 * j..32 * j + 32],
            &mut laux,
            rmin,
            0.1,
            nstep,
        );
        scales[j] = s;
        mins[j] = m;
    }
    let max_scale = scales.iter().copied().fold(0.0f32, f32::max);
    let max_min = mins.iter().copied().fold(0.0f32, f32::max);
    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut packed = [0u8; 12];
    for j in 0..8 {
        let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
        let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
        if j < 4 {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            packed[j - 4] |= (ls >> 4) << 6;
            packed[j] |= (lm >> 4) << 6;
        }
    }
    let d_h = f16::from_f32(max_scale / 63.0);
    let dmin_h = f16::from_f32(max_min / 63.0);
    out.extend_from_slice(&d_h.to_bits().to_le_bytes());
    out.extend_from_slice(&dmin_h.to_bits().to_le_bytes());
    out.extend_from_slice(&packed);
    let (d_all, dmin_all) = (d_h.to_f32(), dmin_h.to_f32());
    for j in 0..8 {
        let (sc, m) = scale_min_k4(j, &packed);
        let d = d_all * f32::from(sc);
        if d == 0.0 {
            continue;
        }
        let dm = dmin_all * f32::from(m);
        for ii in 0..32 {
            let q = nearest_int((x[32 * j + ii] + dm) / d).clamp(0, nmax);
            l[32 * j + ii] = q as u8;
        }
    }
    if five {
        let mut qh = [0u8; 32];
        let mut ql = Vec::with_capacity(128);
        let (mut m1, mut m2) = (1u8, 2u8);
        for n in (0..256).step_by(64) {
            for j in 0..32 {
                let mut l1 = l[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = l[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql.push(l1 | (l2 << 4));
            }
            m1 <<= 2;
            m2 <<= 2;
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&ql);
    } else {
        for j in (0..256).step_by(64) {
            out.extend((0..32).map(|i| l[j + i] | (l[j + i + 32] << 4)));
        }
    }
}

fn scale_min_k4(j: usize, q: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

fn q6_k(x: &[f32], out: &mut Vec<u8>) {
    let mut l = [0u8; 256];
    let mut scales = [0.0f32; 16];
    let (mut max_scale, mut max_abs) = (0.0f32, 0.0f32);
    for ib in 0..16 {
        let s = make_qx_quants(32, &x[16 * ib..16 * ib + 16], &mut l[16 * ib..16 * ib + 16]);
        scales[ib] = s;
        if s.abs() > max_abs {
            max_abs = s.abs();
            max_scale = s;
        }
    }
    if max_abs == 0.0 {
        out.extend_from_slice(&[0u8; 210]);
        return;
    }
    let iscale = -128.0 / max_scale;
    let d_h = f16::from_f32(1.0 / iscale);
    let mut sc = [0i8; 16];
    for ib in 0..16 {
        sc[ib] = nearest_int(iscale * scales[ib]).min(127) as i8;
    }
    let d_all = d_h.to_f32();
    for j in 0..16 {
        let d = d_all * f32::from(sc[j]);
        if d == 0.0 {
            continue;
        }
        for ii in 0..16 {
            let q = nearest_int(x[16 * j + ii] / d).clamp(-32, 31);
            l[16 * j + ii] = (q + 32) as u8;
        }
    }
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for half in 0..2 {
        let j = half * 128;
        for i in 0..32 {
            let q1 = l[j + i];
            let q2 = l[j + i + 32];
            let q3 = l[j + i + 64];
            let q4 = l[j + i + 96];
            ql[half * 64 + i] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            ql[half * 64 + i + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            qh[half * 32 + i] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    out.extend(sc.iter().map(|&s| s as u8));
    out.extend_from_slice(&d_h.to_bits().to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(n: usize) -> Vec<f32> {
        // deterministic pseudo-normal values
        let mut state = 0x1234_5678u32;
        (0..n)
            .map(|_| {
                let mut s = 0.0;
                for _ in 0..4 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    s += (state >> 8) as f32 / (1 << 24) as f32 - 0.5;
                }
                s * 0.5
            })
            .collect()
    }

    fn h(b: &[u8]) -> f32 {
        f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()
    }

    fn dequant(kind: QType, bytes: &[u8]) -> Vec<f32> {
        let mut out = Vec::new();
        for b in bytes.chunks_exact(kind.block_bytes()) {
            match kind {
                QType::Q8_0 => out.extend(b[2..].iter().map(|&q| h(b) * f32::from(q as i8))),
                QType::Q4_0 => {
                    let d = h(b);
                    out.extend(b[2..18].iter().map(|q| d * (f32::from(q & 0xF) - 8.0)));
                    out.extend(b[2..18].iter().map(|q| d * (f32::from(q >> 4) - 8.0)));
                }
                QType::Q4_1 => {
                    let (d, m) = (h(b), h(&b[2..]));
                    out.extend(b[4..20].iter().map(|q| d * f32::from(q & 0xF) + m));
                    out.extend(b[4..20].iter().map(|q| d * f32::from(q >> 4) + m));
                }
                QType::Q5_0 | QType::Q5_1 => {
                    let one = kind == QType::Q5_1;
                    let head = if one { 4 } else { 2 };
                    let (d, m) = (h(b), if one { h(&b[2..]) } else { -16.0 * h(b) });
                    let qh = u32::from_le_bytes(b[head..head + 4].try_into().unwrap());
                    let qs = &b[head + 4..];
                    let mut lo = [0.0; 16];
                    let mut hi = [0.0; 16];
                    for j in 0..16 {
                        let a = (qs[j] & 0xF) | ((((qh >> j) & 1) as u8) << 4);
                        let c = (qs[j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4);
                        lo[j] = d * f32::from(a) + m;
                        hi[j] = d * f32::from(c) + m;
                    }
                    out.extend(lo);
                    out.extend(hi);
                }
                QType::Q4K | QType::Q5K => {
                    let (d, dmin) = (h(b), h(&b[2..]));
                    let sc: [u8; 12] = b[4..16].try_into().unwrap();
                    let five = kind == QType::Q5K;
                    let (qh, qs) = if five {
                        (&b[16..48], &b[48..])
                    } else {
                        (&b[0..0], &b[16..])
                    };
                    for g in 0..4 {
                        for half in 0..2 {
                            let (s, m) = scale_min_k4(2 * g + half, &sc);
                            for i in 0..32 {
                                let byte = qs[32 * g + i];
                                let mut q = if half == 0 { byte & 0xF } else { byte >> 4 };
                                if five && qh[i] & (1 << (2 * g + half)) != 0 {
                                    q += 16;
                                }
                                out.push(d * f32::from(s) * f32::from(q) - dmin * f32::from(m));
                            }
                        }
                    }
                }
                QType::Q6K => {
                    let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
                    let d = h(&b[208..]);
                    let mut y = [0.0f32; 256];
                    for half in 0..2 {
                        for l in 0..32 {
                            let is = l / 16;
                            let h4 = qh[half * 32 + l];
                            let q = [
                                ((ql[half * 64 + l] & 0xF) | ((h4 & 3) << 4)) as i32 - 32,
                                ((ql[half * 64 + l + 32] & 0xF) | (((h4 >> 2) & 3) << 4)) as i32
                                    - 32,
                                ((ql[half * 64 + l] >> 4) | (((h4 >> 4) & 3) << 4)) as i32 - 32,
                                ((ql[half * 64 + l + 32] >> 4) | (((h4 >> 6) & 3) << 4)) as i32
                                    - 32,
                            ];
                            for (k, v) in q.iter().enumerate() {
                                let s = f32::from(sc[half * 8 + is + 2 * k] as i8);
                                y[half * 128 + l + 32 * k] = d * s * *v as f32;
                            }
                        }
                    }
                    out.extend(y);
                }
            }
        }
        out
    }

    #[test]
    fn round_trip_error_shrinks_with_more_bits() {
        let x = sample(1024);
        let rms = |kind| {
            let mut bytes = Vec::new();
            quantize_row(kind, &x, &mut bytes);
            assert_eq!(bytes.len(), kind.row_bytes(x.len()));
            let y = dequant(kind, &bytes);
            assert_eq!(y.len(), x.len());
            (x.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / x.len() as f32).sqrt()
        };
        let signal = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let limits = [
            (QType::Q8_0, 0.006),
            (QType::Q6K, 0.02),
            (QType::Q5_1, 0.04),
            (QType::Q5_0, 0.04),
            (QType::Q5K, 0.04),
            (QType::Q4K, 0.09),
            (QType::Q4_1, 0.09),
            (QType::Q4_0, 0.11),
        ];
        for (kind, limit) in limits {
            let e = rms(kind) / signal;
            assert!(e < limit, "{kind:?}: relative error {e} above {limit}");
        }
    }

    #[test]
    fn zero_blocks_stay_zero() {
        for kind in [
            QType::Q8_0,
            QType::Q4_0,
            QType::Q5_1,
            QType::Q4K,
            QType::Q5K,
            QType::Q6K,
        ] {
            let mut bytes = Vec::new();
            quantize_row(kind, &vec![0.0; 256], &mut bytes);
            assert!(dequant(kind, &bytes).iter().all(|v| *v == 0.0), "{kind:?}");
        }
    }
}
