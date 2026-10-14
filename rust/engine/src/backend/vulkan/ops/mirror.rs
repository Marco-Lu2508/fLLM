//! Rust transcriptions of the trickiest shaders of this package (KV-cache write, attention decode / merge / prompt tiles, MoE routing, DeltaNet,
//! Mamba), executed on the CPU lane by lane and phase by phase (every `barrier()` ends a phase), and compared with the CPU backend / CPU reference
//! kernels. They check the algorithms, index arithmetic and word ownership that the shaders implement; they do not run the GLSL text, so a typo in
//! the shader that the transcription does not share is not caught. Level: unit tests on the CPU only.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::chunks_exact_to_as_chunks,
    clippy::manual_checked_ops
)]

use crate::backend::cpu::CpuBackend;
use crate::backend::{AttentionParams, Backend, DeviceBuffer, DeviceKvCache, Stream};
use crate::cpu_selftest::Lcg;
use half::f16;
use std::sync::Arc;

pub(super) const NEG: f32 = -3.0e38;

pub(super) fn f16_bits(x: f32) -> u32 {
    u32::from(f16::from_f32(x).to_bits())
}

pub(super) fn bytes_to_words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut a = [0u8; 4];
            a[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(a)
        })
        .collect()
}

pub(super) fn upload(cpu: &CpuBackend, st: &mut dyn Stream, bytes: &[u8]) -> DeviceBuffer {
    let mut b = cpu.allocate_buffer(bytes.len().max(4)).unwrap();
    st.upload(&mut b, bytes).unwrap();
    b
}

pub(super) fn upload_f32(cpu: &CpuBackend, st: &mut dyn Stream, v: &[f32]) -> DeviceBuffer {
    upload(
        cpu,
        st,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
    )
}

pub(super) fn download(st: &mut dyn Stream, b: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    st.download(&mut out, b).unwrap();
    out
}

pub(super) fn download_f32(st: &mut dyn Stream, b: &DeviceBuffer, n: usize) -> Vec<f32> {
    download(st, b, n * 4)
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Deterministic shuffle of `0..n` (the order in which "invocations" run: the shaders must not depend on it).
pub(super) fn shuffled(n: usize, rng: &mut Lcg) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// kv_write_f16.comp / kv_write_q8.comp
// ------------------------------------------------------------------------------------------------------------------------------------------

fn atomic_and(w: &mut u32, v: u32) {
    *w &= v;
}

fn atomic_or(w: &mut u32, v: u32) {
    *w |= v;
}

/// `kv_write_f16.comp`; `order` = the order in which the invocations run.
pub(super) fn kv_write_f16(
    pool: &mut [u32],
    src: &[f32],
    slots: &[usize],
    kv: usize,
    base: usize,
    order: &[usize],
) {
    let per_row = kv / 2 + 1;
    for &i in order {
        let (r, j) = (i / per_row, i % per_row);
        let h0 = base + slots[r] * kv;
        let words = ((h0 + kv - 1) >> 1) - (h0 >> 1) + 1;
        if j >= words {
            continue;
        }
        let w = (h0 >> 1) + j;
        let lo_pos = 2 * w;
        let lo_ok = lo_pos >= h0 && lo_pos < h0 + kv;
        let hi_ok = lo_pos + 1 >= h0 && lo_pos + 1 < h0 + kv;
        let lo = if lo_ok {
            f16_bits(src[r * kv + (lo_pos - h0)])
        } else {
            0
        };
        let hi = if hi_ok {
            f16_bits(src[r * kv + (lo_pos + 1 - h0)])
        } else {
            0
        };
        if lo_ok && hi_ok {
            pool[w] = lo | (hi << 16);
        } else {
            if lo_ok {
                atomic_and(&mut pool[w], 0xFFFF_0000);
                atomic_or(&mut pool[w], lo);
            }
            if hi_ok {
                atomic_and(&mut pool[w], 0x0000_FFFF);
                atomic_or(&mut pool[w], hi << 16);
            }
        }
    }
}

fn store_bits(pool: &mut [u32], word: usize, mask: u32, value: u32) {
    if mask == 0xFFFF_FFFF {
        pool[word] = value;
    } else {
        atomic_and(&mut pool[word], !mask);
        atomic_or(&mut pool[word], value & mask);
    }
}

/// `kv_write_q8.comp`.
pub(super) fn kv_write_q8(
    pool: &mut [u32],
    src: &[f32],
    slots: &[usize],
    kv: usize,
    group: usize,
    base: usize,
    scale_base: usize,
    order: &[usize],
) {
    let groups = kv / group;
    for &i in order {
        let (r, g) = (i / groups, i % groups);
        let slot = slots[r];
        let first = r * kv + g * group;
        let mut amax = 0.0f32;
        for j in 0..group {
            amax = amax.max(src[first + j].abs());
        }
        let d = amax / 127.0;
        let sp = base + scale_base + slot * groups + g;
        let sshift = (sp & 1) * 16;
        store_bits(pool, sp >> 1, 0xFFFFu32 << sshift, f16_bits(d) << sshift);
        let byte0 = base * 2 + slot * kv + g * group;
        let mut word = byte0 >> 2;
        let (mut mask, mut value) = (0u32, 0u32);
        for j in 0..group {
            let b = byte0 + j;
            if (b >> 2) != word {
                store_bits(pool, word, mask, value);
                word = b >> 2;
                mask = 0;
                value = 0;
            }
            let mut qv = 0.0f32;
            if d != 0.0 {
                let t = src[first + j] / d;
                qv = (t.signum() * (t.abs() + 0.5).floor()).clamp(-127.0, 127.0);
                if t == 0.0 {
                    qv = 0.0;
                }
            }
            let shift = ((b & 3) * 8) as u32;
            mask |= 0xFFu32 << shift;
            value |= ((qv as i32 as u32) & 0xFF) << shift;
        }
        store_bits(pool, word, mask, value);
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// attention
// ------------------------------------------------------------------------------------------------------------------------------------------

/// Everything the attention shaders read besides q.
pub(super) struct Pools {
    pub k: Vec<u32>,
    pub v: Vec<u32>,
    pub q8: bool,
    pub k_base: usize,
    pub v_base: usize,
    pub k_scale: usize,
    pub v_scale: usize,
    pub k_group: usize,
    pub v_group: usize,
    pub kv_heads: usize,
    pub dk: usize,
    pub dv: usize,
}

fn half_at(pool: &[u32], i: usize) -> f32 {
    f16::from_bits(((pool[i >> 1] >> ((i & 1) * 16)) & 0xFFFF) as u16).to_f32()
}

fn i8_at(pool: &[u32], b: usize) -> f32 {
    let w = pool[b >> 2] << (24 - ((b & 3) * 8));
    ((w as i32) >> 24) as f32
}

fn s8x4(w: u32) -> [f32; 4] {
    [
        ((w << 24) as i32 >> 24) as f32,
        ((w << 16) as i32 >> 24) as f32,
        ((w << 8) as i32 >> 24) as f32,
        (w as i32 >> 24) as f32,
    ]
}

impl Pools {
    /// `key_dot` of attn_decode.comp.
    fn key_dot(&self, qs: &[f32], slot: usize, kh: usize) -> f32 {
        let kvk = self.kv_heads * self.dk;
        let mut s = 0.0f32;
        if !self.q8 {
            let h0 = self.k_base + slot * kvk + kh * self.dk;
            if h0 & 1 == 0 {
                let w0 = h0 >> 1;
                for i in 0..self.dk >> 1 {
                    let w = self.k[w0 + i];
                    let (a, b) = (
                        f16::from_bits((w & 0xFFFF) as u16).to_f32(),
                        f16::from_bits((w >> 16) as u16).to_f32(),
                    );
                    s += qs[2 * i] * a + qs[2 * i + 1] * b;
                }
                if self.dk & 1 != 0 {
                    s += qs[self.dk - 1] * half_at(&self.k, h0 + self.dk - 1);
                }
            } else {
                for d in 0..self.dk {
                    s += qs[d] * half_at(&self.k, h0 + d);
                }
            }
        } else {
            let group = self.k_group;
            let b0 = self.k_base * 2 + slot * kvk + kh * self.dk;
            let sc0 = self.k_base + self.k_scale + slot * (kvk / group) + (kh * self.dk) / group;
            for g in 0..self.dk / group {
                let mut part = 0.0f32;
                let bg = b0 + g * group;
                let qg = g * group;
                if bg & 3 == 0 && group & 3 == 0 {
                    let mut j = 0;
                    while j < group {
                        let kk = s8x4(self.k[(bg + j) >> 2]);
                        part += qs[qg + j] * kk[0]
                            + qs[qg + j + 1] * kk[1]
                            + qs[qg + j + 2] * kk[2]
                            + qs[qg + j + 3] * kk[3];
                        j += 4;
                    }
                } else {
                    for j in 0..group {
                        part += qs[qg + j] * i8_at(&self.k, bg + j);
                    }
                }
                s += part * half_at(&self.k, sc0 + g);
            }
        }
        s
    }

    fn k_at(&self, slot: usize, c: usize) -> f32 {
        let kvk = self.kv_heads * self.dk;
        if !self.q8 {
            half_at(&self.k, self.k_base + slot * kvk + c)
        } else {
            i8_at(&self.k, self.k_base * 2 + slot * kvk + c)
                * half_at(
                    &self.k,
                    self.k_base + self.k_scale + slot * (kvk / self.k_group) + c / self.k_group,
                )
        }
    }

    fn v_at(&self, slot: usize, c: usize) -> f32 {
        let kvv = self.kv_heads * self.dv;
        if !self.q8 {
            half_at(&self.v, self.v_base + slot * kvv + c)
        } else {
            i8_at(&self.v, self.v_base * 2 + slot * kvv + c)
                * half_at(
                    &self.v,
                    self.v_base + self.v_scale + slot * (kvv / self.v_group) + c / self.v_group,
                )
        }
    }
}

pub(super) struct Shape<'a> {
    pub heads: usize,
    pub kv_heads: usize,
    pub stride: usize,
    pub block_size: usize,
    pub scale: f32,
    pub softcap: f32,
    pub window: usize,
    pub chunk: usize,
    pub sinks: Option<&'a [f32]>,
    pub alibi: Option<&'a [f32]>,
}

impl Shape<'_> {
    fn start(&self, n: usize) -> usize {
        if self.chunk != 0 {
            (n - 1) / self.chunk * self.chunk
        } else if self.window != 0 && n > self.window {
            n - self.window
        } else {
            0
        }
    }

    fn slot(&self, tables: &[i64], row: usize, pos: usize) -> usize {
        tables[row * self.stride + pos / self.block_size] as usize * self.block_size
            + pos % self.block_size
    }
}

pub(super) struct Part {
    pub m: f32,
    pub l: f32,
    pub acc: Vec<f32>,
}

const WG: usize = 64;

/// One workgroup of attn_decode.comp: (row `r`, head `h`, split `split` of `nsplit`); `q` is the query of (r, h).
pub(super) fn decode_group(
    sh: &Shape,
    c: &Pools,
    q: &[f32],
    lengths: &[i64],
    tables: &[i64],
    r: usize,
    h: usize,
    split: usize,
    nsplit: usize,
) -> Part {
    let kh = h / (sh.heads / sh.kv_heads);
    let n = lengths[r] as usize;
    let start = sh.start(n);
    let (mut lo, mut hi) = (start, n);
    if nsplit > 1 {
        let mut per = (n - start).div_ceil(nsplit);
        per = per.div_ceil(WG) * WG;
        lo = n.min(start + split * per);
        hi = n.min(lo + per);
    }
    if lo >= hi {
        return Part {
            m: NEG,
            l: 0.0,
            acc: vec![0.0; c.dv],
        };
    }
    let slope = sh.alibi.map_or(0.0, |a| a[h]);
    let mut m = NEG;
    let mut l = 0.0f32;
    let mut acc = vec![[0.0f32; 8]; WG];
    let mut t0 = lo;
    while t0 < hi {
        let mut sc_s = [NEG; WG];
        let mut slot_s = [0usize; WG];
        let mut sc_lane = [NEG; WG];
        // phase 1: one key per lane
        for lane in 0..WG {
            let pos = t0 + lane;
            let mut sc = NEG;
            if pos < hi {
                let slot = sh.slot(tables, r, pos);
                slot_s[lane] = slot;
                sc = c.key_dot(q, slot, kh) * sh.scale;
                if sh.alibi.is_some() {
                    sc += slope * (pos as f32 - (n - 1) as f32);
                }
                if sh.softcap != 0.0 {
                    sc = sh.softcap * (sc / sh.softcap).clamp(-15.0, 15.0).tanh();
                }
            }
            sc_s[lane] = sc;
            sc_lane[lane] = sc;
        }
        // phase 2: tile maximum, probabilities
        let count = WG.min(hi - t0);
        let mut tmax = NEG;
        for j in 0..count {
            tmax = tmax.max(sc_s[j]);
        }
        let m_new = m.max(tmax);
        let corr = (m - m_new).exp();
        let mut e = [0.0f32; WG];
        for lane in 0..WG {
            e[lane] = if t0 + lane < hi {
                (sc_lane[lane] - m_new).exp()
            } else {
                0.0
            };
        }
        let mut tsum = 0.0f32;
        for j in 0..count {
            tsum += e[j];
        }
        l = l * corr + tsum;
        // phase 3: value accumulation, lane owns dimensions lane + 64 i
        for lane in 0..WG {
            for i in 0..8 {
                let d = lane + i * WG;
                if d < c.dv {
                    let mut a = acc[lane][i] * corr;
                    for j in 0..count {
                        a += e[j] * c.v_at(slot_s[j], kh * c.dv + d);
                    }
                    acc[lane][i] = a;
                }
            }
        }
        m = m_new;
        t0 += WG;
    }
    let mut out = vec![0.0f32; c.dv];
    for lane in 0..WG {
        for i in 0..8 {
            let d = lane + i * WG;
            if d < c.dv {
                out[d] = acc[lane][i];
            }
        }
    }
    Part { m, l, acc: out }
}

/// The end of attn_decode.comp for nsplit == 1.
pub(super) fn finish_single(sh: &Shape, h: usize, mut p: Part) -> Vec<f32> {
    if let Some(s) = sh.sinks {
        let sink = s[h];
        let m2 = p.m.max(sink);
        let corr = (p.m - m2).exp();
        p.l = p.l * corr + (sink - m2).exp();
        for a in &mut p.acc {
            *a *= corr;
        }
    }
    let inv = 1.0 / p.l;
    p.acc.iter().map(|a| a * inv).collect()
}

/// attn_merge.comp for one (row, head).
pub(super) fn merge(sh: &Shape, h: usize, parts: &[Part]) -> Vec<f32> {
    let mut mx = NEG;
    for p in parts {
        if p.l > 0.0 {
            mx = mx.max(p.m);
        }
    }
    let mut sink = NEG;
    if let Some(s) = sh.sinks {
        sink = s[h];
        mx = mx.max(sink);
    }
    let mut total = 0.0f32;
    let mut w = Vec::new();
    for p in parts {
        let wi = if p.l > 0.0 { (p.m - mx).exp() } else { 0.0 };
        total += wi * p.l;
        w.push(wi);
    }
    if sh.sinks.is_some() {
        total += (sink - mx).exp();
    }
    let inv = 1.0 / total;
    let dv = parts[0].acc.len();
    (0..dv)
        .map(|d| {
            let mut a = 0.0f32;
            for (i, p) in parts.iter().enumerate() {
                if w[i] != 0.0 {
                    a += w[i] * p.acc[d];
                }
            }
            a * inv
        })
        .collect()
}

const BR: usize = 16;
const BC: usize = 32;
const DC: usize = 32;
const VC: usize = 32;
const PWG: usize = 128;

/// One workgroup of attn_prefill.comp: head `h`, block `block` of 16 rows from `row0`; returns the outputs of the rows of the block.
pub(super) fn prefill_block(
    sh: &Shape,
    c: &Pools,
    q: &[f32],
    lengths: &[i64],
    tables: &[i64],
    h: usize,
    row0: usize,
    rows_end: usize,
    block: usize,
    dvmax: usize,
) -> Vec<Vec<f32>> {
    let (dk, dv) = (c.dk, c.dv);
    let kh = h / (sh.heads / sh.kv_heads);
    let rb0 = row0 + block * BR;
    let rend = (rb0 + BR).min(rows_end);
    let nch = dvmax / VC;
    let vpt = VC / 8;
    // per row (li) of the block
    let rvalid = |li: usize| rb0 + li < rend;
    let n_i = |li: usize| {
        if rvalid(li) {
            lengths[rb0 + li] as usize
        } else {
            1
        }
    };
    let mut n_max = 0usize;
    let mut r_tab = rb0;
    let mut kstart = usize::MAX;
    for ii in 0..BR {
        let rr = rb0 + ii;
        if rr < rend {
            let nn = lengths[rr] as usize;
            if nn > n_max {
                n_max = nn;
                r_tab = rr;
            }
            kstart = kstart.min(sh.start(nn));
        }
    }
    let slope = sh.alibi.map_or(0.0, |a| a[h]);
    let mut m_i = [NEG; PWG];
    let mut l_i = [0.0f32; PWG];
    let mut acc = vec![vec![0.0f32; dvmax / 8]; PWG];
    let mut t0 = kstart;
    while t0 < n_max {
        let mut slot_s = [0usize; BC];
        for lane in 0..BC {
            let pos = t0 + lane;
            if pos < n_max {
                slot_s[lane] = sh.slot(tables, r_tab, pos);
            }
        }
        let mut sacc = vec![[0.0f32; 4]; PWG];
        let mut d0 = 0;
        while d0 < dk {
            let mut kt = vec![0.0f32; BC * (DC + 1)];
            let mut qt = vec![0.0f32; BR * (DC + 1)];
            for lane in 0..PWG {
                for e in 0..(BC * DC) / PWG {
                    let idx = lane + e * PWG;
                    let (kk, dd) = (idx / DC, idx % DC);
                    let d = d0 + dd;
                    let mut val = 0.0;
                    if d < dk && t0 + kk < n_max {
                        val = c.k_at(slot_s[kk], kh * dk + d);
                    }
                    kt[kk * (DC + 1) + dd] = val;
                }
                for e in 0..(BR * DC) / PWG {
                    let idx = lane + e * PWG;
                    let (ii, dd) = (idx / DC, idx % DC);
                    let d = d0 + dd;
                    let rr = rb0 + ii;
                    let mut val = 0.0;
                    if rr < rend && d < dk {
                        val = q[(rr * sh.heads + h) * dk + d];
                    }
                    qt[ii * (DC + 1) + dd] = val;
                }
            }
            for lane in 0..PWG {
                let (li, jl) = (lane >> 3, lane & 7);
                for d in 0..DC {
                    let qv = qt[li * (DC + 1) + d];
                    for cc in 0..4 {
                        sacc[lane][cc] += qv * kt[(jl + 8 * cc) * (DC + 1) + d];
                    }
                }
            }
            d0 += DC;
        }
        // masks and row maxima
        let mut red_m = vec![NEG; BR * 8];
        for lane in 0..PWG {
            let (li, jl) = (lane >> 3, lane & 7);
            let start_i = sh.start(n_i(li));
            let mut sm = NEG;
            for cc in 0..4 {
                let pos = t0 + jl + 8 * cc;
                let mut sc = NEG;
                if rvalid(li) && pos >= start_i && pos < n_i(li) {
                    sc = sacc[lane][cc] * sh.scale;
                    if sh.alibi.is_some() {
                        sc += slope * (pos as f32 - (n_i(li) - 1) as f32);
                    }
                    if sh.softcap != 0.0 {
                        sc = sh.softcap * (sc / sh.softcap).clamp(-15.0, 15.0).tanh();
                    }
                }
                sacc[lane][cc] = sc;
                sm = sm.max(sc);
            }
            red_m[li * 8 + jl] = sm;
        }
        let mut pt = vec![0.0f32; BR * (BC + 1)];
        let mut red_l = vec![0.0f32; BR * 8];
        let mut corr_l = [1.0f32; PWG];
        for lane in 0..PWG {
            let (li, jl) = (lane >> 3, lane & 7);
            let mut tmax = NEG;
            for j in 0..8 {
                tmax = tmax.max(red_m[li * 8 + j]);
            }
            let m_new = m_i[lane].max(tmax);
            let corr = (m_i[lane] - m_new).exp();
            corr_l[lane] = corr;
            let mut psum = 0.0;
            for cc in 0..4 {
                let e = if sacc[lane][cc] != NEG {
                    (sacc[lane][cc] - m_new).exp()
                } else {
                    0.0
                };
                pt[li * (BC + 1) + jl + 8 * cc] = e;
                psum += e;
            }
            red_l[li * 8 + jl] = psum;
            m_i[lane] = m_new; // (the shader updates after summing; the lane's own corr is already fixed)
        }
        for lane in 0..PWG {
            let li = lane >> 3;
            let mut tsum = 0.0;
            for j in 0..8 {
                tsum += red_l[li * 8 + j];
            }
            l_i[lane] = l_i[lane] * corr_l[lane] + tsum;
            for a in acc[lane].iter_mut() {
                *a *= corr_l[lane];
            }
        }
        // P x V in chunks of 32 dimensions
        for cch in 0..nch {
            if cch * VC < dv {
                let mut vt = vec![0.0f32; BC * VC];
                for lane in 0..PWG {
                    for e in 0..(BC * VC) / PWG {
                        let idx = lane + e * PWG;
                        let (kk, dd) = (idx / VC, idx % VC);
                        let d = cch * VC + dd;
                        let mut val = 0.0;
                        if d < dv && t0 + kk < n_max {
                            val = c.v_at(slot_s[kk], kh * dv + d);
                        }
                        vt[kk * VC + dd] = val;
                    }
                }
                for lane in 0..PWG {
                    let (li, jl) = (lane >> 3, lane & 7);
                    for kk in 0..BC {
                        let pk = pt[li * (BC + 1) + kk];
                        for e in 0..vpt {
                            acc[lane][cch * vpt + e] += pk * vt[kk * VC + jl + 8 * e];
                        }
                    }
                }
            }
        }
        t0 += BC;
    }
    let mut outs = vec![vec![0.0f32; dv]; BR];
    for lane in 0..PWG {
        let (li, jl) = (lane >> 3, lane & 7);
        if !rvalid(li) {
            continue;
        }
        let (mut l, mut a) = (l_i[lane], acc[lane].clone());
        if let Some(s) = sh.sinks {
            let sink = s[h];
            let m2 = m_i[lane].max(sink);
            let corr = (m_i[lane] - m2).exp();
            l = l * corr + (sink - m2).exp();
            for x in a.iter_mut() {
                *x *= corr;
            }
        }
        let inv = 1.0 / l;
        for cch in 0..nch {
            for e in 0..vpt {
                let d = cch * VC + jl + 8 * e;
                if d < dv {
                    outs[li][d] = a[cch * vpt + e] * inv;
                }
            }
        }
    }
    outs.truncate(rend - rb0);
    outs
}

/// A KV cache on the CPU backend, filled through `kv_cache_write` with random sequences.
pub(super) struct Fixture {
    pub cpu: CpuBackend,
    pub kv: DeviceKvCache,
    pub layer: usize,
}

impl Fixture {
    pub fn new(
        layers: usize,
        layer: usize,
        pool_blocks: usize,
        block_size: usize,
        kv_heads: usize,
        dk: usize,
        dv: usize,
        kind: i32,
    ) -> Self {
        let cpu = CpuBackend::new();
        let kv = cpu
            .allocate_kv_cache(layers, pool_blocks, block_size, kv_heads, dk, dv, kind)
            .unwrap();
        Self { cpu, kv, layer }
    }

    /// Write random keys / values into `slots` of the layer and return nothing; the pools are read back with `pools()`.
    pub fn fill(&self, slots: &[usize], rng: &mut Lcg) {
        let mut st = self.cpu.create_stream().unwrap();
        let k = rng.vector(slots.len() * self.kv.kv_heads * self.kv.head_dim, 1.5);
        let v = rng.vector(slots.len() * self.kv.kv_heads * self.kv.value_dim, 1.5);
        let (kb, vb) = (
            upload_f32(&self.cpu, st.as_mut(), &k),
            upload_f32(&self.cpu, st.as_mut(), &v),
        );
        st.kv_cache_write(&self.kv, self.layer, slots, &kb, &vb)
            .unwrap();
    }

    pub fn pools(&self) -> Pools {
        let mut st = self.cpu.create_stream().unwrap();
        let kv = &self.kv;
        let q8 = kv.kv_kind == 2;
        let scale = |dim: usize| {
            if q8 {
                (kv.slots() * kv.kv_heads * dim).div_ceil(2)
            } else {
                0
            }
        };
        Pools {
            k: bytes_to_words(&download(st.as_mut(), &kv.key_pool, kv.key_pool.size)),
            v: bytes_to_words(&download(st.as_mut(), &kv.value_pool, kv.value_pool.size)),
            q8,
            k_base: self.layer * kv.words_per_layer_k,
            v_base: self.layer * kv.words_per_layer_v,
            k_scale: scale(kv.head_dim),
            v_scale: scale(kv.value_dim),
            k_group: crate::backend::kv_quant_group(kv.head_dim),
            v_group: crate::backend::kv_quant_group(kv.value_dim),
            kv_heads: kv.kv_heads,
            dk: kv.head_dim,
            dv: kv.value_dim,
        }
    }

    /// The CPU backend's attention for the batch.
    pub fn reference(
        &self,
        q: &[f32],
        tables: &[i64],
        lengths: &[i64],
        p: &AttentionParams,
    ) -> Vec<f32> {
        let mut st = self.cpu.create_stream().unwrap();
        let qb = upload_f32(&self.cpu, st.as_mut(), q);
        let mut out = self
            .cpu
            .allocate_buffer(lengths.len() * p.num_heads * p.value_dim * 4)
            .unwrap();
        st.attention_paged(&mut out, &qb, &self.kv, tables, lengths, self.layer, p)
            .unwrap();
        download_f32(st.as_mut(), &out, lengths.len() * p.num_heads * p.value_dim)
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// tests
// ------------------------------------------------------------------------------------------------------------------------------------------

fn words_to_bytes(w: &[u32], len: usize) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).take(len).collect()
}

#[test]
fn kv_write_matches_the_cpu_backend() {
    let mut rng = Lcg(0x1234_5678);
    let cases = [
        (1, 3, 7, 5),
        (1, 2, 64, 64),
        (1, 1, 1, 1),
        (2, 3, 7, 5),
        (2, 2, 64, 80),
        (2, 1, 33, 32),
        (2, 4, 96, 128),
    ];
    for (kind, kv_heads, dk, dv) in cases {
        for layer in 0..3 {
            let fx = Fixture::new(3, layer, 3, 5, kv_heads, dk, dv, kind);
            let mut st = fx.cpu.create_stream().unwrap();
            // random content in both pools: the writes must leave the neighbours alone
            let mut init: Vec<Vec<u8>> = Vec::new();
            for pool in [&fx.kv.key_pool, &fx.kv.value_pool] {
                let bytes: Vec<u8> = (0..pool.size).map(|_| rng.next() as u8).collect();
                let mut dst = pool.clone();
                st.upload(&mut dst, &bytes).unwrap();
                init.push(bytes);
            }
            let slots = [3usize, 4, 5, 9, 14, 0, 1];
            let k = rng.vector(slots.len() * kv_heads * dk, 2.0);
            let v = rng.vector(slots.len() * kv_heads * dv, 2.0);
            let (kb, vb) = (
                upload_f32(&fx.cpu, st.as_mut(), &k),
                upload_f32(&fx.cpu, st.as_mut(), &v),
            );
            st.kv_cache_write(&fx.kv, layer, &slots, &kb, &vb).unwrap();
            for (pool, bytes, src, dim, words) in [
                (&fx.kv.key_pool, &init[0], &k, dk, fx.kv.words_per_layer_k),
                (&fx.kv.value_pool, &init[1], &v, dv, fx.kv.words_per_layer_v),
            ] {
                let want = download(st.as_mut(), pool, pool.size);
                let mut words_buf = bytes_to_words(bytes);
                let kv = kv_heads * dim;
                let base = layer * words;
                if kind == 1 {
                    let total = slots.len() * (kv / 2 + 1);
                    let order = shuffled(total, &mut rng);
                    kv_write_f16(&mut words_buf, src, &slots, kv, base, &order);
                } else {
                    let group = crate::backend::kv_quant_group(dim);
                    let total = slots.len() * (kv / group);
                    let order = shuffled(total, &mut rng);
                    let scale_base = (fx.kv.slots() * kv).div_ceil(2);
                    kv_write_q8(
                        &mut words_buf,
                        src,
                        &slots,
                        kv,
                        group,
                        base,
                        scale_base,
                        &order,
                    );
                }
                let have = words_to_bytes(&words_buf, pool.size);
                assert_eq!(have, want, "kind {kind} kv_heads {kv_heads} dims {dk}/{dv} layer {layer} (layer base {base} words)");
            }
        }
    }
}

/// Block tables for sequences of `lens` tokens: distinct random blocks of the pool.
fn alloc_tables(
    lens: &[usize],
    bs: usize,
    pool_blocks: usize,
    stride: usize,
    rng: &mut Lcg,
) -> Vec<Vec<usize>> {
    let mut free = shuffled(pool_blocks, rng);
    lens.iter()
        .map(|&n| {
            let need = n.div_ceil(bs);
            assert!(need <= stride && free.len() >= need);
            (0..need).map(|_| free.pop().unwrap()).collect()
        })
        .collect()
}

fn slots_of(blocks: &[usize], n: usize, bs: usize) -> Vec<usize> {
    (0..n).map(|p| blocks[p / bs] * bs + p % bs).collect()
}

fn check_close(have: &[f32], want: &[f32], what: &str) {
    assert_eq!(have.len(), want.len());
    let mut worst = 0.0f32;
    for (i, (a, b)) in have.iter().zip(want).enumerate() {
        let err = (a - b).abs() / b.abs().max(1.0);
        assert!(err < 3e-4, "{what}: element {i}: {a} vs {b}");
        worst = worst.max(err);
    }
    let _ = worst;
}

struct Case {
    kind: i32,
    heads: usize,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    window: usize,
    chunk: usize,
    softcap: f32,
    sinks: bool,
    alibi: bool,
}

fn params_of(c: &Case, bs: usize, stride: usize, rng: &mut Lcg) -> AttentionParams {
    AttentionParams {
        num_heads: c.heads,
        num_kv_heads: c.kv_heads,
        head_dim: c.dk,
        value_dim: c.dv,
        stride,
        block_size: bs,
        scale: 0.9 / (c.dk as f32).sqrt(),
        softcap: (c.softcap != 0.0).then_some(c.softcap),
        window: (c.window != 0).then_some(c.window),
        chunk: (c.chunk != 0).then_some(c.chunk),
        sinks: c.sinks.then(|| Arc::new(rng.vector(c.heads, 2.0))),
        alibi_slopes: c
            .alibi
            .then(|| Arc::new((0..c.heads).map(|h| 0.05 + 0.02 * h as f32).collect())),
    }
}

fn shape_of(p: &AttentionParams) -> Shape<'_> {
    Shape {
        heads: p.num_heads,
        kv_heads: p.num_kv_heads,
        stride: p.stride,
        block_size: p.block_size,
        scale: p.scale,
        softcap: p.softcap.unwrap_or(0.0),
        window: p.window.unwrap_or(0),
        chunk: p.chunk.unwrap_or(0),
        sinks: p.sinks.as_deref().map(|v| v.as_slice()),
        alibi: p.alibi_slopes.as_deref().map(|v| v.as_slice()),
    }
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for kind in [1, 2] {
        for (heads, kv_heads, dk, dv) in [
            (8, 2, 64, 64),
            (4, 4, 80, 48),
            (6, 3, 96, 96),
            (4, 1, 128, 128),
            (2, 1, 256, 256),
            (4, 2, 192, 128),
            (2, 2, 33, 70),
        ] {
            let base = Case {
                kind,
                heads,
                kv_heads,
                dk,
                dv,
                window: 0,
                chunk: 0,
                softcap: 0.0,
                sinks: false,
                alibi: false,
            };
            v.push(Case { ..base });
            v.push(Case {
                window: 37,
                sinks: true,
                ..base
            });
            v.push(Case {
                chunk: 40,
                alibi: true,
                ..base
            });
            v.push(Case {
                softcap: 20.0,
                sinks: true,
                alibi: true,
                ..base
            });
        }
    }
    v
}

#[test]
fn decode_and_merge_match_the_cpu_backend() {
    let mut rng = Lcg(99);
    let (bs, pool_blocks, stride) = (16usize, 40usize, 20usize);
    let lens = [1usize, 5, 17, 64, 130, 300];
    for c in cases() {
        for layer in [0usize, 1] {
            let fx = Fixture::new(2, layer, pool_blocks, bs, c.kv_heads, c.dk, c.dv, c.kind);
            let seqs = alloc_tables(&lens, bs, pool_blocks, stride, &mut rng);
            for (blocks, &n) in seqs.iter().zip(&lens) {
                fx.fill(&slots_of(blocks, n, bs), &mut rng);
            }
            let params = params_of(&c, bs, stride, &mut rng);
            let mut tables = vec![0i64; lens.len() * stride];
            for (r, blocks) in seqs.iter().enumerate() {
                for (b, &id) in blocks.iter().enumerate() {
                    tables[r * stride + b] = id as i64;
                }
            }
            let lengths: Vec<i64> = lens.iter().map(|&n| n as i64).collect();
            let q = rng.vector(lens.len() * c.heads * c.dk, 1.0);
            let want = fx.reference(&q, &tables, &lengths, &params);
            let pools = fx.pools();
            let sh = shape_of(&params);
            for nsplit in [1usize, 2, 3, 8] {
                let mut have = vec![0.0f32; want.len()];
                for r in 0..lens.len() {
                    for h in 0..c.heads {
                        let qv = &q[(r * c.heads + h) * c.dk..][..c.dk];
                        let parts: Vec<Part> = (0..nsplit)
                            .map(|s| {
                                decode_group(&sh, &pools, qv, &lengths, &tables, r, h, s, nsplit)
                            })
                            .collect();
                        let out = if nsplit == 1 {
                            finish_single(&sh, h, parts.into_iter().next().unwrap())
                        } else {
                            merge(&sh, h, &parts)
                        };
                        have[(r * c.heads + h) * c.dv..][..c.dv].copy_from_slice(&out);
                    }
                }
                check_close(
                    &have,
                    &want,
                    &format!("decode kind {} heads {}/{} dims {}/{} win {} chunk {} cap {} sinks {} alibi {} nsplit {nsplit} layer {layer}", c.kind, c.heads, c.kv_heads, c.dk, c.dv, c.window, c.chunk, c.softcap, c.sinks, c.alibi),
                );
            }
        }
    }
}

#[test]
fn prompt_tiles_match_the_cpu_backend() {
    let mut rng = Lcg(2024);
    let (bs, pool_blocks, stride) = (16usize, 24usize, 12usize);
    for c in cases() {
        let dvmax = if c.dv <= 128 {
            128
        } else if c.dv <= 256 {
            256
        } else {
            512
        };
        let fx = Fixture::new(2, 1, pool_blocks, bs, c.kv_heads, c.dk, c.dv, c.kind);
        // one sequence of 150 tokens; the rows 100..145 of it are the prompt chunk (increasing lengths, one shared table)
        let total = 150usize;
        let seqs = alloc_tables(&[total, 40], bs, pool_blocks, stride, &mut rng);
        fx.fill(&slots_of(&seqs[0], total, bs), &mut rng);
        fx.fill(&slots_of(&seqs[1], 40, bs), &mut rng);
        let params = params_of(&c, bs, stride, &mut rng);
        let mut lengths: Vec<i64> = (100..145).map(|n| n + 1).collect();
        let mut tables = Vec::new();
        for _ in 0..lengths.len() {
            let mut row = vec![0i64; stride];
            for (b, &id) in seqs[0].iter().enumerate() {
                row[b] = id as i64;
            }
            tables.extend(row);
        }
        // a second, shorter sequence appended as a decode row (not part of the tiled run)
        lengths.push(40);
        let mut row = vec![0i64; stride];
        for (b, &id) in seqs[1].iter().enumerate() {
            row[b] = id as i64;
        }
        tables.extend(row);
        let rows = lengths.len();
        let q = rng.vector(rows * c.heads * c.dk, 1.0);
        let want = fx.reference(&q, &tables, &lengths, &params);
        let pools = fx.pools();
        let sh = shape_of(&params);
        let run_end = rows - 1;
        let mut have = vec![0.0f32; want.len()];
        for h in 0..c.heads {
            for block in 0..run_end.div_ceil(BR) {
                let outs = prefill_block(
                    &sh, &pools, &q, &lengths, &tables, h, 0, run_end, block, dvmax,
                );
                for (li, o) in outs.iter().enumerate() {
                    let r = block * BR + li;
                    have[(r * c.heads + h) * c.dv..][..c.dv].copy_from_slice(o);
                }
            }
        }
        let n = run_end * c.heads * c.dv;
        check_close(
            &have[..n],
            &want[..n],
            &format!(
                "prefill kind {} heads {}/{} dims {}/{} win {} chunk {} cap {} sinks {} alibi {}",
                c.kind,
                c.heads,
                c.kv_heads,
                c.dk,
                c.dv,
                c.window,
                c.chunk,
                c.softcap,
                c.sinks,
                c.alibi
            ),
        );
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// moe_topk.comp
// ------------------------------------------------------------------------------------------------------------------------------------------

/// One workgroup (64 lanes) of moe_topk.comp for one row of logits.
pub(super) fn moe_topk_row(
    logits: &[f32],
    k: usize,
    normalize: bool,
    scale: f32,
) -> (Vec<i32>, Vec<f32>) {
    const L: usize = 64;
    let experts = logits.len();
    let mut s_logit = logits.to_vec();
    let mut lane_max = [NEG; L];
    for lane in 0..L {
        let mut e = lane;
        while e < experts {
            lane_max[lane] = lane_max[lane].max(s_logit[e]);
            e += L;
        }
    }
    let m = lane_max.iter().fold(NEG, |a, &b| a.max(b));
    let mut lane_sum = [0.0f32; L];
    for lane in 0..L {
        let mut e = lane;
        while e < experts {
            lane_sum[lane] += (s_logit[e] - m).exp();
            e += L;
        }
    }
    let sum: f32 = lane_sum.iter().sum();
    let (mut chosen, mut score) = (Vec::new(), Vec::new());
    for _ in 0..k {
        let mut s_val = [NEG; L];
        let mut s_idx = [0x7FFF_FFFFusize; L];
        for lane in 0..L {
            let mut e = lane;
            while e < experts {
                let v = s_logit[e];
                if v > s_val[lane] || (v == s_val[lane] && e < s_idx[lane]) {
                    s_val[lane] = v;
                    s_idx[lane] = e;
                }
                e += L;
            }
        }
        let mut s = L / 2;
        while s > 0 {
            for lane in 0..s {
                let (ov, oi) = (s_val[lane + s], s_idx[lane + s]);
                if ov > s_val[lane] || (ov == s_val[lane] && oi < s_idx[lane]) {
                    s_val[lane] = ov;
                    s_idx[lane] = oi;
                }
            }
            s /= 2;
        }
        let mut bi = s_idx[0];
        if bi >= experts {
            bi = 0;
        }
        chosen.push(bi as i32);
        score.push((s_val[0] - m).exp() / sum);
        s_logit[bi] = -3.4e38;
    }
    let total: f32 = score.iter().sum();
    let denom = total.max(1e-8);
    let weights = score
        .iter()
        .map(|&s| (if normalize { s / denom } else { s }) * scale)
        .collect();
    (chosen, weights)
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// delta_conv.comp / delta_rec.comp
// ------------------------------------------------------------------------------------------------------------------------------------------

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// delta_conv.comp: all workgroups `(b, chunk)` of a run, in `order`.
pub(super) fn delta_conv(
    qkv: &[f32],
    convw: &[f32],
    state: &mut [f32],
    mixed: &mut [f32],
    (conv_dim, kh, hk, ksz): (usize, usize, usize, usize),
    (row0, n, reset, hist_off): (usize, usize, bool, usize),
    qscale: f32,
    order: &[(usize, usize)],
) {
    const WGC: usize = 128;
    let hist_len = ksz - 1;
    for &(b, ychunk) in order {
        let t0 = ychunk * 16;
        if t0 >= n {
            continue;
        }
        let t1 = n.min(t0 + 16);
        let is_head = b < 2 * kh;
        let (base, width) = if is_head {
            (b * hk, hk)
        } else {
            let base = 2 * kh * hk + (b - 2 * kh) * WGC;
            (base, WGC.min(conv_dim - base))
        };
        let mut wk = vec![[0.0f32; 8]; WGC];
        let mut win = vec![[0.0f32; 8]; WGC];
        for lane in 0..WGC {
            let valid = lane < width;
            let c = base + lane;
            let hist = hist_off + c * hist_len;
            for k in 0..8 {
                wk[lane][k] = if valid && k < ksz {
                    convw[c * ksz + k]
                } else {
                    0.0
                };
            }
            for j in 0..7 {
                if j < hist_len {
                    let mut v = 0.0;
                    if valid {
                        if t0 + j >= hist_len {
                            v = qkv[(row0 + t0 + j - hist_len) * conv_dim + c];
                        } else if !reset {
                            v = state[hist + t0 + j];
                        }
                    }
                    win[lane][j] = v;
                }
            }
        }
        for t in t0..t1 {
            let mut accs = vec![0.0f32; WGC];
            let mut curs = vec![0.0f32; WGC];
            for lane in 0..WGC {
                let valid = lane < width;
                let c = base + lane;
                let cur = if valid {
                    qkv[(row0 + t) * conv_dim + c]
                } else {
                    0.0
                };
                curs[lane] = cur;
                let mut acc = wk[lane][hist_len] * cur;
                for k in 0..7 {
                    if k < hist_len {
                        acc += wk[lane][k] * win[lane][k];
                    }
                }
                accs[lane] = if valid { silu(acc) } else { 0.0 };
            }
            if is_head {
                let ss: f32 = accs.iter().map(|a| a * a).sum();
                let inv = 1.0 / (ss + 1e-6).sqrt();
                for a in accs.iter_mut() {
                    *a = if b < kh { *a * inv * qscale } else { *a * inv };
                }
            }
            for lane in 0..WGC {
                if lane < width {
                    mixed[(row0 + t) * conv_dim + base + lane] = accs[lane];
                }
                for k in 0..6 {
                    if k + 1 < hist_len {
                        win[lane][k] = win[lane][k + 1];
                    }
                }
                win[lane][hist_len - 1] = curs[lane];
            }
        }
        if ychunk == 0 {
            for lane in 0..width {
                let c = base + lane;
                let hist = hist_off + c * hist_len;
                let mut fresh = [0.0f32; 7];
                for k in 0..7 {
                    if k < hist_len {
                        fresh[k] = if n + k >= hist_len {
                            qkv[(row0 + n + k - hist_len) * conv_dim + c]
                        } else if !reset {
                            state[hist + n + k]
                        } else {
                            0.0
                        };
                    }
                }
                for k in 0..7 {
                    if k < hist_len {
                        state[hist + k] = fresh[k];
                    }
                }
            }
        }
    }
}

/// delta_rec.comp for one value head `j` of a run (256 lanes); `hk` = the key width macro.
pub(super) fn delta_rec_head(
    j: usize,
    mixed: &[f32],
    (z, beta, alpha): (&[f32], &[f32], &[f32]),
    (dt_bias, decay, norm_w): (&[f32], &[f32], &[f32]),
    state: &mut [f32],
    out: &mut [f32],
    (kh, vh, hv, hk, conv_dim): (usize, usize, usize, usize, usize),
    (tiled, row0, n, reset, state_off): (bool, usize, usize, bool, usize),
    (beta_scale, eps): (f32, f32),
) {
    const WGR: usize = 256;
    let half = hk / 2;
    let key_dim = kh * hk;
    let value_dim = vh * hv;
    let src = if tiled { j % kh } else { j / (vh / kh) };
    let mut s = vec![vec![0.0f32; half]; WGR];
    for lane in 0..WGR {
        let c = lane >> 1;
        let hid = lane & 1;
        let ok = c < hv;
        let sbase = state_off + (j * hk + hid * half) * hv + c;
        for i in 0..half {
            s[lane][i] = if ok && !reset {
                state[sbase + i * hv]
            } else {
                0.0
            };
        }
    }
    let dtb = dt_bias[j];
    let dec = decay[j];
    for t in 0..n {
        let row = row0 + t;
        let mut qk = vec![0.0f32; 2 * hk];
        for lane in 0..2 * hk {
            let off = if lane < hk {
                src * hk + lane
            } else {
                key_dim + src * hk + (lane - hk)
            };
            qk[lane] = mixed[row * conv_dim + off];
        }
        let raw = alpha[row * vh + j] + dtb;
        let sp = if raw > 20.0 {
            raw
        } else {
            (1.0 + raw.exp()).ln()
        };
        let forget = (dec * sp).exp();
        let sigma = beta_scale / (1.0 + (-beta[row * vh + j]).exp());
        let mut memory = vec![0.0f32; WGR];
        for lane in 0..WGR {
            let (c, hid) = (lane >> 1, lane & 1);
            let mut m = 0.0;
            for i in 0..half {
                s[lane][i] *= forget;
                m += s[lane][i] * qk[hk + hid * half + i];
            }
            memory[lane] = m;
            let _ = c;
        }
        let mut acc = vec![0.0f32; WGR];
        for lane in 0..WGR {
            let (c, hid) = (lane >> 1, lane & 1);
            let mem = memory[lane] + memory[lane ^ 1];
            let v = if c < hv {
                mixed[row * conv_dim + 2 * key_dim + j * hv + c]
            } else {
                0.0
            };
            let delta = (v - mem) * sigma;
            let mut a = 0.0;
            for i in 0..half {
                s[lane][i] += qk[hk + hid * half + i] * delta;
                a += s[lane][i] * qk[hid * half + i];
            }
            acc[lane] = a;
        }
        let acc2: Vec<f32> = (0..WGR).map(|lane| acc[lane] + acc[lane ^ 1]).collect();
        let ss: f32 = (0..WGR)
            .map(|lane| {
                if lane & 1 == 0 && (lane >> 1) < hv {
                    acc2[lane] * acc2[lane]
                } else {
                    0.0
                }
            })
            .sum();
        let inv = 1.0 / (ss / hv as f32 + eps).sqrt();
        for lane in 0..WGR {
            let (c, hid) = (lane >> 1, lane & 1);
            if hid == 0 && c < hv {
                let gate = silu(z[row * value_dim + j * hv + c]);
                out[row * value_dim + j * hv + c] = acc2[lane] * inv * norm_w[c] * gate;
            }
        }
    }
    for lane in 0..WGR {
        let (c, hid) = (lane >> 1, lane & 1);
        if c < hv {
            let sbase = state_off + (j * hk + hid * half) * hv + c;
            for i in 0..half {
                state[sbase + i * hv] = s[lane][i];
            }
        }
    }
}

#[test]
fn moe_topk_matches_the_cpu_backend() {
    let mut rng = Lcg(5);
    for (rows, experts, k) in [
        (3usize, 64usize, 8usize),
        (2, 7, 3),
        (4, 200, 8),
        (1, 1024, 32),
        (5, 10, 10),
    ] {
        let logits = rng.vector(rows * experts, 3.0);
        // ties: duplicate a value inside a row
        let mut logits = logits;
        if experts > 5 {
            // a tie at the top (the lower index must come first) and one lower down
            logits[2] = 5.0;
            logits[experts - 1] = 5.0;
            logits[1] = logits[4];
            if experts > 66 {
                logits[66] = 5.0; // same lane as expert 2: the tie is decided inside one lane
            }
        }
        let cpu = CpuBackend::new();
        let mut st = cpu.create_stream().unwrap();
        let lb = upload_f32(&cpu, st.as_mut(), &logits);
        let mut wbuf = cpu.allocate_buffer(rows * k * 4).unwrap();
        let mut ibuf = cpu.allocate_buffer(rows * k * 4).unwrap();
        st.moe_route(&mut wbuf, &mut ibuf, &lb, experts, k).unwrap();
        let want_w = download_f32(st.as_mut(), &wbuf, rows * k);
        let want_i: Vec<i32> = download(st.as_mut(), &ibuf, rows * k * 4)
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        for r in 0..rows {
            let (ids, w) = moe_topk_row(&logits[r * experts..(r + 1) * experts], k, true, 1.0);
            assert_eq!(
                ids,
                want_i[r * k..(r + 1) * k],
                "ids row {r} ({experts} experts, k {k})"
            );
            check_close(&w, &want_w[r * k..(r + 1) * k], "weights");
            let (ids2, w2) = moe_topk_row(&logits[r * experts..(r + 1) * experts], k, false, 0.5);
            assert_eq!(ids, ids2);
            // unnormalised scores are the softmax probabilities of the chosen experts, times the scale
            let row = &logits[r * experts..(r + 1) * experts];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum: f32 = row.iter().map(|v| (v - mx).exp()).sum();
            for (j, &id) in ids2.iter().enumerate() {
                assert!((w2[j] - 0.5 * (row[id as usize] - mx).exp() / sum).abs() < 1e-6);
            }
        }
    }
}

#[test]
fn delta_net_matches_the_cpu_reference() {
    use crate::cpu_delta::{run, Layer};
    let mut rng = Lcg(77);
    for (kh, vh, hk, hv, ksz, tiled) in [
        (2usize, 2usize, 64usize, 64usize, 4usize, false),
        (2, 4, 64, 32, 4, true),
        (4, 8, 128, 96, 4, false),
        (1, 2, 32, 17, 2, true),
        (2, 4, 96, 128, 8, false),
    ] {
        let conv_dim = 2 * kh * hk + vh * hv;
        let value_dim = vh * hv;
        let conv = rng.vector(conv_dim * ksz, 0.6);
        let dt_bias = rng.vector(vh, 1.0);
        let decay: Vec<f32> = (0..vh).map(|_| -(0.3 + rng.unit().abs() * 2.0)).collect();
        let norm: Vec<f32> = (0..hv).map(|_| 1.0 + 0.25 * rng.unit()).collect();
        let layer = Layer {
            k_heads: kh,
            v_heads: vh,
            k_dim: hk,
            v_dim: hv,
            conv_kernel: ksz,
            beta_scale: 1.0,
            eps: 1e-6,
            tiled_heads: tiled,
            conv: &conv,
            dt_bias: &dt_bias,
            decay: &decay,
            norm: &norm,
        };
        // CPU state
        let mut cpu_hist = vec![0.0f32; conv_dim * (ksz - 1)];
        let mut cpu_rec = vec![0.0f32; vh * hk * hv];
        // mirror state: recurrent matrices, then the history
        let rec_floats = vh * hk * hv;
        let mut state = vec![0.0f32; rec_floats + conv_dim * (ksz - 1)];
        let mut row0 = 0usize;
        for (n, reset) in [
            (19usize, true),
            (1, false),
            (5, false),
            (40, false),
            (2, true),
        ] {
            let qkv = rng.vector(n * conv_dim, 1.0);
            let z = rng.vector(n * value_dim, 1.0);
            let beta = rng.vector(n * vh, 1.0);
            let alpha = rng.vector(n * vh, 1.0);
            if reset {
                cpu_hist.fill(0.0);
                cpu_rec.fill(0.0);
            }
            let mut want = vec![0.0f32; n * value_dim];
            run(
                &layer,
                &mut cpu_hist,
                &mut cpu_rec,
                (&qkv, &z, &beta, &alpha),
                &mut want,
                n,
            );
            // mirror: the run sits at rows row0 .. row0 + n of larger activation buffers
            let total = row0 + n;
            let pad = |v: &[f32], width: usize| -> Vec<f32> {
                let mut out = vec![0.7f32; total * width];
                out[row0 * width..].copy_from_slice(v);
                out
            };
            let (qkv_p, z_p, beta_p, alpha_p) = (
                pad(&qkv, conv_dim),
                pad(&z, value_dim),
                pad(&beta, vh),
                pad(&alpha, vh),
            );
            let mut mixed = vec![0.0f32; total * conv_dim];
            let blocks = 2 * kh + value_dim.div_ceil(128);
            let chunks = n.div_ceil(16);
            let mut order: Vec<(usize, usize)> = (0..blocks)
                .flat_map(|b| (0..chunks).map(move |c| (b, c)))
                .collect();
            let perm = shuffled(order.len(), &mut rng);
            order = perm.iter().map(|&i| order[i]).collect();
            delta_conv(
                &qkv_p,
                &conv,
                &mut state,
                &mut mixed,
                (conv_dim, kh, hk, ksz),
                (row0, n, reset, rec_floats),
                (hk as f32).powf(-0.5),
                &order,
            );
            let mut have = vec![0.0f32; total * value_dim];
            for j in 0..vh {
                delta_rec_head(
                    j,
                    &mixed,
                    (&z_p, &beta_p, &alpha_p),
                    (&dt_bias, &decay, &norm),
                    &mut state,
                    &mut have,
                    (kh, vh, hv, hk, conv_dim),
                    (tiled, row0, n, reset, 0),
                    (1.0, 1e-6),
                );
            }
            check_close(
                &have[row0 * value_dim..],
                &want,
                &format!("delta {kh}/{vh} {hk}x{hv} k{ksz} tiled {tiled} run n={n}"),
            );
            row0 += n;
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------
// mamba_conv.comp / mamba_scan2.comp / mamba_gate_norm.comp / mamba_scan1.comp
// ------------------------------------------------------------------------------------------------------------------------------------------

fn softplus20(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// mamba_conv.comp: all workgroups `(channel block, chunk)` of a run, in `order`.
pub(super) fn mamba_conv(
    x: &[f32],
    (in_stride, in_off): (usize, usize),
    convw: &[f32],
    convb: Option<&[f32]>,
    state: &mut [f32],
    mixed: &mut [f32],
    (channels, ksz): (usize, usize),
    (row0, n, reset, hist_off): (usize, usize, bool, usize),
    order: &[(usize, usize)],
) {
    let hist_len = ksz - 1;
    for &(blk, ychunk) in order {
        let t0 = ychunk * 16;
        for lane in 0..64 {
            let c = blk * 64 + lane;
            if c >= channels || t0 >= n {
                continue;
            }
            let t1 = n.min(t0 + 16);
            let hist = hist_off + c * hist_len;
            let bias = convb.map_or(0.0, |b| b[c]);
            let mut wk = [0.0f32; 8];
            let mut win = [0.0f32; 8];
            for k in 0..8 {
                wk[k] = if k < ksz { convw[c * ksz + k] } else { 0.0 };
            }
            for j in 0..7 {
                if j < hist_len {
                    win[j] = if t0 + j >= hist_len {
                        x[(row0 + t0 + j - hist_len) * in_stride + in_off + c]
                    } else if !reset {
                        state[hist + t0 + j]
                    } else {
                        0.0
                    };
                }
            }
            for t in t0..t1 {
                let cur = x[(row0 + t) * in_stride + in_off + c];
                let mut acc = wk[hist_len] * cur + bias;
                for k in 0..7 {
                    if k < hist_len {
                        acc += wk[k] * win[k];
                    }
                }
                mixed[(row0 + t) * channels + c] = silu(acc);
                for k in 0..6 {
                    if k + 1 < hist_len {
                        win[k] = win[k + 1];
                    }
                }
                win[hist_len - 1] = cur;
            }
            if ychunk == 0 {
                let mut fresh = [0.0f32; 7];
                for k in 0..7 {
                    if k < hist_len {
                        fresh[k] = if n + k >= hist_len {
                            x[(row0 + n + k - hist_len) * in_stride + in_off + c]
                        } else if !reset {
                            state[hist + n + k]
                        } else {
                            0.0
                        };
                    }
                }
                for k in 0..7 {
                    if k < hist_len {
                        state[hist + k] = fresh[k];
                    }
                }
            }
        }
    }
}

pub(super) struct M2 {
    pub inner: usize,
    pub ns: usize,
    pub heads: usize,
    pub groups: usize,
    pub head_dim: usize,
    pub conv_dim: usize,
    pub step_min: f32,
}

/// mamba_scan2.comp for a run: every (head, channel block) workgroup.
pub(super) fn mamba_scan2(
    s: &M2,
    mixed: &[f32],
    projected: &[f32],
    (proj_stride, proj_dt_off): (usize, usize),
    (dt_bias, decay, skip): (&[f32], &[f32], &[f32]),
    state: &mut [f32],
    out: &mut [f32],
    (row0, n, reset, state_off): (usize, usize, bool, usize),
) {
    for h in 0..s.heads {
        for blk in 0..s.head_dim.div_ceil(64) {
            let g = h / (s.heads / s.groups).max(1);
            for lane in 0..64 {
                let pp = blk * 64 + lane;
                if pp >= s.head_dim {
                    continue;
                }
                let cbase = state_off + (h * s.head_dim + pp) * s.ns;
                if reset {
                    for k in 0..s.ns {
                        state[cbase + k] = 0.0;
                    }
                }
                for t in 0..n {
                    let row = row0 + t;
                    let sb = &mixed[row * s.conv_dim + s.inner + g * s.ns..][..s.ns];
                    let sc =
                        &mixed[row * s.conv_dim + s.inner + s.groups * s.ns + g * s.ns..][..s.ns];
                    let raw = projected[row * proj_stride + proj_dt_off + h] + dt_bias[h];
                    let dt = softplus20(raw).max(s.step_min);
                    let da = (dt * decay[h]).exp();
                    let xp = mixed[row * s.conv_dim + h * s.head_dim + pp];
                    let mut acc = 0.0f32;
                    for k in 0..s.ns {
                        let cv = state[cbase + k] * da + dt * xp * sb[k];
                        state[cbase + k] = cv;
                        acc += cv * sc[k];
                    }
                    out[row * s.inner + h * s.head_dim + pp] = acc + skip[h] * xp;
                }
            }
        }
    }
}

/// mamba_gate_norm.comp.
pub(super) fn mamba_gate_norm(
    y: &mut [f32],
    projected: &[f32],
    norm_w: &[f32],
    (inner, groups, rows): (usize, usize, usize),
    (proj_stride, proj_gate_off): (usize, usize),
    eps: f32,
) {
    let size = inner / groups;
    for row in 0..rows {
        for g in 0..groups {
            let v: Vec<f32> = (0..size)
                .map(|i| {
                    y[row * inner + g * size + i]
                        * silu(projected[row * proj_stride + proj_gate_off + g * size + i])
                })
                .collect();
            let ss: f32 = v.iter().map(|x| x * x).sum();
            let inv = 1.0 / (ss / size as f32 + eps).sqrt();
            for i in 0..size {
                y[row * inner + g * size + i] = v[i] * inv * norm_w[g * size + i];
            }
        }
    }
}

pub(super) struct M1 {
    pub inner: usize,
    pub ns: usize,
    pub rank: usize,
}

/// mamba_scan1.comp for a run.
pub(super) fn mamba_scan1(
    s: &M1,
    mixed: &[f32],
    dt: &[f32],
    split: &[f32],
    projected: &[f32],
    (proj_stride, proj_gate_off): (usize, usize),
    (decay, skip): (&[f32], &[f32]),
    state: &mut [f32],
    out: &mut [f32],
    (row0, n, reset, state_off): (usize, usize, bool, usize),
) {
    let split_stride = s.rank + 2 * s.ns;
    for ch in 0..s.inner {
        let cbase = state_off + ch * s.ns;
        if reset {
            for k in 0..s.ns {
                state[cbase + k] = 0.0;
            }
        }
        for t in 0..n {
            let row = row0 + t;
            let step = softplus20(dt[row * s.inner + ch]);
            let xc = mixed[row * s.inner + ch];
            let mut acc = 0.0f32;
            for k in 0..s.ns {
                let b = split[row * split_stride + s.rank + k];
                let c = split[row * split_stride + s.rank + s.ns + k];
                let cv = state[cbase + k] * (step * decay[ch * s.ns + k]).exp() + step * b * xc;
                state[cbase + k] = cv;
                acc += cv * c;
            }
            let z = projected[row * proj_stride + proj_gate_off + ch];
            out[row * s.inner + ch] = (acc + skip[ch] * xc) * silu(z);
        }
    }
}

// ---- references transcribed from `mamba_forward` / `mamba1_forward` in rust_decoder.rs (one token at a time) ----

struct RefState {
    conv: Vec<f32>,
    rec: Vec<f32>,
}

fn ref_conv(x: &[f32], w: &[f32], bias: Option<&[f32]>, ksz: usize, state: &mut [f32]) -> Vec<f32> {
    let mut mixed = vec![0.0f32; x.len()];
    for (c, mixed) in mixed.iter_mut().enumerate() {
        let weights = &w[c * ksz..(c + 1) * ksz];
        let history = &mut state[c * (ksz - 1)..(c + 1) * (ksz - 1)];
        let mut acc = weights[ksz - 1] * x[c] + bias.map_or(0.0, |b| b[c]);
        for (wv, h) in weights.iter().zip(history.iter()) {
            acc += wv * h;
        }
        if ksz > 1 {
            history.copy_within(1.., 0);
            history[ksz - 2] = x[c];
        }
        *mixed = acc / (1.0 + (-acc).exp());
    }
    mixed
}

#[test]
fn mamba2_matches_the_decoder_reference() {
    let mut rng = Lcg(31);
    for (inner, ns, heads, groups, ksz, norm_groups) in [
        (64usize, 16usize, 4usize, 2usize, 4usize, 2usize),
        (128, 48, 8, 1, 2, 1),
        (192, 128, 6, 3, 4, 3),
        (96, 16, 3, 3, 8, 3),
    ] {
        let conv_dim = inner + 2 * groups * ns;
        let width = inner + conv_dim + heads;
        let head_dim = inner / heads;
        let per_group = heads / groups;
        let conv = rng.vector(conv_dim * ksz, 0.5);
        let conv_bias = rng.vector(conv_dim, 0.3);
        let dt_bias = rng.vector(heads, 1.0);
        let decay: Vec<f32> = (0..heads).map(|_| -(0.2 + rng.unit().abs())).collect();
        let skip = rng.vector(heads, 1.0);
        let norm_w: Vec<f32> = (0..inner).map(|_| 1.0 + 0.2 * rng.unit()).collect();
        let (step_min, eps) = (0.001f32, 1e-5f32);
        let mut rf = RefState {
            conv: vec![0.0; conv_dim * (ksz - 1)],
            rec: vec![0.0; heads * head_dim * ns],
        };
        let hist_floats = conv_dim * (ksz - 1);
        let mut state = vec![0.0f32; 7 + hist_floats + heads * head_dim * ns];
        let (hist_off, rec_off) = (7usize, 7 + hist_floats);
        let mut row0 = 0;
        for (n, reset) in [(18usize, true), (1, false), (3, false), (17, true)] {
            if reset {
                rf.conv.fill(0.0);
                rf.rec.fill(0.0);
            }
            let projected = rng.vector(n * width, 1.0);
            // reference
            let mut want = vec![0.0f32; n * inner];
            for r in 0..n {
                let row = &projected[r * width..(r + 1) * width];
                let (gate, rest) = row.split_at(inner);
                let (xbc, dt_raw) = rest.split_at(conv_dim);
                let mixed = ref_conv(xbc, &conv, Some(&conv_bias), ksz, &mut rf.conv);
                let (xs, bc) = mixed.split_at(inner);
                let (b_all, c_all) = bc.split_at(groups * ns);
                let scan = &mut want[r * inner..(r + 1) * inner];
                for (h, (y, s)) in scan
                    .chunks_mut(head_dim)
                    .zip(rf.rec.chunks_mut(head_dim * ns))
                    .enumerate()
                {
                    let raw = dt_raw[h] + dt_bias[h];
                    let dt = (if raw > 20.0 { raw } else { raw.exp().ln_1p() }).max(step_min);
                    let da = (dt * decay[h]).exp();
                    let g = h / per_group.max(1);
                    let b = &b_all[g * ns..(g + 1) * ns];
                    let c = &c_all[g * ns..(g + 1) * ns];
                    for (p, y) in y.iter_mut().enumerate() {
                        let xp = xs[h * head_dim + p];
                        let cell = &mut s[p * ns..(p + 1) * ns];
                        let mut acc = 0.0f32;
                        for k in 0..ns {
                            cell[k] = cell[k] * da + dt * xp * b[k];
                            acc += cell[k] * c[k];
                        }
                        *y = acc + skip[h] * xp;
                    }
                }
                for (v, g) in scan.iter_mut().zip(gate) {
                    *v *= g / (1.0 + (-g).exp());
                }
                let size = inner / norm_groups;
                for (chunk, weight) in scan.chunks_mut(size).zip(norm_w.chunks(size)) {
                    let inv = (chunk.iter().map(|v| v * v).sum::<f32>() / size as f32 + eps)
                        .sqrt()
                        .recip();
                    for (v, w) in chunk.iter_mut().zip(weight) {
                        *v *= inv * w;
                    }
                }
            }
            // mirror: the run sits at rows row0 .. row0 + n of larger buffers
            let total = row0 + n;
            let mut proj_p = vec![0.3f32; total * width];
            proj_p[row0 * width..].copy_from_slice(&projected);
            let mut mixed = vec![0.0f32; total * conv_dim];
            let chunks = n.div_ceil(16);
            let blocks = conv_dim.div_ceil(64);
            let mut order: Vec<(usize, usize)> = (0..blocks)
                .flat_map(|b| (0..chunks).map(move |c| (b, c)))
                .collect();
            let perm = shuffled(order.len(), &mut rng);
            order = perm.iter().map(|&i| order[i]).collect();
            mamba_conv(
                &proj_p,
                (width, inner),
                &conv,
                Some(&conv_bias),
                &mut state,
                &mut mixed,
                (conv_dim, ksz),
                (row0, n, reset, hist_off),
                &order,
            );
            let mut y = vec![0.0f32; total * inner];
            let sh = M2 {
                inner,
                ns,
                heads,
                groups,
                head_dim,
                conv_dim,
                step_min,
            };
            mamba_scan2(
                &sh,
                &mixed,
                &proj_p,
                (width, inner + conv_dim),
                (&dt_bias, &decay, &skip),
                &mut state,
                &mut y,
                (row0, n, reset, rec_off),
            );
            let mut gated = y[row0 * inner..].to_vec();
            let proj_run = &proj_p[row0 * width..];
            mamba_gate_norm(
                &mut gated,
                proj_run,
                &norm_w,
                (inner, norm_groups, n),
                (width, 0),
                eps,
            );
            check_close(
                &gated,
                &want,
                &format!("mamba2 inner {inner} ns {ns} heads {heads}/{groups} k{ksz} run n={n}"),
            );
            row0 += n;
        }
    }
}

#[test]
fn mamba1_matches_the_decoder_reference() {
    let mut rng = Lcg(414);
    for (inner, ns, rank, ksz) in [
        (64usize, 16usize, 4usize, 4usize),
        (100, 8, 6, 2),
        (70, 20, 3, 5),
    ] {
        let conv = rng.vector(inner * ksz, 0.5);
        let conv_bias = rng.vector(inner, 0.3);
        let decay: Vec<f32> = (0..inner * ns).map(|_| -(0.1 + rng.unit().abs())).collect();
        let skip = rng.vector(inner, 1.0);
        let hist_floats = inner * (ksz - 1);
        let mut rf = RefState {
            conv: vec![0.0; hist_floats],
            rec: vec![0.0; inner * ns],
        };
        let mut state = vec![0.0f32; 3 + hist_floats + inner * ns];
        let (hist_off, rec_off) = (3usize, 3 + hist_floats);
        let mut row0 = 0;
        for (n, reset) in [(9usize, true), (1, false), (20, false)] {
            if reset {
                rf.conv.fill(0.0);
                rf.rec.fill(0.0);
            }
            let projected = rng.vector(n * 2 * inner, 1.0);
            let dt = rng.vector(n * inner, 2.0);
            let split = rng.vector(n * (rank + 2 * ns), 1.0);
            let mut want = vec![0.0f32; n * inner];
            for r in 0..n {
                let row = &projected[r * 2 * inner..(r + 1) * 2 * inner];
                let (xs, gate) = row.split_at(inner);
                let mixed = ref_conv(xs, &conv, Some(&conv_bias), ksz, &mut rf.conv);
                let b = &split[r * (rank + 2 * ns) + rank..][..ns];
                let c = &split[r * (rank + 2 * ns) + rank + ns..][..ns];
                for ch in 0..inner {
                    let raw = dt[r * inner + ch];
                    let step = if raw > 20.0 { raw } else { raw.exp().ln_1p() };
                    let xc = mixed[ch];
                    let s = &mut rf.rec[ch * ns..(ch + 1) * ns];
                    let mut acc = 0.0f32;
                    for k in 0..ns {
                        s[k] = s[k] * (step * decay[ch * ns + k]).exp() + step * b[k] * xc;
                        acc += s[k] * c[k];
                    }
                    let z = gate[ch];
                    want[r * inner + ch] = (acc + skip[ch] * xc) * (z / (1.0 + (-z).exp()));
                }
            }
            let total = row0 + n;
            let pad = |v: &[f32], width: usize| -> Vec<f32> {
                let mut out = vec![0.3f32; total * width];
                out[row0 * width..].copy_from_slice(v);
                out
            };
            let (proj_p, dt_p, split_p) = (
                pad(&projected, 2 * inner),
                pad(&dt, inner),
                pad(&split, rank + 2 * ns),
            );
            let mut mixed = vec![0.0f32; total * inner];
            let chunks = n.div_ceil(16);
            let order: Vec<(usize, usize)> = (0..inner.div_ceil(64))
                .flat_map(|b| (0..chunks).map(move |c| (b, c)))
                .collect();
            mamba_conv(
                &proj_p,
                (2 * inner, 0),
                &conv,
                Some(&conv_bias),
                &mut state,
                &mut mixed,
                (inner, ksz),
                (row0, n, reset, hist_off),
                &order,
            );
            let mut out = vec![0.0f32; total * inner];
            mamba_scan1(
                &M1 { inner, ns, rank },
                &mixed,
                &dt_p,
                &split_p,
                &proj_p,
                (2 * inner, inner),
                (&decay, &skip),
                &mut state,
                &mut out,
                (row0, n, reset, rec_off),
            );
            check_close(
                &out[row0 * inner..],
                &want,
                &format!("mamba1 inner {inner} ns {ns} k{ksz} run n={n}"),
            );
            row0 += n;
        }
    }
}
