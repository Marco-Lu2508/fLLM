//! Self-test cases for the DeepSeek-V4 attention operations (`v4_rope`, `v4_compress`, `v4_attention`, `v4_ring_write` and the indexer selection through
//! `dsa_score_select`) against an f64 reference of `v4_attention` and `pool_window` (rust_decoder/deepseek4.rs). A layer is run over two batches of two
//! sequences each (the second batch continues the state of the first), so the buffered projections, the overlap series, the ring and the entries carry over.
use super::*;

struct V4Case {
    heads: usize,
    hd: usize,
    rope: usize,
    window: usize,
    ratio: usize,
    o_groups: usize,
    /// Indexer: heads, width, top-k.
    index: Option<(usize, usize, usize)>,
    /// Rows of the two sequences in the first and the second batch.
    first: [usize; 2],
    second: [usize; 2],
}

const CASES: [V4Case; 11] = [
    V4Case { heads: 4, hd: 32, rope: 8, window: 5, ratio: 0, o_groups: 2, index: None, first: [7, 11], second: [9, 1] },
    V4Case { heads: 4, hd: 32, rope: 8, window: 6, ratio: 4, o_groups: 2, index: None, first: [7, 11], second: [9, 1] },
    V4Case { heads: 4, hd: 32, rope: 8, window: 6, ratio: 3, o_groups: 4, index: None, first: [20, 5], second: [1, 14] },
    V4Case { heads: 4, hd: 32, rope: 8, window: 6, ratio: 4, o_groups: 2, index: Some((3, 16, 3)), first: [26, 9], second: [14, 1] },
    V4Case { heads: 8, hd: 64, rope: 16, window: 16, ratio: 4, o_groups: 4, index: Some((4, 32, 4)), first: [70, 3], second: [1, 40] },
    V4Case { heads: 8, hd: 128, rope: 32, window: 12, ratio: 8, o_groups: 4, index: None, first: [90, 6], second: [11, 2] },
    // the widths of DeepSeek-V4-Flash: 64 heads of 512, window 128, indexer 64 x 128 (top-k cut to 32 so that the selection is exercised)
    V4Case { heads: 64, hd: 512, rope: 64, window: 128, ratio: 4, o_groups: 8, index: Some((64, 128, 32)), first: [150, 30], second: [20, 1] },
    V4Case { heads: 8, hd: 512, rope: 64, window: 16, ratio: 4, o_groups: 4, index: None, first: [40, 10], second: [5, 1] },
    V4Case { heads: 8, hd: 64, rope: 16, window: 128, ratio: 4, o_groups: 4, index: None, first: [150, 30], second: [5, 1] },
    V4Case { heads: 64, hd: 64, rope: 16, window: 16, ratio: 4, o_groups: 8, index: None, first: [40, 10], second: [5, 1] },
    V4Case { heads: 8, hd: 64, rope: 16, window: 16, ratio: 4, o_groups: 4, index: Some((64, 128, 32)), first: [150, 30], second: [5, 1] },
];

/// Reference state of one sequence: everything the sequence has fed in, by position.
#[derive(Default)]
struct RefSeq {
    kv: Vec<Vec<f64>>,
    ckv: Vec<Vec<f64>>,
    cgate: Vec<Vec<f64>>,
    ikv: Vec<Vec<f64>>,
    igate: Vec<Vec<f64>>,
    cs_entry: Vec<Vec<f64>>,
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| f64::from(x)).collect()
}

/// Entry `e` of a compressor from the projections of positions `e * ratio ..` (and the window before for two series).
#[allow(clippy::too_many_arguments)]
fn pool_ref(kv: &[Vec<f64>], gate: &[Vec<f64>], cs_entry: &[Vec<f64>], pos: &[f64], norm: &[f64], e: usize, ratio: usize, dim: usize, coff: usize, rope: usize) -> Vec<f64> {
    let w = coff * dim;
    let off = if coff == 2 { dim } else { 0 };
    let mut entry = vec![0.0f64; dim];
    for (c, out) in entry.iter_mut().enumerate() {
        let mut logits: Vec<(f64, f64)> = Vec::new();
        if coff == 2 && e > 0 {
            for j in 0..ratio {
                let p = (e - 1) * ratio + j;
                logits.push((gate[p][c] + pos[j * w + c], kv[p][c]));
            }
        }
        for j in 0..ratio {
            let p = e * ratio + j;
            logits.push((gate[p][off + c] + pos[j * w + off + c], kv[p][off + c]));
        }
        let max = logits.iter().map(|l| l.0).fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l.0 - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        *out = exps.iter().zip(&logits).map(|(x, l)| x / sum * l.1).sum();
    }
    let inv = (entry.iter().map(|v| v * v).sum::<f64>() / dim as f64 + 1e-6).sqrt().recip();
    entry.iter_mut().zip(norm).for_each(|(v, n)| *v *= inv * n);
    let cs = &cs_entry[e * ratio + ratio - 1];
    for j in 0..rope / 2 {
        let at = dim - rope + 2 * j;
        let (a, b) = (entry[at], entry[at + 1]);
        entry[at] = a * cs[2 * j] - b * cs[2 * j + 1];
        entry[at + 1] = b * cs[2 * j] + a * cs[2 * j + 1];
    }
    entry
}

impl Tester<'_> {
    pub(super) fn v4_cases(&mut self) {
        for (rows, groups, dim, rope, sign) in [(3usize, 2usize, 16usize, 8usize, 1.0f32), (5, 4, 64, 64, -1.0), (2, 1, 130, 30, 1.0)] {
            let supported = self.device.create_stream().map(|s| s.supports_v4(&V4Shape { heads: 2, head_dim: dim, rope, window: 4, ratio: 0, index: None, max_entries: 1, o_groups: 1 })).unwrap_or(false);
            let label = format!("{rows} rows x {groups} groups x {dim}, rope {rope}, sign {sign}");
            if !supported {
                self.record_skip("v4_rope", &label, "supports_v4 is false");
                continue;
            }
            let r = self.v4_rope_check(rows, groups, dim, rope, sign);
            self.record("v4_rope", &label, r);
        }
        for case in &CASES {
            let shape = V4Shape {
                heads: case.heads,
                head_dim: case.hd,
                rope: case.rope,
                window: case.window,
                ratio: case.ratio,
                index: case.index.map(|(heads, dim, topk)| V4Index { heads, dim, topk }),
                max_entries: case.first.iter().chain(&case.second).sum::<usize>().checked_div(case.ratio).map_or(1, |n| n + 2),
                o_groups: case.o_groups,
            };
            let label = format!(
                "{} heads x {}, window {}, {}{}, {} + {} rows",
                case.heads,
                case.hd,
                case.window,
                if case.ratio == 0 { "no entries".to_owned() } else { format!("ratio {}", case.ratio) },
                case.index.map_or(String::new(), |(h, d, k)| format!(", indexer {h} x {d} top-{k}")),
                case.first.iter().sum::<usize>(),
                case.second.iter().sum::<usize>()
            );
            if !self.device.create_stream().map(|s| s.supports_v4(&shape)).unwrap_or(false) {
                self.record_skip("v4_layer", &label, "supports_v4 is false");
                continue;
            }
            let r = self.v4_layer_check(case, &shape);
            self.record("v4_layer", &label, r);
        }
    }

    fn v4_rope_check(&mut self, rows: usize, groups: usize, dim: usize, rope: usize, sign: f32) -> Result<Outcome, BackendError> {
        let x = self.rng.vector(rows * groups * dim, 2.0);
        let angles = self.rng.vector(rows * rope / 2, 3.0);
        let cs: Vec<f32> = angles.iter().flat_map(|a| [a.cos(), a.sin()]).collect();
        let mut want = f64s(&x);
        for g in 0..rows * groups {
            let r = g / groups;
            for j in 0..rope / 2 {
                let (c, s) = (f64::from(cs[(r * (rope / 2) + j) * 2]), f64::from(cs[(r * (rope / 2) + j) * 2 + 1]) * f64::from(sign));
                let at = g * dim + dim - rope + 2 * j;
                let (a, b) = (f64::from(x[at]), f64::from(x[at + 1]));
                want[at] = a * c - b * s;
                want[at + 1] = b * c + a * s;
            }
        }
        let mut stream = self.device.create_stream()?;
        let mut xb = put(self.device, stream.as_mut(), &x)?;
        let cb = put(self.device, stream.as_mut(), &cs)?;
        let mut ms = 0.0;
        timed(&mut ms, stream.as_mut(), |s| s.v4_rope(&mut xb, &cb, rows, groups, dim, rope, sign))?;
        let have = get(stream.as_mut(), &xb, x.len())?;
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(Outcome { device: worst(&have, &want, &scale), cpu: 0.0, tol: 2e-6, cpu_tol: 1.0, ms })
    }

    fn v4_layer_check(&mut self, case: &V4Case, shape: &V4Shape) -> Result<Outcome, BackendError> {
        let (heads, hd, rope, window, ratio) = (case.heads, case.hd, case.rope, case.window, case.ratio);
        let coff = shape.coff();
        let idx = shape.index;
        let (ih, idim, topk) = idx.map_or((0, 0, 0), |i| (i.heads, i.dim, i.topk));
        let w = coff * hd;
        let iw_width = 2 * idim;
        let hpg = heads / case.o_groups;
        let lay = shape.layout();
        let state_offset = lay.total * 4;
        // parameters of the layer
        let pos = self.rng.vector(ratio.max(1) * w, 1.0);
        let norm: Vec<f32> = self.rng.vector(hd, 0.3).iter().map(|v| 1.0 + v).collect();
        let ipos = self.rng.vector(ratio.max(1) * iw_width.max(1), 1.0);
        let inorm: Vec<f32> = self.rng.vector(idim.max(1), 0.3).iter().map(|v| 1.0 + v).collect();
        let sinks = self.rng.vector(heads, 1.0);
        let mut stream = self.device.create_stream()?;
        let pos_b = put(self.device, stream.as_mut(), &pos)?;
        let norm_b = put(self.device, stream.as_mut(), &norm)?;
        let ipos_b = put(self.device, stream.as_mut(), &ipos)?;
        let inorm_b = put(self.device, stream.as_mut(), &inorm)?;
        let sinks_b = put(self.device, stream.as_mut(), &sinks)?;
        // two sequences, each with a state buffer holding a fake earlier layer first (checks the layer offset)
        let mut states = Vec::new();
        for _ in 0..2 {
            let mut b = self.device.allocate_buffer(2 * lay.total * 4)?;
            stream.upload(&mut b, &f32_bytes(&vec![0.0f32; 2 * lay.total]))?;
            states.push(b);
        }
        let mut refs = [RefSeq::default(), RefSeq::default()];
        let mut worst_err = 0.0f64;
        let mut total_ms = 0.0;
        let mut position = [0usize; 2];
        for counts in [case.first, case.second] {
            let rows: usize = counts.iter().sum();
            // inputs of the rows (sequence 0's rows first)
            let q = self.rng.vector(rows * heads * hd, 1.0);
            let kv = self.rng.vector(rows * hd, 1.0);
            let ckv = self.rng.vector(rows * w, 1.0);
            let cgate = self.rng.vector(rows * w, 1.0);
            let ikv = self.rng.vector(rows * iw_width, 1.0);
            let igate = self.rng.vector(rows * iw_width, 1.0);
            let iq = self.rng.vector(rows * ih * idim, 1.0);
            let iwt = self.rng.vector(rows * ih, 1.0);
            let angles = self.rng.vector(rows * rope / 2, 2.0);
            let cs: Vec<f32> = angles.iter().flat_map(|a| [a.cos(), a.sin()]).collect();
            let angles_e = self.rng.vector(rows * rope / 2, 2.0);
            let cs_entry: Vec<f32> = angles_e.iter().flat_map(|a| [a.cos(), a.sin()]).collect();
            let qb = put(self.device, stream.as_mut(), &q)?;
            let kvb = put(self.device, stream.as_mut(), &kv)?;
            let ckvb = put(self.device, stream.as_mut(), &ckv)?;
            let cgb = put(self.device, stream.as_mut(), &cgate)?;
            let pad = |v: &[f32]| if v.is_empty() { vec![0.0f32] } else { v.to_vec() };
            let ikvb = put(self.device, stream.as_mut(), &pad(&ikv))?;
            let igb = put(self.device, stream.as_mut(), &pad(&igate))?;
            let iqb = put(self.device, stream.as_mut(), &pad(&iq))?;
            let iwb = put(self.device, stream.as_mut(), &pad(&iwt))?;
            let csb = put(self.device, stream.as_mut(), &cs)?;
            let ceb = put(self.device, stream.as_mut(), &cs_entry)?;
            let mut out = put(self.device, stream.as_mut(), &vec![f32::NAN; rows * heads * hd])?;
            let mut sel = self.device.allocate_buffer((rows * topk).max(1) * 4)?;
            stream.upload(&mut sel, &vec![0xffu8; (rows * topk).max(1) * 4])?;
            let mut first = 0;
            let mut specs = Vec::new();
            for (s, &n) in counts.iter().enumerate() {
                specs.push((s, first, n, position[s]));
                first += n;
            }
            let runs: Vec<V4Run<'_>> = specs.iter().map(|&(s, first_row, n, p)| V4Run { state: &states[s], first_row, rows: n, position: p }).collect();
            let mut ms = 0.0;
            let started = std::time::Instant::now();
            if ratio > 0 {
                stream.v4_compress(V4Pool::Attention, shape, &runs, state_offset, &ckvb, &cgb, &pos_b, &norm_b, &ceb, 1e-6)?;
            }
            if let Some(index) = idx {
                stream.v4_compress(V4Pool::Index, shape, &runs, state_offset, &ikvb, &igb, &ipos_b, &inorm_b, &ceb, 1e-6)?;
                let params = DsaParams {
                    heads: ih,
                    dim: idim,
                    topk: index.topk,
                    stride: 1,
                    block_size: shape.entry_cap(),
                    scale: (idim as f32).powf(-0.5),
                    head_scale: (ih as f32).powf(-0.5),
                    kpool: 1,
                    always_tail: false,
                };
                for run in &runs {
                    let lengths: Vec<i64> = (0..run.rows).map(|i| (((run.position + i + 1) / ratio).max(1)) as i64).collect();
                    let store = run.state.slice(state_offset + lay.index_entries * 4, shape.entry_cap() * idim * 4)?;
                    let q_run = iqb.slice(run.first_row * ih * idim * 4, run.rows * ih * idim * 4)?;
                    let w_run = iwb.slice(run.first_row * ih * 4, run.rows * ih * 4)?;
                    let mut sel_run = sel.slice(run.first_row * topk * 4, run.rows * topk * 4)?;
                    stream.dsa_score_select(&mut sel_run, &store, &q_run, &w_run, &vec![0i64; run.rows], &lengths, &params)?;
                }
            }
            stream.v4_attention(&mut out, &qb, &kvb, shape, &runs, state_offset, &sinks_b, idx.map(|_| &sel), &csb)?;
            stream.v4_ring_write(shape, &runs, state_offset, &kvb)?;
            stream.synchronize()?;
            ms += started.elapsed().as_secs_f64() * 1e3;
            total_ms += ms;
            let have = get(stream.as_mut(), &out, rows * heads * hd)?;
            // reference
            let mut want = vec![0.0f64; rows * heads * hd];
            for &(s, first_row, n, p0) in &specs {
                for i in 0..n {
                    let row = first_row + i;
                    let sq = &mut refs[s];
                    sq.kv.push(f64s(&kv[row * hd..(row + 1) * hd]));
                    sq.ckv.push(f64s(&ckv[row * w..(row + 1) * w]));
                    sq.cgate.push(f64s(&cgate[row * w..(row + 1) * w]));
                    sq.ikv.push(f64s(&ikv[row * iw_width..(row + 1) * iw_width]));
                    sq.igate.push(f64s(&igate[row * iw_width..(row + 1) * iw_width]));
                    sq.cs_entry.push(f64s(&cs_entry[row * rope..(row + 1) * rope]));
                    let p = p0 + i;
                    let sq = &refs[s];
                    let count = (p + 1).checked_div(ratio).unwrap_or(0);
                    let entries: Vec<Vec<f64>> = (0..count).map(|e| pool_ref(&sq.ckv, &sq.cgate, &sq.cs_entry, &f64s(&pos), &f64s(&norm), e, ratio, hd, coff, rope)).collect();
                    let picked: Vec<usize> = match idx {
                        Some(index) if count > 0 => {
                            let ientries: Vec<Vec<f64>> = (0..count).map(|e| pool_ref(&sq.ikv, &sq.igate, &sq.cs_entry, &f64s(&ipos), &f64s(&inorm), e, ratio, idim, 2, rope)).collect();
                            let mut scored: Vec<(f64, usize)> = (0..count)
                                .map(|e| {
                                    let mut score = 0.0;
                                    for h in 0..ih {
                                        let dot: f64 = (0..idim).map(|d| f64::from(iq[(row * ih + h) * idim + d]) * ientries[e][d]).sum::<f64>() * (idim as f64).powf(-0.5);
                                        if dot > 0.0 {
                                            score += dot * f64::from(iwt[row * ih + h]) * (ih as f64).powf(-0.5);
                                        }
                                    }
                                    (score, e)
                                })
                                .collect();
                            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
                            let mut chosen: Vec<usize> = scored.iter().take(index.topk).map(|x| x.1).collect();
                            chosen.sort_unstable();
                            chosen
                        }
                        _ => (0..count).collect(),
                    };
                    let mut keys: Vec<&Vec<f64>> = Vec::new();
                    for pp in (p + 1).saturating_sub(window)..=p {
                        keys.push(&sq.kv[pp]);
                    }
                    for &e in &picked {
                        keys.push(&entries[e]);
                    }
                    let scale = (hd as f64).powf(-0.5);
                    for h in 0..heads {
                        let query: Vec<f64> = f64s(&q[(row * heads + h) * hd..(row * heads + h + 1) * hd]);
                        let mut scores: Vec<f64> = keys.iter().map(|k| k.iter().zip(&query).map(|(a, b)| a * b).sum::<f64>() * scale).collect();
                        let sink = f64::from(sinks[h]);
                        let max = scores.iter().copied().fold(sink, f64::max);
                        let mut sum = (sink - max).exp();
                        for sc in &mut scores {
                            *sc = (*sc - max).exp();
                            sum += *sc;
                        }
                        let mut o = vec![0.0f64; hd];
                        for (sc, k) in scores.iter().zip(&keys) {
                            o.iter_mut().zip(k.iter()).for_each(|(o, x)| *o += sc / sum * x);
                        }
                        for j in 0..rope / 2 {
                            let at = hd - rope + 2 * j;
                            let (c, sn) = (f64::from(cs[row * rope + 2 * j]), -f64::from(cs[row * rope + 2 * j + 1]));
                            let (a, b) = (o[at], o[at + 1]);
                            o[at] = a * c - b * sn;
                            o[at + 1] = b * c + a * sn;
                        }
                        // group-major result
                        let (g, lh) = (h / hpg, h % hpg);
                        let dst = (g * rows + row) * hpg * hd + lh * hd;
                        want[dst..dst + hd].copy_from_slice(&o);
                    }
                }
                position[s] = p0 + n;
            }
            let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
            worst_err = worst_err.max(worst(&have, &want, &scale));
            if std::env::var_os("FLLM_V4_DEBUG").is_some() {
                // (row, head) pairs with a wrong value: where the output of a failing case goes wrong
                let mut bad: std::collections::BTreeMap<(usize, usize), usize> = Default::default();
                for (i, (h, w)) in have.iter().zip(&want).enumerate() {
                    if (f64::from(*h) - w).abs() > 1e-3 * w.abs().max(1.0) {
                        let (g, rest) = (i / (rows * hpg * hd), i % (rows * hpg * hd));
                        let (row, rest) = (rest / (hpg * hd), rest % (hpg * hd));
                        *bad.entry((row, g * hpg + rest / hd)).or_default() += 1;
                    }
                }
                eprintln!("v4 debug: {} bad (row, head) pairs; first {:?}", bad.len(), bad.iter().take(8).collect::<Vec<_>>());
            }
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 3e-4, cpu_tol: 1.0, ms: total_ms })
    }
}
