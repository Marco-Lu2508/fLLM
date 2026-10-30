//! Self-test cases for the sparse-attention operations of DeepSeek-V3.2 / GLM-5: the lightning indexer (`dsa_select`) and the attention over its
//! selection (`dsa_attention`) against an f64 reference.
use super::*;

/// One batch: sequences of `lengths` cached tokens; `prefill` rows per sequence (the last positions, causally; 0: all positions).
struct DsaCase {
    heads: usize,
    dim: usize,
    topk: usize,
    /// Attention: heads, key and value width.
    att_heads: usize,
    dk: usize,
    dv: usize,
    block_size: usize,
    stride: usize,
    pool_blocks: usize,
    lengths: &'static [usize],
    /// Rows per sequence (the last `rows` positions); 0 = every position.
    rows: usize,
    /// Keys and queries that make many scores exactly zero (the tie rule decides).
    ties: bool,
    /// Attention sinks (one logit per head that adds to the softmax denominator).
    sinks: bool,
}

const CASES: [DsaCase; 5] = [
    DsaCase { heads: 4, dim: 16, topk: 8, att_heads: 4, dk: 24, dv: 16, block_size: 4, stride: 24, pool_blocks: 40, lengths: &[37, 12], rows: 0, ties: false, sinks: false },
    DsaCase { heads: 3, dim: 20, topk: 8, att_heads: 6, dk: 24, dv: 16, block_size: 5, stride: 24, pool_blocks: 40, lengths: &[50], rows: 6, ties: true, sinks: true },
    DsaCase { heads: 8, dim: 32, topk: 64, att_heads: 4, dk: 80, dv: 64, block_size: 16, stride: 24, pool_blocks: 48, lengths: &[300, 70], rows: 4, ties: false, sinks: true },
    DsaCase { heads: 2, dim: 8, topk: 16, att_heads: 2, dk: 16, dv: 16, block_size: 8, stride: 24, pool_blocks: 48, lengths: &[16, 3, 90], rows: 1, ties: true, sinks: false },
    DsaCase { heads: 64, dim: 128, topk: 512, att_heads: 16, dk: 192, dv: 128, block_size: 16, stride: 160, pool_blocks: 170, lengths: &[1500], rows: 3, ties: false, sinks: false },
];

impl Tester<'_> {
    pub(super) fn dsa_cases(&mut self) {
        for (i, case) in CASES.iter().enumerate() {
            let params = DsaParams {
                heads: case.heads,
                dim: case.dim,
                topk: case.topk,
                stride: case.stride,
                block_size: case.block_size,
                scale: (case.dim as f32).powf(-0.5),
                head_scale: (case.heads as f32).powf(-0.5),
            };
            let supported = self.device.create_stream().map(|s| s.supports_dsa(&params, case.dk, 1)).unwrap_or(false);
            let label = format!("{} heads x {}, top-{}, {} tokens", case.heads, case.dim, case.topk, case.lengths.iter().sum::<usize>());
            if !supported {
                self.record_skip("dsa", &label, "supports_dsa is false");
                continue;
            }
            let r = self.dsa_check(case, &params, i == 1);
            self.record("dsa", &label, r);
        }
    }

    fn dsa_check(&mut self, case: &DsaCase, p: &DsaParams, decode_after: bool) -> Result<Outcome, BackendError> {
        let (heads, dim, topk) = (case.heads, case.dim, case.topk);
        // random block tables
        let mut free: Vec<usize> = (0..case.pool_blocks).collect();
        for i in (1..free.len()).rev() {
            let j = (self.rng.next() % (i as u64 + 1)) as usize;
            free.swap(i, j);
        }
        let mut tables_seq = Vec::new();
        for &n in case.lengths {
            let mut table = vec![-1i64; case.stride];
            for entry in table.iter_mut().take(n.div_ceil(case.block_size)) {
                *entry = free.pop().ok_or_else(|| BackendError::InvalidShape("pool too small".into()))? as i64;
            }
            tables_seq.push(table);
        }
        let slot_of = |s: usize, pos: usize| tables_seq[s][pos / case.block_size] as usize * case.block_size + pos % case.block_size;
        // keys of every token, the rows of the first call: (sequence, position)
        let keys: Vec<Vec<f32>> = case.lengths.iter().map(|&n| self.rng.vector(n * case.dim, 1.0)).collect();
        // every token is a row of the first call (it builds the key store); only the last `case.rows` rows of a sequence are checked (0: all)
        let mut row_list: Vec<(usize, usize)> = Vec::new();
        for (s, &n) in case.lengths.iter().enumerate() {
            for pos in 0..n {
                row_list.push((s, pos));
            }
        }
        let mut stream = self.device.create_stream()?;
        let mut store = self.device.allocate_buffer(case.pool_blocks * case.block_size * case.dim * 4)?;
        let zero = vec![0.0f32; case.pool_blocks * case.block_size * case.dim];
        stream.upload(&mut store, &f32_bytes(&zero))?;
        let mut worst_err = 0.0f64;
        let mut total_ms = 0.0;
        // call 1: all tokens as rows (builds the key store); call 2 (some cases): the last token of every sequence alone, as a decode step
        for decode in [false, true] {
            if decode && !decode_after {
                break;
            }
            let rows_here: Vec<(usize, usize)> = if decode { case.lengths.iter().enumerate().map(|(s, &n)| (s, n - 1)).collect() } else { row_list.clone() };
            let checked = |s: usize, pos: usize| decode || case.rows == 0 || pos + case.rows >= case.lengths[s];
            let rows = rows_here.len();
            let q = self.rng.vector(rows * heads * dim, 1.0);
            let w = self.rng.vector(rows * heads, 1.0);
            // ties: a negative query direction gives many exactly-zero scores (the ReLU)
            let q: Vec<f32> = if case.ties { q.iter().enumerate().map(|(i, &v)| if (i / (heads * dim)) % 2 == 0 { -v.abs() } else { v }).collect() } else { q };
            let k_new: Vec<f32> = rows_here.iter().flat_map(|&(s, pos)| keys[s][pos * dim..(pos + 1) * dim].to_vec()).collect();
            // for ties the keys are made non-negative so that a negative query scores exactly zero
            let k_new: Vec<f32> = if case.ties { k_new.iter().map(|v| v.abs()).collect() } else { k_new };
            let slots: Vec<usize> = rows_here.iter().map(|&(s, pos)| slot_of(s, pos)).collect();
            let lengths: Vec<i64> = rows_here.iter().map(|&(_, pos)| pos as i64 + 1).collect();
            let tables: Vec<i64> = rows_here.iter().flat_map(|&(s, _)| tables_seq[s].clone()).collect();
            let qb = put(self.device, stream.as_mut(), &q)?;
            let wb = put(self.device, stream.as_mut(), &w)?;
            let kb = put(self.device, stream.as_mut(), &k_new)?;
            let mut sel = self.device.allocate_buffer(rows * topk * 4)?;
            stream.upload(&mut sel, &vec![0xffu8; rows * topk * 4])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.dsa_select(&mut sel, &mut store, &qb, &kb, &wb, &slots, &tables, &lengths, p))?;
            total_ms += ms;
            let mut bytes = vec![0u8; rows * topk * 4];
            stream.download(&mut bytes, &sel)?;
            let have: Vec<i32> = bytes.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
            // reference
            let mut violations = 0usize;
            for (r, &(s, pos)) in rows_here.iter().enumerate() {
                let n = pos + 1;
                if !checked(s, pos) {
                    continue;
                }
                let expected: Vec<i32>;
                let got = &have[r * topk..(r + 1) * topk];
                if n <= topk {
                    let mut list: Vec<i32> = (0..n as i32).collect();
                    list.resize(topk, -1);
                    expected = list;
                    if got != expected.as_slice() {
                        violations += 1;
                    }
                } else {
                    let key_of = |position: usize| -> Vec<f32> {
                        let raw = &keys[s][position * dim..(position + 1) * dim];
                        if case.ties { raw.iter().map(|v| v.abs()).collect() } else { raw.to_vec() }
                    };
                    let scores: Vec<f64> = (0..n)
                        .map(|position| {
                            let key = key_of(position);
                            let mut score = 0.0f64;
                            for h in 0..heads {
                                let dot: f64 = (0..dim).map(|d| f64::from(q[(r * heads + h) * dim + d]) * f64::from(key[d])).sum::<f64>() * f64::from(p.scale);
                                if dot > 0.0 {
                                    score += dot * f64::from(w[r * heads + h]) * f64::from(p.head_scale);
                                }
                            }
                            score
                        })
                        .collect();
                    let mut order: Vec<usize> = (0..n).collect();
                    order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
                    let kth = scores[order[topk - 1]];
                    let mut best: Vec<i32> = order[..topk].iter().map(|&x| x as i32).collect();
                    best.sort_unstable();
                    expected = best;
                    let sorted_unique = got.windows(2).all(|w2| w2[0] < w2[1]) && got.iter().all(|&x| x >= 0 && (x as usize) < n);
                    if !sorted_unique {
                        violations += 1;
                    } else if kth.abs() < 1e-30 {
                        // a tie at zero: the lower positions win, exactly
                        if got != expected.as_slice() {
                            violations += 1;
                        }
                    } else {
                        let eps = 2e-5 * kth.abs().max(1e-3);
                        let in_got: std::collections::HashSet<i32> = got.iter().copied().collect();
                        let in_want: std::collections::HashSet<i32> = expected.iter().copied().collect();
                        for &x in in_got.symmetric_difference(&in_want) {
                            if (scores[x as usize] - kth).abs() > eps {
                                violations += 1;
                            }
                        }
                    }
                }
            }
            worst_err = worst_err.max(violations as f64);
            // attention over the device selection (the reference selection can differ at near ties; the device list is what the kernel gets)
            let att = self.dsa_attention_check(case, stream.as_mut(), &rows_here, &have, &tables_seq, &tables, &checked)?;
            worst_err = worst_err.max(att);
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-5, cpu_tol: 1.0, ms: total_ms })
    }

    #[allow(clippy::too_many_arguments)]
    fn dsa_attention_check(
        &mut self,
        case: &DsaCase,
        stream: &mut dyn Stream,
        rows_here: &[(usize, usize)],
        sel: &[i32],
        tables_seq: &[Vec<i64>],
        tables: &[i64],
        checked: &dyn Fn(usize, usize) -> bool,
    ) -> Result<f64, BackendError> {
        let (heads, dk, dv, topk) = (case.att_heads, case.dk, case.dv, case.topk);
        let rows = rows_here.len();
        let kv_heads = heads;
        let slot_of = |s: usize, pos: usize| tables_seq[s][pos / case.block_size] as usize * case.block_size + pos % case.block_size;
        let cache = self.device.allocate_kv_cache(1, case.pool_blocks, case.block_size, kv_heads, dk, dv, 1)?;
        // keys and values of every cached token, written through the device cache write
        let mut slot_list = Vec::new();
        let mut k = Vec::new();
        let mut v = Vec::new();
        for (s, &n) in case.lengths.iter().enumerate() {
            for pos in 0..n {
                slot_list.push(slot_of(s, pos));
            }
            k.extend(self.rng.vector(n * kv_heads * dk, 1.0));
            v.extend(self.rng.vector(n * kv_heads * dv, 1.0));
        }
        let kb = put(self.device, stream, &k)?;
        let vb = put(self.device, stream, &v)?;
        stream.kv_cache_write(&cache, 0, &slot_list, &kb, &vb)?;
        let q = self.rng.vector(rows * heads * dk, 1.0);
        let sinks = case.sinks.then(|| Arc::new(self.rng.vector(heads, 2.0)));
        let qb = put(self.device, stream, &q)?;
        let mut selb = self.device.allocate_buffer(rows * topk * 4)?;
        stream.upload(&mut selb, &sel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        let mut out = put(self.device, stream, &vec![f32::NAN; rows * heads * dv])?;
        let params = AttentionParams {
            num_heads: heads,
            num_kv_heads: kv_heads,
            head_dim: dk,
            value_dim: dv,
            stride: case.stride,
            block_size: case.block_size,
            scale: 1.0 / (dk as f32).sqrt(),
            softcap: None,
            window: None,
            chunk: None,
            sinks: sinks.clone(),
            alibi_slopes: None,
        };
        stream.dsa_attention(&mut out, &qb, &cache, tables, 0, &selb, topk, &params)?;
        stream.synchronize()?;
        let have = get(stream, &out, rows * heads * dv)?;
        // reference over the f16-rounded cache values
        let rounded = |x: f32| f64::from(f16::from_f32(x).to_f32());
        let offsets: Vec<usize> = case.lengths.iter().scan(0usize, |acc, &n| { let o = *acc; *acc += n; Some(o) }).collect();
        let mut want = vec![0.0f64; rows * heads * dv];
        for (r, &(s, pos)) in rows_here.iter().enumerate() {
            if !checked(s, pos) {
                continue;
            }
            let list: Vec<usize> = sel[r * topk..(r + 1) * topk].iter().take_while(|&&x| x >= 0).map(|&x| x as usize).collect();
            for h in 0..heads {
                let scores: Vec<f64> = list
                    .iter()
                    .map(|&pos| {
                        let base = ((offsets[s] + pos) * kv_heads + h) * dk;
                        (0..dk).map(|d| f64::from(q[(r * heads + h) * dk + d]) * rounded(k[base + d])).sum::<f64>() * f64::from(params.scale)
                    })
                    .collect();
                let sink = sinks.as_ref().map(|s| f64::from(s[h]));
                let max = scores.iter().copied().fold(sink.unwrap_or(f64::NEG_INFINITY), f64::max);
                let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = exps.iter().sum::<f64>() + sink.map_or(0.0, |s| (s - max).exp());
                for d in 0..dv {
                    let acc: f64 = list.iter().zip(&exps).map(|(&pos, e)| e * rounded(v[((offsets[s] + pos) * kv_heads + h) * dv + d])).sum();
                    want[(r * heads + h) * dv + d] = acc / sum;
                }
            }
        }
        // rows that are not checked take the device value (no deviation)
        for (r, &(s, pos)) in rows_here.iter().enumerate() {
            if !checked(s, pos) {
                for i in r * heads * dv..(r + 1) * heads * dv {
                    want[i] = f64::from(have[i]);
                }
            }
        }
        let scale: Vec<f64> = want.iter().map(|v| v.abs().max(1.0)).collect();
        Ok(worst(&have, &want, &scale))
    }
}
