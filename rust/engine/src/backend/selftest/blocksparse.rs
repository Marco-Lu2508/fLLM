//! Self-test cases for the block-sparse attention of MiniMax-M3: the block indexer (`block_select`) and the attention over the selected blocks
//! (`block_attention`) against an f64 reference.
//!
//! The selection is compared as sets of blocks: the device scores in f32, so a block whose score is within a few ulp of the threshold may take the
//! place of another one; every difference must lie within that margin, except for blocks whose scores are exactly equal in the reference (keys repeated
//! block for block), where the lower block has to win exactly. The attention is compared over the selection the device wrote (so that one near tie
//! cannot fail two checks) against the f16-rounded cache.
use super::*;

struct BlockCase {
    /// Indexer heads and width.
    heads: usize,
    dim: usize,
    /// Keys per block, blocks kept (the local ones included), newest blocks always kept.
    block: usize,
    topk: usize,
    local: usize,
    /// Attention heads, key / value heads, key and value width.
    att_heads: usize,
    kv_heads: usize,
    dk: usize,
    dv: usize,
    block_size: usize,
    stride: usize,
    pool_blocks: usize,
    lengths: &'static [usize],
    /// Rows per sequence (the last `rows` positions); 0 = every position.
    rows: usize,
    /// The keys repeat every two blocks, so that many block scores are exactly equal.
    ties: bool,
}

const CASES: [BlockCase; 7] = [
    BlockCase { heads: 2, dim: 16, block: 4, topk: 3, local: 1, att_heads: 4, kv_heads: 2, dk: 16, dv: 16, block_size: 16, stride: 24, pool_blocks: 40, lengths: &[37, 12], rows: 0, ties: false },
    BlockCase { heads: 2, dim: 16, block: 4, topk: 3, local: 1, att_heads: 4, kv_heads: 2, dk: 16, dv: 16, block_size: 5, stride: 40, pool_blocks: 60, lengths: &[50, 7], rows: 0, ties: true },
    BlockCase { heads: 4, dim: 30, block: 16, topk: 4, local: 2, att_heads: 8, kv_heads: 4, dk: 24, dv: 24, block_size: 16, stride: 24, pool_blocks: 40, lengths: &[300, 90], rows: 5, ties: false },
    BlockCase { heads: 1, dim: 8, block: 8, topk: 2, local: 1, att_heads: 3, kv_heads: 1, dk: 80, dv: 64, block_size: 4, stride: 80, pool_blocks: 90, lengths: &[16, 3, 90], rows: 2, ties: false },
    BlockCase { heads: 4, dim: 128, block: 128, topk: 16, local: 1, att_heads: 16, kv_heads: 4, dk: 128, dv: 128, block_size: 16, stride: 400, pool_blocks: 410, lengths: &[5000], rows: 3, ties: false },
    BlockCase { heads: 4, dim: 128, block: 64, topk: 8, local: 1, att_heads: 8, kv_heads: 8, dk: 64, dv: 64, block_size: 16, stride: 160, pool_blocks: 230, lengths: &[2000, 1500], rows: 2, ties: false },
    BlockCase { heads: 2, dim: 16, block: 4, topk: 3, local: 0, att_heads: 2, kv_heads: 1, dk: 16, dv: 16, block_size: 16, stride: 24, pool_blocks: 40, lengths: &[30], rows: 0, ties: false },
];

impl Tester<'_> {
    pub(super) fn block_sparse_cases(&mut self) {
        for case in &CASES {
            let params = BlockSparseParams { heads: case.heads, dim: case.dim, block: case.block, topk: case.topk, local: case.local, stride: case.stride, block_size: case.block_size };
            let supported = self.device.create_stream().map(|s| s.supports_block_sparse(&params, case.att_heads, case.kv_heads, case.dk, 1)).unwrap_or(false);
            let label = format!("{} heads x {}, {} of blocks of {}{}, {} tokens", case.heads, case.dim, case.topk, case.block, if case.ties { " ties" } else { "" }, case.lengths.iter().sum::<usize>());
            if !supported {
                self.record_skip("block sparse", &label, "supports_block_sparse is false");
                continue;
            }
            let r = self.block_check(case, &params);
            self.record("block sparse", &label, r);
        }
    }

    fn block_check(&mut self, case: &BlockCase, p: &BlockSparseParams) -> Result<Outcome, BackendError> {
        let (heads, dim, topk, block) = (case.heads, case.dim, case.topk, case.block);
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
        let keys: Vec<Vec<f32>> = case
            .lengths
            .iter()
            .map(|&n| {
                let raw = self.rng.vector(n * dim, 1.0);
                if case.ties {
                    // repeat the keys of the first two blocks: blocks b and b + 2 score exactly the same
                    (0..n * dim).map(|i| raw[(i / (dim * block) % 2) * dim * block + i % (dim * block)]).collect()
                } else {
                    raw
                }
            })
            .collect();
        let mut row_list: Vec<(usize, usize)> = Vec::new();
        for (s, &n) in case.lengths.iter().enumerate() {
            for pos in 0..n {
                row_list.push((s, pos));
            }
        }
        let mut stream = self.device.create_stream()?;
        let mut store = self.device.allocate_buffer(case.pool_blocks * case.block_size * dim * 4)?;
        stream.upload(&mut store, &f32_bytes(&vec![0.0f32; case.pool_blocks * case.block_size * dim]))?;
        let mut worst_err = 0.0f64;
        let mut total_ms = 0.0;
        // every token is a row of the first calls (they build the key store, batch after batch); only the last `case.rows` rows of a sequence are checked
        let batches: Vec<Vec<(usize, usize)>> = row_list.chunks(256).map(<[(usize, usize)]>::to_vec).collect();
        for rows_here in batches {
            let checked = |s: usize, pos: usize| case.rows == 0 || pos + case.rows >= case.lengths[s];
            let rows = rows_here.len();
            let q = self.rng.vector(rows * heads * dim, 1.0);
            let k_new: Vec<f32> = rows_here.iter().flat_map(|&(s, pos)| keys[s][pos * dim..(pos + 1) * dim].to_vec()).collect();
            let slots: Vec<usize> = rows_here.iter().map(|&(s, pos)| slot_of(s, pos)).collect();
            let lengths: Vec<i64> = rows_here.iter().map(|&(_, pos)| pos as i64 + 1).collect();
            let tables: Vec<i64> = rows_here.iter().flat_map(|&(s, _)| tables_seq[s].clone()).collect();
            let qb = put(self.device, stream.as_mut(), &q)?;
            let kb = put(self.device, stream.as_mut(), &k_new)?;
            let mut sel = self.device.allocate_buffer(rows * heads * topk * 4)?;
            stream.upload(&mut sel, &vec![0xffu8; rows * heads * topk * 4])?;
            let mut ms = 0.0;
            timed(&mut ms, stream.as_mut(), |s| s.block_select(&mut sel, &mut store, &qb, &kb, &slots, &tables, &lengths, p))?;
            total_ms += ms;
            let mut bytes = vec![0u8; rows * heads * topk * 4];
            stream.download(&mut bytes, &sel)?;
            let have: Vec<i32> = bytes.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c)).collect();
            let mut violations = 0usize;
            for (r, &(s, pos)) in rows_here.iter().enumerate() {
                if !checked(s, pos) {
                    continue;
                }
                let n = pos + 1;
                let blocks = n.div_ceil(block);
                for h in 0..heads {
                    let got = &have[(r * heads + h) * topk..(r * heads + h + 1) * topk];
                    // reference: the maximum dot product per block, the newest blocks forced in
                    let mut score = vec![f64::NEG_INFINITY; blocks];
                    for j in 0..n {
                        let dot: f64 = (0..dim).map(|d| f64::from(q[(r * heads + h) * dim + d]) * f64::from(keys[s][j * dim + d])).sum();
                        score[j / block] = score[j / block].max(dot);
                    }
                    let q_block = (n - 1) / block;
                    for l in 0..case.local {
                        score[q_block.saturating_sub(l)] = f64::INFINITY;
                    }
                    let mut order: Vec<usize> = (0..blocks).collect();
                    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
                    let take = topk.min(blocks);
                    let kth = score[order[take - 1]];
                    let mut want: Vec<i32> = order[..take].iter().map(|&b| b as i32).collect();
                    want.sort_unstable();
                    let valid = got.iter().take_while(|&&x| x >= 0).count();
                    let shape_ok = valid == take && got[valid..].iter().all(|&x| x == -1) && got[..valid].windows(2).all(|w| w[0] < w[1]) && got[..valid].iter().all(|&x| (x as usize) < blocks);
                    if !shape_ok {
                        violations += 1;
                        continue;
                    }
                    if got[..valid] == want[..] {
                        continue;
                    }
                    // a difference is allowed where the scores are the same up to f32 rounding, but not where they are exactly the same (the lower block wins then)
                    let eps = 2e-5 * kth.abs().max(1e-3);
                    let in_got: std::collections::HashSet<i32> = got[..valid].iter().copied().collect();
                    let in_want: std::collections::HashSet<i32> = want.iter().copied().collect();
                    for &b in in_got.symmetric_difference(&in_want) {
                        let s_b = score[b as usize];
                        if (s_b - kth).abs() > eps || s_b == kth {
                            violations += 1;
                        }
                    }
                }
            }
            worst_err = worst_err.max(violations as f64);
            if rows_here.iter().any(|&(s, pos)| checked(s, pos)) {
                let att = self.block_attention_check(case, p, stream.as_mut(), &rows_here, &have, &tables_seq, &tables, &lengths, &checked)?;
                worst_err = worst_err.max(att);
            }
        }
        Ok(Outcome { device: worst_err, cpu: 0.0, tol: 2e-5, cpu_tol: 1.0, ms: total_ms })
    }

    #[allow(clippy::too_many_arguments)]
    fn block_attention_check(
        &mut self,
        case: &BlockCase,
        p: &BlockSparseParams,
        stream: &mut dyn Stream,
        rows_here: &[(usize, usize)],
        sel: &[i32],
        tables_seq: &[Vec<i64>],
        tables: &[i64],
        lengths: &[i64],
        checked: &dyn Fn(usize, usize) -> bool,
    ) -> Result<f64, BackendError> {
        let (heads, kv_heads, dk, dv) = (case.att_heads, case.kv_heads, case.dk, case.dv);
        let rows = rows_here.len();
        let slot_of = |s: usize, pos: usize| tables_seq[s][pos / case.block_size] as usize * case.block_size + pos % case.block_size;
        let cache = self.device.allocate_kv_cache(1, case.pool_blocks, case.block_size, kv_heads, dk, dv, 1)?;
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
        let qb = put(self.device, stream, &q)?;
        let mut selb = self.device.allocate_buffer(rows * case.heads * case.topk * 4)?;
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
            sinks: None,
            alibi_slopes: None,
        };
        stream.block_attention(&mut out, &qb, &cache, tables, lengths, 0, &selb, p, &params)?;
        stream.synchronize()?;
        let have = get(stream, &out, rows * heads * dv)?;
        let rounded = |x: f32| f64::from(f16::from_f32(x).to_f32());
        let offsets: Vec<usize> = case.lengths.iter().scan(0usize, |acc, &n| { let o = *acc; *acc += n; Some(o) }).collect();
        let mut want = vec![0.0f64; rows * heads * dv];
        for (r, &(s, pos)) in rows_here.iter().enumerate() {
            if !checked(s, pos) {
                continue;
            }
            let n = pos + 1;
            for h in 0..heads {
                let group = h / (heads / case.heads);
                let kh = h / (heads / kv_heads);
                let list = &sel[(r * case.heads + group) * case.topk..(r * case.heads + group + 1) * case.topk];
                let positions: Vec<usize> = list.iter().take_while(|&&b| b >= 0).flat_map(|&b| (b as usize * case.block..(b as usize + 1) * case.block).take_while(|&j| j < n)).collect();
                let scores: Vec<f64> = positions
                    .iter()
                    .map(|&j| (0..dk).map(|d| f64::from(q[(r * heads + h) * dk + d]) * rounded(k[((offsets[s] + j) * kv_heads + kh) * dk + d])).sum::<f64>() * f64::from(params.scale))
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for d in 0..dv {
                    let acc: f64 = positions.iter().zip(&exps).map(|(&j, e)| e * rounded(v[((offsets[s] + j) * kv_heads + kh) * dv + d])).sum();
                    want[(r * heads + h) * dv + d] = acc / sum;
                }
            }
        }
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
