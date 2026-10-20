//! Dynamically claim small panels so P/E cores finish together.
use std::sync::atomic::{AtomicUsize, Ordering};

struct Output {
    pointer: *mut f32,
    // Only the flat panel scheduler (x86 Q8 decode and tests) reads the length.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    len: usize,
}
// SAFETY: Output is private. Both schedulers only access an index claimed
// exactly once from the shared atomic. The backing slice stays exclusively
// borrowed until worker dispatch joins every loop; no resize or alias is allowed.
unsafe impl Sync for Output {}
impl Output {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    unsafe fn panel<S, F: Fn(&mut S, usize, &mut [f32])>(
        &self,
        index: usize,
        size: usize,
        state: &mut S,
        body: &F,
    ) {
        let start = index * size;
        let len = size.min(self.len - start);
        body(
            state,
            index,
            std::slice::from_raw_parts_mut(self.pointer.add(start), len),
        );
    }
}

#[cfg(test)]
pub(crate) fn panels<S, I, F>(output: &mut [f32], size: usize, init: I, body: F)
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &mut [f32]) + Sync,
{
    panels_impl(output, size, init, body, false);
}

/// Decode-only variant: worker loops are stealable scope jobs rather than a
/// broadcast that requires every pool worker to participate.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn q8_decode_panels<S, I, F>(output: &mut [f32], size: usize, init: I, body: F)
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &mut [f32]) + Sync,
{
    panels_impl(output, size, init, body, true);
}

#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn panels_impl<S, I, F>(output: &mut [f32], size: usize, init: I, body: F, scoped: bool)
where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, &mut [f32]) + Sync,
{
    assert!(size > 0);
    if output.is_empty() {
        return;
    }
    let count = output.len().div_ceil(size);
    let next = AtomicUsize::new(0);
    let buffer = Output {
        pointer: output.as_mut_ptr(),
        len: output.len(),
    };
    dispatch_workers(scoped, || {
        let mut state = init();
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                break;
            }
            // Atomic ownership proves this panel cannot overlap another worker.
            unsafe {
                buffer.panel(index, size, &mut state, &body);
            }
        }
    });
}

/// Divisor of the guided claim size: with this many claims per worker left in the queue the
/// claim size starts to fall below its maximum.
const GUIDE: usize = 4;

/// Rows a worker claims at once while plenty of work is left (64 bytes of f32 outputs).
const DEFAULT_CLAIM_ROWS: usize = 16;

/// The same at batch one, where the outputs of a claim are one contiguous run and a claim costs
/// an atomic read-modify-write on a line every worker hits: with 16-row claims the queue itself
/// was a third of a decode step on a fast machine. The guided claim size still shrinks the last
/// claims, so the tail of a projection stays balanced.
const DECODE_CLAIM_ROWS: usize = 256;

/// Exclusive columns of one projection across all token rows. Only the
/// scheduler can construct this view; its atomic claim never repeats a tile.
pub(crate) struct Rows<'a> {
    output: &'a Output,
    first: usize,
    width: usize,
    stride: usize,
    batch: usize,
}
impl Rows<'_> {
    pub(crate) fn row(&mut self, token: usize) -> &mut [f32] {
        assert!(token < self.batch);
        let start = token * self.stride + self.first;
        // SAFETY: Construction validates batch*stride == output.len and the
        // column range stays inside stride. Other claims own different column
        // ranges or different allocations. The mutable view prevents aliasing
        // rows within this claim; outputs cannot be resized until workers join.
        unsafe { std::slice::from_raw_parts_mut(self.output.pointer.add(start), self.width) }
    }
}

pub(crate) fn row_panels<S, I, F>(
    output: &mut [Vec<f32>],
    rows: &[usize],
    batch: usize,
    size: usize,
    init: I,
    body: F,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, usize, &mut Rows<'_>) + Sync,
{
    let claim = if batch == 1 {
        DECODE_CLAIM_ROWS
    } else {
        DEFAULT_CLAIM_ROWS
    };
    row_panels_impl(output, rows, (batch, None), size, claim, init, body, false);
}

/// Like [`row_panels`] but claims up to `claim_rows` rows at a time while plenty of work is
/// left. Streaming expert weights are read in long contiguous blocks per worker this way.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_panels_claim<S, I, F>(
    output: &mut [Vec<f32>],
    rows: &[usize],
    batch: usize,
    size: usize,
    claim_rows: usize,
    init: I,
    body: F,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, usize, &mut Rows<'_>) + Sync,
{
    row_panels_impl(output, rows, (batch, None), size, claim_rows, init, body, false);
}

/// [`row_panels`] for members with their own token counts: member `m` holds `batches[m]` rows
/// (all non-zero) of `rows[m]` columns.
#[cfg(target_arch = "x86_64")]
pub(crate) fn row_panels_ragged<S, I, F>(
    output: &mut [Vec<f32>],
    rows: &[usize],
    batches: &[usize],
    size: usize,
    init: I,
    body: F,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, usize, &mut Rows<'_>) + Sync,
{
    assert_eq!(batches.len(), rows.len());
    row_panels_impl(
        output,
        rows,
        (1, Some(batches)),
        size,
        DEFAULT_CLAIM_ROWS,
        init,
        body,
        false,
    );
}

/// Q8 sibling panels use stealable scope jobs for batch-one decode.
pub(crate) fn q8_decode_row_panels<S, I, F>(
    output: &mut [Vec<f32>],
    rows: &[usize],
    batch: usize,
    size: usize,
    init: I,
    body: F,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, usize, &mut Rows<'_>) + Sync,
{
    let claim = if batch == 1 {
        DECODE_CLAIM_ROWS
    } else {
        DEFAULT_CLAIM_ROWS
    };
    row_panels_impl(output, rows, (batch, None), size, claim, init, body, batch == 1);
}

#[allow(clippy::too_many_arguments)]
fn row_panels_impl<S, I, F>(
    output: &mut [Vec<f32>],
    rows: &[usize],
    (batch, batches): (usize, Option<&[usize]>),
    size: usize,
    claim_rows: usize,
    init: I,
    body: F,
    scoped: bool,
) where
    I: Fn() -> S + Sync,
    F: Fn(&mut S, usize, usize, &mut Rows<'_>) + Sync,
{
    assert!(size > 0 && batch > 0);
    assert_eq!(output.len(), rows.len());
    let mut task_count = 0usize;
    let buffers = output
        .iter_mut()
        .zip(rows)
        .enumerate()
        .map(|(member, (out, &stride))| {
            assert!(stride > 0);
            let batch = batches.map_or(batch, |b| b[member]);
            assert!(batch > 0);
            assert_eq!(
                out.len(),
                batch
                    .checked_mul(stride)
                    .expect("projection dimensions overflow")
            );
            task_count = task_count
                .checked_add(stride.div_ceil(size))
                .expect("projection panel count overflow");
            (
                Output {
                    pointer: out.as_mut_ptr(),
                    len: out.len(),
                },
                task_count,
            )
        })
        .collect::<Vec<_>>();
    let next = AtomicUsize::new(0);
    // Keep neighboring output columns on one worker. Four-row tiles are only
    // 16 bytes wide; assigning each to a different core causes output cache
    // line contention across every token row. A claim spans at least 64 bytes
    // while plenty of work is left (`claim`), and shrinks to a single tile
    // towards the end of the queue: on a hybrid CPU the last claim of a slow
    // E-core would otherwise be the whole tail of the projection, since a
    // claim is 16 rows of every token (several hundred microseconds for a
    // prompt) while the P-cores sit idle. A worker takes at most a
    // `workers * GUIDE`-th of what remains.
    let claim = claim_rows.max(1).div_ceil(size);
    let workers = rayon::current_num_threads().max(1);
    dispatch_workers(scoped, || {
        let mut state = init();
        loop {
            let seen = next.load(Ordering::Relaxed);
            if seen >= task_count {
                break;
            }
            let take = ((task_count - seen) / (workers * GUIDE)).clamp(1, claim);
            let index = next.fetch_add(take, Ordering::Relaxed);
            if index >= task_count {
                break;
            }
            // Store one cumulative end per projection instead of materializing
            // every (member, row) task on each call. A Q8 QKV/gate-up group
            // has thousands of tiles but only two or three member records.
            let end = index.saturating_add(take).min(task_count);
            let mut member = buffers.partition_point(|(_, limit)| *limit <= index);
            let mut cursor = index;
            while cursor < end {
                let begin = if member == 0 {
                    0
                } else {
                    buffers[member - 1].1
                };
                let member_end = end.min(buffers[member].1);
                for tile in cursor..member_end {
                    let first = (tile - begin) * size;
                    let mut view = Rows {
                        output: &buffers[member].0,
                        first,
                        width: size.min(rows[member] - first),
                        stride: rows[member],
                        batch: batches.map_or(batch, |b| b[member]),
                    };
                    body(&mut state, member, first, &mut view);
                }
                cursor = member_end;
                member += 1;
            }
        }
    });
}

fn dispatch_workers<F>(scoped: bool, body: F)
where
    F: Fn() + Sync,
{
    // Decode runs a hundred short projections per token: keep the workers awake between them.
    if crate::cpu_spin::run(&body) {
        return;
    }
    if scoped {
        let workers = rayon::current_num_threads();
        rayon::scope(|scope| {
            for _ in 1..workers {
                scope.spawn(|_| body());
            }
            body();
        });
    } else {
        rayon::broadcast(|_| body());
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[should_panic(expected = "projection dimensions overflow")]
    fn strided_panels_reject_overflow_before_constructing_views() {
        super::row_panels(
            &mut [Vec::new()],
            &[2],
            usize::MAX / 2 + 1,
            4,
            || (),
            |_, _, _, _| {},
        );
    }
    #[test]
    fn strided_sibling_panels_write_each_cell_once() {
        let rows = [1, 3, 4, 13, 37];
        for threads in [1, 3, 10] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for batch in [1, 2, 9] {
                let mut output = rows
                    .iter()
                    .map(|&r| vec![0.0; batch * r])
                    .collect::<Vec<_>>();
                pool.install(|| {
                    super::row_panels(
                        &mut output,
                        &rows,
                        batch,
                        4,
                        || (),
                        |_, member, first, view| {
                            for b in 0..batch {
                                for (j, value) in view.row(b).iter_mut().enumerate() {
                                    *value +=
                                        (member * 1000 + b * rows[member] + first + j + 1) as f32;
                                }
                            }
                        },
                    )
                });
                for (member, out) in output.iter().enumerate() {
                    assert!(out
                        .iter()
                        .enumerate()
                        .all(|(i, &v)| v == (member * 1000 + i + 1) as f32));
                }
            }
        }
    }
    #[test]
    fn guided_claims_cover_long_queues_exactly_once() {
        // Thousands of tiles: the claim size shrinks from 4 to 1 towards the end of the queue.
        let rows = [4099, 1031];
        for threads in [3, 10] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for batch in [1, 4] {
                let mut output = rows
                    .iter()
                    .map(|&r| vec![0.0; batch * r])
                    .collect::<Vec<_>>();
                pool.install(|| {
                    super::row_panels(
                        &mut output,
                        &rows,
                        batch,
                        4,
                        || (),
                        |_, member, first, view| {
                            for b in 0..batch {
                                for (j, value) in view.row(b).iter_mut().enumerate() {
                                    *value += (member * 100_000 + b * rows[member] + first + j + 1)
                                        as f32;
                                }
                            }
                        },
                    )
                });
                for (member, out) in output.iter().enumerate() {
                    assert!(out
                        .iter()
                        .enumerate()
                        .all(|(i, &v)| v == (member * 100_000 + i + 1) as f32));
                }
            }
        }
    }
    #[test]
    fn dynamic_panels_claim_each_output_once_including_tail() {
        for threads in [1, 3, 10] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut output = vec![0.0; 10003];
            pool.install(|| {
                super::panels(
                    &mut output,
                    8,
                    || (),
                    |_, index, panel| {
                        for (j, value) in panel.iter_mut().enumerate() {
                            *value += (index * 8 + j + 1) as f32;
                        }
                    },
                )
            });
            assert!(output.iter().enumerate().all(|(i, v)| *v == (i + 1) as f32));
        }
    }

    #[test]
    fn q8_scope_schedulers_claim_panels_and_sibling_tails_once() {
        for threads in [1, 3, 10] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut flat = vec![0.0; 10003];
            pool.install(|| {
                super::q8_decode_panels(
                    &mut flat,
                    8,
                    || (),
                    |_, index, panel| {
                        for (offset, value) in panel.iter_mut().enumerate() {
                            *value = (index * 8 + offset + 1) as f32;
                        }
                    },
                )
            });
            assert!(flat
                .iter()
                .enumerate()
                .all(|(index, value)| *value == (index + 1) as f32));

            let rows = [9, 33, 70];
            let mut grouped = rows.iter().map(|&row| vec![0.0; row]).collect::<Vec<_>>();
            pool.install(|| {
                super::q8_decode_row_panels(
                    &mut grouped,
                    &rows,
                    1,
                    8,
                    || (),
                    |_, member, first, view| {
                        for (offset, value) in view.row(0).iter_mut().enumerate() {
                            *value = (member * 1000 + first + offset + 1) as f32;
                        }
                    },
                )
            });
            for (member, output) in grouped.iter().enumerate() {
                assert!(output
                    .iter()
                    .enumerate()
                    .all(|(index, value)| { *value == (member * 1000 + index + 1) as f32 }));
            }
        }
    }
}
