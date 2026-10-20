//! Fork-join over threads that spin briefly between jobs.
//!
//! A batch-one decode step is about a hundred short projections in a row. rayon's idle workers
//! go to sleep within a few microseconds, so every projection pays a futex wake-up per thread
//! (tens of microseconds on a virtual machine), which is as much as the weights take to read.
//! The workers of this pool poll for the next job for [`SPIN`] after the last one and park only
//! after that, so the projections of one token find them awake.
//!
//! The caller takes part in the job like a worker and the job claims its own work (as the
//! schedulers of `cpu_schedule` do), so a worker that is late to start does not delay the end of
//! the job: [`SpinPool::run`] only waits for workers that have entered the job. `FLLM_NO_SPIN`
//! switches the pool off (rayon dispatch is used instead), for comparisons.
use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{
    AtomicBool, AtomicU64, AtomicUsize,
    Ordering::{Relaxed, SeqCst},
};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

/// How long an idle worker polls for the next job before it parks.
const SPIN: Duration = Duration::from_micros(2000);

type Job = *const (dyn Fn() + Sync);

struct Slot(UnsafeCell<Option<Job>>);
// SAFETY: the slot is written only by the dispatcher while no worker is inside a job (`active`
// is zero and `open` is false) and read only by workers that entered the job (`active` raised,
// `open` seen set), so reads and writes never overlap.
unsafe impl Sync for Slot {}
// SAFETY: as above; the pointer is only dereferenced inside a job.
unsafe impl Send for Slot {}

struct Shared {
    /// Counts jobs; a worker that sees a new value looks at `open`.
    generation: AtomicU64,
    /// The job in `job` may be entered.
    open: AtomicBool,
    /// Workers inside the job.
    active: AtomicUsize,
    panicked: AtomicBool,
    job: Slot,
    asleep: Vec<AtomicBool>,
}

pub(crate) struct SpinPool {
    shared: Arc<Shared>,
    threads: Vec<Thread>,
    dispatch: Mutex<()>,
}

thread_local! {
    static IN_POOL: Cell<bool> = const { Cell::new(false) };
}

fn worker(shared: &Shared, index: usize) {
    IN_POOL.set(true);
    let mut seen = 0;
    loop {
        let mut since = Instant::now();
        let mut polls = 0u32;
        seen = loop {
            let generation = shared.generation.load(SeqCst);
            if generation != seen {
                break generation;
            }
            polls = polls.wrapping_add(1);
            if !polls.is_multiple_of(256) || since.elapsed() < SPIN {
                std::hint::spin_loop();
                continue;
            }
            // Park. The dispatcher publishes the generation before it looks at `asleep`, so
            // either it sees this flag and unparks us, or we see its generation here.
            shared.asleep[index].store(true, SeqCst);
            if shared.generation.load(SeqCst) == seen {
                thread::park();
            }
            shared.asleep[index].store(false, SeqCst);
            since = Instant::now();
        };
        // Enter the job, then check that it is still open: the dispatcher closes it and then
        // waits for `active` to drain, so a job is never left running behind its owner.
        shared.active.fetch_add(1, SeqCst);
        if shared.open.load(SeqCst) {
            // SAFETY: `open` was set after the slot was written and the dispatcher keeps the
            // closure alive until `active` is zero.
            let job = unsafe { (*shared.job.0.get()).expect("an open job has a closure") };
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { (*job)() }))
                .is_err()
            {
                shared.panicked.store(true, SeqCst);
            }
        }
        shared.active.fetch_sub(1, SeqCst);
    }
}

/// Closes the job and waits for the workers inside it, also when the caller's share panics.
struct Close<'a>(&'a Shared);
impl Drop for Close<'_> {
    fn drop(&mut self) {
        self.0.open.store(false, SeqCst);
        while self.0.active.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
    }
}

impl SpinPool {
    fn new(threads: usize) -> Option<Self> {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            open: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            job: Slot(UnsafeCell::new(None)),
            asleep: (1..threads).map(|_| AtomicBool::new(false)).collect(),
        });
        let mut handles = Vec::new();
        for index in 0..threads - 1 {
            let shared = shared.clone();
            let handle = thread::Builder::new()
                .name("fllm-spin".into())
                .spawn(move || worker(&shared, index))
                .ok()?;
            handles.push(handle.thread().clone());
        }
        Some(Self {
            shared,
            threads: handles,
            dispatch: Mutex::new(()),
        })
    }

    /// Run `body` once on every thread of the pool (the caller included). `false` when another
    /// job is in flight on this pool.
    fn run(&self, body: &(dyn Fn() + Sync)) -> bool {
        let Ok(_exclusive) = self.dispatch.try_lock() else {
            return false;
        };
        let shared = &*self.shared;
        // SAFETY: the closure outlives the job: `Close` waits for every worker inside it.
        let erased: &'static (dyn Fn() + Sync) = unsafe { std::mem::transmute(body) };
        unsafe { *shared.job.0.get() = Some(erased as Job) };
        shared.panicked.store(false, SeqCst);
        shared.open.store(true, SeqCst);
        shared.generation.fetch_add(1, SeqCst);
        for (flag, thread) in shared.asleep.iter().zip(&self.threads) {
            if flag.load(SeqCst) {
                thread.unpark();
            }
        }
        {
            let _close = Close(shared);
            body();
        }
        assert!(!shared.panicked.load(SeqCst), "a spin pool worker panicked");
        true
    }
}

/// Run `body` once on every thread of the current rayon pool, on spinning threads. `false` when
/// the pool does not apply (a single thread, more threads than cores, disabled, or busy with
/// another job); the caller then dispatches through rayon.
pub(crate) fn run(body: &(dyn Fn() + Sync)) -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    static CORES: OnceLock<usize> = OnceLock::new();
    static POOLS: Mutex<Vec<(usize, Arc<SpinPool>)>> = Mutex::new(Vec::new());
    let threads = rayon::current_num_threads();
    if threads < 2
        || IN_POOL.get()
        || threads > *CORES.get_or_init(|| thread::available_parallelism().map_or(1, |n| n.get()))
        || *OFF.get_or_init(|| {
            std::env::var_os("FLLM_NO_SPIN").is_some_and(|v| !v.is_empty() && v != "0")
        })
    {
        return false;
    }
    let pool = {
        let mut pools = POOLS.lock().unwrap_or_else(|e| e.into_inner());
        match pools.iter().find(|(n, _)| *n == threads) {
            Some((_, pool)) => pool.clone(),
            None => {
                let Some(pool) = SpinPool::new(threads) else {
                    return false;
                };
                let pool = Arc::new(pool);
                pools.push((threads, pool.clone()));
                pool
            }
        }
    };
    pool.run(body)
}

/// A raw pointer that the claiming loops below hand to several threads; every index or chunk is
/// claimed by exactly one of them.
struct Claimed<T>(*mut T);
// SAFETY: see above; the pointee is only touched through disjoint claims.
unsafe impl<T: Send> Sync for Claimed<T> {}
impl<T> Claimed<T> {
    /// The pointer `n` elements in; a method so closures capture the whole wrapper.
    fn at(&self, n: usize) -> *mut T {
        // SAFETY: callers pass an index inside the allocation.
        unsafe { self.0.add(n) }
    }
}

/// `(0..count).into_par_iter().map(f).collect()` on the spinning threads (rayon when they do not
/// apply). Results keep the index order.
pub(crate) fn collect<T: Send>(count: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    use rayon::prelude::*;
    if count > 1 {
        let mut slots: Vec<Option<T>> = (0..count).map(|_| None).collect();
        let base = Claimed(slots.as_mut_ptr());
        let next = AtomicUsize::new(0);
        let claimed = run(&|| loop {
            let index = next.fetch_add(1, Relaxed);
            if index >= count {
                break;
            }
            // SAFETY: `index` is claimed once and is in range.
            unsafe { *base.at(index) = Some(f(index)) };
        });
        if claimed {
            return slots
                .into_iter()
                .map(|slot| slot.expect("every index was claimed"))
                .collect();
        }
    }
    (0..count).into_par_iter().map(f).collect()
}

/// `data.par_chunks_mut(chunk).enumerate().for_each(f)` on the spinning threads (rayon when they
/// do not apply).
pub(crate) fn chunks_mut<T: Send>(data: &mut [T], chunk: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    use rayon::prelude::*;
    let (len, count) = (data.len(), data.len().div_ceil(chunk));
    if count > 1 {
        let base = Claimed(data.as_mut_ptr());
        let next = AtomicUsize::new(0);
        let claimed = run(&|| loop {
            let index = next.fetch_add(1, Relaxed);
            if index >= count {
                break;
            }
            let start = index * chunk;
            // SAFETY: chunk `index` is claimed once; chunks are disjoint and in range.
            f(index, unsafe {
                std::slice::from_raw_parts_mut(base.at(start), chunk.min(len - start))
            });
        });
        if claimed {
            return;
        }
    }
    data.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(index, part)| f(index, part));
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_thread_joins_and_the_job_ends_with_its_owner() {
        let threads = 4.min(std::thread::available_parallelism().map_or(1, |n| n.get()));
        if threads < 2 {
            return;
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            for round in 0..200 {
                let next = AtomicUsize::new(0);
                let done = AtomicUsize::new(0);
                assert!(super::run(&|| {
                    while next.fetch_add(1, Ordering::Relaxed) < 1000 {
                        done.fetch_add(1, Ordering::Relaxed);
                    }
                }));
                assert_eq!(done.load(Ordering::Relaxed), 1000, "round {round}");
                if round % 50 == 0 {
                    // let the workers park, then wake them again
                    std::thread::sleep(std::time::Duration::from_millis(10));
                }
            }
        });
    }
}
