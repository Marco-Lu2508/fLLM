//! CPU topology: logical processors grouped into cores, cores grouped into performance classes.
//!
//! Hybrid processors (Intel 12th generation and later, Apple M-series, Snapdragon, big.LITTLE servers) mix fast and slow cores. A
//! decode step is a series of short parallel projections that all wait for their slowest thread, and decoding is limited by memory
//! bandwidth, so "one thread per logical processor" is the worst default there: the slowest cores and the second hardware thread of
//! every core add waiting, not bandwidth. This module only *describes* the machine and derives a default thread count from it
//! ([`Topology::default_threads`]); `cpu_threads` measures and remembers the real optimum.
//!
//! Sources, all parsed by pure functions that the tests feed with sample data:
//! * Windows: `GetSystemCpuSetInformation` (core index and `EfficiencyClass` of every logical processor).
//! * Linux: sysfs `cpu*/topology/thread_siblings_list` and `cpu*/cpu_capacity` (else `cpufreq/cpuinfo_max_freq`).
//! * macOS: `hw.perflevelN.physicalcpu` (Apple Silicon), else `hw.physicalcpu` / `hw.logicalcpu`.
//! * anything else, or a process restricted to fewer processors than the machine has: every logical processor counts as one
//!   core of one class (the former behaviour).
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// One physical core: its performance rank (0 = slowest class of this machine) and its logical processors in the OS's numbering
/// (Windows: CPU set ids).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Core {
    pub rank: usize,
    pub logical: Vec<usize>,
}

/// Summary of one performance class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    pub rank: usize,
    pub cores: usize,
    pub threads: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Topology {
    pub cores: Vec<Core>,
    /// Where the data came from (`windows cpu sets`, `linux sysfs`, `macos sysctl`, `flat`).
    pub source: &'static str,
}

/// One logical processor as an OS reports it, before grouping.
#[derive(Clone, Debug)]
pub struct RawCpu {
    pub id: usize,
    /// Identifies the physical core; logical processors with equal keys are SMT siblings.
    pub core: (u32, u32),
    /// Performance value; only the order (and, for `cluster`, the ratio) matters.
    pub perf: u64,
}

/// Dense ranks (0 = slowest) of performance values. With `cluster`, values within 17.6 % of the lowest value of a group count as
/// one class (the cores of one kind differ a little in `cpu_capacity` or boost frequency).
pub fn perf_ranks(values: &[u64], cluster: bool) -> Vec<usize> {
    let mut distinct: Vec<u64> = values.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    // class index of every distinct value
    let mut class_of = Vec::with_capacity(distinct.len());
    let mut class = 0usize;
    let mut base = 0u64;
    for (i, &v) in distinct.iter().enumerate() {
        if i == 0 {
            base = v;
        } else if !cluster || (v as f64) > (base as f64) / 0.85 {
            class += 1;
            base = v;
        }
        class_of.push(class);
    }
    values
        .iter()
        .map(|v| class_of[distinct.binary_search(v).unwrap_or(0)])
        .collect()
}

impl Topology {
    /// Groups raw logical processors into cores (a core's performance is its fastest logical processor's).
    pub fn from_raw(cpus: &[RawCpu], cluster: bool, source: &'static str) -> Self {
        let mut by_core: BTreeMap<(u32, u32), (u64, Vec<usize>)> = BTreeMap::new();
        for cpu in cpus {
            let entry = by_core.entry(cpu.core).or_insert((0, Vec::new()));
            entry.0 = entry.0.max(cpu.perf);
            entry.1.push(cpu.id);
        }
        let mut grouped: Vec<(u64, Vec<usize>)> = by_core.into_values().collect();
        for (_, ids) in &mut grouped {
            ids.sort_unstable();
        }
        grouped.sort_by_key(|(_, ids)| ids[0]);
        let ranks = perf_ranks(&grouped.iter().map(|(p, _)| *p).collect::<Vec<_>>(), cluster);
        Self {
            cores: grouped
                .into_iter()
                .zip(ranks)
                .map(|((_, logical), rank)| Core { rank, logical })
                .collect(),
            source,
        }
    }

    /// `physical` equal cores with `logical / physical` hardware threads each; ids are positions (not usable for pinning).
    pub fn uniform(physical: usize, logical: usize, source: &'static str) -> Self {
        let physical = physical.clamp(1, logical.max(1));
        let smt = (logical / physical).max(1);
        Self {
            cores: (0..physical)
                .map(|c| Core { rank: 0, logical: (c * smt..(c + 1) * smt).collect() })
                .collect(),
            source,
        }
    }

    /// Performance levels from the highest to the lowest as `(physical cores, logical processors)` (macOS `hw.perflevelN`).
    pub fn from_levels(levels: &[(usize, usize)]) -> Self {
        let mut cores = Vec::new();
        let mut next = 0;
        for (level, &(physical, logical)) in levels.iter().enumerate() {
            let smt = (logical / physical.max(1)).max(1);
            for _ in 0..physical {
                cores.push(Core { rank: levels.len() - 1 - level, logical: (next..next + smt).collect() });
                next += smt;
            }
        }
        Self { cores, source: "macos sysctl" }
    }

    pub fn logical_count(&self) -> usize {
        self.cores.iter().map(|c| c.logical.len()).sum()
    }

    pub fn physical_cores(&self) -> usize {
        self.cores.len()
    }

    /// Classes from the slowest to the fastest.
    pub fn classes(&self) -> Vec<Class> {
        let mut out: Vec<Class> = Vec::new();
        for core in &self.cores {
            match out.iter_mut().find(|c| c.rank == core.rank) {
                Some(c) => {
                    c.cores += 1;
                    c.threads += core.logical.len();
                }
                None => out.push(Class { rank: core.rank, cores: 1, threads: core.logical.len() }),
            }
        }
        out.sort_by_key(|c| c.rank);
        out
    }

    pub fn is_hybrid(&self) -> bool {
        self.classes().len() > 1
    }

    /// True when the logical ids can be handed to the OS to restrict the process.
    pub fn can_pin(&self) -> bool {
        matches!(self.source, "windows cpu sets" | "linux sysfs")
    }

    /// The default number of decode threads (see the module documentation and docs/cpu-kernels.md).
    ///
    /// * One class (every desktop, laptop and server part without efficiency cores): one thread per physical core. Decoding is
    ///   limited by memory bandwidth, which a few threads per memory channel already saturate; the second hardware thread of a
    ///   core adds no bandwidth and its work contends for the core's load ports.
    /// * Two or more classes: every core of the fastest class, plus three quarters of the cores of the other classes, except that
    ///   the slowest class is left out when there are three (the low-power cores of Intel Core Ultra sit outside the compute tile
    ///   and are several times slower than the rest: with them every parallel step waits for them). The remaining quarter is left
    ///   to the operating system, the HTTP server and the sampler, which compete with spinning workers for the same cores.
    pub fn default_threads(&self) -> usize {
        let classes = self.classes();
        let n = match classes.len() {
            0 => 1,
            1 => classes[0].cores,
            n => {
                let skip = usize::from(n >= 3);
                let (top, rest) = classes[skip..].split_last().map_or((0, 0), |(top, rest)| {
                    (top.cores, rest.iter().map(|c| c.cores).sum::<usize>())
                });
                top + rest * 3 / 4
            }
        };
        n.max(1)
    }

    /// Threads for batches of many rows (prompt processing). That work is compute-bound, so the second hardware thread of a core
    /// helps (typically 10 to 25 %) on a processor whose cores are all alike; on a hybrid processor the decode default also won
    /// for prompts in the one measurement taken (a 2P+8E+2LPE laptop: 8 threads beat 12 and 14), so it is kept there.
    pub fn prefill_threads(&self) -> usize {
        if self.is_hybrid() {
            self.default_threads()
        } else {
            self.logical_count().max(1)
        }
    }

    /// The logical processors in the order in which they are given to a pool: the first thread of every core of the fast classes
    /// (fastest first), then their remaining hardware threads, then the slowest class when there are three or more classes.
    /// The first `n` entries are the processors of an `n`-thread pool.
    pub fn preferred_order(&self) -> Vec<usize> {
        let classes = self.classes();
        let low_rank = if classes.len() >= 3 { Some(classes[0].rank) } else { None };
        let mut cores: Vec<&Core> = self.cores.iter().collect();
        cores.sort_by_key(|c| std::cmp::Reverse(c.rank)); // stable: ties keep the core order
        let (low, high): (Vec<&Core>, Vec<&Core>) = cores.into_iter().partition(|c| Some(c.rank) == low_rank);
        let mut out = Vec::new();
        for group in [&high, &low] {
            for core in group.iter() {
                out.extend(core.logical.first());
            }
            for core in group.iter() {
                out.extend(core.logical.iter().skip(1));
            }
        }
        out
    }

    /// "hybrid processor: 2 performance cores, 8 efficiency cores, 2 low-power efficiency cores" or "8 cores, 16 threads".
    pub fn describe(&self) -> String {
        let classes = self.classes();
        if classes.len() < 2 {
            let (cores, threads) = (self.physical_cores(), self.logical_count());
            return if cores == threads {
                format!("{cores} cores")
            } else {
                format!("{cores} cores, {threads} threads")
            };
        }
        let parts: Vec<String> = classes
            .iter()
            .rev()
            .enumerate()
            .map(|(i, c)| {
                let name = if i == 0 {
                    "performance"
                } else if i == classes.len() - 1 && classes.len() >= 3 {
                    "low-power efficiency"
                } else {
                    "efficiency"
                };
                format!("{} {name} core{}", c.cores, if c.cores == 1 { "" } else { "s" })
            })
            .collect();
        format!("hybrid processor: {}", parts.join(", "))
    }

    pub fn to_json(&self) -> serde_json::Value {
        let classes: Vec<_> = self
            .classes()
            .iter()
            .rev()
            .map(|c| serde_json::json!({"rank": c.rank, "cores": c.cores, "threads": c.threads}))
            .collect();
        serde_json::json!({
            "source": self.source,
            "hybrid": self.is_hybrid(),
            "physical_cores": self.physical_cores(),
            "logical_processors": self.logical_count(),
            "classes": classes,
            "description": self.describe(),
            "default_threads": self.default_threads(),
        })
    }
}

/// The thread counts the auto-tuner tries around `policy`: `policy`, one step below, and one and two steps above, a step being a
/// quarter of `policy`; limited to `1..=max`, sorted, without duplicates.
pub fn candidates(policy: usize, max: usize) -> Vec<usize> {
    let step = (policy / 4).max(1);
    let mut out: Vec<usize> = [policy.saturating_sub(step), policy, policy + step, policy + 2 * step]
        .into_iter()
        .filter(|&n| n >= 1 && n <= max.max(1))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Fewest rows of output one thread should get; below this a parallel step costs more than it saves.
pub const MIN_ROWS_PER_THREAD: usize = 32;

/// `threads`, but not more than a projection of `rows` output rows can feed.
pub fn limit_for_rows(threads: usize, rows: usize) -> usize {
    threads.min((rows / MIN_ROWS_PER_THREAD).max(1)).max(1)
}

// ---------------------------------------------------------------------------------------------------------------------------
// Parsers
// ---------------------------------------------------------------------------------------------------------------------------

/// A sysfs CPU list such as `0-3,8,10-11`.
pub fn parse_cpu_list(text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in text.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            Some((a, b)) => {
                if let (Ok(a), Ok(b)) = (a.trim().parse::<usize>(), b.trim().parse::<usize>()) {
                    out.extend(a..=b.min(a + 4096));
                }
            }
            None => out.extend(part.parse::<usize>()),
        }
    }
    out
}

/// What sysfs says about one logical processor.
#[derive(Clone, Debug, Default)]
pub struct LinuxCpu {
    pub id: usize,
    /// `topology/thread_siblings_list` (empty when missing).
    pub siblings: Vec<usize>,
    /// `cpu_capacity` (ARM and some x86 kernels).
    pub capacity: Option<u64>,
    /// `cpufreq/cpuinfo_max_freq` in kHz.
    pub max_khz: Option<u64>,
}

/// Topology from sysfs data. Capacity wins over frequency, but only when every processor has one.
pub fn from_linux(cpus: &[LinuxCpu]) -> Topology {
    let use_capacity = !cpus.is_empty() && cpus.iter().all(|c| c.capacity.is_some());
    let raw: Vec<RawCpu> = cpus
        .iter()
        .map(|c| RawCpu {
            id: c.id,
            core: (0, c.siblings.iter().copied().chain([c.id]).min().unwrap_or(c.id) as u32),
            perf: if use_capacity { c.capacity.unwrap_or(0) } else { c.max_khz.unwrap_or(0) },
        })
        .collect();
    Topology::from_raw(&raw, true, "linux sysfs")
}

/// Topology from the bytes `GetSystemCpuSetInformation` returns: a series of `SYSTEM_CPU_SET_INFORMATION` records (size at 0,
/// type at 4, and for type 0 the CPU set: id at 8, group at 12, logical processor index at 14, core index at 15, efficiency class
/// at 18).
pub fn from_windows_cpu_sets(buffer: &[u8]) -> Topology {
    // (id, group, core, last level cache index, efficiency class)
    let mut sets = Vec::new();
    let mut at = 0usize;
    while at + 8 <= buffer.len() {
        let size = u32::from_le_bytes(buffer[at..at + 4].try_into().unwrap_or([0; 4])) as usize;
        if size < 8 || at + size > buffer.len() {
            break;
        }
        let kind = u32::from_le_bytes(buffer[at + 4..at + 8].try_into().unwrap_or([0; 4]));
        if kind == 0 && size >= 20 {
            let record = &buffer[at..at + size];
            let id = u32::from_le_bytes(record[8..12].try_into().unwrap_or([0; 4])) as usize;
            let group = u16::from_le_bytes(record[12..14].try_into().unwrap_or([0; 2]));
            sets.push((id, u32::from(group), u32::from(record[15]), record[16], record[18]));
        }
        at += size;
    }
    // Windows gives the efficiency cores and the low-power cores of Intel Core Ultra the same efficiency class. The low-power
    // cores sit outside the compute tile and do not share its last level cache, so a core below the top class that does not
    // share a cache with a top class core is ranked under the other efficiency cores.
    let top = sets.iter().map(|s| s.4).max().unwrap_or(0);
    let top_caches: Vec<u8> = sets.iter().filter(|s| s.4 == top).map(|s| s.3).collect();
    let raw: Vec<RawCpu> = sets
        .iter()
        .map(|&(id, group, core, cache, class)| RawCpu {
            id,
            core: (group, core),
            perf: u64::from(class) * 4 + if class == top || top_caches.contains(&cache) { 2 } else { 0 },
        })
        .collect();
    Topology::from_raw(&raw, false, "windows cpu sets")
}

// ---------------------------------------------------------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------------------------------------------------------

#[cfg(target_os = "linux")]
fn detect_os() -> Option<Topology> {
    let root = std::path::Path::new("/sys/devices/system/cpu");
    let online = parse_cpu_list(&std::fs::read_to_string(root.join("online")).ok()?);
    let read = |id: usize, file: &str| std::fs::read_to_string(root.join(format!("cpu{id}")).join(file)).ok();
    let cpus: Vec<LinuxCpu> = online
        .into_iter()
        .map(|id| LinuxCpu {
            id,
            siblings: read(id, "topology/thread_siblings_list").map(|t| parse_cpu_list(&t)).unwrap_or_default(),
            capacity: read(id, "cpu_capacity").and_then(|t| t.trim().parse().ok()),
            max_khz: read(id, "cpufreq/cpuinfo_max_freq").and_then(|t| t.trim().parse().ok()),
        })
        .collect();
    (!cpus.is_empty()).then(|| from_linux(&cpus))
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    extern "system" {
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn GetSystemCpuSetInformation(info: *mut u8, length: u32, returned: *mut u32, process: *mut c_void, flags: u32) -> i32;
        pub fn SetProcessDefaultCpuSets(process: *mut c_void, ids: *const u32, count: u32) -> i32;
        pub fn SetProcessInformation(process: *mut c_void, class: i32, info: *const c_void, size: u32) -> i32;
    }
}

#[cfg(windows)]
fn detect_os() -> Option<Topology> {
    let mut length = 0u32;
    // SAFETY: the first call only asks for the size; the second fills a buffer of that size.
    unsafe {
        let process = win::GetCurrentProcess();
        win::GetSystemCpuSetInformation(std::ptr::null_mut(), 0, &mut length, process, 0);
        if length == 0 || length > 1 << 20 {
            return None;
        }
        let mut buffer = vec![0u8; length as usize];
        if win::GetSystemCpuSetInformation(buffer.as_mut_ptr(), length, &mut length, process, 0) == 0 {
            return None;
        }
        buffer.truncate(length as usize);
        Some(from_windows_cpu_sets(&buffer))
    }
}

#[cfg(target_os = "macos")]
fn detect_os() -> Option<Topology> {
    extern "C" {
        fn sysctlbyname(name: *const std::ffi::c_char, old: *mut std::ffi::c_void, len: *mut usize, new: *mut std::ffi::c_void, newlen: usize) -> i32;
    }
    let get = |name: &str| -> Option<usize> {
        let name = std::ffi::CString::new(name).ok()?;
        let (mut value, mut len) = (0u32, std::mem::size_of::<u32>());
        // SAFETY: `value` is a 4-byte buffer and `len` says so.
        let rc = unsafe { sysctlbyname(name.as_ptr(), (&raw mut value).cast(), &mut len, std::ptr::null_mut(), 0) };
        (rc == 0).then_some(value as usize)
    };
    let mut levels = Vec::new();
    for level in 0..8 {
        match (get(&format!("hw.perflevel{level}.physicalcpu")), get(&format!("hw.perflevel{level}.logicalcpu"))) {
            (Some(p), Some(l)) if p > 0 => levels.push((p, l.max(p))),
            _ => break,
        }
    }
    if !levels.is_empty() {
        return Some(Topology::from_levels(&levels));
    }
    Some(Topology::uniform(get("hw.physicalcpu")?, get("hw.logicalcpu")?, "macos sysctl"))
}

#[cfg(not(any(target_os = "linux", windows, target_os = "macos")))]
fn detect_os() -> Option<Topology> {
    None
}

/// The topology of this machine, as far as the process may use it.
pub fn detect() -> &'static Topology {
    static TOPOLOGY: OnceLock<Topology> = OnceLock::new();
    TOPOLOGY.get_or_init(|| {
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        match detect_os() {
            // fewer processors available than exist (container limits, affinity): the details no longer apply
            Some(t) if t.logical_count() == available => t,
            _ => Topology::uniform(available, available, "flat"),
        }
    })
}

// ---------------------------------------------------------------------------------------------------------------------------
// Keeping the process on the chosen cores
// ---------------------------------------------------------------------------------------------------------------------------

/// Restricts the process to the first `n` logical processors of [`Topology::preferred_order`] (Windows CPU sets), so the workers
/// stay on the chosen cores and off the low-power ones. `n` of at least every processor lifts the restriction. Returns whether
/// the operating system accepted it; always `false` where this is not implemented (Linux uses `--pin-cores`). `FLLM_NO_PIN=1`
/// disables it.
pub fn pin_first(topology: &Topology, n: usize) -> bool {
    if std::env::var_os("FLLM_NO_PIN").is_some_and(|v| !v.is_empty() && v != "0") || !topology.can_pin() {
        return false;
    }
    pin_ids(&topology.preferred_order(), n, topology.logical_count())
}

#[cfg(windows)]
fn pin_ids(order: &[usize], n: usize, all: usize) -> bool {
    let ids: Vec<u32> = order.iter().take(n).map(|&id| id as u32).collect();
    let (ptr, count) = if n >= all { (std::ptr::null(), 0) } else { (ids.as_ptr(), ids.len() as u32) };
    // SAFETY: `ptr` points to `count` ids that live until the call returns.
    unsafe { win::SetProcessDefaultCpuSets(win::GetCurrentProcess(), ptr, count) != 0 }
}

#[cfg(not(windows))]
fn pin_ids(_order: &[usize], _n: usize, _all: usize) -> bool {
    false
}

/// Windows: opts the process out of power throttling (EcoQoS), which moves the threads of a background-looking process to the
/// efficiency cores. No effect elsewhere. Returns whether the call was accepted.
#[cfg(windows)]
pub fn disable_power_throttling() -> bool {
    #[repr(C)]
    struct State {
        version: u32,
        control_mask: u32,
        state_mask: u32,
    }
    let state = State { version: 1, control_mask: 1, state_mask: 0 };
    // SAFETY: `state` is a PROCESS_POWER_THROTTLING_STATE (ProcessPowerThrottling = 4) of the size passed.
    unsafe { win::SetProcessInformation(win::GetCurrentProcess(), 4, (&raw const state).cast(), std::mem::size_of::<State>() as u32) != 0 }
}

#[cfg(not(windows))]
pub fn disable_power_throttling() -> bool {
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linux_cpu(id: usize, siblings: &[usize], capacity: Option<u64>, khz: Option<u64>) -> LinuxCpu {
        LinuxCpu { id, siblings: siblings.to_vec(), capacity, max_khz: khz }
    }

    /// Core Ultra 5 235U as Linux shows it: 2 P cores with SMT (cpu0-3), 8 E cores (4-11), 2 low-power E cores (12-13).
    fn ultra_235u() -> Topology {
        let mut cpus = Vec::new();
        for core in 0..2 {
            for t in 0..2 {
                cpus.push(linux_cpu(core * 2 + t, &[core * 2, core * 2 + 1], None, Some(4_400_000 - (t as u64) * 10_000)));
            }
        }
        for id in 4..12 {
            cpus.push(linux_cpu(id, &[id], None, Some(3_600_000)));
        }
        for id in 12..14 {
            cpus.push(linux_cpu(id, &[id], None, Some(2_500_000)));
        }
        from_linux(&cpus)
    }

    /// `p` P cores with SMT, `e` E cores, `l` low-power cores, as Windows CPU sets (efficiency classes 2, 1 and, for the
    /// low-power cores, 0 or, as on real Core Ultra parts, 1 again). The low-power cores have a cache of their own.
    fn windows_buffer(p: usize, e: usize, l: usize) -> Vec<u8> {
        windows_buffer_with(p, e, l, 0)
    }

    fn windows_buffer_with(p: usize, e: usize, l: usize, lpe_class: u8) -> Vec<u8> {
        let mut out = Vec::new();
        let mut index = 0u32;
        let mut core = 0u8;
        let push = |class: u8, cache: u8, smt: usize, out: &mut Vec<u8>, index: &mut u32, core: &mut u8| {
            for _ in 0..smt {
                let mut record = [0u8; 32];
                record[0..4].copy_from_slice(&32u32.to_le_bytes());
                record[8..12].copy_from_slice(&(0x100 + *index).to_le_bytes());
                record[14] = *index as u8;
                record[15] = *core;
                record[16] = cache;
                record[18] = class;
                out.extend_from_slice(&record);
                *index += 1;
            }
            *core += 1;
        };
        for _ in 0..p {
            push(2, 0, 2, &mut out, &mut index, &mut core);
        }
        for _ in 0..e {
            push(1, 0, 1, &mut out, &mut index, &mut core);
        }
        for _ in 0..l {
            push(lpe_class, 1, 1, &mut out, &mut index, &mut core);
        }
        out
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(parse_cpu_list("0-3,8,10-11\n"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
        assert_eq!(parse_cpu_list("5"), vec![5]);
    }

    #[test]
    fn ranks_cluster_similar_values() {
        assert_eq!(perf_ranks(&[1024, 1009, 640, 640, 300], true), vec![2, 2, 1, 1, 0]);
        assert_eq!(perf_ranks(&[2, 1, 0, 2], false), vec![2, 1, 0, 2]);
        assert_eq!(perf_ranks(&[1024, 1009], false), vec![1, 0]);
    }

    #[test]
    fn linux_sample() {
        let t = ultra_235u();
        assert_eq!((t.physical_cores(), t.logical_count()), (12, 14));
        let classes: Vec<_> = t.classes().iter().map(|c| (c.cores, c.threads)).collect();
        assert_eq!(classes, vec![(2, 2), (8, 8), (2, 4)]);
        assert_eq!(t.describe(), "hybrid processor: 2 performance cores, 8 efficiency cores, 2 low-power efficiency cores");
        // capacity is preferred when every processor has one
        let arm: Vec<_> = (0..8).map(|i| linux_cpu(i, &[i], Some(if i < 4 { 430 } else { 1024 }), Some(1))).collect();
        let t = from_linux(&arm);
        assert_eq!(t.classes().iter().map(|c| c.cores).collect::<Vec<_>>(), vec![4, 4]);
        assert_eq!(t.cores[0].rank, 0);
    }

    #[test]
    fn windows_sample() {
        let t = from_windows_cpu_sets(&windows_buffer(2, 8, 2));
        assert_eq!((t.physical_cores(), t.logical_count()), (12, 14));
        assert_eq!(t.classes().iter().map(|c| (c.cores, c.threads)).collect::<Vec<_>>(), vec![(2, 2), (8, 8), (2, 4)]);
        assert_eq!(t.cores[0].logical, vec![0x100, 0x101]);
        // the same classes as Windows reports them on a Core Ultra: efficiency and low-power cores share a class
        let t = from_windows_cpu_sets(&windows_buffer_with(2, 8, 2, 1));
        assert_eq!(t.classes().iter().map(|c| (c.cores, c.threads)).collect::<Vec<_>>(), vec![(2, 2), (8, 8), (2, 4)]);
        assert_eq!(t.default_threads(), 8);
        // a truncated or foreign record ends the walk instead of panicking
        let mut cut = windows_buffer(1, 1, 0);
        cut.truncate(40);
        assert_eq!(from_windows_cpu_sets(&cut).logical_count(), 1);
        assert_eq!(from_windows_cpu_sets(&[]).logical_count(), 0);
    }

    #[test]
    fn apple_levels() {
        let t = Topology::from_levels(&[(8, 8), (4, 4)]);
        assert_eq!(t.describe(), "hybrid processor: 8 performance cores, 4 efficiency cores");
        assert_eq!(t.default_threads(), 11);
        assert_eq!(t.prefill_threads(), 11);
        assert_eq!(Topology::uniform(8, 16, "x").prefill_threads(), 16);
        assert_eq!(ultra_235u().prefill_threads(), 8);
    }

    #[test]
    fn default_thread_policy_table() {
        // (name, topology, expected default)
        let table: Vec<(&str, Topology, usize)> = vec![
            ("core ultra 5 235u", ultra_235u(), 8),
            ("14th gen 6P+8E+2LPE", from_windows_cpu_sets(&windows_buffer(6, 8, 2)), 12),
            ("8 equal cores with SMT", Topology::uniform(8, 16, "x"), 8),
            ("4 equal cores", Topology::uniform(4, 4, "x"), 4),
            ("apple m2 pro 8P+4E", Topology::from_levels(&[(8, 8), (4, 4)]), 11),
            ("64-core server with SMT", Topology::uniform(64, 128, "x"), 64),
            ("flat fallback", Topology::uniform(6, 6, "flat"), 6),
            ("2P+4E (two classes)", from_windows_cpu_sets(&windows_buffer(2, 4, 0)), 5),
        ];
        for (name, topology, expected) in table {
            assert_eq!(topology.default_threads(), expected, "{name}");
            assert!(topology.default_threads() <= topology.logical_count(), "{name}");
            assert!(topology.prefill_threads() >= topology.default_threads(), "{name}");
        }
    }

    #[test]
    fn preferred_order_fills_fast_cores_first() {
        let t = from_windows_cpu_sets(&windows_buffer(2, 8, 2));
        let order = t.preferred_order();
        assert_eq!(order.len(), 14);
        // 8 threads: both P cores and 6 E cores, no hardware thread twice on one core
        let first8: Vec<_> = order.iter().take(8).copied().collect();
        assert_eq!(first8, vec![0x100, 0x102, 0x104, 0x105, 0x106, 0x107, 0x108, 0x109]);
        // then the other E cores, then the P siblings, the low-power cores last
        assert_eq!(&order[8..10], &[0x10a, 0x10b]);
        assert_eq!(&order[10..12], &[0x101, 0x103]);
        assert_eq!(&order[12..], &[0x10c, 0x10d]);
        // equal cores: first threads before siblings
        assert_eq!(Topology::uniform(2, 4, "x").preferred_order(), vec![0, 2, 1, 3]);
    }

    #[test]
    fn candidates_and_row_limit() {
        assert_eq!(candidates(8, 14), vec![6, 8, 10, 12]);
        assert_eq!(candidates(8, 8), vec![6, 8]);
        assert_eq!(candidates(1, 1), vec![1]);
        assert_eq!(candidates(4, 16), vec![3, 4, 5, 6]);
        assert_eq!(candidates(64, 128), vec![48, 64, 80, 96]);
        assert_eq!(limit_for_rows(16, 2048), 16);
        assert_eq!(limit_for_rows(16, 128), 4);
        assert_eq!(limit_for_rows(16, 10), 1);
    }

    #[test]
    fn detection_is_consistent_with_this_machine() {
        let t = detect();
        assert!(t.logical_count() >= 1);
        assert!(t.default_threads() >= 1 && t.default_threads() <= t.logical_count());
        assert_eq!(t.to_json()["logical_processors"], t.logical_count());
    }
}
