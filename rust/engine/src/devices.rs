//! Compute device inventory: the host CPU, the CUDA / ROCm devices of the cuda backend, every hardware GPU that wgpu can reach
//! through Vulkan, Metal or DirectX 12 (feature `rust-gpu`) and the devices of the native backends (features `vulkan`, `metal`,
//! `hip`, `sycl`, `npu-*`, `tpu-pjrt`).
//!
//! # Device numbers
//!
//! `fllm-rs devices` prints two numbers per device:
//!
//! * the **inventory number** in square brackets: the position in the list of all devices of all backends. It only identifies
//!   a line of that output; no command line option takes it.
//! * the **device specification** `--device <backend>:<N>`: `N` counts the devices of *that backend* (and, where a backend has a
//!   native and a wgpu implementation, of the implementation the name selects), starting at 0. `vulkan:1` is the second Vulkan
//!   device, whatever its inventory number is. [`select_device`] and [`device_candidates`] implement this rule; every consumer of a
//!   `--device` value (the decoder, `selftest-gpu`, the memory planner) goes through them.
//!
//! The names: `cpu`, `cuda`, `hip` (`rocm`), `metal`, `vulkan`, `dx12`, `sycl` (`xpu`), `npu` (`openvino`), `tpu`, `gpu` (`wgpu`, any
//! wgpu adapter), the explicit `vulkan-native` / `metal-native` (no wgpu fallback), `wgpu-vulkan` / `wgpu-metal` / `wgpu-dx12`
//! (the wgpu implementation even where a native one exists) and `auto`.
//!
//! # `auto`
//!
//! `auto` takes the first backend of `CUDA > ROCm (hip) > Metal > Vulkan > DX12 > CPU` that has a hardware GPU (override the
//! order with `FLLM_AUTO_ORDER=cuda,vulkan,cpu`; `FLLM_AUTO_IGPU=0` leaves integrated GPUs other than Apple's to an explicit
//! `--device`), and there the best device (discrete before integrated, then the larger memory).
//! `metal` and `vulkan` prefer the native backend and use wgpu where no native device is listed. NPUs and TPUs are never chosen
//! automatically. [`device_candidates`] returns the whole chain, ending in `cpu`; `RustDecoder::set_device` tries the entries in
//! turn and reports why a device was skipped.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceType {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
}

impl DeviceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "cpu",
            DeviceType::Gpu => "gpu",
            DeviceType::IntegratedGpu => "igpu",
            DeviceType::Accelerator => "accel",
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub description: String,
    pub dev_type: DeviceType,
    pub free_memory_bytes: u64,
    pub total_memory_bytes: u64,
}

/// How a backend reaches its hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Api {
    /// The host CPU.
    Host,
    /// A native backend of this crate (cuda, hip, vulkan, metal, sycl, npu, tpu).
    Native,
    /// The wgpu layer (feature `rust-gpu`).
    Wgpu,
}

impl Api {
    pub fn as_str(self) -> &'static str {
        match self {
            Api::Host => "host",
            Api::Native => "native",
            Api::Wgpu => "wgpu",
        }
    }
}

/// One line of the device inventory: a [`DeviceInfo`] plus the backend that drives it and its number within that backend.
#[derive(Debug, Clone)]
pub struct InventoryEntry {
    pub info: DeviceInfo,
    /// Backend name as used by `--device`: `cpu`, `cuda`, `hip`, `vulkan`, `metal`, `dx12`, `sycl`, `npu`, `tpu` (wgpu adapters carry the name of their graphics API).
    pub backend: &'static str,
    pub api: Api,
    /// Number among the entries with the same `backend` and `api`.
    pub local_index: usize,
    /// What to pass to `--device` to get exactly this device (`vulkan:0`, `wgpu-vulkan:0`, `cuda:1`, `cpu`).
    pub spec: String,
}

impl InventoryEntry {
    /// Capabilities of this device; see [`DeviceCaps`].
    pub fn caps(&self) -> DeviceCaps {
        #[cfg_attr(not(feature = "cuda"), allow(unused_mut))]
        let mut caps = caps_for(self.backend, self.api, &self.info);
        #[cfg(feature = "cuda")]
        if self.backend == "cuda" && self.api == Api::Native {
            if let Some(cc) = number_after(&self.info.description.to_lowercase(), "sm_") {
                caps.compute_capability = Some(format!("{}.{}", cc / 10, cc % 10));
            }
            let (kernels, paths) = crate::backend::cuda::device_report(self.local_index);
            caps.kernels = Some(kernels).filter(|k| !k.is_empty());
            caps.fast_paths = paths;
        }
        caps
    }
}

/// Backends compiled into this binary.
pub fn compiled_backends() -> Vec<String> {
    let mut backends = vec!["cpu".to_owned()];
    if cfg!(feature = "rust-gpu") {
        backends.extend(["vulkan", "metal", "dx12", "wgpu"].map(str::to_owned));
    }
    if cfg!(feature = "cuda") {
        backends.push("cuda".to_owned());
    }
    backends.extend(crate::backend::native_backend_names().into_iter().map(str::to_owned));
    backends
}

/// Does the inventory list a device for the backend `name` (`cpu` always)?
pub fn is_backend_available(name: &str) -> bool {
    let name = name.trim().to_lowercase();
    if name == "cpu" {
        return true;
    }
    !lookup(cached_inventory(), &name).is_empty()
}

// ---------------------------------------------------------------------------------------------------------------------------
// Specification and resolution
// ---------------------------------------------------------------------------------------------------------------------------

/// Default order of `--device auto` (backends without a hardware device are skipped; `cpu` is the end of the chain). `sycl`, `npu` and
/// `tpu` are not part of it: they are chosen by name (`--device sycl`) or put into `FLLM_AUTO_ORDER` (`sycl` only; the NPU and TPU never
/// take part in `auto`). An Intel GPU is reached through Vulkan.
pub const DEFAULT_AUTO_ORDER: &[&str] = &["cuda", "hip", "metal", "vulkan", "dx12", "cpu"];

/// How `auto` chooses: the backends in the order tried, and whether integrated GPUs take part.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutoPolicy {
    pub order: Vec<String>,
    /// Integrated GPUs (shared system memory) may be chosen. They are the default; `FLLM_AUTO_IGPU=0` leaves them to an explicit
    /// `--device` (on x86 machines the CPU kernels can be the faster choice, and nothing in the inventory says which is).
    pub integrated: bool,
}

impl Default for AutoPolicy {
    fn default() -> Self {
        AutoPolicy { order: DEFAULT_AUTO_ORDER.iter().map(|n| (*n).to_owned()).collect(), integrated: true }
    }
}

impl AutoPolicy {
    /// `FLLM_AUTO_ORDER` (comma separated backend names) and `FLLM_AUTO_IGPU` (`0`, `off`, `no` exclude integrated GPUs).
    pub fn from_env() -> Self {
        Self::from_values(std::env::var("FLLM_AUTO_ORDER").ok().as_deref(), std::env::var("FLLM_AUTO_IGPU").ok().as_deref())
    }

    /// [`AutoPolicy::from_env`] over explicit values.
    pub fn from_values(order: Option<&str>, integrated: Option<&str>) -> Self {
        let mut policy = AutoPolicy::default();
        if let Some(text) = order.filter(|t| !t.trim().is_empty()) {
            policy.order = text.split(',').map(|n| n.trim().to_lowercase()).filter(|n| !n.is_empty()).collect();
        }
        if let Some(value) = integrated {
            policy.integrated = !matches!(value.trim().to_lowercase().as_str(), "0" | "off" | "no" | "false");
        }
        policy
    }
}

/// The order `auto` follows: `FLLM_AUTO_ORDER` (comma separated backend names) or [`DEFAULT_AUTO_ORDER`].
pub fn auto_order() -> Vec<String> {
    AutoPolicy::from_env().order
}

/// Is `spec` the request for automatic selection?
pub fn is_auto(spec: &str) -> bool {
    spec.trim().eq_ignore_ascii_case("auto")
}

/// Canonical backend name for the aliases `--device` accepts (the inventory uses the canonical names).
fn canonical(kind: &str) -> &str {
    match kind {
        "rocm" => "hip",
        "xpu" => "sycl",
        "openvino" => "npu",
        "vulkan-native" => "vulkan",
        "metal-native" => "metal",
        "gpu" => "wgpu",
        other => other,
    }
}

/// The inventory entries that `kind` (a name accepted by `--device`) numbers: index `N` of `kind:N` is a position in this list.
fn lookup<'a>(inventory: &'a [InventoryEntry], kind: &str) -> Vec<&'a InventoryEntry> {
    let base = canonical(kind);
    let native_only = matches!(kind, "vulkan-native" | "metal-native");
    if base == "wgpu" {
        return inventory.iter().filter(|e| e.api == Api::Wgpu).collect();
    }
    if let Some(api) = base.strip_prefix("wgpu-") {
        return inventory.iter().filter(|e| e.api == Api::Wgpu && e.backend == api).collect();
    }
    if base == "cpu" {
        return inventory.iter().filter(|e| e.api == Api::Host).collect();
    }
    let native: Vec<&InventoryEntry> = inventory.iter().filter(|e| e.api == Api::Native && e.backend == base).collect();
    if !native.is_empty() || native_only {
        return native;
    }
    // vulkan / metal / dx12 without a native device: the wgpu adapters of that API
    inventory.iter().filter(|e| e.api == Api::Wgpu && e.backend == base).collect()
}

/// `--device` value split into backend name and optional device number.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSpec {
    pub kind: String,
    pub index: Option<usize>,
}

/// Parse `kind` or `kind:N` (case-insensitive). `auto` is returned as kind `auto`.
pub fn parse_spec(spec: &str) -> Result<DeviceSpec, String> {
    let lower = spec.trim().to_lowercase();
    if lower.is_empty() {
        return Err("empty --device value".into());
    }
    match lower.split_once(':') {
        Some((kind, number)) => {
            let index = number.parse::<usize>().map_err(|_| format!("invalid device number in '{spec}' (expected <backend>:<N>, N counts the devices of that backend)"))?;
            Ok(DeviceSpec { kind: kind.to_owned(), index: Some(index) })
        }
        None => Ok(DeviceSpec { kind: lower, index: None }),
    }
}

/// Position of the best device among `entries` (the numbers `kind:N` takes): discrete before integrated, then more memory, then the lower number.
/// Integrated GPUs only count when `integrated` is set; Apple GPUs always count, since their unified memory is what the whole machine has.
fn best_position(entries: &[&InventoryEntry], integrated: bool) -> Option<usize> {
    let rank = |e: &InventoryEntry| match e.info.dev_type {
        DeviceType::Gpu => 0,
        DeviceType::IntegratedGpu => 1,
        DeviceType::Accelerator => 2,
        DeviceType::Cpu => 3,
    };
    let allowed = |e: &InventoryEntry| match e.info.dev_type {
        DeviceType::Gpu => true,
        DeviceType::IntegratedGpu => integrated || e.backend == "metal",
        DeviceType::Accelerator | DeviceType::Cpu => false,
    };
    entries
        .iter()
        .enumerate()
        .filter(|(_, e)| allowed(e))
        .min_by_key(|(position, e)| (rank(e), std::cmp::Reverse(e.info.total_memory_bytes), *position))
        .map(|(position, _)| position)
}

/// The chain `auto` follows over `inventory`: the best device of each backend in the policy's order that has a hardware GPU, then `cpu`.
/// NPUs and TPUs are never part of it.
pub fn auto_chain(inventory: &[InventoryEntry], policy: &AutoPolicy) -> Vec<(String, usize)> {
    let mut chain: Vec<(String, usize)> = Vec::new();
    for kind in &policy.order {
        if kind == "cpu" {
            break;
        }
        if matches!(canonical(kind), "npu" | "tpu") || chain.iter().any(|(k, _)| k == kind) {
            continue;
        }
        let entries = lookup(inventory, kind);
        if let Some(position) = best_position(&entries, policy.integrated) {
            chain.push((kind.clone(), position));
        }
    }
    chain.push(("cpu".to_owned(), 0));
    chain
}

/// Candidates for `spec` over `inventory`, best first. An explicit specification yields exactly one entry; `auto` the whole chain
/// (see [`auto_chain`]). The numbers are those of `--device kind:N` (devices of that backend).
pub fn resolve_over(spec: &str, inventory: &[InventoryEntry], policy: &AutoPolicy) -> Result<Vec<(String, usize)>, String> {
    let parsed = parse_spec(spec)?;
    if parsed.kind == "auto" {
        if parsed.index.is_some() {
            return Err("--device auto takes no device number".into());
        }
        return Ok(auto_chain(inventory, policy));
    }
    if parsed.kind == "cpu" {
        return Ok(vec![("cpu".to_owned(), 0)]);
    }
    let index = parsed.index.unwrap_or(0);
    let entries = lookup(inventory, &parsed.kind);
    // An empty list means the backend cannot be enumerated here (or is not present): the creation reports the real reason.
    if index > 0 && !entries.is_empty() && index >= entries.len() {
        let list: Vec<String> = entries.iter().enumerate().map(|(n, e)| format!("{}:{n} = {}", parsed.kind, e.info.name)).collect();
        return Err(format!(
            "--device {}:{index}: this machine has {} {} device(s) ({}); N counts the devices of one backend, not the inventory numbers of `fllm-rs devices`",
            parsed.kind,
            entries.len(),
            parsed.kind,
            list.join(", ")
        ));
    }
    Ok(vec![(parsed.kind, index)])
}

/// All devices to try for `spec`, best first (see [`resolve_over`]). Lists the hardware only where the specification needs it
/// (`auto`, or a device number above 0), and only once per process.
pub fn device_candidates(spec: &str) -> Result<Vec<(String, usize)>, String> {
    let parsed = parse_spec(spec)?;
    if parsed.kind != "auto" && parsed.index.unwrap_or(0) == 0 {
        return Ok(vec![(if parsed.kind == "cpu" { "cpu".to_owned() } else { parsed.kind }, 0)]);
    }
    resolve_over(spec, cached_inventory(), &AutoPolicy::from_env())
}

/// The inventory of this process, listed on first use. Selection happens at start-up, so the free-memory figures going stale do not matter
/// here; [`find_entry`] and [`inventory`] list afresh.
fn cached_inventory() -> &'static [InventoryEntry] {
    static CACHE: std::sync::OnceLock<Vec<InventoryEntry>> = std::sync::OnceLock::new();
    CACHE.get_or_init(inventory)
}

/// Select a device from a `--device` value: `(backend, number within that backend)`. For `auto` this is the first entry of the
/// chain, the best device; use [`device_candidates`] for the fallbacks.
pub fn select_device(spec: &str) -> Result<(String, usize), String> {
    device_candidates(spec).map(|mut chain| chain.remove(0))
}

/// The inventory entry behind `backend:index` as `select_device` returns it, listed afresh so that the memory figures are current
/// (`None` for the CPU or a device the inventory does not list).
pub fn find_entry(backend: &str, index: usize) -> Option<InventoryEntry> {
    lookup(&inventory(), backend).get(index).map(|e| (*e).clone())
}

/// `vulkan:0 (NVIDIA GeForce RTX 4090)` for log lines.
pub fn describe_choice(backend: &str, index: usize) -> String {
    if backend == "cpu" {
        return "cpu".to_owned();
    }
    match lookup(cached_inventory(), backend).get(index) {
        Some(entry) => format!("{backend}:{index} ({})", entry.info.name),
        None => format!("{backend}:{index}"),
    }
}

/// `cuda:0 (RTX 4090) -> vulkan:0 (RTX 4090) -> cpu` for log lines.
pub fn describe_chain(chain: &[(String, usize)]) -> String {
    chain.iter().map(|(backend, index)| describe_choice(backend, *index)).collect::<Vec<_>>().join(" -> ")
}

// ---------------------------------------------------------------------------------------------------------------------------
// Inventory
// ---------------------------------------------------------------------------------------------------------------------------

fn meminfo(key: &str) -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix(key)?.trim().strip_suffix("kB"))
                .and_then(|value| value.trim().parse::<u64>().ok())
        })
        .map_or(0, |kib| kib * 1024)
}

fn cpu_name() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| line.strip_prefix("model name"))
                .and_then(|rest| rest.split_once(':'))
                .map(|(_, name)| name.trim().to_owned())
        })
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned())
}

fn cpu_info() -> DeviceInfo {
    DeviceInfo {
        index: 0,
        name: cpu_name(),
        description: format!(
            "{} threads, {}",
            std::thread::available_parallelism().map_or(1, usize::from),
            crate::cpu_kernels::isa()
        ),
        dev_type: DeviceType::Cpu,
        free_memory_bytes: host_memory(true),
        total_memory_bytes: host_memory(false),
    }
}

/// Physical memory of the host that is free right now (what an integrated GPU can still be given).
pub fn host_free_memory_bytes() -> u64 {
    host_memory(true)
}

/// Free (`available`) or total physical memory of the host: `/proc/meminfo` on Linux, the system call of the platform elsewhere.
fn host_memory(available: bool) -> u64 {
    let from_proc = meminfo(if available { "MemAvailable:" } else { "MemTotal:" });
    if from_proc > 0 {
        return from_proc;
    }
    platform_memory(available)
}

#[cfg(windows)]
fn platform_memory(available: bool) -> u64 {
    #[repr(C)]
    #[allow(dead_code)] // laid out for the system call; only two fields are read back
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `status` is a valid MEMORYSTATUSEX with `length` set, as the call requires.
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return 0;
    }
    if available {
        status.avail_phys
    } else {
        status.total_phys
    }
}

#[cfg(target_vendor = "apple")]
fn platform_memory(available: bool) -> u64 {
    extern "C" {
        fn sysctlbyname(name: *const std::ffi::c_char, oldp: *mut std::ffi::c_void, oldlenp: *mut usize, newp: *mut std::ffi::c_void, newlen: usize) -> i32;
    }
    let mut value: u64 = 0;
    let mut size = std::mem::size_of::<u64>();
    // SAFETY: `value` and `size` describe a writable u64; the name is a NUL-terminated literal.
    let code = unsafe { sysctlbyname(c"hw.memsize".as_ptr(), (&mut value as *mut u64).cast(), &mut size, std::ptr::null_mut(), 0) };
    if code != 0 {
        return 0;
    }
    // macOS has no cheap "available" figure; the total is the honest upper bound the memory planner works with
    let _ = available;
    value
}

#[cfg(not(any(windows, target_vendor = "apple")))]
fn platform_memory(_available: bool) -> u64 {
    0
}

/// The graphics API an adapter of wgpu reports in its description (`Vulkan, driver ...`, `Metal, ...`, `Dx12, ...`).
#[cfg(any(feature = "rust-gpu", test))]
fn wgpu_api_of(description: &str) -> &'static str {
    let lower = description.to_lowercase();
    if lower.starts_with("vulkan") {
        "vulkan"
    } else if lower.starts_with("metal") {
        "metal"
    } else if lower.starts_with("dx12") {
        "dx12"
    } else {
        "wgpu"
    }
}

/// Name of the backend that drives the AMD devices of the cuda module's inventory: `hip` when the feature is on, else the CUDA backend
/// itself (a `FLLM_HIP=1` build of the `cuda` feature answers to `--device cuda`).
#[cfg(any(feature = "cuda", test))]
fn rocm_backend() -> &'static str {
    if cfg!(feature = "hip") {
        "hip"
    } else {
        "cuda"
    }
}

/// Number the raw list: the number within (backend, api) and the `--device` text of every entry.
fn index_entries(raw: Vec<(&'static str, Api, DeviceInfo)>) -> Vec<InventoryEntry> {
    let native: Vec<&'static str> = raw.iter().filter(|r| r.1 == Api::Native).map(|r| r.0).collect();
    let mut counts: Vec<((&'static str, Api), usize)> = Vec::new();
    raw.into_iter()
        .map(|(backend, api, info)| {
            let local_index = match counts.iter_mut().find(|(key, _)| *key == (backend, api)) {
                Some((_, n)) => {
                    *n += 1;
                    *n - 1
                }
                None => {
                    counts.push(((backend, api), 1));
                    0
                }
            };
            let spec = match api {
                Api::Host => "cpu".to_owned(),
                // the plain name selects the native backend where there is one; the wgpu devices are then `wgpu-<api>:N`
                Api::Wgpu if native.contains(&backend) => format!("wgpu-{backend}:{local_index}"),
                Api::Wgpu | Api::Native => format!("{backend}:{local_index}"),
            };
            InventoryEntry { info, backend, api, local_index, spec }
        })
        .collect()
}

/// Every device of every backend compiled into this binary, with the backend that drives it.
pub fn inventory() -> Vec<InventoryEntry> {
    #[allow(unused_mut)]
    let mut raw: Vec<(&'static str, Api, DeviceInfo)> = vec![("cpu", Api::Host, cpu_info())];
    #[cfg(feature = "cuda")]
    for device in crate::backend::cuda::list_devices(raw.len()) {
        let backend = if device.description.starts_with("CUDA/ROCm") { rocm_backend() } else { "cuda" };
        raw.push((backend, Api::Native, device));
    }
    #[cfg(feature = "rust-gpu")]
    for device in crate::gpu::list_adapters(raw.len()) {
        let backend = wgpu_api_of(&device.description);
        raw.push((backend, Api::Wgpu, device));
    }
    for (backend, device) in crate::backend::native_list_devices_tagged(raw.len()) {
        raw.push((backend, Api::Native, device));
    }
    // some Windows drivers announce the same integrated GPU twice (two driver registrations): list it once
    let mut seen: Vec<(&'static str, Api, String, u64)> = Vec::new();
    raw.retain(|(backend, api, info)| {
        if info.dev_type != DeviceType::IntegratedGpu {
            return true;
        }
        let key = (*backend, *api, info.name.clone(), info.total_memory_bytes);
        if seen.contains(&key) {
            false
        } else {
            seen.push(key);
            true
        }
    });
    index_entries(raw)
}

/// Every device (see [`inventory`]), numbered in the global inventory.
pub fn list_devices() -> Vec<DeviceInfo> {
    inventory().into_iter().map(|entry| entry.info).collect()
}

// ---------------------------------------------------------------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------------------------------------------------------------

/// What a device can do, as far as the backends report it. `None` is "not known": the figure is not part of what the backend tells
/// `fllm-rs devices` (and for most GPUs the bandwidth is not either; `fllm-rs devices --measure` measures it for the CPU).
///
/// Most values come from the description a backend gives its devices ([`DeviceInfo::description`]): the architecture name (`sm_89`,
/// `gfx1100`, `Apple8`), feature words (`fp16`, `int-dot`, `coop-matrix`, `XMX`, `unified-memory`) and a `N GB/s` figure. They are
/// inferred, not queried from the hardware again; `notes` says where a value is a guess.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceCaps {
    /// Backend name (`cuda`, `hip`, `vulkan`, ...).
    pub backend: String,
    /// `host`, `native` or `wgpu`.
    pub api: String,
    /// Native FP16 arithmetic (the CPU: FP16 conversion on x86, FP16 arithmetic on ARM).
    pub fp16: Option<bool>,
    pub bf16: Option<bool>,
    /// A hardware integer dot-product instruction (dp4a, VNNI, dotprod, DP4A on Intel, `int-dot` of Vulkan).
    pub int8_dot: Option<bool>,
    /// Matrix units (`tensor cores (FP8)`, `MFMA`, `WMMA`, `XMX 8x16x16`, `simdgroup_matrix`, `cooperative matrix`), `Some("none")` when there are none.
    pub matrix_cores: Option<String>,
    /// CPU and device share one memory (APUs, Apple silicon, the CPU itself).
    pub unified_memory: Option<bool>,
    pub total_memory_bytes: u64,
    pub free_memory_bytes: u64,
    /// Memory bandwidth in GB/s, when the backend reports it (`~273 GB/s` in the description) or it was measured.
    pub bandwidth_gbs: Option<f64>,
    /// `true` when `bandwidth_gbs` was measured by this process instead of read from the description.
    pub bandwidth_measured: bool,
    /// CUDA: compute capability (`9.0`), the cubin of this binary the device loads, and the fast paths this binary has for it
    /// (`compute_capability`, `kernels`, `fast_paths` stay empty for other backends).
    pub compute_capability: Option<String>,
    pub kernels: Option<String>,
    pub fast_paths: Vec<String>,
    pub notes: Vec<String>,
}

fn yes_no(value: Option<bool>) -> &'static str {
    match value {
        Some(true) => "yes",
        Some(false) => "no",
        None => "?",
    }
}

impl DeviceCaps {
    /// `FP16 yes | BF16 ? | Int8 dot yes | Matrix cores tensor cores | Unified memory no | Bandwidth ? GB/s` for `fllm-rs devices`.
    pub fn summary(&self) -> String {
        let bandwidth = match self.bandwidth_gbs {
            Some(gbs) if self.bandwidth_measured => format!("{gbs:.0} GB/s (measured)"),
            Some(gbs) => format!("{gbs:.0} GB/s"),
            None => "? GB/s".to_owned(),
        };
        format!(
            "FP16 {} | BF16 {} | Int8 dot {} | Matrix cores {} | Unified memory {} | Bandwidth {}",
            yes_no(self.fp16),
            yes_no(self.bf16),
            yes_no(self.int8_dot),
            self.matrix_cores.as_deref().unwrap_or("?"),
            yes_no(self.unified_memory),
            bandwidth
        )
    }

    /// The same facts as a JSON object (`fllm-rs devices --json`).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "backend": self.backend,
            "api": self.api,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "int8_dot": self.int8_dot,
            "matrix_cores": self.matrix_cores,
            "unified_memory": self.unified_memory,
            "total_memory_bytes": self.total_memory_bytes,
            "free_memory_bytes": self.free_memory_bytes,
            "bandwidth_gbs": self.bandwidth_gbs,
            "bandwidth_measured": self.bandwidth_measured,
            "compute_capability": self.compute_capability,
            "kernels": self.kernels,
            "fast_paths": self.fast_paths,
            "notes": self.notes,
        })
    }
}

/// The backend and API a bare [`DeviceInfo`] belongs to, recognised by the description its backend gives it.
pub fn infer_backend(info: &DeviceInfo) -> (&'static str, Api) {
    let desc = info.description.to_lowercase();
    if info.dev_type == DeviceType::Cpu {
        return ("cpu", Api::Host);
    }
    if desc.starts_with("cuda/rocm") || desc.starts_with("rocm") {
        return (if cfg!(feature = "hip") { "hip" } else { "cuda" }, Api::Native);
    }
    if desc.starts_with("cuda") {
        return ("cuda", Api::Native);
    }
    if desc.starts_with("native,") && desc.contains("vulkan") {
        return ("vulkan", Api::Native);
    }
    if desc.starts_with("metal (native)") {
        return ("metal", Api::Native);
    }
    if desc.starts_with("sycl") {
        return ("sycl", Api::Native);
    }
    if desc.starts_with("tpu") {
        return ("tpu", Api::Native);
    }
    if desc.starts_with("npu") || desc.contains("openvino") || info.dev_type == DeviceType::Accelerator {
        return ("npu", Api::Native);
    }
    for (prefix, api) in [("vulkan", "vulkan"), ("metal", "metal"), ("dx12", "dx12")] {
        if desc.starts_with(prefix) {
            return (api, Api::Wgpu);
        }
    }
    ("wgpu", Api::Wgpu)
}

/// Capabilities of one device (see [`DeviceCaps`]). The backend is recognised from the description, so any [`DeviceInfo`] works;
/// [`InventoryEntry::caps`] knows the backend for certain.
pub fn device_caps(info: &DeviceInfo) -> DeviceCaps {
    let (backend, api) = infer_backend(info);
    caps_for(backend, api, info)
}

/// The number written directly before `marker` in `text` (`~273 GB/s` -> 273.0).
fn number_before(text: &str, marker: &str) -> Option<f64> {
    let end = text.find(marker)?;
    let head = text[..end].trim_end();
    let start = head.char_indices().rev().find(|(_, c)| !(c.is_ascii_digit() || *c == '.')).map_or(0, |(i, c)| i + c.len_utf8());
    head[start..].parse::<f64>().ok().filter(|v| v.is_finite() && *v > 0.0)
}

/// The digits following the first occurrence of `marker` in `text` that has any (`sm_89` after `sm_` -> 89).
fn number_after(text: &str, marker: &str) -> Option<u32> {
    text.match_indices(marker).find_map(|(at, _)| {
        let digits: String = text[at + marker.len()..].chars().take_while(char::is_ascii_digit).collect();
        digits.parse().ok()
    })
}

/// The `gfxNNNN` architecture in a description or name.
fn gfx_arch(text: &str) -> Option<String> {
    let start = text.find("gfx")?;
    let arch: String = text[start..].chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
    (arch.len() >= 6 && arch[3..].chars().all(|c| c.is_ascii_hexdigit())).then_some(arch)
}

fn caps_for(backend: &str, api: Api, info: &DeviceInfo) -> DeviceCaps {
    let desc = info.description.to_lowercase();
    let name = info.name.to_lowercase();
    let mut caps = DeviceCaps {
        backend: backend.to_owned(),
        api: api.as_str().to_owned(),
        total_memory_bytes: info.total_memory_bytes,
        free_memory_bytes: info.free_memory_bytes,
        bandwidth_gbs: number_before(&desc, "gb/s"),
        ..DeviceCaps::default()
    };
    let integrated = info.dev_type == DeviceType::IntegratedGpu;
    match (backend, api) {
        ("cpu", _) => {
            let features = cpu_features();
            let has = |f: &str| features.contains(&f);
            caps.fp16 = Some(if cfg!(target_arch = "aarch64") { has("fp16") } else { has("f16c") });
            caps.int8_dot = Some(has("avxvnni") || has("avx512vnni") || has("avxvnniint8") || has("dotprod"));
            caps.matrix_cores = Some(if has("i8mm") { "i8mm (SMMLA/UMMLA)".to_owned() } else { "none (SIMD only)".to_owned() });
            caps.unified_memory = Some(true);
            caps.notes.push("x86: FP16 means F16C conversion; BF16 and AMX are not probed".to_owned());
        }
        ("cuda", _) => {
            let cc = number_after(&desc, "sm_");
            if let Some(cc) = cc {
                caps.fp16 = Some(cc >= 53);
                caps.bf16 = Some(cc >= 80);
                caps.int8_dot = Some(cc >= 61);
                caps.matrix_cores = Some(
                    match cc {
                        0..=69 => "none",
                        70..=74 => "tensor cores (FP16)",
                        75..=79 => "tensor cores (FP16, INT8)",
                        80..=88 => "tensor cores (TF32, BF16)",
                        89 => "tensor cores (FP8)",
                        90..=99 => "tensor cores (FP8, wgmma, TMA)",
                        100..=119 => "tensor cores (FP8, FP4, tcgen05)",
                        _ => "tensor cores (FP8, FP4)",
                    }
                    .to_owned(),
                );
                caps.notes.push("inferred from the compute capability".to_owned());
            }
            caps.unified_memory = Some(["gb10", "orin", "thor", "tegra"].iter().any(|n| name.contains(n)));
        }
        ("hip", _) => {
            let arch = gfx_arch(&desc).or_else(|| gfx_arch(&name));
            if let Some(arch) = &arch {
                let family = |p: &str| arch.starts_with(p);
                let cdna = matches!(arch.as_str(), "gfx908" | "gfx90a") || family("gfx94") || family("gfx95");
                let wmma = family("gfx11") || family("gfx12");
                caps.fp16 = Some(true);
                caps.bf16 = Some(arch != "gfx908" && (cdna || wmma));
                caps.int8_dot = Some(!(family("gfx900") || family("gfx902") || family("gfx904") || family("gfx909") || family("gfx90c") || family("gfx100") || family("gfx101")));
                caps.matrix_cores = Some(if cdna { "MFMA" } else if wmma { "WMMA" } else { "none" }.to_owned());
                caps.unified_memory = Some(matches!(arch.as_str(), "gfx90c" | "gfx909" | "gfx1103" | "gfx1150" | "gfx1151") || name.contains("mi300a"));
                caps.notes.push("inferred from the architecture name".to_owned());
            }
        }
        ("vulkan", Api::Native) => {
            // the description lists exactly the features the probe found (fp16+int8+int-dot+coop-matrix+unified-memory)
            caps.fp16 = Some(desc.contains("fp16"));
            caps.int8_dot = Some(desc.contains("int-dot"));
            caps.matrix_cores = Some(if desc.contains("coop-matrix") { "cooperative matrix" } else { "none" }.to_owned());
            caps.unified_memory = Some(desc.contains("unified-memory") || integrated);
        }
        ("metal", Api::Native) => {
            caps.fp16 = Some(true);
            let generation = number_after(&desc, "apple");
            let metal3 = desc.contains("metal3") || desc.contains("metal4");
            let msl_ok = number_after(&desc, "msl 3.").is_some_and(|minor| minor >= 1) || desc.contains("metal4");
            caps.bf16 = Some(metal3 && msl_ok);
            caps.matrix_cores = Some(if generation.is_some_and(|g| g >= 7) || desc.contains("mac2") { "simdgroup_matrix" } else { "none" }.to_owned());
            caps.unified_memory = Some(desc.contains("unified memory"));
        }
        ("sycl", _) => {
            caps.fp16 = Some(true);
            caps.int8_dot = Some(true);
            // "XMX 8x16x16" or "no XMX"
            let has_xmx = desc.contains("xmx") && !desc.contains("no xmx");
            let shape = desc.find("xmx ").map(|at| desc[at + 4..].split([',', ' ']).next().unwrap_or("").to_owned()).filter(|s| !s.is_empty());
            caps.bf16 = Some(has_xmx);
            caps.matrix_cores = Some(match shape {
                Some(shape) if has_xmx => format!("XMX {shape}"),
                _ if has_xmx => "XMX".to_owned(),
                _ => "none".to_owned(),
            });
            caps.unified_memory = Some(integrated);
            caps.notes.push("FP16 and DP4A are assumed for every Intel GPU the SYCL backend lists".to_owned());
        }
        ("tpu", _) => {
            caps.bf16 = Some(true);
            caps.matrix_cores = Some("MXU".to_owned());
            caps.unified_memory = Some(false);
        }
        ("npu", _) => {
            caps.fp16 = Some(true);
            caps.int8_dot = Some(true);
            caps.matrix_cores = Some("NPU array".to_owned());
            caps.unified_memory = Some(true);
            caps.notes.push("NPU capabilities are assumed from the device class".to_owned());
        }
        // wgpu adapters (and anything unrecognised): the adapter limits are not part of the inventory
        _ => {
            caps.unified_memory = Some(integrated).filter(|u| *u);
            caps.notes.push("wgpu adapter: feature support is not queried".to_owned());
        }
    }
    caps
}

/// A rough read bandwidth of the host memory in GB/s (best of three sweeps over up to 256 MiB with all cores). The measurement
/// takes a fraction of a second; it is what `fllm-rs devices --measure` prints for the CPU.
pub fn measure_cpu_bandwidth_gbs() -> f64 {
    use rayon::prelude::*;
    let free = host_memory(true);
    let limit: u64 = if free > 0 { (256u64 << 20).min(free / 8) } else { 64 << 20 };
    let words = (limit.max(16 << 20) / 8) as usize;
    let data: Vec<u64> = vec![1; words];
    let mut best = 0.0f64;
    for _ in 0..3 {
        let start = std::time::Instant::now();
        let sum: u64 = data.par_chunks(1 << 15).map(|chunk| chunk.iter().fold(0u64, |a, &b| a.wrapping_add(b))).sum();
        let seconds = start.elapsed().as_secs_f64();
        std::hint::black_box(sum);
        if seconds > 0.0 {
            best = best.max((words * 8) as f64 / seconds / 1e9);
        }
    }
    best
}

/// Host CPU capabilities from runtime detection: the kernel tier in use, every tier this CPU
/// supports and the SIMD extensions the kernels dispatch on.
pub fn cpu_capabilities() -> String {
    let tiers = crate::cpu_kernels::supported_tiers()
        .iter()
        .map(|t| t.name())
        .collect::<Vec<_>>()
        .join(", ");
    let features = cpu_features();
    format!(
        "kernel tier {} (supported: {tiers}); SIMD: {}",
        crate::cpu_kernels::active_tier().name(),
        if features.is_empty() {
            "none detected".to_owned()
        } else {
            features.join(" ")
        }
    )
}

/// SIMD extensions relevant to the CPU kernels that this CPU reports at runtime.
pub fn cpu_features() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut found = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        macro_rules! probe {
            ($($name:tt),*) => {
                $(if std::arch::is_x86_feature_detected!($name) { found.push($name); })*
            };
        }
        probe!(
            "avx2",
            "fma",
            "f16c",
            "avxvnni",
            "avx512f",
            "avx512bw",
            "avx512vl",
            "avx512vnni",
            "avxvnniint8"
        );
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! probe {
            ($($name:tt),*) => {
                $(if std::arch::is_aarch64_feature_detected!($name) { found.push($name); })*
            };
        }
        probe!("neon", "dotprod", "i8mm", "bf16", "fp16", "fhm", "sve", "sve2");
    }
    found
}

/// What the wgpu backend of this build runs on a device (the WGSL kernel set); adapter
/// specifics such as memory are not queried here.
pub fn gpu_capabilities() -> String {
    #[cfg(feature = "rust-gpu")]
    {
        format!(
            "WGSL kernels: linear/embedding {} (other formats dequantized to F32 on upload), \
             RMSNorm, RoPE, SwiGLU/GeGLU, residual, KV-cache write and paged attention \
             (f16/q8_0 cache, head width <= 512); check with `fllm-rs selftest-gpu`",
            crate::backend::wgpu_backend::native_formats().join("/")
        )
    }
    #[cfg(not(feature = "rust-gpu"))]
    {
        "no GPU backend in this build (rebuild with --features rust-gpu)".to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(name: &str, description: &str, dev_type: DeviceType, total_gib: u64) -> DeviceInfo {
        DeviceInfo {
            index: 0,
            name: name.to_owned(),
            description: description.to_owned(),
            dev_type,
            free_memory_bytes: total_gib << 30,
            total_memory_bytes: total_gib << 30,
        }
    }

    fn inventory_of(parts: &[(&'static str, Api, DeviceInfo)]) -> Vec<InventoryEntry> {
        let mut raw = vec![("cpu", Api::Host, info("Test CPU", "8 threads, avx2", DeviceType::Cpu, 32))];
        raw.extend(parts.iter().cloned());
        for (i, entry) in raw.iter_mut().enumerate() {
            entry.2.index = i;
        }
        index_entries(raw)
    }

    fn gpu(name: &str, description: &str, gib: u64) -> DeviceInfo {
        info(name, description, DeviceType::Gpu, gib)
    }

    fn default_order() -> AutoPolicy {
        AutoPolicy::default()
    }

    fn pair(spec: &str, index: usize) -> (String, usize) {
        (spec.to_owned(), index)
    }

    #[test]
    fn spec_parsing() {
        assert_eq!(parse_spec("Vulkan:2").unwrap(), DeviceSpec { kind: "vulkan".into(), index: Some(2) });
        assert_eq!(parse_spec(" CUDA ").unwrap(), DeviceSpec { kind: "cuda".into(), index: None });
        assert!(parse_spec("vulkan:x").is_err());
        assert!(parse_spec("").is_err());
        assert!(resolve_over("auto:1", &inventory_of(&[]), &default_order()).is_err());
    }

    #[test]
    fn numbers_count_within_one_backend() {
        let inv = inventory_of(&[
            ("cuda", Api::Native, gpu("A", "CUDA, sm_89, 76 SMs", 24)),
            ("vulkan", Api::Native, gpu("A", "native, Vulkan 1.3.0", 24)),
            ("vulkan", Api::Native, gpu("B", "native, Vulkan 1.3.0", 8)),
            ("vulkan", Api::Wgpu, gpu("A", "Vulkan, driver 1", 24)),
            ("vulkan", Api::Wgpu, gpu("B", "Vulkan, driver 1", 8)),
        ]);
        let specs: Vec<&str> = inv.iter().map(|e| e.spec.as_str()).collect();
        assert_eq!(specs, ["cpu", "cuda:0", "vulkan:0", "vulkan:1", "wgpu-vulkan:0", "wgpu-vulkan:1"]);
        // inventory number 3 is vulkan:1 (B): the number after the colon is not the inventory number
        assert_eq!(resolve_over("vulkan:1", &inv, &default_order()).unwrap(), vec![pair("vulkan", 1)]);
        assert_eq!(lookup(&inv, "vulkan")[1].info.name, "B");
        assert_eq!(lookup(&inv, "wgpu-vulkan")[1].info.name, "B");
        assert_eq!(lookup(&inv, "gpu").len(), 2);
        let err = resolve_over("vulkan:2", &inv, &default_order()).unwrap_err();
        assert!(err.contains("2 vulkan device") && err.contains("vulkan:1 = B"), "{err}");
        // the inventory number of vulkan:1 is no valid CUDA number either
        assert!(resolve_over("cuda:3", &inv, &default_order()).is_err());
    }

    #[test]
    fn wgpu_adapters_are_numbered_without_a_native_backend() {
        let inv = inventory_of(&[("vulkan", Api::Wgpu, gpu("A", "Vulkan, driver 1", 8)), ("dx12", Api::Wgpu, gpu("A", "Dx12, driver 1", 8))]);
        assert_eq!(inv[1].spec, "vulkan:0");
        assert_eq!(inv[2].spec, "dx12:0");
        assert_eq!(lookup(&inv, "vulkan").len(), 1);
        assert_eq!(lookup(&inv, "vulkan-native").len(), 0);
    }

    #[test]
    fn unknown_or_unlisted_devices_pass_through() {
        let inv = inventory_of(&[]);
        assert_eq!(resolve_over("cuda:1", &inv, &default_order()).unwrap(), vec![pair("cuda", 1)]);
        assert_eq!(resolve_over("rocm", &inv, &default_order()).unwrap(), vec![pair("rocm", 0)]);
        assert_eq!(resolve_over("cpu:3", &inv, &default_order()).unwrap(), vec![pair("cpu", 0)]);
        assert_eq!(resolve_over("quantum", &inv, &default_order()).unwrap(), vec![pair("quantum", 0)]);
    }

    #[test]
    fn aliases_use_the_same_devices() {
        let inv = inventory_of(&[("hip", Api::Native, gpu("R", "ROCm gfx1100, wave32, WMMA", 24)), ("sycl", Api::Native, gpu("I", "SYCL, Level Zero", 16))]);
        assert_eq!(lookup(&inv, "rocm").len(), 1);
        assert_eq!(lookup(&inv, "xpu").len(), 1);
        assert!(resolve_over("rocm:1", &inv, &default_order()).is_err());
    }

    #[test]
    fn auto_follows_the_priority_and_ends_in_cpu() {
        let inv = inventory_of(&[
            ("vulkan", Api::Wgpu, gpu("A", "Vulkan, driver 1", 24)),
            ("dx12", Api::Wgpu, gpu("A", "Dx12, driver 1", 24)),
            ("cuda", Api::Native, gpu("A", "CUDA, sm_89", 24)),
            ("vulkan", Api::Native, gpu("A", "native, Vulkan 1.3.0", 24)),
            ("sycl", Api::Native, gpu("I", "SYCL, Level Zero", 16)),
            ("npu", Api::Native, info("N", "npu-x", DeviceType::Accelerator, 8)),
            ("tpu", Api::Native, info("T", "tpu pjrt", DeviceType::Accelerator, 16)),
        ]);
        let chain = resolve_over("auto", &inv, &default_order()).unwrap();
        assert_eq!(chain, vec![pair("cuda", 0), pair("vulkan", 0), pair("dx12", 0), pair("cpu", 0)]);
        // SYCL takes part only when the order names it
        let policy = AutoPolicy::from_values(Some("vulkan,sycl,cpu"), None);
        assert_eq!(resolve_over("auto", &inv, &policy).unwrap(), vec![pair("vulkan", 0), pair("sycl", 0), pair("cpu", 0)]);
    }

    #[test]
    fn auto_without_hardware_is_the_cpu() {
        assert_eq!(resolve_over("auto", &inventory_of(&[]), &default_order()).unwrap(), vec![pair("cpu", 0)]);
    }

    #[test]
    fn auto_prefers_discrete_then_memory() {
        let inv = inventory_of(&[
            ("vulkan", Api::Native, info("iGPU", "native, Vulkan 1.3.0", DeviceType::IntegratedGpu, 64)),
            ("vulkan", Api::Native, gpu("small", "native, Vulkan 1.3.0", 8)),
            ("vulkan", Api::Native, gpu("big", "native, Vulkan 1.3.0", 24)),
        ]);
        // vulkan:2 (the 24 GiB discrete card) although the integrated GPU has more memory and comes first
        assert_eq!(resolve_over("auto", &inv, &default_order()).unwrap(), vec![pair("vulkan", 2), pair("cpu", 0)]);
    }

    #[test]
    fn auto_order_can_be_changed() {
        let inv = inventory_of(&[("cuda", Api::Native, gpu("A", "CUDA, sm_89", 24)), ("vulkan", Api::Native, gpu("A", "native, Vulkan 1.3.0", 24))]);
        let policy = AutoPolicy::from_values(Some("vulkan, cpu ,cuda"), None);
        assert_eq!(policy.order, ["vulkan", "cpu", "cuda"]);
        assert_eq!(resolve_over("auto", &inv, &policy).unwrap(), vec![pair("vulkan", 0), pair("cpu", 0)]);
        // NPU / TPU in a custom order are still never chosen
        let inv = inventory_of(&[("npu", Api::Native, info("N", "npu-x", DeviceType::Accelerator, 8)), ("cuda", Api::Native, gpu("A", "CUDA, sm_89", 24))]);
        let policy = AutoPolicy::from_values(Some("npu,tpu,cuda"), None);
        assert_eq!(resolve_over("auto", &inv, &policy).unwrap(), vec![pair("cuda", 0), pair("cpu", 0)]);
        // an empty value keeps the default order
        assert_eq!(AutoPolicy::from_values(Some("  "), None), AutoPolicy::default());
    }

    #[test]
    fn integrated_gpus_can_be_left_out_of_auto() {
        let inv = inventory_of(&[
            ("vulkan", Api::Wgpu, info("iGPU", "Vulkan, driver 1", DeviceType::IntegratedGpu, 64)),
            ("metal", Api::Native, info("Apple M2", "Metal (native), Apple8", DeviceType::IntegratedGpu, 32)),
        ]);
        assert_eq!(resolve_over("auto", &inv, &AutoPolicy::default()).unwrap(), vec![pair("metal", 0), pair("vulkan", 0), pair("cpu", 0)]);
        let policy = AutoPolicy::from_values(None, Some("0"));
        assert!(!policy.integrated);
        // Apple GPUs stay: their unified memory is the whole machine; the x86 integrated GPU goes
        assert_eq!(resolve_over("auto", &inv, &policy).unwrap(), vec![pair("metal", 0), pair("cpu", 0)]);
        // an explicit device is not affected
        assert_eq!(resolve_over("vulkan", &inv, &policy).unwrap(), vec![pair("vulkan", 0)]);
        assert!(AutoPolicy::from_values(None, Some("1")).integrated && !AutoPolicy::from_values(None, Some("off")).integrated);
    }

    #[test]
    fn rocm_devices_of_a_plain_cuda_build_answer_to_cuda() {
        assert_eq!(rocm_backend(), if cfg!(feature = "hip") { "hip" } else { "cuda" });
    }

    #[test]
    fn wgpu_api_names() {
        assert_eq!(wgpu_api_of("Vulkan, driver 535"), "vulkan");
        assert_eq!(wgpu_api_of("Metal, driver 1"), "metal");
        assert_eq!(wgpu_api_of("Dx12, driver 31"), "dx12");
        assert_eq!(wgpu_api_of("Gl, driver"), "wgpu");
    }

    #[test]
    fn backend_is_recognised_from_the_description() {
        let of = |description: &str, dev_type: DeviceType| infer_backend(&info("x", description, dev_type, 8));
        assert_eq!(of("8 threads, avx2", DeviceType::Cpu), ("cpu", Api::Host));
        assert_eq!(of("CUDA, sm_89, 76 SMs", DeviceType::Gpu), ("cuda", Api::Native));
        assert_eq!(of("native, Vulkan 1.3.0, radv", DeviceType::Gpu), ("vulkan", Api::Native));
        assert_eq!(of("Vulkan, driver 1", DeviceType::Gpu), ("vulkan", Api::Wgpu));
        assert_eq!(of("Metal (native), Apple8", DeviceType::IntegratedGpu), ("metal", Api::Native));
        assert_eq!(of("Metal, driver 1", DeviceType::IntegratedGpu), ("metal", Api::Wgpu));
        assert_eq!(of("Dx12, driver 1", DeviceType::Gpu), ("dx12", Api::Wgpu));
        assert_eq!(of("SYCL, Level Zero", DeviceType::Gpu), ("sycl", Api::Native));
        assert_eq!(of("tpu pjrt tpu 1", DeviceType::Accelerator), ("tpu", Api::Native));
        assert_eq!(of("npu-ane: Core ML", DeviceType::Accelerator), ("npu", Api::Native));
    }

    #[test]
    fn number_parsing_helpers() {
        assert_eq!(number_before("apple m2, ~273 gb/s", "gb/s"), Some(273.0));
        assert_eq!(number_before("bandwidth 936.5 gb/s", "gb/s"), Some(936.5));
        assert_eq!(number_before("no figure", "gb/s"), None);
        assert_eq!(number_before("gb/s", "gb/s"), None);
        assert_eq!(number_after("cuda, sm_120, 84 sms", "sm_"), Some(120));
        assert_eq!(number_after("apple8, metal3", "apple"), Some(8));
        assert_eq!(number_after("none", "sm_"), None);
        assert_eq!(gfx_arch("rocm gfx1100, wave32"), Some("gfx1100".to_owned()));
        assert_eq!(gfx_arch("amd radeon (gfx90a:sramecc+:xnack-)"), Some("gfx90a".to_owned()));
        assert_eq!(gfx_arch("no arch"), None);
    }

    #[test]
    fn cuda_caps_follow_the_compute_capability() {
        let ada = device_caps(&gpu("RTX 4090", "CUDA, sm_89, 128 SMs", 24));
        assert_eq!((ada.backend.as_str(), ada.fp16, ada.bf16, ada.int8_dot), ("cuda", Some(true), Some(true), Some(true)));
        assert_eq!(ada.matrix_cores.as_deref(), Some("tensor cores (FP8)"));
        assert_eq!(ada.unified_memory, Some(false));
        let blackwell = device_caps(&gpu("RTX 5090", "CUDA, sm_120, 170 SMs", 32));
        assert_eq!(blackwell.matrix_cores.as_deref(), Some("tensor cores (FP8, FP4)"));
        assert_eq!(device_caps(&gpu("H100", "CUDA, sm_90, 132 SMs", 80)).matrix_cores.as_deref(), Some("tensor cores (FP8, wgmma, TMA)"));
        assert_eq!(device_caps(&gpu("B200", "CUDA, sm_100, 148 SMs", 180)).matrix_cores.as_deref(), Some("tensor cores (FP8, FP4, tcgen05)"));
        let pascal = device_caps(&gpu("GTX 1080", "CUDA, sm_61, 20 SMs", 8));
        assert_eq!((pascal.fp16, pascal.bf16, pascal.int8_dot), (Some(true), Some(false), Some(true)));
        assert_eq!(pascal.matrix_cores.as_deref(), Some("none"));
        assert_eq!(device_caps(&gpu("NVIDIA GB10", "CUDA, sm_121, 48 SMs", 120)).unified_memory, Some(true));
    }

    #[test]
    fn vulkan_caps_follow_the_feature_words() {
        let full = device_caps(&gpu("A", "native, Vulkan 1.3.0, radv, subgroup 64, fp16+int8+int-dot+coop-matrix", 16));
        assert_eq!(full.api, "native");
        assert_eq!((full.fp16, full.int8_dot), (Some(true), Some(true)));
        assert_eq!(full.matrix_cores.as_deref(), Some("cooperative matrix"));
        let bare = device_caps(&gpu("B", "native, Vulkan 1.2.0, subgroup 32", 4));
        assert_eq!((bare.fp16, bare.int8_dot), (Some(false), Some(false)));
        assert_eq!(bare.matrix_cores.as_deref(), Some("none"));
        let wgpu = device_caps(&gpu("C", "Vulkan, driver 1", 4));
        assert_eq!((wgpu.api.as_str(), wgpu.fp16, wgpu.matrix_cores.clone()), ("wgpu", None, None));
    }

    #[test]
    fn metal_and_amd_and_intel_caps() {
        let mac = device_caps(&info("Apple M2 Pro", "Metal (native), Apple M2 Pro (Apple8, Metal3, unified memory, 21.3 GiB working set, MSL 3.2), ~200 GB/s", DeviceType::IntegratedGpu, 32));
        assert_eq!((mac.fp16, mac.bf16, mac.unified_memory), (Some(true), Some(true), Some(true)));
        assert_eq!(mac.matrix_cores.as_deref(), Some("simdgroup_matrix"));
        assert_eq!(mac.bandwidth_gbs, Some(200.0));
        let amd = device_caps(&gpu("Radeon RX 7900 XTX (gfx1100)", "ROCm gfx1100, wave32, WMMA, CUDA/ROCm, 96 CUs", 24));
        assert_eq!(amd.backend, if cfg!(feature = "hip") { "hip" } else { "cuda" });
        let amd = caps_for("hip", Api::Native, &gpu("Radeon RX 7900 XTX (gfx1100)", "ROCm gfx1100, wave32, WMMA", 24));
        assert_eq!((amd.matrix_cores.as_deref(), amd.bf16, amd.int8_dot), (Some("WMMA"), Some(true), Some(true)));
        let mi = caps_for("hip", Api::Native, &gpu("AMD Instinct MI250 (gfx90a:sramecc+:xnack-)", "ROCm gfx90a", 64));
        assert_eq!(mi.matrix_cores.as_deref(), Some("MFMA"));
        let intel = device_caps(&info("Arc A770", "SYCL, Level Zero, Xe-HPG, 512 EUs, XMX 8x8x16, driver 1.3", DeviceType::Gpu, 16));
        assert_eq!(intel.matrix_cores.as_deref(), Some("XMX 8x8x16"));
        let plain = device_caps(&info("Iris Xe", "SYCL, Level Zero, 96 EUs, no XMX, driver 1.3", DeviceType::IntegratedGpu, 16));
        assert_eq!((plain.matrix_cores.as_deref(), plain.unified_memory), (Some("none"), Some(true)));
    }

    #[test]
    fn cpu_caps_are_filled_from_the_runtime_probe() {
        let caps = device_caps(&info("CPU", "8 threads, avx2", DeviceType::Cpu, 16));
        assert_eq!(caps.backend, "cpu");
        assert!(caps.fp16.is_some() && caps.int8_dot.is_some() && caps.matrix_cores.is_some());
        assert_eq!(caps.unified_memory, Some(true));
        assert!(caps.summary().starts_with("FP16 "), "{}", caps.summary());
    }

    #[test]
    fn caps_summary_and_json() {
        let mut caps = DeviceCaps { bandwidth_gbs: Some(51.2), ..DeviceCaps::default() };
        assert!(caps.summary().contains("Bandwidth 51 GB/s"));
        caps.bandwidth_measured = true;
        assert!(caps.summary().contains("(measured)"));
        caps.bandwidth_gbs = None;
        assert!(caps.summary().contains("Bandwidth ? GB/s"));
        let json = caps.to_json();
        assert!(json["fp16"].is_null() && json["notes"].is_array());
    }

    #[test]
    fn host_memory_is_reported() {
        let cpu = cpu_info();
        // every supported host reports its memory; a zero here would make the memory planner refuse every model
        assert!(cpu.total_memory_bytes > 0 || cfg!(not(any(target_os = "linux", windows, target_vendor = "apple"))));
    }
}
