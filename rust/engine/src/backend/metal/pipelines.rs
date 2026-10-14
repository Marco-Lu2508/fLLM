//! Compute pipelines of the native Metal backend (objc2-metal; compiled on Apple targets only).
//!
//! The Metal shaders are embedded as source text (`shaders::SOURCES`) and compiled at run time with
//! `newLibraryWithSource`. Each source file is its own library that is compiled on first use (or all in
//! parallel through [`PipelineCache::prewarm_libraries`]); the files named `common*` / `_*` are shared code that is
//! put in front of every unit (see [`source`]). A kernel becomes a `MTLComputePipelineState` per distinct set of
//! function-constant values, cached under a [`PipelineKey`]. `FLLM_METAL_LIB` swaps the whole source path for a
//! precompiled `.metallib` (`xcrun -sdk macosx metal -c *.metal; xcrun -sdk macosx metallib *.air -o fllm.metallib`).
//!
//! Level of verification: L0 (compiles for `aarch64-apple-darwin`; never run on a Mac).

pub mod key;
pub mod source;

pub use key::{ConstantValue, FunctionConstant, PipelineKey};

use super::device_info::platform::msl_version_code;
use super::device_info::{MetalConfig, MetalDevice, MetalError};
use super::shaders::SOURCES;
use objc2::rc::{autoreleasepool, Retained};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineState, MTLDataType, MTLDevice, MTLFunction,
    MTLFunctionConstantValues, MTLLanguageVersion, MTLLibrary,
};
use std::collections::HashMap;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::time::Instant;

/// A compiled compute pipeline with the limits the dispatcher needs.
pub struct Pipeline {
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    id: u64,
    key: PipelineKey,
    max_threads: usize,
    simd_width: usize,
    static_threadgroup_memory: usize,
}

impl std::fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pipeline({} max_threads={} simd={})",
            self.key, self.max_threads, self.simd_width
        )
    }
}

impl Pipeline {
    /// The `MTLComputePipelineState` for direct API calls.
    pub fn raw(&self) -> &ProtocolObject<dyn MTLComputePipelineState> {
        &self.state
    }

    /// Process-unique id (the recorder skips `setComputePipelineState` when it repeats).
    pub fn id(&self) -> u64 {
        self.id
    }

    pub fn key(&self) -> &PipelineKey {
        &self.key
    }

    /// `maxTotalThreadsPerThreadgroup`: the largest threadgroup this pipeline can run (register pressure may put it
    /// below the device limit).
    pub fn max_threads(&self) -> usize {
        self.max_threads
    }

    /// `threadExecutionWidth`: threads that execute in lock step (32 on Apple GPUs). Threadgroup sizes should be a
    /// multiple of it.
    pub fn simd_width(&self) -> usize {
        self.simd_width
    }

    /// Threadgroup memory the kernel declares statically; dynamic memory comes on top
    /// (`maxThreadgroupMemoryLength` bounds the sum).
    pub fn static_threadgroup_memory(&self) -> usize {
        self.static_threadgroup_memory
    }
}

struct Library {
    raw: Retained<ProtocolObject<dyn MTLLibrary>>,
}

struct UnitSlot {
    unit: source::Unit,
    library: OnceLock<Result<Library, MetalError>>,
}

/// Counters of the compile work done so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PipelineStats {
    pub units: usize,
    pub libraries_built: usize,
    pub pipelines_built: usize,
    pub compile_seconds: f64,
}

/// Libraries and pipelines of one device.
pub struct PipelineCache {
    device: MetalDevice,
    fast_math: bool,
    labels: bool,
    language: (u16, u16),
    units: Vec<UnitSlot>,
    kernel_index: RwLock<HashMap<String, usize>>,
    external: Option<Library>,
    pipelines: RwLock<HashMap<PipelineKey, Arc<Pipeline>>>,
    next_id: AtomicU64,
    libraries_built: AtomicUsize,
    pipelines_built: AtomicUsize,
    compile_nanos: AtomicU64,
}

fn load_metallib(device: &MetalDevice, path: &Path) -> Result<Library, MetalError> {
    let text = path.to_str().ok_or_else(|| MetalError::Compile {
        unit: path.display().to_string(),
        message: "path is not valid UTF-8".to_owned(),
    })?;
    autoreleasepool(|_| {
        let url = NSURL::fileURLWithPath(&NSString::from_str(text));
        device
            .raw()
            .newLibraryWithURL_error(&url)
            .map(|raw| Library { raw })
            .map_err(|e| MetalError::Compile {
                unit: text.to_owned(),
                message: e.localizedDescription().to_string(),
            })
    })
}

/// Run `work(i)` for `i in 0..count` on up to `threads` threads; the first error wins.
fn parallel_for<F: Fn(usize) -> Result<(), MetalError> + Sync>(
    count: usize,
    work: F,
) -> Result<(), MetalError> {
    let threads = std::thread::available_parallelism()
        .map_or(2, usize::from)
        .clamp(1, 8)
        .min(count.max(1));
    let next = AtomicUsize::new(0);
    let first_error: Mutex<Option<MetalError>> = Mutex::new(None);
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= count {
                    break;
                }
                if let Err(error) = work(index) {
                    first_error
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .get_or_insert(error);
                }
            });
        }
    });
    match first_error
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn constant_values(constants: &[FunctionConstant]) -> Option<Retained<MTLFunctionConstantValues>> {
    if constants.is_empty() {
        return None;
    }
    let values = MTLFunctionConstantValues::new();
    for constant in constants {
        let index = constant.index as usize;
        // SAFETY: every pointer refers to a live local of exactly the size of the Metal data type named next to
        // it (bool = 1 byte, 32-bit int / uint / float = 4 bytes); Metal copies the value during the call.
        unsafe {
            match constant.value {
                ConstantValue::Bool(v) => values.setConstantValue_type_atIndex(
                    NonNull::from(&v).cast(),
                    MTLDataType::Bool,
                    index,
                ),
                ConstantValue::I32(v) => values.setConstantValue_type_atIndex(
                    NonNull::from(&v).cast(),
                    MTLDataType::Int,
                    index,
                ),
                ConstantValue::U32(v) => values.setConstantValue_type_atIndex(
                    NonNull::from(&v).cast(),
                    MTLDataType::UInt,
                    index,
                ),
                ConstantValue::F32(v) => values.setConstantValue_type_atIndex(
                    NonNull::from(&v).cast(),
                    MTLDataType::Float,
                    index,
                ),
            }
        }
    }
    Some(values)
}

impl PipelineCache {
    /// Prepare the compile units from the embedded sources (nothing is compiled yet), or load the precompiled
    /// library named by `FLLM_METAL_LIB`.
    pub fn new(device: &MetalDevice, config: &MetalConfig) -> Result<Self, MetalError> {
        match &config.metallib_path {
            Some(path) => Ok(Self::assemble(
                device,
                config,
                Some(load_metallib(device, path)?),
                Vec::new(),
            )),
            None => Ok(Self::with_sources(device, config, SOURCES)),
        }
    }

    /// A cache over other sources than the embedded ones (self-tests, experiments). Same file-name rules as the
    /// embedded table: `common*` / `_*` files are shared prefixes.
    pub fn with_sources(
        device: &MetalDevice,
        config: &MetalConfig,
        sources: &[(&str, &str)],
    ) -> Self {
        let units = source::build_units(
            sources,
            &device.info().shader_defines(),
            config.single_library,
        );
        Self::assemble(device, config, None, units)
    }

    fn assemble(
        device: &MetalDevice,
        config: &MetalConfig,
        external: Option<Library>,
        units: Vec<source::Unit>,
    ) -> Self {
        let mut index = HashMap::new();
        for (i, unit) in units.iter().enumerate() {
            for kernel in &unit.kernels {
                index.entry(kernel.clone()).or_insert(i);
            }
        }
        Self {
            device: device.clone(),
            fast_math: config.fast_math,
            labels: config.debug_labels,
            language: device.info().msl_version,
            units: units
                .into_iter()
                .map(|unit| UnitSlot {
                    unit,
                    library: OnceLock::new(),
                })
                .collect(),
            kernel_index: RwLock::new(index),
            external,
            pipelines: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            libraries_built: AtomicUsize::new(0),
            pipelines_built: AtomicUsize::new(0),
            compile_nanos: AtomicU64::new(0),
        }
    }
    /// Kernel names found in the sources (macro-generated kernels may be missing from the list but still resolve).
    pub fn kernel_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .kernel_index
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            units: self.units.len(),
            libraries_built: self.libraries_built.load(Ordering::Relaxed),
            pipelines_built: self.pipelines_built.load(Ordering::Relaxed),
            compile_seconds: self.compile_nanos.load(Ordering::Relaxed) as f64 * 1e-9,
        }
    }

    fn compile_unit(&self, unit: &source::Unit) -> Result<Library, MetalError> {
        let started = Instant::now();
        let result = autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            #[allow(deprecated)]
            // `mathMode` exists only since macOS 15; this setter works everywhere
            options.setFastMathEnabled(self.fast_math);
            options.setLanguageVersion(MTLLanguageVersion(msl_version_code(self.language)));
            let text = NSString::from_str(&unit.text);
            self.device
                .raw()
                .newLibraryWithSource_options_error(&text, Some(&options))
                .map(|raw| Library { raw })
                .map_err(|e| MetalError::Compile {
                    unit: unit.name.clone(),
                    message: e.localizedDescription().to_string(),
                })
        });
        self.compile_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if result.is_ok() {
            self.libraries_built.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    fn library_for_unit(&self, index: usize) -> Result<&Library, MetalError> {
        let slot = &self.units[index];
        slot.library
            .get_or_init(|| self.compile_unit(&slot.unit))
            .as_ref()
            .map_err(Clone::clone)
    }

    /// Compile all source units now, on several threads. Without this, each unit compiles when its first kernel is
    /// requested (serially). The OS caches compiled shaders, so only the first start of a given source pays in full.
    pub fn prewarm_libraries(&self) -> Result<(), MetalError> {
        parallel_for(self.units.len(), |i| self.library_for_unit(i).map(|_| ()))
    }

    /// Create these pipelines now, on several threads (libraries are compiled as needed).
    pub fn prewarm(&self, requests: &[(&str, &[FunctionConstant])]) -> Result<(), MetalError> {
        parallel_for(requests.len(), |i| {
            self.get(requests[i].0, requests[i].1).map(|_| ())
        })
    }

    /// The function `key` names, from `library`; `Ok(None)` if the library has no function of that name.
    fn function_in(
        &self,
        library: &Library,
        key: &PipelineKey,
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLFunction>>>, MetalError> {
        let name = NSString::from_str(key.function());
        let Some(plain) = library.raw.newFunctionWithName(&name) else {
            return Ok(None);
        };
        let function = match constant_values(key.constants()) {
            None => plain,
            Some(values) => library
                .raw
                .newFunctionWithName_constantValues_error(&name, &values)
                .map_err(|e| MetalError::Pipeline {
                    function: key.to_string(),
                    message: e.localizedDescription().to_string(),
                })?,
        };
        if self.labels {
            function.setLabel(Some(&NSString::from_str(&key.to_string())));
        }
        Ok(Some(function))
    }

    fn find_function(
        &self,
        key: &PipelineKey,
    ) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, MetalError> {
        if let Some(library) = &self.external {
            return self
                .function_in(library, key)?
                .ok_or_else(|| MetalError::Pipeline {
                    function: key.to_string(),
                    message: "no function of this name in the precompiled library".to_owned(),
                });
        }
        let hinted = self
            .kernel_index
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(key.function())
            .copied();
        if let Some(index) = hinted {
            if let Some(function) = self.function_in(self.library_for_unit(index)?, key)? {
                return Ok(function);
            }
        }
        // not where the scan expected it (macro-generated or templated names): ask every unit
        for index in (0..self.units.len()).filter(|i| Some(*i) != hinted) {
            if let Some(function) = self.function_in(self.library_for_unit(index)?, key)? {
                self.kernel_index
                    .write()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .insert(key.function().to_owned(), index);
                return Ok(function);
            }
        }
        Err(MetalError::Pipeline {
            function: key.to_string(),
            message: format!(
                "no kernel of this name in the {} shader units",
                self.units.len()
            ),
        })
    }

    fn build(&self, key: PipelineKey) -> Result<Pipeline, MetalError> {
        let started = Instant::now();
        let result = autoreleasepool(|_| {
            let function = self.find_function(&key)?;
            let state = self
                .device
                .raw()
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| MetalError::Pipeline {
                    function: key.to_string(),
                    message: e.localizedDescription().to_string(),
                })?;
            Ok::<_, MetalError>(Pipeline {
                id: self.next_id.fetch_add(1, Ordering::Relaxed),
                max_threads: state.maxTotalThreadsPerThreadgroup(),
                simd_width: state.threadExecutionWidth(),
                static_threadgroup_memory: state.staticThreadgroupMemoryLength(),
                state,
                key,
            })
        });
        self.compile_nanos
            .fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if result.is_ok() {
            self.pipelines_built.fetch_add(1, Ordering::Relaxed);
        }
        result
    }

    /// The pipeline of kernel `function` specialised with `constants`; built on the first request. Resolve pipelines
    /// once at start-up and keep the `Arc` instead of calling this per dispatch (it hashes the key every time).
    pub fn get(
        &self,
        function: &str,
        constants: &[FunctionConstant],
    ) -> Result<Arc<Pipeline>, MetalError> {
        let key =
            PipelineKey::new(function, constants).map_err(|message| MetalError::Pipeline {
                function: function.to_owned(),
                message,
            })?;
        if let Some(found) = self
            .pipelines
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&key)
        {
            return Ok(Arc::clone(found));
        }
        let pipeline = Arc::new(self.build(key.clone())?);
        let mut pipelines = self
            .pipelines
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Ok(Arc::clone(pipelines.entry(key).or_insert(pipeline)))
    }

    /// Shorthand for a kernel without function constants.
    pub fn get_plain(&self, function: &str) -> Result<Arc<Pipeline>, MetalError> {
        self.get(function, &[])
    }
}
