//! Device memory of the TPU backend: cells (the arrays behind `DeviceBuffer`), views onto them, and the shared runtime state.
//!
//! A PJRT buffer is immutable: a program reads buffers and produces new ones. The backend's `DeviceBuffer` is mutable and has byte
//! windows, so it is modelled as a [`Cell`]: a lock around the *current* PJRT buffer of an array of bytes. An operation replaces the
//! cell's buffer by a program output (`Replace`), or updates it in place with `dynamic_update_slice` while the program donates the old
//! buffer to the output (`tf.aliasing_output`), so a cache update does not copy the cache. A `DeviceBuffer` is a window
//! (`offset`, `size` in bytes) onto a cell; slices of a buffer share the cell.
//!
//! The element type of a cell is that of the first thing written to it (f32 for activations, i32 for index arrays, f16 for the KV
//! pools); a cell that was never written reads as zeros.
//!
//! Level of verification: L0.

use super::plan::Plan;
use crate::backend::tpu::pjrt::{
    self, proto::CompileOptions, proto::MatrixPrecision, Buffer, ElementType, Executable,
};
use crate::backend::tpu::stablehlo::ops::Numerics;
use crate::backend::tpu::stablehlo::{self, DType};
use crate::backend::{BackendError, DeviceBuffer};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

/// The PJRT element type of a StableHLO type.
pub fn element_type(dtype: DType) -> ElementType {
    match dtype {
        DType::Pred => ElementType::Pred,
        DType::I8 => ElementType::S8,
        DType::U8 => ElementType::U8,
        DType::I16 => ElementType::S16,
        DType::U16 => ElementType::U16,
        DType::I32 => ElementType::S32,
        DType::U32 => ElementType::U32,
        DType::I64 => ElementType::S64,
        DType::F16 => ElementType::F16,
        DType::Bf16 => ElementType::Bf16,
        DType::F32 => ElementType::F32,
    }
}

/// The StableHLO type of a PJRT element type.
pub fn dtype_of(element: ElementType) -> DType {
    match element {
        ElementType::Pred => DType::Pred,
        ElementType::S8 => DType::I8,
        ElementType::U8 => DType::U8,
        ElementType::S16 => DType::I16,
        ElementType::U16 => DType::U16,
        ElementType::S32 => DType::I32,
        ElementType::U32 => DType::U32,
        ElementType::S64 => DType::I64,
        ElementType::F16 => DType::F16,
        ElementType::Bf16 => DType::Bf16,
        ElementType::F32 => DType::F32,
    }
}

pub fn driver(error: impl std::fmt::Display) -> BackendError {
    BackendError::DriverError(error.to_string())
}

pub fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidShape(message.into())
}

/// What a cell holds right now.
pub struct CellState {
    /// The current buffer; `None` until the first write (reads as zeros).
    pub buf: Option<Buffer>,
    pub dtype: Option<DType>,
}

/// The array behind a `DeviceBuffer`.
pub struct Cell {
    /// Bytes of the array (the allocation size rounded up to whole 32-bit words).
    pub bytes: usize,
    /// For arrays that are not flat (the KV pools): the fixed shape and element type.
    pub shaped: Option<(Vec<i64>, DType)>,
    pub state: Mutex<CellState>,
}

impl Cell {
    /// A flat byte array of at least `size` bytes.
    pub fn flat(size: usize) -> Arc<Self> {
        Arc::new(Self {
            bytes: size.max(1).next_multiple_of(4),
            shaped: None,
            state: Mutex::new(CellState {
                buf: None,
                dtype: None,
            }),
        })
    }

    /// An array of fixed shape (the dtype is fixed too).
    pub fn shaped(dims: &[i64], dtype: DType) -> Arc<Self> {
        let bytes = dims.iter().map(|&d| d as usize).product::<usize>() * dtype.bytes();
        Arc::new(Self {
            bytes,
            shaped: Some((dims.to_vec(), dtype)),
            state: Mutex::new(CellState {
                buf: None,
                dtype: Some(dtype),
            }),
        })
    }

    pub fn lock(&self) -> Result<MutexGuard<'_, CellState>, BackendError> {
        self.state
            .lock()
            .map_err(|_| BackendError::BufferError("TPU buffer lock poisoned".into()))
    }

    /// Dimensions and element type of the array for element type `dtype` when it is flat.
    pub fn signature(&self, dtype: DType) -> (Vec<i64>, DType) {
        match &self.shaped {
            Some((dims, dtype)) => (dims.clone(), *dtype),
            None => (vec![(self.bytes / dtype.bytes()) as i64], dtype),
        }
    }

    pub fn is_flat(&self) -> bool {
        self.shaped.is_none()
    }
}

/// A byte window onto a cell.
#[derive(Clone)]
pub struct View {
    pub cell: Arc<Cell>,
    pub offset: usize,
    pub size: usize,
}

impl View {
    /// The window a `DeviceBuffer` describes.
    pub fn of(buffer: &DeviceBuffer) -> Result<Self, BackendError> {
        let cell = Arc::clone(&buffer.handle).downcast::<Cell>().map_err(|_| {
            BackendError::BufferError("the buffer does not belong to the TPU backend".into())
        })?;
        if buffer.offset + buffer.size > cell.bytes.max(buffer.capacity) {
            return Err(BackendError::BufferError(
                "buffer window outside its allocation".into(),
            ));
        }
        Ok(Self {
            cell,
            offset: buffer.offset,
            size: buffer.size,
        })
    }

    /// The window covers the whole array.
    pub fn is_whole(&self) -> bool {
        self.offset == 0 && self.size == self.cell.bytes
    }

    /// A window of the same cell starting `delta` bytes further, `size` bytes long.
    pub fn sub(&self, delta: usize, size: usize) -> Self {
        Self {
            cell: Arc::clone(&self.cell),
            offset: self.offset + delta,
            size,
        }
    }
}

/// How the backend treats arithmetic and weights, from the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settings {
    pub numerics: Numerics,
    pub weights: super::weights::WeightFormat,
}

impl Settings {
    /// `FLLM_TPU_EXACT=1`: f32 weights and full-precision products (what `selftest-gpu` needs). Otherwise bf16 weights and one-pass
    /// bf16 products; `FLLM_TPU_WEIGHTS=bf16|f32|int8` and `FLLM_TPU_NUMERICS=exact|fast` override the parts.
    pub fn from_env() -> Self {
        use super::weights::WeightFormat;
        let exact = std::env::var_os("FLLM_TPU_EXACT").is_some_and(|v| v != "0");
        let numerics = match std::env::var("FLLM_TPU_NUMERICS").ok().as_deref() {
            Some("exact") => Numerics::Exact,
            Some("fast") => Numerics::Fast,
            _ if exact => Numerics::Exact,
            _ => Numerics::Fast,
        };
        let weights = match std::env::var("FLLM_TPU_WEIGHTS").ok().as_deref() {
            Some("f32") => WeightFormat::F32,
            Some("int8") => WeightFormat::Int8,
            Some("bf16") => WeightFormat::Bf16,
            _ if exact => WeightFormat::F32,
            _ => WeightFormat::Bf16,
        };
        Self { numerics, weights }
    }
}

/// A host array that was uploaded once and is looked up by the identity of its `Arc` (RoPE tables, sinks, ALiBi slopes).
struct TableEntry {
    source: Arc<Vec<f32>>,
    columns: usize,
    buffer: Arc<Buffer>,
}

/// State shared by the backend and all its streams.
pub struct Shared {
    pub runtime: Arc<pjrt::Runtime>,
    pub device: pjrt::Device,
    pub index: usize,
    pub settings: Settings,
    programs: Mutex<HashMap<String, Arc<Executable>>>,
    tables: Mutex<Vec<TableEntry>>,
    next_id: AtomicU64,
    /// The cell written last, for `synchronize`.
    last: Mutex<Option<Arc<Cell>>>,
    /// Dry-run mode: programs are built and recorded here instead of being compiled and run.
    dry: Option<Mutex<Vec<String>>>,
    /// `FLLM_TPU_DUMP_DIR`: every compiled program is written there.
    dump_dir: Option<std::path::PathBuf>,
}

impl Shared {
    pub fn new(
        runtime: Arc<pjrt::Runtime>,
        device: pjrt::Device,
        index: usize,
        settings: Settings,
    ) -> Self {
        Self {
            runtime,
            device,
            index,
            settings,
            programs: Mutex::new(HashMap::new()),
            tables: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(1),
            last: Mutex::new(None),
            dry: None,
            dump_dir: std::env::var_os("FLLM_TPU_DUMP_DIR")
                .filter(|v| !v.is_empty())
                .map(std::path::PathBuf::from),
        }
    }

    /// A backend state without a plugin: operations build their programs and record them (see [Shared::recorded]); buffers exist only
    /// on paper and read as zeros. Used by the unit tests to check every program the operations emit.
    pub fn dry(settings: Settings) -> Self {
        let client = pjrt::Client::dry();
        let device = client.devices()[0].clone();
        let runtime = Arc::new(pjrt::Runtime {
            api: Arc::clone(client.api()),
            client,
        });
        let mut shared = Self::new(runtime, device, 0, settings);
        shared.dry = Some(Mutex::new(Vec::new()));
        shared.dump_dir = None;
        shared
    }

    /// The programs a dry run has recorded so far, in order (empty outside dry runs).
    pub fn recorded(&self) -> Vec<String> {
        self.dry
            .as_ref()
            .and_then(|d| d.lock().ok().map(|v| v.clone()))
            .unwrap_or_default()
    }

    fn is_dry(&self) -> bool {
        self.dry.is_some()
    }

    fn record(&self, text: &str) {
        if let Some(dry) = &self.dry {
            if let Ok(mut programs) = dry.lock() {
                programs.push(text.to_owned());
            }
        }
    }

    pub fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Compile (or fetch) a program given as StableHLO text.
    pub fn executable(&self, text: &str) -> Result<Arc<Executable>, BackendError> {
        let mut programs = self
            .programs
            .lock()
            .map_err(|_| driver("TPU program cache lock poisoned"))?;
        if let Some(found) = programs.get(text) {
            return Ok(Arc::clone(found));
        }
        let options = CompileOptions {
            device_id: Some(i64::from(self.device.id)),
            matrix_precision: (self.settings.numerics == Numerics::Exact)
                .then_some(MatrixPrecision::Highest),
            ..CompileOptions::default()
        };
        let start = Instant::now();
        let executable = self.runtime.client.compile(text, &options).map_err(|e| {
            let head: String = text.lines().take(60).collect::<Vec<_>>().join("\n");
            driver(format!(
                "compiling a StableHLO program failed: {e}\n--- program (first lines) ---\n{head}"
            ))
        })?;
        if pjrt::debug_enabled() {
            eprintln!(
                "[fllm-tpu] compiled a program of {} bytes in {:.0} ms ({} cached)",
                text.len(),
                start.elapsed().as_secs_f64() * 1e3,
                programs.len() + 1
            );
        }
        if let Some(dir) = &self.dump_dir {
            let name = text
                .lines()
                .next()
                .unwrap_or("module")
                .replace(['@', ' ', '{'], "");
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(format!("{:04}_{name}.mlir", programs.len())), text);
        }
        let executable = Arc::new(executable);
        programs.insert(text.to_owned(), Arc::clone(&executable));
        Ok(executable)
    }

    /// Upload a dense host array to the device.
    pub fn host_buffer(
        &self,
        bytes: &[u8],
        dtype: DType,
        dims: &[i64],
    ) -> Result<Buffer, BackendError> {
        if self.is_dry() {
            return Ok(Buffer::dry(
                self.runtime.client.api(),
                element_type(dtype),
                dims,
            ));
        }
        self.runtime
            .client
            .buffer_from_host(bytes, element_type(dtype), dims, &self.device)
            .map_err(driver)
    }

    /// Give a never-written cell its zero buffer of element type `dtype`.
    pub fn materialize(&self, cell: &Arc<Cell>, dtype: DType) -> Result<(), BackendError> {
        let mut state = cell.lock()?;
        if state.buf.is_some() {
            return Ok(());
        }
        let (dims, dtype) = cell.signature(dtype);
        if self.is_dry() {
            state.buf = Some(Buffer::dry(
                self.runtime.client.api(),
                element_type(dtype),
                &dims,
            ));
            state.dtype = Some(dtype);
            return Ok(());
        }
        let ty = stablehlo::TensorType::new(&dims, dtype);
        let mut f = stablehlo::Func::new("main");
        let zeros = f.constant(0.0, &dims, dtype);
        let text = stablehlo::module("zeros", &f, &[&zeros]);
        let executable = self.executable(&text)?;
        let mut outputs = executable.execute(&[]).map_err(driver)?;
        let buffer = outputs
            .pop()
            .ok_or_else(|| driver(format!("the zero program for {ty} returned nothing")))?;
        state.buf = Some(buffer);
        state.dtype = Some(dtype);
        Ok(())
    }

    /// A device copy of a host table (`columns` per row; 0 for a plain vector), uploaded once per `Arc`.
    pub fn table(
        &self,
        source: &Arc<Vec<f32>>,
        columns: usize,
    ) -> Result<Arc<Buffer>, BackendError> {
        let mut tables = self
            .tables
            .lock()
            .map_err(|_| driver("TPU table cache lock poisoned"))?;
        if let Some(entry) = tables
            .iter()
            .find(|e| Arc::ptr_eq(&e.source, source) && e.columns == columns)
        {
            return Ok(Arc::clone(&entry.buffer));
        }
        let dims: Vec<i64> = match source.len().checked_div(columns) {
            Some(rows) => vec![rows as i64, columns as i64],
            None => vec![source.len() as i64],
        };
        let bytes: Vec<u8> = source.iter().flat_map(|v| v.to_le_bytes()).collect();
        let buffer = Arc::new(self.host_buffer(&bytes, DType::F32, &dims)?);
        if tables.len() >= 64 {
            tables.remove(0);
        }
        tables.push(TableEntry {
            source: Arc::clone(source),
            columns,
            buffer: Arc::clone(&buffer),
        });
        Ok(buffer)
    }

    /// Run a finished plan: compile it if needed, lock the cells it touches, execute, and store the outputs. Returns the buffers of
    /// the plan's host results, in order.
    pub fn run(&self, plan: Plan<'_>) -> Result<Vec<Buffer>, BackendError> {
        let finished = plan.finish()?;
        if self.is_dry() {
            return self.run_dry(finished);
        }
        let executable = self.executable(&finished.text)?;
        // lock every distinct cell once, in a fixed order, and keep the guards while the program runs
        let mut cells: Vec<Arc<Cell>> = Vec::new();
        let mut note = |cell: &Arc<Cell>| {
            if !cells.iter().any(|c| Arc::ptr_eq(c, cell)) {
                cells.push(Arc::clone(cell));
            }
        };
        for arg in &finished.args {
            if let super::plan::Arg::Cell(cell) = arg {
                note(cell);
            }
        }
        for sink in &finished.sinks {
            if let super::plan::Sink::Cell(cell) = sink {
                note(cell);
            }
        }
        cells.sort_by_key(|c| Arc::as_ptr(c) as usize);
        let mut guards: Vec<MutexGuard<'_, CellState>> = Vec::with_capacity(cells.len());
        for cell in &cells {
            guards.push(cell.lock()?);
        }
        let position =
            |cell: &Arc<Cell>| cells.iter().position(|c| Arc::ptr_eq(c, cell)).unwrap_or(0);
        let meta_buffer = if finished.meta.is_empty() {
            None
        } else {
            let bytes: Vec<u8> = finished.meta.iter().flat_map(|v| v.to_le_bytes()).collect();
            Some(self.host_buffer(&bytes, DType::I32, &[finished.meta.len() as i64])?)
        };
        let outputs = {
            let mut arguments: Vec<&Buffer> = Vec::with_capacity(finished.args.len());
            for arg in &finished.args {
                match arg {
                    super::plan::Arg::Meta => arguments.push(
                        meta_buffer
                            .as_ref()
                            .ok_or_else(|| driver("plan without meta data"))?,
                    ),
                    super::plan::Arg::Buffer(buffer) => arguments.push(buffer),
                    super::plan::Arg::Cell(cell) => {
                        let state = &guards[position(cell)];
                        arguments.push(state.buf.as_ref().ok_or_else(|| {
                            BackendError::BufferError(
                                "a program input was never materialised".into(),
                            )
                        })?);
                    }
                }
            }
            executable.execute(&arguments).map_err(driver)?
        };
        if outputs.len() != finished.sinks.len() {
            return Err(driver(format!(
                "a program returned {} outputs, {} were planned",
                outputs.len(),
                finished.sinks.len()
            )));
        }
        let mut host = Vec::new();
        let mut last = None;
        for (output, sink) in outputs.into_iter().zip(&finished.sinks) {
            match sink {
                super::plan::Sink::Cell(cell) => {
                    let state = &mut guards[position(cell)];
                    state.dtype = Some(dtype_of(output.element));
                    state.buf = Some(output);
                    last = Some(Arc::clone(cell));
                }
                super::plan::Sink::Host => host.push(output),
            }
        }
        drop(guards);
        if let Some(cell) = last {
            *self
                .last
                .lock()
                .map_err(|_| driver("TPU stream lock poisoned"))? = Some(cell);
        }
        Ok(host)
    }

    /// `run` for dry runs: record the program and give every result an array that exists on paper.
    fn run_dry(&self, finished: super::plan::Finished) -> Result<Vec<Buffer>, BackendError> {
        self.record(&finished.text);
        let api = self.runtime.client.api();
        let mut host = Vec::new();
        for (sink, ty) in finished.sinks.iter().zip(&finished.result_types) {
            let buffer = Buffer::dry(api, element_type(ty.dtype), &ty.dims);
            match sink {
                super::plan::Sink::Cell(cell) => {
                    let mut state = cell.lock()?;
                    state.dtype = Some(ty.dtype);
                    state.buf = Some(buffer);
                }
                super::plan::Sink::Host => host.push(buffer),
            }
        }
        Ok(host)
    }

    /// Wait until the work submitted so far has finished.
    pub fn synchronize(&self) -> Result<(), BackendError> {
        let cell = self
            .last
            .lock()
            .map_err(|_| driver("TPU stream lock poisoned"))?
            .clone();
        if let Some(cell) = cell {
            let state = cell.lock()?;
            if let Some(buffer) = &state.buf {
                buffer.wait_ready().map_err(driver)?;
            }
        }
        Ok(())
    }
}
