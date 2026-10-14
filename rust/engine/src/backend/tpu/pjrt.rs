//! Google TPU through PJRT / StableHLO: safe wrappers of the PJRT C API.
//!
//! A PJRT plugin (`libtpu.so` of the `libtpu` pip package for TPUs) is a shared library that exports `GetPjrtApi`, which returns a
//! table of C functions (see [`sys`]). This file loads the library with `libloading`, copies the table, and wraps the handles in
//! types that release themselves:
//!
//! - [`Api`]: the loaded plugin (the table, the library handle). Every call goes through it; errors are turned into [`Error`].
//! - [`Client`]: `PJRT_Client`, the devices it exposes, program compilation and host-to-device transfers.
//! - [`Buffer`]: `PJRT_Buffer`, one array in device memory (destroyed on drop).
//! - [`Executable`]: a compiled program (`PJRT_LoadedExecutable`, destroyed on drop).
//! - [`runtime`]: the process-wide plugin and client (a TPU can only be opened by one client per process).
//!
//! Nothing here knows about LLMs. Level of verification: L0 (compiles; never run against a plugin).

pub mod proto;
pub mod sys;

use libloading::Library;
use std::ffi::{c_char, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

/// An error reported by the plugin or by this wrapper.
#[derive(Debug, Clone)]
pub struct Error {
    pub message: String,
    /// `PJRT_Error_Code` when the plugin reported one.
    pub code: Option<i32>,
}

impl Error {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            code: None,
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.code {
            Some(code) => write!(f, "{} (PJRT error code {code})", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Element types of buffers (`PJRT_Buffer_Type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    Pred,
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    S64,
    F16,
    Bf16,
    F32,
}

impl ElementType {
    /// The `PJRT_Buffer_Type` value.
    pub fn code(self) -> i32 {
        use sys::buffer_type as t;
        match self {
            Self::Pred => t::PRED,
            Self::S8 => t::S8,
            Self::U8 => t::U8,
            Self::S16 => t::S16,
            Self::U16 => t::U16,
            Self::S32 => t::S32,
            Self::U32 => t::U32,
            Self::S64 => t::S64,
            Self::F16 => t::F16,
            Self::Bf16 => t::BF16,
            Self::F32 => t::F32,
        }
    }

    pub fn from_code(code: i32) -> Option<Self> {
        use sys::buffer_type as t;
        Some(match code {
            t::PRED => Self::Pred,
            t::S8 => Self::S8,
            t::U8 => Self::U8,
            t::S16 => Self::S16,
            t::U16 => Self::U16,
            t::S32 => Self::S32,
            t::U32 => Self::U32,
            t::S64 => Self::S64,
            t::F16 => Self::F16,
            t::BF16 => Self::Bf16,
            t::F32 => Self::F32,
            _ => return None,
        })
    }

    /// Bytes of one element.
    pub fn bytes(self) -> usize {
        match self {
            Self::Pred | Self::S8 | Self::U8 => 1,
            Self::S16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::S32 | Self::U32 | Self::F32 => 4,
            Self::S64 => 8,
        }
    }
}

/// Number of elements of an array of shape `dims` (1 for a scalar).
pub fn element_count(dims: &[i64]) -> usize {
    dims.iter().map(|&d| d.max(0) as usize).product()
}

/// A loaded PJRT plugin.
pub struct Api {
    table: sys::PjrtApi,
    version: (i32, i32),
    path: PathBuf,
    // keeps the shared library mapped for as long as any handle of the plugin exists
    _library: Library,
}

// SAFETY: the table is plain data (function pointers) copied out of the plugin; the PJRT C API is thread-safe by contract and the
// library handle is only used to keep the mapping alive.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// Calls table entry `$field` with the args struct `$args`; a plugin without the entry reports an error.
macro_rules! call {
    ($api:expr, $field:ident, $args:expr) => {{
        match $api.table.$field {
            Some(function) => {
                // SAFETY: `function` comes from the plugin's table and `$args` is the args struct that entry takes, with
                // `struct_size` set; the pointers inside are valid for the duration of the call.
                let error = unsafe { function(&mut $args) };
                $api.check(error)
            }
            None => Err(Error::new(format!("the PJRT plugin does not implement {}", stringify!($field)))),
        }
    }};
}

/// A handle on the running program itself (a library that is always there), for the dry-run API.
fn this_library() -> Library {
    #[cfg(unix)]
    {
        libloading::os::unix::Library::this().into()
    }
    #[cfg(windows)]
    {
        libloading::os::windows::Library::this()
            .expect("a handle on the current process")
            .into()
    }
}

impl Api {
    /// A plugin-less API for dry runs (programs are built and recorded, nothing is sent anywhere): the table is empty, so every call
    /// reports that the plugin does not implement it.
    pub fn dry() -> Arc<Self> {
        // SAFETY: the table consists of an integer, pointers, `Option`s of function pointers and a version: all-zero is valid.
        let table: sys::PjrtApi = unsafe { std::mem::zeroed() };
        Arc::new(Self {
            table,
            version: (sys::API_MAJOR, sys::API_MINOR),
            path: PathBuf::from("<dry run>"),
            _library: this_library(),
        })
    }

    /// Load the plugin at `path` and run `PJRT_Plugin_Initialize`.
    pub fn load(path: &Path) -> Result<Arc<Self>> {
        // SAFETY: loading a plugin runs its initialisers; that is what the user asked for by naming it.
        let library = unsafe { Library::new(path) }
            .map_err(|e| Error::new(format!("cannot load {}: {e}", path.display())))?;
        // SAFETY: `GetPjrtApi` has this signature in every PJRT plugin.
        let get: libloading::Symbol<'_, sys::GetPjrtApiFn> =
            unsafe { library.get(b"GetPjrtApi\0") }.map_err(|e| {
                Error::new(format!(
                    "{} is not a PJRT plugin (no GetPjrtApi): {e}",
                    path.display()
                ))
            })?;
        // SAFETY: see above; the returned table lives as long as the library.
        let raw = unsafe { get() };
        if raw.is_null() {
            return Err(Error::new(format!(
                "GetPjrtApi of {} returned null",
                path.display()
            )));
        }
        // SAFETY: the table starts with its own `struct_size`; only that many bytes are copied into a zeroed table of this
        // transcription's size, so entries the plugin does not have stay `None` and are never read out of bounds.
        let (table, version) = unsafe {
            let plugin_size = (*raw).struct_size;
            let mut table: sys::PjrtApi = std::mem::zeroed();
            let n = plugin_size.min(sys::PjrtApi::KNOWN_SIZE);
            std::ptr::copy_nonoverlapping(
                raw.cast::<u8>(),
                (&mut table as *mut sys::PjrtApi).cast::<u8>(),
                n,
            );
            let version = (
                (*raw).pjrt_api_version.major_version,
                (*raw).pjrt_api_version.minor_version,
            );
            (table, version)
        };
        let api = Arc::new(Self {
            table,
            version,
            path: path.to_owned(),
            _library: library,
        });
        if api.table.plugin_initialize.is_some() {
            let mut args = sys::PluginInitializeArgs::init();
            call!(api, plugin_initialize, args)?;
        }
        Ok(api)
    }

    /// `(major, minor)` of the API the plugin implements.
    pub fn version(&self) -> (i32, i32) {
        self.version
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Turn a returned `PJRT_Error*` into a `Result`, releasing the error.
    fn check(&self, error: *mut sys::PjrtError) -> Result<()> {
        if error.is_null() {
            return Ok(());
        }
        let mut message = String::from("unknown PJRT error");
        if let Some(function) = self.table.error_message {
            let mut args = sys::ErrorMessageArgs::init();
            args.error = error;
            // SAFETY: `error` is a live error returned by the plugin; the message stays valid until the error is destroyed.
            unsafe { function(&mut args) };
            if !args.message.is_null() {
                // SAFETY: the plugin returns `message_size` bytes at `message`.
                let bytes = unsafe {
                    std::slice::from_raw_parts(args.message.cast::<u8>(), args.message_size)
                };
                message = String::from_utf8_lossy(bytes).into_owned();
            }
        }
        let mut code = None;
        if let Some(function) = self.table.error_get_code {
            let mut args = sys::ErrorGetCodeArgs::init();
            args.error = error;
            // SAFETY: as above. The call itself may return an error, which is ignored (and released).
            let inner = unsafe { function(&mut args) };
            if inner.is_null() {
                code = Some(args.code);
            } else if let Some(destroy) = self.table.error_destroy {
                let mut d = sys::ErrorDestroyArgs::init();
                d.error = inner;
                // SAFETY: `inner` is a live error.
                unsafe { destroy(&mut d) };
            }
        }
        if let Some(destroy) = self.table.error_destroy {
            let mut args = sys::ErrorDestroyArgs::init();
            args.error = error;
            // SAFETY: `error` is destroyed exactly once.
            unsafe { destroy(&mut args) };
        }
        Err(Error { message, code })
    }

    /// Release `event` (which may be null) without waiting for it.
    fn release_event(&self, event: *mut sys::PjrtEvent) {
        if event.is_null() {
            return;
        }
        let mut destroy_args = sys::EventDestroyArgs::init();
        destroy_args.event = event;
        let _ = call!(self, event_destroy, destroy_args);
    }

    /// Wait for `event` (which may be null) and release it.
    fn await_event(&self, event: *mut sys::PjrtEvent) -> Result<()> {
        if event.is_null() {
            return Ok(());
        }
        let mut await_args = sys::EventAwaitArgs::init();
        await_args.event = event;
        let waited = call!(self, event_await, await_args);
        let mut destroy_args = sys::EventDestroyArgs::init();
        destroy_args.event = event;
        let destroyed = call!(self, event_destroy, destroy_args);
        waited.and(destroyed)
    }
}

/// Text at `ptr` / `len` as a `String` (lossy), empty for a null pointer.
fn text(ptr: *const c_char, len: usize) -> String {
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: the plugin returns `len` valid bytes at `ptr`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), len) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// One addressable device of a client.
#[derive(Debug, Clone)]
pub struct Device {
    raw: *mut sys::PjrtDevice,
    /// `PJRT_DeviceDescription_Id`: the id programs name in their device assignment.
    pub id: i32,
    /// `PJRT_DeviceDescription_Kind`, for example `TPU v5 lite`.
    pub kind: String,
    pub debug_string: String,
    pub local_hardware_id: i32,
}

// SAFETY: a `PJRT_Device*` is an immutable handle owned by the client and usable from any thread.
unsafe impl Send for Device {}
unsafe impl Sync for Device {}

/// Memory the device reports (`PJRT_Device_MemoryStats`); `None` when the plugin does not.
#[derive(Debug, Clone, Copy)]
pub struct MemoryStats {
    pub bytes_in_use: i64,
    pub bytes_limit: Option<i64>,
}

/// A PJRT client: the plugin's view of the machine's devices.
pub struct Client {
    api: Arc<Api>,
    raw: *mut sys::PjrtClient,
    devices: Vec<Device>,
    platform_name: String,
    platform_version: String,
}

// SAFETY: the PJRT client is thread-safe by contract.
unsafe impl Send for Client {}
unsafe impl Sync for Client {}

impl Client {
    /// Create a client with the plugin's default options.
    pub fn create(api: &Arc<Api>) -> Result<Arc<Self>> {
        let mut args = sys::ClientCreateArgs::init();
        call!(api, client_create, args)?;
        let raw = args.client;
        if raw.is_null() {
            return Err(Error::new("PJRT_Client_Create returned no client"));
        }
        let mut client = Self {
            api: Arc::clone(api),
            raw,
            devices: Vec::new(),
            platform_name: String::new(),
            platform_version: String::new(),
        };
        client.platform_name = client.query_text(true).unwrap_or_default();
        client.platform_version = client.query_text(false).unwrap_or_default();
        client.devices = client.query_devices()?;
        Ok(Arc::new(client))
    }

    fn query_text(&self, name: bool) -> Result<String> {
        if name {
            let mut args = sys::ClientPlatformNameArgs::init();
            args.client = self.raw;
            call!(self.api, client_platform_name, args)?;
            Ok(text(args.platform_name, args.platform_name_size))
        } else {
            let mut args = sys::ClientPlatformVersionArgs::init();
            args.client = self.raw;
            call!(self.api, client_platform_version, args)?;
            Ok(text(args.platform_version, args.platform_version_size))
        }
    }

    fn query_devices(&self) -> Result<Vec<Device>> {
        let mut args = sys::ClientAddressableDevicesArgs::init();
        args.client = self.raw;
        call!(self.api, client_addressable_devices, args)?;
        let mut devices = Vec::with_capacity(args.num_addressable_devices);
        for i in 0..args.num_addressable_devices {
            // SAFETY: the plugin returned `num_addressable_devices` device pointers.
            let raw = unsafe { *args.addressable_devices.add(i) };
            let mut description = sys::DeviceGetDescriptionArgs::init();
            description.device = raw;
            call!(self.api, device_get_description, description)?;
            let desc = description.device_description;
            let mut id = sys::DeviceDescriptionIdArgs::init();
            id.device_description = desc;
            call!(self.api, device_description_id, id)?;
            let mut kind = sys::DeviceDescriptionKindArgs::init();
            kind.device_description = desc;
            call!(self.api, device_description_kind, kind)?;
            let mut debug = sys::DeviceDescriptionDebugStringArgs::init();
            debug.device_description = desc;
            let debug_string = match call!(self.api, device_description_debug_string, debug) {
                Ok(()) => text(debug.debug_string, debug.debug_string_size),
                Err(_) => String::new(),
            };
            let mut local = sys::DeviceLocalHardwareIdArgs::init();
            local.device = raw;
            let local_hardware_id = match call!(self.api, device_local_hardware_id, local) {
                Ok(()) => local.local_hardware_id,
                Err(_) => -1,
            };
            devices.push(Device {
                raw,
                id: id.id,
                kind: text(kind.device_kind, kind.device_kind_size),
                debug_string,
                local_hardware_id,
            });
        }
        Ok(devices)
    }

    /// A client without a plugin for dry runs: one device that exists only on paper.
    pub fn dry() -> Arc<Self> {
        let device = Device {
            raw: std::ptr::null_mut(),
            id: 0,
            kind: "TPU (dry run)".into(),
            debug_string: String::new(),
            local_hardware_id: 0,
        };
        Arc::new(Self {
            api: Api::dry(),
            raw: std::ptr::null_mut(),
            devices: vec![device],
            platform_name: "dry".into(),
            platform_version: "dry run".into(),
        })
    }

    pub fn api(&self) -> &Arc<Api> {
        &self.api
    }

    /// The devices this process can use (`PJRT_Client_AddressableDevices`).
    pub fn devices(&self) -> &[Device] {
        &self.devices
    }

    pub fn platform_name(&self) -> &str {
        &self.platform_name
    }

    pub fn platform_version(&self) -> &str {
        &self.platform_version
    }

    /// Memory use and limit of `device`, when the plugin reports them.
    pub fn memory_stats(&self, device: &Device) -> Option<MemoryStats> {
        let mut args = sys::DeviceMemoryStatsArgs::init();
        args.device = device.raw;
        call!(self.api, device_memory_stats, args).ok()?;
        Some(MemoryStats {
            bytes_in_use: args.bytes_in_use,
            bytes_limit: args.bytes_limit_is_set.then_some(args.bytes_limit),
        })
    }

    /// Compile a StableHLO module given as MLIR text (`format = "mlir"`).
    pub fn compile(&self, mlir: &str, options: &proto::CompileOptions) -> Result<Executable> {
        let format = b"mlir";
        let mut program = sys::Program::init();
        program.code = mlir.as_ptr().cast_mut().cast::<c_char>();
        program.code_size = mlir.len();
        program.format = format.as_ptr().cast::<c_char>();
        program.format_size = format.len();
        let encoded = options.encode();
        let mut args = sys::ClientCompileArgs::init();
        args.client = self.raw;
        args.program = &program;
        args.compile_options = encoded.as_ptr().cast::<c_char>();
        args.compile_options_size = encoded.len();
        call!(self.api, client_compile, args)?;
        let raw = args.executable;
        if raw.is_null() {
            return Err(Error::new("PJRT_Client_Compile returned no executable"));
        }
        let mut executable = Executable {
            api: Arc::clone(&self.api),
            raw,
            num_outputs: 0,
        };
        executable.num_outputs = executable.query_num_outputs()?;
        Ok(executable)
    }

    /// Copy `data` (dense, row-major, `dims` of `element` values) to a new buffer on `device`.
    pub fn buffer_from_host(
        &self,
        data: &[u8],
        element: ElementType,
        dims: &[i64],
        device: &Device,
    ) -> Result<Buffer> {
        let expected = element_count(dims) * element.bytes();
        if data.len() != expected {
            return Err(Error::new(format!(
                "host buffer holds {} bytes, shape {dims:?} of {element:?} needs {expected}",
                data.len()
            )));
        }
        let mut args = sys::ClientBufferFromHostBufferArgs::init();
        args.client = self.raw;
        // a null data pointer is rejected by some plugins for empty arrays: give them a valid dangling one
        args.data = if data.is_empty() {
            std::ptr::NonNull::<u8>::dangling()
                .as_ptr()
                .cast_const()
                .cast::<c_void>()
        } else {
            data.as_ptr().cast::<c_void>()
        };
        args.type_ = element.code();
        args.dims = dims.as_ptr();
        args.num_dims = dims.len();
        args.host_buffer_semantics = sys::host_semantics::IMMUTABLE_ONLY_DURING_CALL;
        args.device = device.raw;
        call!(self.api, client_buffer_from_host_buffer, args)?;
        let buffer = Buffer {
            api: Arc::clone(&self.api),
            raw: args.buffer,
            element,
            dims: dims.to_vec(),
        };
        // IMMUTABLE_ONLY_DURING_CALL: the plugin has copied the data when the call returns, so the event is only released
        self.api.release_event(args.done_with_host_buffer);
        if buffer.raw.is_null() {
            return Err(Error::new(
                "PJRT_Client_BufferFromHostBuffer returned no buffer",
            ));
        }
        Ok(buffer)
    }
}

impl Drop for Client {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let mut args = sys::ClientDestroyArgs::init();
        args.client = self.raw;
        let _ = call!(self.api, client_destroy, args);
    }
}

/// One array in device memory.
pub struct Buffer {
    api: Arc<Api>,
    raw: *mut sys::PjrtBuffer,
    pub element: ElementType,
    pub dims: Vec<i64>,
}

// SAFETY: a PJRT buffer handle may be used from any thread; the wrapper owns it exclusively.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    /// An array that exists only on paper (dry runs): it reads as zeros.
    pub fn dry(api: &Arc<Api>, element: ElementType, dims: &[i64]) -> Self {
        Self {
            api: Arc::clone(api),
            raw: std::ptr::null_mut(),
            element,
            dims: dims.to_vec(),
        }
    }

    /// Bytes of the array (dense).
    pub fn byte_len(&self) -> usize {
        element_count(&self.dims) * self.element.bytes()
    }

    /// Copy the whole array to the host and wait for it.
    pub fn to_host(&self) -> Result<Vec<u8>> {
        if self.raw.is_null() {
            return Ok(vec![0u8; self.byte_len()]);
        }
        // dense row-major data on the host, whatever tiling the device uses (the layout JAX requests too)
        let mut out = vec![0u8; self.byte_len()];
        let minor_to_major: Vec<i64> = (0..self.dims.len() as i64).rev().collect();
        let mut layout = sys::MemoryLayout {
            struct_size: std::mem::size_of::<sys::MemoryLayout>(),
            extension_start: std::ptr::null_mut(),
            layout: sys::MemoryLayoutUnion {
                tiled: std::mem::ManuallyDrop::new(sys::MemoryLayoutTiled {
                    struct_size: std::mem::size_of::<sys::MemoryLayoutTiled>(),
                    extension_start: std::ptr::null_mut(),
                    minor_to_major: minor_to_major.as_ptr(),
                    minor_to_major_size: minor_to_major.len(),
                    tile_dims: std::ptr::null(),
                    tile_dim_sizes: std::ptr::null(),
                    num_tiles: 0,
                }),
            },
            type_: sys::layout_type::TILED,
        };
        let mut args = sys::BufferToHostBufferArgs::init();
        args.src = self.raw;
        if !self.dims.is_empty() {
            args.host_layout = (&mut layout as *mut sys::MemoryLayout).cast::<c_void>();
        }
        args.dst = out.as_mut_ptr().cast::<c_void>();
        args.dst_size = out.len();
        call!(self.api, buffer_to_host_buffer, args)?;
        self.api.await_event(args.event)?;
        Ok(out)
    }

    /// Block until the computation that produces this buffer has finished.
    pub fn wait_ready(&self) -> Result<()> {
        if self.raw.is_null() {
            return Ok(());
        }
        let mut args = sys::BufferReadyEventArgs::init();
        args.buffer = self.raw;
        call!(self.api, buffer_ready_event, args)?;
        self.api.await_event(args.event)
    }

    /// Whether the buffer was deleted (donated to an execution).
    pub fn is_deleted(&self) -> bool {
        let mut args = sys::BufferIsDeletedArgs::init();
        args.buffer = self.raw;
        call!(self.api, buffer_is_deleted, args)
            .map(|()| args.is_deleted)
            .unwrap_or(false)
    }

    /// Bytes the array occupies on the device (tiling can pad it).
    pub fn on_device_size(&self) -> Option<usize> {
        let mut args = sys::BufferOnDeviceSizeInBytesArgs::init();
        args.buffer = self.raw;
        call!(self.api, buffer_on_device_size_in_bytes, args).ok()?;
        Some(args.on_device_size_in_bytes)
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let mut args = sys::BufferDestroyArgs::init();
        args.buffer = self.raw;
        let _ = call!(self.api, buffer_destroy, args);
    }
}

/// A compiled program, loaded on the devices it was compiled for.
pub struct Executable {
    api: Arc<Api>,
    raw: *mut sys::PjrtLoadedExecutable,
    pub num_outputs: usize,
}

// SAFETY: a loaded executable may be executed from several threads.
unsafe impl Send for Executable {}
unsafe impl Sync for Executable {}

impl Executable {
    fn query_num_outputs(&self) -> Result<usize> {
        let mut get = sys::LoadedExecutableGetExecutableArgs::init();
        get.loaded_executable = self.raw;
        call!(self.api, loaded_executable_get_executable, get)?;
        let mut count = sys::ExecutableNumOutputsArgs::init();
        count.executable = get.executable;
        let counted = call!(self.api, executable_num_outputs, count);
        let mut destroy = sys::ExecutableDestroyArgs::init();
        destroy.executable = get.executable;
        let _ = call!(self.api, executable_destroy, destroy);
        counted?;
        Ok(count.num_outputs)
    }

    /// Run the program on one device with `args` and return its outputs (the call returns when the work is enqueued; reading an
    /// output waits for it). Arguments that the program donates (`tf.aliasing_output`) are consumed: their buffers are deleted.
    pub fn execute(&self, args: &[&Buffer]) -> Result<Vec<Buffer>> {
        let mut options = sys::ExecuteOptions::init();
        let argument_ptrs: Vec<*mut sys::PjrtBuffer> = args.iter().map(|b| b.raw).collect();
        let argument_list: *const *mut sys::PjrtBuffer = argument_ptrs.as_ptr();
        let argument_lists = [argument_list];
        let mut outputs: Vec<*mut sys::PjrtBuffer> = vec![std::ptr::null_mut(); self.num_outputs];
        let output_list: *mut *mut sys::PjrtBuffer = outputs.as_mut_ptr();
        let output_lists = [output_list];
        let mut call_args = sys::LoadedExecutableExecuteArgs::init();
        call_args.executable = self.raw;
        call_args.options = &mut options;
        call_args.argument_lists = argument_lists.as_ptr();
        call_args.num_devices = 1;
        call_args.num_args = args.len();
        call_args.output_lists = output_lists.as_ptr();
        call!(self.api, loaded_executable_execute, call_args)?;
        let mut buffers = Vec::with_capacity(outputs.len());
        for raw in outputs {
            if raw.is_null() {
                return Err(Error::new(
                    "PJRT_LoadedExecutable_Execute returned a null output buffer",
                ));
            }
            let mut buffer = Buffer {
                api: Arc::clone(&self.api),
                raw,
                element: ElementType::F32,
                dims: Vec::new(),
            };
            buffer.refresh_shape()?;
            buffers.push(buffer);
        }
        Ok(buffers)
    }
}

impl Buffer {
    /// Read element type and dimensions from the plugin (outputs of an execution carry no shape of their own).
    fn refresh_shape(&mut self) -> Result<()> {
        let mut element = sys::BufferElementTypeArgs::init();
        element.buffer = self.raw;
        call!(self.api, buffer_element_type, element)?;
        self.element = ElementType::from_code(element.type_).ok_or_else(|| {
            Error::new(format!("unsupported buffer element type {}", element.type_))
        })?;
        let mut dims = sys::BufferDimensionsArgs::init();
        dims.buffer = self.raw;
        call!(self.api, buffer_dimensions, dims)?;
        self.dims = if dims.num_dims == 0 {
            Vec::new()
        } else {
            // SAFETY: the plugin returns `num_dims` dimensions at `dims`.
            unsafe { std::slice::from_raw_parts(dims.dims, dims.num_dims) }.to_vec()
        };
        Ok(())
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        let mut args = sys::LoadedExecutableDestroyArgs::init();
        args.executable = self.raw;
        let _ = call!(self.api, loaded_executable_destroy, args);
    }
}

/// Where to look for the PJRT plugin: `FLLM_TPU_LIBRARY`, `TPU_LIBRARY_PATH`, the `libtpu` Python package, then the dynamic
/// loader's search path.
pub fn plugin_candidates() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for var in ["FLLM_TPU_LIBRARY", "TPU_LIBRARY_PATH"] {
        if let Some(value) = std::env::var_os(var).filter(|v| !v.is_empty()) {
            found.push(PathBuf::from(value));
        }
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(venv) = std::env::var_os("VIRTUAL_ENV") {
        roots.push(PathBuf::from(venv).join("lib"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(home).join(".local/lib"));
    }
    roots.extend(["/usr/local/lib", "/usr/lib", "/usr/lib/python3"].map(PathBuf::from));
    for root in roots {
        // <root>/python3.X/{site,dist}-packages/libtpu/libtpu.so, and <root>/{site,dist}-packages/libtpu/libtpu.so
        let mut python_dirs: Vec<PathBuf> = vec![root.clone()];
        if let Ok(entries) = std::fs::read_dir(&root) {
            python_dirs.extend(entries.flatten().map(|e| e.path()).filter(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("python3"))
            }));
        }
        for dir in python_dirs {
            for packages in ["site-packages", "dist-packages"] {
                let candidate = dir.join(packages).join("libtpu").join("libtpu.so");
                if candidate.is_file() && !found.contains(&candidate) {
                    found.push(candidate);
                }
            }
        }
    }
    if let Some(dir) = python_package_dir() {
        let candidate = dir.join("libtpu.so");
        if candidate.is_file() && !found.contains(&candidate) {
            found.push(candidate);
        }
    }
    found.push(PathBuf::from("libtpu.so"));
    found
}

/// Directory of the `libtpu` Python package, asked from the interpreter (empty when there is no Python or no package).
fn python_package_dir() -> Option<PathBuf> {
    let output = std::process::Command::new("python3")
        .args([
            "-c",
            "import libtpu, os; print(os.path.dirname(libtpu.__file__))",
        ])
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    (!line.is_empty()).then(|| PathBuf::from(line))
}

/// The process-wide plugin and client.
pub struct Runtime {
    pub api: Arc<Api>,
    pub client: Arc<Client>,
}

static RUNTIME: OnceLock<std::result::Result<Arc<Runtime>, String>> = OnceLock::new();

/// Load the TPU plugin and create the client, once per process. The error text says which candidates were tried.
pub fn runtime() -> std::result::Result<Arc<Runtime>, String> {
    RUNTIME
        .get_or_init(|| {
            let candidates = plugin_candidates();
            let mut tried = Vec::new();
            for path in &candidates {
                match Api::load(path) {
                    Ok(api) => {
                        return match Client::create(&api) {
                            Ok(client) => Ok(Arc::new(Runtime { api, client })),
                            Err(e) => Err(format!("{}: creating the PJRT client failed: {e}", path.display())),
                        }
                    }
                    Err(e) => tried.push(e.to_string()),
                }
            }
            Err(format!(
                "no PJRT TPU plugin found (set FLLM_TPU_LIBRARY to libtpu.so, or `pip install libtpu`): {}",
                tried.join("; ")
            ))
        })
        .clone()
}

/// Whether the failure text of [`runtime`] means "no plugin on this machine" (as opposed to a plugin that failed to start).
pub fn is_missing_plugin(message: &str) -> bool {
    message.starts_with("no PJRT TPU plugin found")
}

/// Debug messages of the TPU backend (`FLLM_TPU_DEBUG=1`).
pub fn debug_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var_os("FLLM_TPU_DEBUG").is_some_and(|v| v != "0"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_types_round_trip() {
        for element in [
            ElementType::Pred,
            ElementType::S8,
            ElementType::U8,
            ElementType::S16,
            ElementType::U16,
            ElementType::S32,
            ElementType::U32,
            ElementType::S64,
            ElementType::F16,
            ElementType::Bf16,
            ElementType::F32,
        ] {
            assert_eq!(ElementType::from_code(element.code()), Some(element));
        }
        assert_eq!(ElementType::F32.bytes(), 4);
        assert_eq!(ElementType::Bf16.bytes(), 2);
        assert_eq!(element_count(&[]), 1);
        assert_eq!(element_count(&[3, 4]), 12);
        assert_eq!(element_count(&[3, 0]), 0);
    }

    #[test]
    fn missing_plugin_is_reported_with_a_hint() {
        let path = Path::new("/nonexistent/libtpu.so");
        let error = Api::load(path)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("cannot load"), "{error}");
    }

    #[test]
    fn candidates_end_with_the_loader_search() {
        assert_eq!(
            plugin_candidates().last().map(PathBuf::as_path),
            Some(Path::new("libtpu.so"))
        );
    }
}
