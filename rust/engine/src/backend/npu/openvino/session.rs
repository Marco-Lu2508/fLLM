//! Safe (RAII) wrappers over the OpenVINO C API: the core, tensors and one compiled single-input / single-output model with its
//! inference request.
//!
//! Verification level: L0 (compiled; never run against a real runtime by this code).

use super::ffi::{Api, OvCompiledModel, OvCore, OvInferRequest, OvModel, OvShape, OvTensor};
use std::ffi::{c_char, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// An `ov_core_t`. The core outlives every object created from it (they keep an `Arc<Runtime>`), because OpenVINO unloads the device
/// plugins when the core is destroyed.
pub struct Runtime {
    api: Arc<Api>,
    core: *mut OvCore,
}

// SAFETY: `ov::Core` is documented as thread safe (read_model / compile_model / get_property may be called concurrently).
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: `core` was created by `ov_core_create` and is freed exactly once.
        unsafe { (self.api.core_free)(self.core) };
    }
}

/// An owned `ov_tensor_t`.
struct Tensor {
    api: Arc<Api>,
    ptr: *mut OvTensor,
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: the handle is owned by this value (created by `ov_tensor_create` or returned by a `get_*_tensor` call).
            unsafe { (self.api.tensor_free)(self.ptr) };
        }
    }
}

impl Tensor {
    /// A tensor of `bytes.len()` rounded up to four bytes, holding a copy of `bytes` (element type `f32`; the IR reader only looks
    /// at the byte size of the weights tensor, see `ov::util::read_model`).
    fn weights_from_bytes(api: &Arc<Api>, bytes: &[u8]) -> Result<Self, String> {
        let types = api.element_types()?;
        let words = bytes.len().div_ceil(4).max(1);
        let mut dims = [words as i64];
        let mut shape = OvShape {
            rank: 1,
            dims: std::ptr::null_mut(),
        };
        // SAFETY: `dims` has `rank` entries; the shape allocation is released below.
        api.check(
            unsafe { (api.shape_create)(1, dims.as_mut_ptr(), &mut shape) },
            "ov_shape_create",
        )?;
        let mut ptr: *mut OvTensor = std::ptr::null_mut();
        // SAFETY: `shape` is valid for the call; the resulting tensor owns its memory.
        let status = unsafe { (api.tensor_create)(types.f32, shape, &mut ptr) };
        // SAFETY: `shape` came from `ov_shape_create`.
        unsafe { (api.shape_free)(&mut shape) };
        api.check(status, "ov_tensor_create")?;
        let tensor = Tensor {
            api: Arc::clone(api),
            ptr,
        };
        let target = tensor.data()?;
        if tensor.byte_size()? < bytes.len() {
            return Err("weights tensor is smaller than the IR weights".into());
        }
        // SAFETY: the tensor holds at least `bytes.len()` bytes (checked) and does not overlap `bytes`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), target.cast::<u8>(), bytes.len());
            // zero the padding behind the weights
            std::ptr::write_bytes(
                target.cast::<u8>().add(bytes.len()),
                0,
                words * 4 - bytes.len(),
            );
        }
        Ok(tensor)
    }

    fn data(&self) -> Result<*mut c_void, String> {
        let mut data: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a live tensor.
        self.api.check(
            unsafe { (self.api.tensor_data)(self.ptr, &mut data) },
            "ov_tensor_data",
        )?;
        if data.is_null() {
            return Err("ov_tensor_data returned no pointer".into());
        }
        Ok(data)
    }

    fn byte_size(&self) -> Result<usize, String> {
        let mut size = 0usize;
        // SAFETY: `ptr` is a live tensor.
        self.api.check(
            unsafe { (self.api.tensor_byte_size)(self.ptr, &mut size) },
            "ov_tensor_get_byte_size",
        )?;
        Ok(size)
    }
}

/// An owned `ov_model_t`.
struct Model {
    api: Arc<Api>,
    ptr: *mut OvModel,
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: owned handle, freed once.
            unsafe { (self.api.model_free)(self.ptr) };
        }
    }
}

/// Temporary IR files written when the runtime cannot read from memory; removed when dropped.
struct TempIr {
    xml: PathBuf,
    bin: PathBuf,
}

impl Drop for TempIr {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.xml);
        let _ = std::fs::remove_file(&self.bin);
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn c_string(text: &str, what: &str) -> Result<CString, String> {
    CString::new(text).map_err(|_| format!("{what} contains a NUL byte"))
}

impl Runtime {
    /// Load the C runtime (once per process) and create a core.
    pub fn new() -> Result<Arc<Runtime>, String> {
        let api = Api::get()?;
        let mut core: *mut OvCore = std::ptr::null_mut();
        // SAFETY: `core` is a valid out-parameter.
        api.check(unsafe { (api.core_create)(&mut core) }, "ov_core_create")?;
        if core.is_null() {
            return Err("ov_core_create returned no core".into());
        }
        Ok(Arc::new(Runtime { api, core }))
    }

    /// The process-wide core (created on first use); `Err` while the C runtime cannot be loaded.
    pub fn shared() -> Result<Arc<Runtime>, String> {
        static SHARED: std::sync::OnceLock<Result<Arc<Runtime>, String>> =
            std::sync::OnceLock::new();
        SHARED.get_or_init(Runtime::new).clone()
    }

    pub fn api(&self) -> &Arc<Api> {
        &self.api
    }

    /// Names of the devices the runtime enumerates (`CPU`, `GPU.0`, `NPU`, ...).
    pub fn available_devices(&self) -> Result<Vec<String>, String> {
        let mut list = super::ffi::OvAvailableDevices {
            devices: std::ptr::null_mut(),
            size: 0,
        };
        // SAFETY: `list` is a valid out-parameter and is released with `ov_available_devices_free` below.
        self.api.check(
            unsafe { (self.api.available_devices)(self.core, &mut list) },
            "ov_core_get_available_devices",
        )?;
        let mut names = Vec::with_capacity(list.size);
        for index in 0..list.size {
            // SAFETY: `devices` points to `size` NUL-terminated strings owned by the list.
            let entry = unsafe { *list.devices.add(index) };
            if !entry.is_null() {
                // SAFETY: non-null, NUL-terminated.
                names.push(
                    unsafe { CStr::from_ptr(entry) }
                        .to_string_lossy()
                        .into_owned(),
                );
            }
        }
        // SAFETY: `list` was filled by `ov_core_get_available_devices`.
        unsafe { (self.api.available_devices_free)(&mut list) };
        Ok(names)
    }

    /// A string property of a device (`FULL_DEVICE_NAME`, `NPU_DRIVER_VERSION`, ...); an error when the device does not know it.
    pub fn property(&self, device: &str, key: &str) -> Result<String, String> {
        let (device, key) = (
            c_string(device, "device name")?,
            c_string(key, "property key")?,
        );
        let mut value: *mut c_char = std::ptr::null_mut();
        // SAFETY: valid C strings and out-parameter; the returned string is copied and released with `ov_free`.
        let status = unsafe {
            (self.api.get_property)(self.core, device.as_ptr(), key.as_ptr(), &mut value)
        };
        self.api.check(status, "ov_core_get_property")?;
        if value.is_null() {
            return Ok(String::new());
        }
        // SAFETY: non-null, NUL-terminated, owned by the runtime until `ov_free`.
        let text = unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: `value` was allocated by the runtime for this call.
        unsafe { (self.api.free_string)(value) };
        Ok(text)
    }

    fn read_model(&self, xml: &str, bin: &[u8]) -> Result<(Model, Option<TempIr>), String> {
        let mut ptr: *mut OvModel = std::ptr::null_mut();
        if self.api.reads_from_memory() {
            let weights = if bin.is_empty() {
                None
            } else {
                Some(Tensor::weights_from_bytes(&self.api, bin)?)
            };
            let weights_ptr = weights
                .as_ref()
                .map_or(std::ptr::null(), |t| t.ptr.cast_const());
            // SAFETY: `core` is live; `xml` outlives the call; `weights_ptr` is null or a live tensor (kept alive by `weights`).
            let status = unsafe {
                self.api
                    .read_from_memory(self.core, xml.as_bytes(), weights_ptr, &mut ptr)
            };
            if let Some(status) = status {
                self.api
                    .check(status, "ov_core_read_model_from_memory_buffer")?;
                // the constants of the model share the weights tensor, which the model keeps alive itself
                return Ok((
                    Model {
                        api: Arc::clone(&self.api),
                        ptr,
                    },
                    None,
                ));
            }
        }
        let id = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let base = std::env::temp_dir().join(format!("fllm-ov-{}-{id}", std::process::id()));
        let files = TempIr {
            xml: base.with_extension("xml"),
            bin: base.with_extension("bin"),
        };
        std::fs::write(&files.xml, xml)
            .map_err(|e| format!("cannot write {}: {e}", files.xml.display()))?;
        std::fs::write(&files.bin, bin)
            .map_err(|e| format!("cannot write {}: {e}", files.bin.display()))?;
        let xml_c = c_string(&files.xml.to_string_lossy(), "IR path")?;
        let bin_c = c_string(&files.bin.to_string_lossy(), "weights path")?;
        // SAFETY: valid C strings; `ptr` is a valid out-parameter.
        let status =
            unsafe { (self.api.read_file)(self.core, xml_c.as_ptr(), bin_c.as_ptr(), &mut ptr) };
        self.api.check(status, "ov_core_read_model")?;
        // the model may memory-map the .bin, so the files stay until the model is gone
        Ok((
            Model {
                api: Arc::clone(&self.api),
                ptr,
            },
            Some(files),
        ))
    }

    /// Parse an IR without compiling it: `Ok` when the OpenVINO reader accepts the graph (types, shapes and constants are checked by
    /// the reader's own shape inference). No device is involved.
    pub fn check_ir(&self, xml: &str, bin: &[u8]) -> Result<(), String> {
        self.read_model(xml, bin).map(|_| ())
    }

    /// Compile an IR (`xml` text and `.bin` bytes) for `device` with the given properties (at most three).
    pub fn compile_ir(
        self: &Arc<Self>,
        xml: &str,
        bin: &[u8],
        device: &str,
        props: &[(String, String)],
    ) -> Result<CompiledLinear, String> {
        if props.len() > 3 {
            return Err("at most three compile properties are supported".into());
        }
        let (model, _temp_files) = self.read_model(xml, bin)?;
        let device_c = c_string(device, "device name")?;
        let mut keys = Vec::with_capacity(props.len());
        for (key, value) in props {
            keys.push((
                c_string(key, "property key")?,
                c_string(value, "property value")?,
            ));
        }
        let mut compiled: *mut OvCompiledModel = std::ptr::null_mut();
        let core = self.core.cast_const();
        let model_ptr = model.ptr.cast_const();
        let dev = device_c.as_ptr();
        let compile = self.api.compile;
        let key = |i: usize| keys[i].0.as_ptr();
        let value = |i: usize| keys[i].1.as_ptr();
        // SAFETY: the variadic arguments are NUL-terminated strings, `property_args_size` counts them (key and value of each
        // property); `core` and `model_ptr` are live for the call; `compiled` is a valid out-parameter.
        let status = unsafe {
            match keys.len() {
                0 => compile(core, model_ptr, dev, 0, &mut compiled),
                1 => compile(core, model_ptr, dev, 2, &mut compiled, key(0), value(0)),
                2 => compile(
                    core,
                    model_ptr,
                    dev,
                    4,
                    &mut compiled,
                    key(0),
                    value(0),
                    key(1),
                    value(1),
                ),
                _ => compile(
                    core,
                    model_ptr,
                    dev,
                    6,
                    &mut compiled,
                    key(0),
                    value(0),
                    key(1),
                    value(1),
                    key(2),
                    value(2),
                ),
            }
        };
        self.api
            .check(status, &format!("ov_core_compile_model({device})"))?;
        if compiled.is_null() {
            return Err("ov_core_compile_model returned no model".into());
        }
        // from here on the compiled model owns what it needs; the IR model can go
        drop(model);
        let mut request: *mut OvInferRequest = std::ptr::null_mut();
        // SAFETY: `compiled` is live; `request` is a valid out-parameter.
        let status = unsafe { (self.api.create_request)(compiled.cast_const(), &mut request) };
        if let Err(error) = self
            .api
            .check(status, "ov_compiled_model_create_infer_request")
        {
            // SAFETY: `compiled` is owned here and not used afterwards.
            unsafe { (self.api.compiled_free)(compiled) };
            return Err(error);
        }
        Ok(CompiledLinear {
            runtime: Arc::clone(self),
            compiled,
            request: Mutex::new(RequestPtr(request)),
        })
    }
}

struct RequestPtr(*mut OvInferRequest);

// SAFETY: an inference request may be used from any thread, but only by one at a time; the `Mutex` in `CompiledLinear` guarantees it.
unsafe impl Send for RequestPtr {}

/// A compiled model with one input and one output of 16-bit floats and the inference request that runs it.
pub struct CompiledLinear {
    runtime: Arc<Runtime>,
    compiled: *mut OvCompiledModel,
    request: Mutex<RequestPtr>,
}

// SAFETY: the compiled model is only read after creation; the request is behind a `Mutex`; the runtime is `Sync`.
unsafe impl Send for CompiledLinear {}
unsafe impl Sync for CompiledLinear {}

impl Drop for CompiledLinear {
    fn drop(&mut self) {
        let api = &self.runtime.api;
        let request = self
            .request
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // SAFETY: both handles are owned by this value and freed once; the request goes first.
        unsafe {
            if !request.0.is_null() {
                (api.request_free)(request.0);
            }
            (api.compiled_free)(self.compiled);
        }
    }
}

impl CompiledLinear {
    /// Run one inference: `input` (f16 bits) is copied into the request's input tensor, `output` receives the result. Both slices
    /// must have exactly the element counts the model was compiled for.
    pub fn run(&self, input: &[u16], output: &mut [u16]) -> Result<(), String> {
        let api = &self.runtime.api;
        let request = self
            .request
            .lock()
            .map_err(|_| "inference request lock poisoned".to_owned())?;
        let tensor_of = |output_side: bool| -> Result<Tensor, String> {
            let mut ptr: *mut OvTensor = std::ptr::null_mut();
            // SAFETY: the request is live and locked; `ptr` is a valid out-parameter.
            let status = unsafe {
                if output_side {
                    (api.request_output)(request.0.cast_const(), 0, &mut ptr)
                } else {
                    (api.request_input)(request.0.cast_const(), 0, &mut ptr)
                }
            };
            api.check(status, "ov_infer_request_get_*_tensor_by_index")?;
            Ok(Tensor {
                api: Arc::clone(api),
                ptr,
            })
        };
        let input_tensor = tensor_of(false)?;
        if input_tensor.byte_size()? != input.len() * 2 {
            return Err(format!(
                "model input holds {} bytes, expected {}",
                input_tensor.byte_size()?,
                input.len() * 2
            ));
        }
        // SAFETY: the tensor memory has `input.len() * 2` bytes (checked) and stays valid while the request lives.
        unsafe {
            std::ptr::copy_nonoverlapping(
                input.as_ptr(),
                input_tensor.data()?.cast::<u16>(),
                input.len(),
            )
        };
        // SAFETY: the request is live and locked.
        api.check(unsafe { (api.infer)(request.0) }, "ov_infer_request_infer")?;
        let output_tensor = tensor_of(true)?;
        if output_tensor.byte_size()? != output.len() * 2 {
            return Err(format!(
                "model output holds {} bytes, expected {}",
                output_tensor.byte_size()?,
                output.len() * 2
            ));
        }
        // SAFETY: the tensor memory has `output.len() * 2` bytes (checked).
        unsafe {
            std::ptr::copy_nonoverlapping(
                output_tensor.data()?.cast::<u16>().cast_const(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        Ok(())
    }
}

/// `ov_core_get_property`-independent check that the loaded runtime works at all: returns its version string.
pub fn runtime_version() -> Result<String, String> {
    let api = Api::get()?;
    Ok(api
        .version_string()
        .unwrap_or_else(|| "unknown version".to_owned()))
}
