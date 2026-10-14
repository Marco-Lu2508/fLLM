//! Core ML probe for the Apple Neural Engine (package 11, feature `npu-ane`, Apple platforms only).
//!
//! Core ML decides itself which compute unit runs which operation; an application can only state a preference
//! (`MLModelConfiguration.computeUnits`). This file writes the one-layer fp16 ML Program of [`super::mlpackage`] to a temporary
//! `.mlpackage`, compiles it (`+[MLModel compileModelAtURL:error:]`), loads it with `computeUnits = cpuAndNeuralEngine`, runs one
//! prediction through `MLMultiArray` inputs and compares the result with a CPU reference. It proves that the Core ML route works end to
//! end on the machine; whether the layer really ran on the Neural Engine is not observable from the API (use Xcode Instruments or
//! `powermetrics` for that). Verification level: L0 (type-checked for `aarch64-apple-darwin`, never run).

use super::mlpackage::{write_linear_package, LinearSpec};
use super::objc::{self, msg_send, sel, Id, Obj, Pool};
use crate::backend::npu::xdna::testdata;
use half::f16;
use std::ffi::c_void;
use std::path::Path;
use std::ptr;
use std::time::Instant;

/// `MLComputeUnits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(isize)]
pub enum ComputeUnits {
    CpuOnly = 0,
    CpuAndGpu = 1,
    All = 2,
    CpuAndNeuralEngine = 3,
}

/// `MLMultiArrayDataType`.
const MULTI_ARRAY_FLOAT16: isize = 65552;
const MULTI_ARRAY_FLOAT32: isize = 65568;

/// What is known about the Neural Engine of this Mac.
#[derive(Debug, Clone)]
pub struct AneInfo {
    /// `machdep.cpu.brand_string` ("Apple M3 Pro").
    pub chip: String,
    /// `hw.memsize` (unified memory shared with the CPU and GPU).
    pub memory_bytes: u64,
    /// `MLNeuralEngineComputeDevice.totalCoreCount` (macOS 14 and later), if it could be read.
    pub neural_engine_cores: Option<usize>,
}

/// The Neural Engine of an Apple Silicon Mac (every M-series chip has one); `None` on Intel Macs or if Core ML cannot be loaded.
pub fn detect() -> Option<AneInfo> {
    if !cfg!(target_arch = "aarch64") {
        return None;
    }
    let frameworks = objc::frameworks().ok()?;
    let chip = objc::sysctl_string(c"machdep.cpu.brand_string")
        .unwrap_or_else(|| "Apple Silicon".to_owned());
    let memory_bytes = objc::sysctl_u64(c"hw.memsize").unwrap_or(0);
    let _pool = Pool::new();
    let neural_engine_cores = (|| {
        let all = frameworks.symbol(c"MLAllComputeDevices");
        if all.is_null() {
            return None;
        }
        // SAFETY: `MLAllComputeDevices` is `NSArray<id<MLComputeDeviceProtocol>>* MLAllComputeDevices(void)` (macOS 14); the messages sent
        // below are `count`, `objectAtIndex:`, `isKindOfClass:` and `totalCoreCount` (NSInteger) of the returned objects.
        unsafe {
            let all = std::mem::transmute::<*mut c_void, unsafe extern "C" fn() -> Id>(all);
            let devices = all();
            let neural_engine = objc::class(c"MLNeuralEngineComputeDevice").ok()?;
            if devices.is_null() {
                return None;
            }
            let count = msg_send!(devices, sel(c"count"), () -> usize);
            for i in 0..count {
                let device = msg_send!(devices, sel(c"objectAtIndex:"), (usize) -> Id, i);
                if msg_send!(device, sel(c"isKindOfClass:"), (Id) -> u8, neural_engine) != 0 {
                    let cores = msg_send!(device, sel(c"totalCoreCount"), () -> isize);
                    return usize::try_from(cores).ok();
                }
            }
            None
        }
    })();
    Some(AneInfo {
        chip,
        memory_bytes,
        neural_engine_cores,
    })
}

/// Result of [`probe_linear`].
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub compute_units: ComputeUnits,
    /// `compileModelAtURL:` time (the Core ML compiler, ahead-of-time part).
    pub compile_ms: f64,
    /// `modelWithContentsOfURL:configuration:` time (includes the on-device specialization for the Neural Engine on first load).
    pub load_ms: f64,
    /// One `predictionFromFeatures:` call.
    pub predict_ms: f64,
    /// Largest absolute difference to the f32 CPU reference computed from the fp16-rounded inputs.
    pub max_abs_err: f32,
    /// The MIL text of the program (informational).
    pub mil: String,
}

/// Removes a directory when dropped.
struct RemoveDirOnDrop(String);

impl Drop for RemoveDirOnDrop {
    fn drop(&mut self) {
        if !self.0.is_empty() {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

fn ns_integer(array: Id, index: usize) -> isize {
    // SAFETY: `array` is an NSArray of NSNumber; `objectAtIndex:` / `integerValue` have the signatures used.
    unsafe {
        let number = msg_send!(array, sel(c"objectAtIndex:"), (usize) -> Id, index);
        msg_send!(number, sel(c"integerValue"), () -> isize)
    }
}

/// Builds the package for `y = x . w^T` (fp16 random data), compiles, loads and runs it once.
pub fn probe_linear(
    m: usize,
    k: usize,
    n: usize,
    units: ComputeUnits,
) -> Result<ProbeReport, String> {
    objc::frameworks()?;
    let x_bits: Vec<u16> = testdata::pseudo_random(m * k, 0x2545_f491_4f6c_dd1d)
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect();
    let w_bits: Vec<u16> = testdata::pseudo_random(n * k, 0x9e37_79b9_7f4a_7c15)
        .iter()
        .map(|&v| f16::from_f32(v).to_bits())
        .collect();
    let x_f32: Vec<f32> = x_bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
    let mut w_t = vec![0f32; k * n];
    for j in 0..n {
        for p in 0..k {
            w_t[p * n + j] = f16::from_bits(w_bits[j * k + p]).to_f32();
        }
    }
    let expected = testdata::matmul_reference(&x_f32, &w_t, m, k, n);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir =
        std::env::temp_dir().join(format!("fllm-ane-{}-{nanos}.mlpackage", std::process::id()));
    let spec = LinearSpec {
        m,
        k,
        n,
        weight: &w_bits,
        bias: None,
    };
    let mil = write_linear_package(&dir, &spec)?;
    let result = run_package(&dir, &x_bits, &expected, (m, n), units, mil);
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn run_package(
    dir: &Path,
    x_bits: &[u16],
    expected: &[f32],
    (m, n): (usize, usize),
    units: ComputeUnits,
    mil: String,
) -> Result<ProbeReport, String> {
    let _pool = Pool::new();
    let dir_text = dir.to_str().ok_or("temporary path is not valid UTF-8")?;
    let k = x_bits.len() / m;
    // SAFETY: every message below is sent with the argument and return types of its documented Objective-C declaration (Foundation and
    // Core ML, macOS 13 or later); objects created with alloc/init are owned by `Obj`, the rest is autoreleased into `_pool`.
    unsafe {
        let mut error: Id = ptr::null_mut();
        let url = msg_send!(objc::class(c"NSURL")?, sel(c"fileURLWithPath:isDirectory:"), (Id, u8) -> Id, objc::ns_string(dir_text)?, 1);
        let ml_model = objc::class(c"MLModel")?;

        let started = Instant::now();
        let compiled = msg_send!(ml_model, sel(c"compileModelAtURL:error:"), (Id, *mut Id) -> Id, url, &mut error);
        if compiled.is_null() {
            return Err(format!(
                "compileModelAtURL failed: {}",
                objc::error_message(error)
            ));
        }
        let compile_ms = started.elapsed().as_secs_f64() * 1e3;
        // the compiled model is deleted when the probe is done (it must outlive the prediction: the runtime may map its files)
        let _compiled_dir = RemoveDirOnDrop(objc::to_rust_string(
            msg_send!(compiled, sel(c"path"), () -> Id),
        ));

        let configuration = Obj::owned(msg_send!(msg_send!(objc::class(c"MLModelConfiguration")?, sel(c"alloc"), () -> Id), sel(c"init"), () -> Id))
            .ok_or("MLModelConfiguration init failed")?;
        msg_send!(configuration.id(), sel(c"setComputeUnits:"), (isize) -> (), units as isize);

        let started = Instant::now();
        let model = msg_send!(ml_model, sel(c"modelWithContentsOfURL:configuration:error:"), (Id, Id, *mut Id) -> Id, compiled, configuration.id(), &mut error);
        let load_ms = started.elapsed().as_secs_f64() * 1e3;
        if model.is_null() {
            return Err(format!(
                "modelWithContentsOfURL failed: {}",
                objc::error_message(error)
            ));
        }

        // Input: MLMultiArray [m, k] float16, filled through dataPointer (a fresh array is contiguous).
        let number_class = objc::class(c"NSNumber")?;
        let numbers: Vec<Id> = [m as i64, k as i64]
            .iter()
            .map(|&d| msg_send!(number_class, sel(c"numberWithLongLong:"), (i64) -> Id, d))
            .collect();
        let shape = msg_send!(objc::class(c"NSArray")?, sel(c"arrayWithObjects:count:"), (*const Id, usize) -> Id, numbers.as_ptr(), numbers.len());
        let input = Obj::owned(msg_send!(
            msg_send!(objc::class(c"MLMultiArray")?, sel(c"alloc"), () -> Id),
            sel(c"initWithShape:dataType:error:"),
            (Id, isize, *mut Id) -> Id,
            shape,
            MULTI_ARRAY_FLOAT16,
            &mut error
        ))
        .ok_or_else(|| {
            format!(
                "MLMultiArray creation failed: {}",
                objc::error_message(error)
            )
        })?;
        let data = msg_send!(input.id(), sel(c"dataPointer"), () -> *mut c_void);
        ptr::copy_nonoverlapping(
            x_bits.as_ptr().cast::<u8>(),
            data.cast::<u8>(),
            x_bits.len() * 2,
        );

        let feature = msg_send!(objc::class(c"MLFeatureValue")?, sel(c"featureValueWithMultiArray:"), (Id) -> Id, input.id());
        let dictionary = msg_send!(objc::class(c"NSDictionary")?, sel(c"dictionaryWithObject:forKey:"), (Id, Id) -> Id, feature, objc::ns_string("x")?);
        let provider = Obj::owned(msg_send!(
            msg_send!(objc::class(c"MLDictionaryFeatureProvider")?, sel(c"alloc"), () -> Id),
            sel(c"initWithDictionary:error:"),
            (Id, *mut Id) -> Id,
            dictionary,
            &mut error
        ))
        .ok_or_else(|| {
            format!(
                "MLDictionaryFeatureProvider creation failed: {}",
                objc::error_message(error)
            )
        })?;

        let started = Instant::now();
        let output = msg_send!(model, sel(c"predictionFromFeatures:error:"), (Id, *mut Id) -> Id, provider.id(), &mut error);
        let predict_ms = started.elapsed().as_secs_f64() * 1e3;
        if output.is_null() {
            return Err(format!(
                "predictionFromFeatures failed: {}",
                objc::error_message(error)
            ));
        }
        let y_value =
            msg_send!(output, sel(c"featureValueForName:"), (Id) -> Id, objc::ns_string("y")?);
        let y = msg_send!(y_value, sel(c"multiArrayValue"), () -> Id);
        if y.is_null() {
            return Err("output \"y\" is not a multi-array".to_owned());
        }
        let y_type = msg_send!(y, sel(c"dataType"), () -> isize);
        let strides = msg_send!(y, sel(c"strides"), () -> Id);
        let (row_stride, column_stride) = (ns_integer(strides, 0), ns_integer(strides, 1));
        let y_data = msg_send!(y, sel(c"dataPointer"), () -> *mut c_void);
        let mut max_abs_err = 0f32;
        for i in 0..m {
            for j in 0..n {
                let at = i as isize * row_stride + j as isize * column_stride;
                let value = match y_type {
                    MULTI_ARRAY_FLOAT16 => {
                        f16::from_bits(*y_data.cast::<u16>().offset(at)).to_f32()
                    }
                    MULTI_ARRAY_FLOAT32 => *y_data.cast::<f32>().offset(at),
                    other => return Err(format!("unexpected output data type {other}")),
                };
                max_abs_err = max_abs_err.max((value - expected[i * n + j]).abs());
            }
        }
        Ok(ProbeReport {
            compute_units: units,
            compile_ms,
            load_ms,
            predict_ms,
            max_abs_err,
            mil,
        })
    }
}
