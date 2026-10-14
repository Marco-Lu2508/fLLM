//! A single-MatMul QNN graph on the HTP (package 11, feature `npu-qnn`): the "single MatMul model" of the Hexagon study.
//!
//! `y[m, n] = x[m, k] . w[n, k]^T` with fp16 tensors: `x` is an application-write input, `w` a static (constant) tensor, `y` an
//! application-read output; the HTP graph precision is set to fp16. The graph is built at run time with `QnnGraph_addNode` (the QNN way
//! to build a model without an offline conversion), compiled by `QnnGraph_finalize` (this is the slow step on the device), and executed once
//! with `QnnGraph_execute`. It proves that the backend accepts and runs a linear layer; it is not an LLM path (see `STUDY_HEXAGON.md`: the
//! HTP wants static shapes, 4-bit blockwise weights and whole-model context binaries). Verification level: L0.

use super::qnn::{
    dtype, tensor_type, ApiVersion, GraphConfig, Handle, HtpGraphConfig, OpConfig, OpConfigV1,
    Param, Tensor, GRAPH_CONFIG_OPTION_CUSTOM, OPCONFIG_VERSION_1, OP_MAT_MUL,
    OP_MAT_MUL_PARAM_TRANSPOSE_IN0, OP_MAT_MUL_PARAM_TRANSPOSE_IN1, OP_PACKAGE_QTI_AISW,
};
use super::runtime::Qnn;
use crate::backend::npu::xdna::testdata;
use half::f16;
use std::ffi::c_void;
use std::ptr;
use std::time::Instant;

/// Result of [`Qnn::probe_matmul_f16`].
#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub provider: String,
    pub api_version: ApiVersion,
    /// `QnnGraph_finalize` time (graph compilation for the HTP).
    pub finalize_ms: f64,
    /// `QnnGraph_execute` time of one run.
    pub run_ms: f64,
    /// Largest absolute difference to the f32 CPU reference computed from the fp16-rounded inputs.
    pub max_abs_err: f32,
}

/// Backend + device + context, released in reverse order on drop (the graph is owned by the context).
struct Instance<'a> {
    qnn: &'a Qnn,
    backend: Handle,
    device: Handle,
    context: Handle,
}

impl Drop for Instance<'_> {
    fn drop(&mut self) {
        // SAFETY: the handles were created by the same interface table and are released once; failures during teardown are ignored.
        unsafe {
            if let (Some(free), false) = (self.qnn.table.context_free, self.context.is_null()) {
                free(self.context, ptr::null_mut());
            }
            if let (Some(free), false) = (self.qnn.table.device_free, self.device.is_null()) {
                free(self.device);
            }
            if let (Some(free), false) = (self.qnn.table.backend_free, self.backend.is_null()) {
                free(self.backend);
            }
        }
    }
}

impl Qnn {
    /// Builds, compiles and runs `y = x . w^T` (fp16, random data) on the HTP and compares it with a CPU reference.
    pub fn probe_matmul_f16(&self, m: usize, k: usize, n: usize) -> Result<ProbeReport, String> {
        let t = &self.table;
        let backend_create = t.backend_create.ok_or("backendCreate missing")?;
        let context_create = t.context_create.ok_or("contextCreate missing")?;
        let graph_create = t.graph_create.ok_or("graphCreate missing")?;
        let create_tensor = t
            .tensor_create_graph_tensor
            .ok_or("tensorCreateGraphTensor missing")?;
        let add_node = t.graph_add_node.ok_or("graphAddNode missing")?;
        let finalize = t.graph_finalize.ok_or("graphFinalize missing")?;
        let execute = t.graph_execute.ok_or("graphExecute missing")?;
        let to_u32 = |v: usize, what: &str| {
            u32::try_from(v).map_err(|_| format!("{what} {v} does not fit into 32 bits"))
        };
        let (m32, k32, n32) = (to_u32(m, "m")?, to_u32(k, "k")?, to_u32(n, "n")?);
        let x_bytes = u32::try_from(m * k * 2).map_err(|_| "x is larger than 4 GiB".to_owned())?;
        let w_bytes = u32::try_from(n * k * 2).map_err(|_| "w is larger than 4 GiB".to_owned())?;
        let y_bytes = u32::try_from(m * n * 2).map_err(|_| "y is larger than 4 GiB".to_owned())?;

        // Data: x [m,k], w [n,k] (so that y = x . w^T), both rounded to fp16; the reference uses the rounded values.
        let x_bits: Vec<u16> = testdata::pseudo_random(m * k, 0x2545_f491_4f6c_dd1d)
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();
        let w_bits: Vec<u16> = testdata::pseudo_random(n * k, 0x9e37_79b9_7f4a_7c15)
            .iter()
            .map(|&v| f16::from_f32(v).to_bits())
            .collect();
        let x_f32: Vec<f32> = x_bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        let w_f32_t: Vec<f32> = {
            // transpose [n,k] -> [k,n] for the shared [k,n] reference helper
            let mut out = vec![0f32; k * n];
            for j in 0..n {
                for p in 0..k {
                    out[p * n + j] = f16::from_bits(w_bits[j * k + p]).to_f32();
                }
            }
            out
        };
        let expected = testdata::matmul_reference(&x_f32, &w_f32_t, m, k, n);
        let mut y_bits = vec![0u16; m * n];

        // SAFETY: all pointers handed to the backend (names, dimensions, buffers, parameter arrays) outlive the calls; handles are
        // released by `Instance::drop`. The structure layouts are those of qnn.rs (size-checked by unit tests).
        unsafe {
            let mut inst = Instance {
                qnn: self,
                backend: ptr::null_mut(),
                device: ptr::null_mut(),
                context: ptr::null_mut(),
            };
            self.check(
                backend_create(ptr::null_mut(), ptr::null(), &mut inst.backend),
                "backendCreate",
            )?;
            if let Some(device_create) = t.device_create {
                self.check(
                    device_create(ptr::null_mut(), ptr::null(), &mut inst.device),
                    "deviceCreate",
                )?;
            }
            self.check(
                context_create(inst.backend, inst.device, ptr::null(), &mut inst.context),
                "contextCreate",
            )?;

            let htp = HtpGraphConfig::precision_fp16();
            let config = GraphConfig {
                option: GRAPH_CONFIG_OPTION_CUSTOM,
                custom_config: &htp as *const HtpGraphConfig as *mut c_void,
            };
            let configs: [*const GraphConfig; 2] = [&config, ptr::null()];
            let mut graph: Handle = ptr::null_mut();
            self.check(
                graph_create(
                    inst.context,
                    c"fllm_matmul".as_ptr(),
                    configs.as_ptr(),
                    &mut graph,
                ),
                "graphCreate",
            )?;

            let mut dims_x = [m32, k32];
            let mut dims_w = [n32, k32];
            let mut dims_y = [m32, n32];
            let mut x = Tensor::v1(
                c"x",
                tensor_type::APP_WRITE,
                dtype::FLOAT_16,
                &mut dims_x,
                ptr::null_mut(),
                0,
            );
            let mut w = Tensor::v1(
                c"w",
                tensor_type::STATIC,
                dtype::FLOAT_16,
                &mut dims_w,
                w_bits.as_ptr() as *mut c_void,
                w_bytes,
            );
            let mut y = Tensor::v1(
                c"y",
                tensor_type::APP_READ,
                dtype::FLOAT_16,
                &mut dims_y,
                ptr::null_mut(),
                0,
            );
            for (tensor, what) in [(&mut x, "x"), (&mut w, "w"), (&mut y, "y")] {
                self.check(
                    create_tensor(graph, tensor),
                    &format!("tensorCreateGraphTensor({what})"),
                )?;
            }

            let mut params = [
                Param::boolean(OP_MAT_MUL_PARAM_TRANSPOSE_IN0, false),
                Param::boolean(OP_MAT_MUL_PARAM_TRANSPOSE_IN1, true),
            ];
            let mut node_inputs = [x, w];
            let mut node_outputs = [y];
            let op = OpConfig {
                version: OPCONFIG_VERSION_1,
                v1: OpConfigV1 {
                    name: c"matmul0".as_ptr(),
                    package_name: OP_PACKAGE_QTI_AISW.as_ptr(),
                    type_name: OP_MAT_MUL.as_ptr(),
                    num_params: params.len() as u32,
                    params: params.as_mut_ptr(),
                    num_inputs: node_inputs.len() as u32,
                    input_tensors: node_inputs.as_mut_ptr(),
                    num_outputs: node_outputs.len() as u32,
                    output_tensors: node_outputs.as_mut_ptr(),
                },
            };
            self.check(add_node(graph, op), "graphAddNode(MatMul)")?;
            let started = Instant::now();
            self.check(
                finalize(graph, ptr::null_mut(), ptr::null_mut()),
                "graphFinalize",
            )?;
            let finalize_ms = started.elapsed().as_secs_f64() * 1e3;

            // Execution: the application-write input and the application-read output get their buffers now.
            x.bind_buffer(x_bits.as_ptr() as *mut c_void, x_bytes);
            y.bind_buffer(y_bits.as_mut_ptr().cast(), y_bytes);
            let inputs = [x];
            let mut outputs = [y];
            let started = Instant::now();
            self.check(
                execute(
                    graph,
                    inputs.as_ptr(),
                    1,
                    outputs.as_mut_ptr(),
                    1,
                    ptr::null_mut(),
                    ptr::null_mut(),
                ),
                "graphExecute",
            )?;
            let run_ms = started.elapsed().as_secs_f64() * 1e3;

            let max_abs_err = y_bits
                .iter()
                .zip(&expected)
                .map(|(&b, &e)| (f16::from_bits(b).to_f32() - e).abs())
                .fold(0.0f32, f32::max);
            Ok(ProbeReport {
                provider: self.provider.clone(),
                api_version: self.api_version,
                finalize_ms,
                run_ms,
                max_abs_err,
            })
        }
    }
}
