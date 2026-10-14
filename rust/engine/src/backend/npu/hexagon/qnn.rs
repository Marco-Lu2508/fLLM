//! Memory layouts of the Qualcomm AI Engine Direct (QNN) C API that the Hexagon study needs (package 11).
//!
//! Written from the public interface of the QAIRT SDK 2.37 headers (`QnnInterface.h`, `QnnTypes.h`, `QnnGraph.h`, `QnnDevice.h`,
//! `HTP/QnnHtpDevice.h`, `HTP/QnnHtpGraph.h`; core API 2.27); the headers are Qualcomm-proprietary and are neither copied nor shipped, only
//! the binary interface (member order, sizes, constants) is restated here. Everything in this file is plain data and constants, so it is
//! compiled on every host and the struct sizes are checked by unit tests against the sizes derived by hand from the C declarations
//! (LP64 and LLP64 agree: no `long` is involved).
//!
//! The interface table `QnnInterface_t` is a union of versioned function tables; this file declares the `v2.x` table up to
//! `errorFreeVerboseMessage` (51 members, in header order) and types only the members the probe calls; the others are untyped slots.

use std::ffi::{c_char, c_void};

/// `Qnn_ErrorHandle_t`: 0 is success, otherwise the low 16 bits are the error code.
pub type ErrorHandle = u64;
/// `Qnn_Handle_t` and all handle typedefs derived from it.
pub type Handle = *mut c_void;

pub const QNN_SUCCESS: ErrorHandle = 0;

/// `QNN_GET_ERROR_CODE(errorHandle)`.
pub fn error_code(error: ErrorHandle) -> u32 {
    (error & 0xFFFF) as u32
}

/// The core API major version this binding is written for (`QNN_API_VERSION_MAJOR`).
pub const API_MAJOR: u32 = 2;
/// Lowest core API minor version for which all 51 members of the declared table are assumed to exist. Assumption, not checked against the
/// SDK changelog (every QAIRT release for Windows on Snapdragon X and for current Android SoCs is newer); revisit on the first test run.
pub const API_MIN_MINOR: u32 = 17;

/// `Qnn_Version_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

/// `Qnn_ApiVersion_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ApiVersion {
    pub core: Version,
    pub backend: Version,
}

impl ApiVersion {
    /// Whether the declared interface table can be read from a provider with this version.
    pub fn is_supported(&self) -> bool {
        self.core.major == API_MAJOR && self.core.minor >= API_MIN_MINOR
    }
}

/// `Qnn_DataType_t` values.
pub mod dtype {
    pub const FLOAT_16: u32 = 0x0216;
    pub const FLOAT_32: u32 = 0x0232;
    pub const BOOL_8: u32 = 0x0508;
    pub const UNDEFINED: u32 = 0x7FFF_FFFF;
}

/// `Qnn_TensorType_t` values.
pub mod tensor_type {
    pub const APP_WRITE: u32 = 0;
    pub const APP_READ: u32 = 1;
    pub const STATIC: u32 = 4;
}

pub const TENSOR_VERSION_1: u32 = 1;
pub const OPCONFIG_VERSION_1: u32 = 1;
pub const PARAMTYPE_SCALAR: u32 = 0;
pub const TENSOR_DATA_FORMAT_DENSE: u32 = 0;
pub const TENSORMEMTYPE_RAW: u32 = 0;
pub const DEFINITION_UNDEFINED: u32 = 0x7FFF_FFFF;
pub const ENCODING_UNDEFINED: u32 = 0x7FFF_FFFF;
/// `QNN_GRAPH_CONFIG_OPTION_CUSTOM`.
pub const GRAPH_CONFIG_OPTION_CUSTOM: u32 = 0;
/// `QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION`.
pub const HTP_GRAPH_CONFIG_OPTION_PRECISION: u32 = 2;
/// `QNN_PRECISION_FLOAT16`.
pub const PRECISION_FLOAT16: u32 = 1;
/// `QNN_HTP_DEVICE_TYPE_ON_CHIP`.
pub const HTP_DEVICE_TYPE_ON_CHIP: u32 = 0;

/// Op package and op type names (`QnnOpDef.h`).
pub const OP_PACKAGE_QTI_AISW: &std::ffi::CStr = c"qti.aisw";
pub const OP_MAT_MUL: &std::ffi::CStr = c"MatMul";
pub const OP_MAT_MUL_PARAM_TRANSPOSE_IN0: &std::ffi::CStr = c"transpose_in0";
pub const OP_MAT_MUL_PARAM_TRANSPOSE_IN1: &std::ffi::CStr = c"transpose_in1";

/// `Qnn_ScaleOffset_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScaleOffset {
    pub scale: f32,
    pub offset: i32,
}

/// Payload of `Qnn_QuantizeParams_t`: the largest member is `Qnn_BwAxisScaleOffset_t` (32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
pub union QuantizeUnion {
    pub scale_offset: ScaleOffset,
    pub largest_member: [u64; 4],
}

/// `Qnn_QuantizeParams_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QuantizeParams {
    pub encoding_definition: u32,
    pub quantization_encoding: u32,
    pub payload: QuantizeUnion,
}

impl QuantizeParams {
    /// `QNN_QUANTIZE_PARAMS_INIT`: undefined (the tensor is not quantized).
    pub fn undefined() -> Self {
        Self {
            encoding_definition: DEFINITION_UNDEFINED,
            quantization_encoding: ENCODING_UNDEFINED,
            payload: QuantizeUnion {
                largest_member: [0; 4],
            },
        }
    }
}

/// `Qnn_ClientBuffer_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ClientBuffer {
    pub data: *mut c_void,
    pub data_size: u32,
}

/// The anonymous union at the end of `Qnn_TensorV1_t` / `Qnn_TensorV2_t` (client buffer, memory handle or retrieve-raw callbacks).
#[repr(C)]
#[derive(Clone, Copy)]
pub union TensorMemory {
    pub client_buf: ClientBuffer,
    pub mem_handle: Handle,
    pub retrieve_raw: *mut c_void,
}

/// `Qnn_TensorV1_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TensorV1 {
    pub id: u32,
    pub name: *const c_char,
    pub tensor_type: u32,
    pub data_format: u32,
    pub data_type: u32,
    pub quantize_params: QuantizeParams,
    pub rank: u32,
    pub dimensions: *mut u32,
    pub mem_type: u32,
    pub memory: TensorMemory,
}

/// `Qnn_SparseParams_t` (type + the hybrid-COO member).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SparseParams {
    pub layout_type: u32,
    pub num_specified_elements: u32,
    pub num_sparse_dimensions: u32,
}

/// `Qnn_TensorV2_t`. Only its size matters here: it fixes the stride of `Qnn_Tensor_t` arrays.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TensorV2 {
    pub id: u32,
    pub name: *const c_char,
    pub tensor_type: u32,
    pub data_format: u32,
    pub data_type: u32,
    pub quantize_params: QuantizeParams,
    pub rank: u32,
    pub dimensions: *mut u32,
    pub mem_type: u32,
    pub memory: TensorMemory,
    pub is_dynamic_dimensions: *mut u8,
    pub sparse_params: SparseParams,
    pub is_produced: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union TensorVersions {
    pub v1: TensorV1,
    pub v2: TensorV2,
}

/// `Qnn_Tensor_t`: version tag + union of the versioned layouts (this binding always writes version 1, which every SDK accepts).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Tensor {
    pub version: u32,
    pub body: TensorVersions,
}

impl Tensor {
    /// A version-1 tensor description.
    ///
    /// `dimensions` and `name` must outlive every use of the tensor (the SDK keeps reading them through the pointers until the graph is
    /// finalized); `data` is the client buffer (null for graph-internal and not yet bound tensors).
    pub fn v1(
        name: &std::ffi::CStr,
        tensor_type: u32,
        data_type: u32,
        dimensions: &mut [u32],
        data: *mut c_void,
        data_size: u32,
    ) -> Self {
        Self {
            version: TENSOR_VERSION_1,
            body: TensorVersions {
                v1: TensorV1 {
                    id: 0,
                    name: name.as_ptr(),
                    tensor_type,
                    data_format: TENSOR_DATA_FORMAT_DENSE,
                    data_type,
                    quantize_params: QuantizeParams::undefined(),
                    rank: dimensions.len() as u32,
                    dimensions: dimensions.as_mut_ptr(),
                    mem_type: TENSORMEMTYPE_RAW,
                    memory: TensorMemory {
                        client_buf: ClientBuffer { data, data_size },
                    },
                },
            },
        }
    }

    /// Points the client buffer of a version-1 tensor at `data`.
    pub fn bind_buffer(&mut self, data: *mut c_void, data_size: u32) {
        // tensors built by `Tensor::v1` carry the v1 member; assigning a `Copy` union field is safe
        self.body.v1.memory.client_buf = ClientBuffer { data, data_size };
    }

    /// The id the backend assigned in `tensorCreateGraphTensor`.
    pub fn id(&self) -> u32 {
        // SAFETY: see `bind_buffer`.
        unsafe { self.body.v1.id }
    }
}

/// `Qnn_Scalar_t`: data type tag + 8-byte union.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Scalar {
    pub data_type: u32,
    pub value: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union ParamPayload {
    pub scalar: Scalar,
    pub tensor: Tensor,
}

/// `Qnn_Param_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Param {
    pub param_type: u32,
    pub name: *const c_char,
    pub payload: ParamPayload,
}

impl Param {
    /// A boolean scalar parameter (`QNN_DATATYPE_BOOL_8`).
    pub fn boolean(name: &'static std::ffi::CStr, value: bool) -> Self {
        Self {
            param_type: PARAMTYPE_SCALAR,
            name: name.as_ptr(),
            payload: ParamPayload {
                scalar: Scalar {
                    data_type: dtype::BOOL_8,
                    value: u64::from(value),
                },
            },
        }
    }
}

/// `Qnn_OpConfigV1_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpConfigV1 {
    pub name: *const c_char,
    pub package_name: *const c_char,
    pub type_name: *const c_char,
    pub num_params: u32,
    pub params: *mut Param,
    pub num_inputs: u32,
    pub input_tensors: *mut Tensor,
    pub num_outputs: u32,
    pub output_tensors: *mut Tensor,
}

/// `Qnn_OpConfig_t` (passed by value to `graphAddNode`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OpConfig {
    pub version: u32,
    pub v1: OpConfigV1,
}

/// `QnnGraph_Config_t` with the custom option: a pointer to a backend specific configuration (see [`HtpGraphConfig`]).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GraphConfig {
    pub option: u32,
    pub custom_config: *mut c_void,
}

/// `QnnHtpGraph_CustomConfig_t` as far as the probe needs it: the option tag at offset 0 and the 8-byte aligned union at offset 8
/// (its largest member is 44 bytes; 48 are reserved). The precision enum is the first member of the union.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HtpGraphConfig {
    pub option: u32,
    pub reserved_padding: u32,
    pub union_bytes: [u64; 6],
}

impl HtpGraphConfig {
    /// `{ .option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION, .precision = QNN_PRECISION_FLOAT16 }`: run float graphs in fp16 on the HTP.
    pub fn precision_fp16() -> Self {
        Self {
            option: HTP_GRAPH_CONFIG_OPTION_PRECISION,
            reserved_padding: 0,
            union_bytes: [u64::from(PRECISION_FLOAT16), 0, 0, 0, 0, 0],
        }
    }
}

/// `QnnHtpDevice_OnChipDeviceInfoExtension_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HtpOnChipDevice {
    pub vtcm_size: usize,
    pub soc_model: u32,
    pub signed_pd_support: bool,
    pub dlbc_support: bool,
    pub arch: u32,
}

/// `QnnHtpDevice_DeviceInfoExtension_t` (the structure behind `QnnDevice_DeviceInfoExtension_t`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HtpDeviceInfoExtension {
    pub device_type: u32,
    pub on_chip: HtpOnChipDevice,
}

/// `QnnDevice_HardwareDeviceInfoV1_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HardwareDeviceInfoV1 {
    pub device_id: u32,
    pub device_type: u32,
    pub num_cores: u32,
    pub cores: *mut c_void,
    pub device_info_extension: *mut HtpDeviceInfoExtension,
}

/// `QnnDevice_HardwareDeviceInfo_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HardwareDeviceInfo {
    pub version: u32,
    pub v1: HardwareDeviceInfoV1,
}

/// `QnnDevice_PlatformInfoV1_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlatformInfoV1 {
    pub num_hw_devices: u32,
    pub hw_devices: *mut HardwareDeviceInfo,
}

/// `QnnDevice_PlatformInfo_t`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PlatformInfo {
    pub version: u32,
    pub v1: PlatformInfoV1,
}

/// An untyped member of the interface table (function pointer, never called).
pub type Slot = Option<unsafe extern "C" fn()>;

pub type BackendCreateFn = unsafe extern "C" fn(
    logger: Handle,
    config: *const *const c_void,
    backend: *mut Handle,
) -> ErrorHandle;
pub type BackendFreeFn = unsafe extern "C" fn(backend: Handle) -> ErrorHandle;
pub type ContextCreateFn = unsafe extern "C" fn(
    backend: Handle,
    device: Handle,
    config: *const *const c_void,
    context: *mut Handle,
) -> ErrorHandle;
pub type ContextFreeFn = unsafe extern "C" fn(context: Handle, profile: Handle) -> ErrorHandle;
pub type GraphCreateFn = unsafe extern "C" fn(
    context: Handle,
    graph_name: *const c_char,
    config: *const *const GraphConfig,
    graph: *mut Handle,
) -> ErrorHandle;
pub type GraphAddNodeFn = unsafe extern "C" fn(graph: Handle, op_config: OpConfig) -> ErrorHandle;
pub type GraphFinalizeFn =
    unsafe extern "C" fn(graph: Handle, profile: Handle, signal: Handle) -> ErrorHandle;
pub type GraphExecuteFn = unsafe extern "C" fn(
    graph: Handle,
    inputs: *const Tensor,
    num_inputs: u32,
    outputs: *mut Tensor,
    num_outputs: u32,
    profile: Handle,
    signal: Handle,
) -> ErrorHandle;
pub type TensorCreateGraphTensorFn =
    unsafe extern "C" fn(graph: Handle, tensor: *mut Tensor) -> ErrorHandle;
pub type DeviceGetPlatformInfoFn =
    unsafe extern "C" fn(logger: Handle, info: *mut *const PlatformInfo) -> ErrorHandle;
pub type DeviceFreePlatformInfoFn =
    unsafe extern "C" fn(logger: Handle, info: *const PlatformInfo) -> ErrorHandle;
pub type DeviceCreateFn = unsafe extern "C" fn(
    logger: Handle,
    config: *const *const c_void,
    device: *mut Handle,
) -> ErrorHandle;
pub type DeviceFreeFn = unsafe extern "C" fn(device: Handle) -> ErrorHandle;
pub type ErrorGetMessageFn =
    unsafe extern "C" fn(error: ErrorHandle, message: *mut *const c_char) -> ErrorHandle;

/// `QnnInterface_ImplementationV2_x_t`, members 0..=50 in header order (`propertyHasCapability` .. `errorFreeVerboseMessage`).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct InterfaceTable {
    pub property_has_capability: Slot,
    pub backend_create: Option<BackendCreateFn>,
    pub backend_set_config: Slot,
    pub backend_get_api_version: Slot,
    pub backend_get_build_id: Slot,
    pub backend_register_op_package: Slot,
    pub backend_get_supported_operations: Slot,
    pub backend_validate_op_config: Slot,
    pub backend_free: Option<BackendFreeFn>,
    pub context_create: Option<ContextCreateFn>,
    pub context_set_config: Slot,
    pub context_get_binary_size: Slot,
    pub context_get_binary: Slot,
    pub context_create_from_binary: Slot,
    pub context_free: Option<ContextFreeFn>,
    pub graph_create: Option<GraphCreateFn>,
    pub graph_create_subgraph: Slot,
    pub graph_set_config: Slot,
    pub graph_add_node: Option<GraphAddNodeFn>,
    pub graph_finalize: Option<GraphFinalizeFn>,
    pub graph_retrieve: Slot,
    pub graph_execute: Option<GraphExecuteFn>,
    pub graph_execute_async: Slot,
    pub tensor_create_context_tensor: Slot,
    pub tensor_create_graph_tensor: Option<TensorCreateGraphTensorFn>,
    pub log_create: Slot,
    pub log_set_log_level: Slot,
    pub log_free: Slot,
    pub profile_create: Slot,
    pub profile_set_config: Slot,
    pub profile_get_events: Slot,
    pub profile_get_sub_events: Slot,
    pub profile_get_event_data: Slot,
    pub profile_get_extended_event_data: Slot,
    pub profile_free: Slot,
    pub mem_register: Slot,
    pub mem_de_register: Slot,
    pub device_get_platform_info: Option<DeviceGetPlatformInfoFn>,
    pub device_free_platform_info: Option<DeviceFreePlatformInfoFn>,
    pub device_get_infrastructure: Slot,
    pub device_create: Option<DeviceCreateFn>,
    pub device_set_config: Slot,
    pub device_get_info: Slot,
    pub device_free: Option<DeviceFreeFn>,
    pub signal_create: Slot,
    pub signal_set_config: Slot,
    pub signal_trigger: Slot,
    pub signal_free: Slot,
    pub error_get_message: Option<ErrorGetMessageFn>,
    pub error_get_verbose_message: Slot,
    pub error_free_verbose_message: Slot,
}

/// `QnnInterface_t`: `{ backendId, providerName, apiVersion, union { v2_x table } }`. The table is read through a pointer into the
/// provider's own static storage; never copy a `QnnInterface` by value unless [`ApiVersion::is_supported`] holds.
#[repr(C)]
pub struct Interface {
    pub backend_id: u32,
    pub provider_name: *const c_char,
    pub api_version: ApiVersion,
    pub table: InterfaceTable,
}

/// `Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList, uint32_t* numProviders)`.
pub type GetProvidersFn =
    unsafe extern "C" fn(providers: *mut *const *const Interface, count: *mut u32) -> ErrorHandle;

/// `QnnHtpDevice_Arch_t` as a printable version ("v73"); 0 means unknown.
pub fn htp_arch_name(arch: u32) -> String {
    match arch {
        0 | 0x7fff_ffff => "unknown".to_owned(),
        v => format!("v{v}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    #[test]
    fn struct_sizes_match_the_c_declarations() {
        // sizes derived by hand from QnnTypes.h on a 64-bit target (see the field walk-through in the comments below)
        assert_eq!(size_of::<ScaleOffset>(), 8);
        assert_eq!(size_of::<QuantizeParams>(), 40); // 2 x u32 + 32-byte union, 8-aligned
        assert_eq!(size_of::<ClientBuffer>(), 16);
        assert_eq!(size_of::<TensorV1>(), 112);
        assert_eq!(size_of::<SparseParams>(), 12);
        assert_eq!(size_of::<TensorV2>(), 136);
        assert_eq!(size_of::<Tensor>(), 144); // 4 + pad 4 + the larger of the two versions
        assert_eq!(size_of::<Scalar>(), 16);
        assert_eq!(size_of::<Param>(), 160); // u32 + pad + name + the 144-byte tensor member
        assert_eq!(size_of::<OpConfigV1>(), 72);
        assert_eq!(size_of::<OpConfig>(), 80);
        assert_eq!(size_of::<GraphConfig>(), 16);
        assert_eq!(size_of::<HtpGraphConfig>(), 56);
    }

    #[test]
    fn tensor_v1_offsets() {
        assert_eq!(offset_of!(TensorV1, name), 8);
        assert_eq!(offset_of!(TensorV1, tensor_type), 16);
        assert_eq!(offset_of!(TensorV1, data_type), 24);
        assert_eq!(offset_of!(TensorV1, quantize_params), 32);
        assert_eq!(offset_of!(TensorV1, rank), 72);
        assert_eq!(offset_of!(TensorV1, dimensions), 80);
        assert_eq!(offset_of!(TensorV1, mem_type), 88);
        assert_eq!(offset_of!(TensorV1, memory), 96);
        assert_eq!(offset_of!(Tensor, body), 8);
        assert_eq!(offset_of!(Param, payload), 16);
        assert_eq!(offset_of!(OpConfig, v1), 8);
        assert_eq!(offset_of!(HtpGraphConfig, union_bytes), 8);
    }

    #[test]
    fn interface_and_device_info_layout() {
        assert_eq!(size_of::<Version>(), 12);
        assert_eq!(size_of::<ApiVersion>(), 24);
        assert_eq!(offset_of!(Interface, provider_name), 8);
        assert_eq!(offset_of!(Interface, api_version), 16);
        assert_eq!(offset_of!(Interface, table), 40);
        assert_eq!(size_of::<InterfaceTable>(), 51 * size_of::<usize>());
        assert_eq!(
            offset_of!(InterfaceTable, error_get_message),
            48 * size_of::<usize>()
        );
        assert_eq!(
            offset_of!(InterfaceTable, graph_execute),
            21 * size_of::<usize>()
        );
        assert_eq!(
            offset_of!(InterfaceTable, device_create),
            40 * size_of::<usize>()
        );
        assert_eq!(size_of::<HtpOnChipDevice>(), 24);
        assert_eq!(offset_of!(HtpDeviceInfoExtension, on_chip), 8);
        assert_eq!(size_of::<HardwareDeviceInfoV1>(), 32);
        assert_eq!(size_of::<HardwareDeviceInfo>(), 40);
        assert_eq!(offset_of!(HardwareDeviceInfo, v1), 8);
        assert_eq!(offset_of!(PlatformInfo, v1), 8);
        assert_eq!(align_of::<Tensor>(), 8);
    }

    #[test]
    fn api_version_gate() {
        let v = |major, minor| ApiVersion {
            core: Version {
                major,
                minor,
                patch: 0,
            },
            backend: Version::default(),
        };
        assert!(v(2, 27).is_supported());
        assert!(v(2, API_MIN_MINOR).is_supported());
        assert!(!v(2, API_MIN_MINOR - 1).is_supported());
        assert!(!v(3, 27).is_supported());
    }

    #[test]
    fn error_codes_and_arch_names() {
        assert_eq!(error_code(0x0001_1770), 0x1770);
        assert_eq!(htp_arch_name(73), "v73");
        assert_eq!(htp_arch_name(0), "unknown");
    }

    #[test]
    fn htp_precision_config_puts_the_enum_first_in_the_union() {
        let c = HtpGraphConfig::precision_fp16();
        assert_eq!(c.option, 2);
        assert_eq!(c.union_bytes[0], 1);
    }
}
