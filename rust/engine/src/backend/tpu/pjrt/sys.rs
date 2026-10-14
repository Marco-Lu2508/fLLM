// Copyright 2022 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Raw bindings of the PJRT C API (`xla/pjrt/c/pjrt_c_api.h` of OpenXLA, API 0.116, Apache-2.0).
//!
//! This file is a hand-written Rust transcription of the structs and the function table of that header (the layout, the field
//! order and the enum values were copied from it; nothing is generated at build time and no header is needed to build).
//! Every call goes through the `PjrtApi` table that the plugin (`libtpu.so`) returns from `GetPjrtApi`. The table and the
//! `*Args` structs only ever grow at the end, so a table or args struct of this version is a valid prefix of a newer plugin's and
//! a newer prefix of an older plugin's: `struct_size` tells the plugin how much of the struct this caller knows.
//!
//! Level of verification: L0 (compiles; never run against a plugin).

#![allow(
    non_snake_case,
    non_camel_case_types,
    dead_code,
    clippy::upper_case_acronyms,
    clippy::too_many_arguments
)]

use std::ffi::{c_char, c_void};
use std::mem::size_of;

/// Opaque handle types of the C API.
macro_rules! opaque {
    ($($name:ident),* $(,)?) => {
        $(
            #[repr(C)]
            pub struct $name {
                _private: [u8; 0],
            }
        )*
    };
}

opaque!(
    PjrtError,
    PjrtEvent,
    PjrtClient,
    PjrtDevice,
    PjrtDeviceDescription,
    PjrtMemory,
    PjrtExecutable,
    PjrtLoadedExecutable,
    PjrtBuffer
);

/// `PJRT_Extension_Base`.
#[repr(C)]
pub struct ExtensionBase {
    pub struct_size: usize,
    pub type_: i32,
    pub next: *mut ExtensionBase,
}

/// `PJRT_Api_Version`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ApiVersion {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub major_version: i32,
    pub minor_version: i32,
}

/// The major / minor version of the header this file transcribes.
pub const API_MAJOR: i32 = 0;
pub const API_MINOR: i32 = 116;

/// `PJRT_Buffer_Type`.
pub mod buffer_type {
    pub const INVALID: i32 = 0;
    pub const PRED: i32 = 1;
    pub const S8: i32 = 2;
    pub const S16: i32 = 3;
    pub const S32: i32 = 4;
    pub const S64: i32 = 5;
    pub const U8: i32 = 6;
    pub const U16: i32 = 7;
    pub const U32: i32 = 8;
    pub const U64: i32 = 9;
    pub const F16: i32 = 10;
    pub const F32: i32 = 11;
    pub const F64: i32 = 12;
    pub const BF16: i32 = 13;
}

/// `PJRT_HostBufferSemantics`.
pub mod host_semantics {
    pub const IMMUTABLE_ONLY_DURING_CALL: i32 = 0;
    pub const IMMUTABLE_UNTIL_TRANSFER_COMPLETES: i32 = 1;
    pub const IMMUTABLE_ZERO_COPY: i32 = 2;
    pub const MUTABLE_ZERO_COPY: i32 = 3;
}

/// Defines an args struct with the two common header fields and a zero-initialising constructor that fills `struct_size`.
macro_rules! args {
    ($name:ident { $($field:ident : $ty:ty),* $(,)? }) => {
        #[repr(C)]
        pub struct $name {
            pub struct_size: usize,
            pub extension_start: *mut ExtensionBase,
            $(pub $field: $ty,)*
        }
        impl $name {
            /// All fields zero (null pointers, `false`, 0) except `struct_size`.
            pub fn init() -> Self {
                // SAFETY: every field is a raw pointer, an integer, a bool or an `Option` of a function pointer, for which the
                // all-zero bit pattern is valid.
                let mut value: Self = unsafe { std::mem::zeroed() };
                value.struct_size = size_of::<Self>();
                value
            }
        }
    };
}

args!(ErrorDestroyArgs { error: *mut PjrtError });
args!(ErrorMessageArgs { error: *const PjrtError, message: *const c_char, message_size: usize });
args!(ErrorGetCodeArgs { error: *const PjrtError, code: i32 });
args!(PluginInitializeArgs {});
args!(EventDestroyArgs { event: *mut PjrtEvent });
args!(EventIsReadyArgs { event: *mut PjrtEvent, is_ready: bool });
args!(EventAwaitArgs { event: *mut PjrtEvent });

args!(ClientCreateArgs {
    create_options: *const c_void,
    num_options: usize,
    kv_get_callback: *const c_void,
    kv_get_user_arg: *mut c_void,
    kv_put_callback: *const c_void,
    kv_put_user_arg: *mut c_void,
    client: *mut PjrtClient,
    kv_try_get_callback: *const c_void,
    kv_try_get_user_arg: *mut c_void,
});
args!(ClientDestroyArgs { client: *mut PjrtClient });
args!(ClientPlatformNameArgs { client: *mut PjrtClient, platform_name: *const c_char, platform_name_size: usize });
args!(ClientPlatformVersionArgs { client: *mut PjrtClient, platform_version: *const c_char, platform_version_size: usize });
args!(ClientDevicesArgs { client: *mut PjrtClient, devices: *const *mut PjrtDevice, num_devices: usize });
args!(ClientAddressableDevicesArgs {
    client: *mut PjrtClient,
    addressable_devices: *const *mut PjrtDevice,
    num_addressable_devices: usize,
});

/// `PJRT_Program`.
#[repr(C)]
pub struct Program {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub code: *mut c_char,
    pub code_size: usize,
    pub format: *const c_char,
    pub format_size: usize,
}

impl Program {
    pub fn init() -> Self {
        // SAFETY: raw pointers and integers only.
        let mut value: Self = unsafe { std::mem::zeroed() };
        value.struct_size = size_of::<Self>();
        value
    }
}

args!(ClientCompileArgs {
    client: *mut PjrtClient,
    program: *const Program,
    compile_options: *const c_char,
    compile_options_size: usize,
    executable: *mut PjrtLoadedExecutable,
});
args!(ClientBufferFromHostBufferArgs {
    client: *mut PjrtClient,
    data: *const c_void,
    type_: i32,
    dims: *const i64,
    num_dims: usize,
    byte_strides: *const i64,
    num_byte_strides: usize,
    host_buffer_semantics: i32,
    device: *mut PjrtDevice,
    memory: *mut PjrtMemory,
    device_layout: *mut c_void,
    done_with_host_buffer: *mut PjrtEvent,
    buffer: *mut PjrtBuffer,
});

args!(DeviceDescriptionIdArgs { device_description: *mut PjrtDeviceDescription, id: i32 });
args!(DeviceDescriptionKindArgs {
    device_description: *mut PjrtDeviceDescription,
    device_kind: *const c_char,
    device_kind_size: usize,
});
args!(DeviceDescriptionDebugStringArgs {
    device_description: *mut PjrtDeviceDescription,
    debug_string: *const c_char,
    debug_string_size: usize,
});
args!(DeviceDescriptionToStringArgs {
    device_description: *mut PjrtDeviceDescription,
    to_string: *const c_char,
    to_string_size: usize,
});
args!(DeviceGetDescriptionArgs { device: *mut PjrtDevice, device_description: *mut PjrtDeviceDescription });
args!(DeviceIsAddressableArgs { device: *mut PjrtDevice, is_addressable: bool });
args!(DeviceLocalHardwareIdArgs { device: *mut PjrtDevice, local_hardware_id: i32 });
args!(DeviceMemoryStatsArgs {
    device: *mut PjrtDevice,
    bytes_in_use: i64,
    peak_bytes_in_use: i64,
    peak_bytes_in_use_is_set: bool,
    num_allocs: i64,
    num_allocs_is_set: bool,
    largest_alloc_size: i64,
    largest_alloc_size_is_set: bool,
    bytes_limit: i64,
    bytes_limit_is_set: bool,
    bytes_reserved: i64,
    bytes_reserved_is_set: bool,
    peak_bytes_reserved: i64,
    peak_bytes_reserved_is_set: bool,
    bytes_reservable_limit: i64,
    bytes_reservable_limit_is_set: bool,
    largest_free_block_bytes: i64,
    largest_free_block_bytes_is_set: bool,
    pool_bytes: i64,
    pool_bytes_is_set: bool,
    peak_pool_bytes: i64,
    peak_pool_bytes_is_set: bool,
    peak_allocated_bytes: i64,
    peak_allocated_bytes_is_set: bool,
});

args!(ExecutableDestroyArgs { executable: *mut PjrtExecutable });
args!(ExecutableNumOutputsArgs { executable: *mut PjrtExecutable, num_outputs: usize });
args!(LoadedExecutableDestroyArgs { executable: *mut PjrtLoadedExecutable });
args!(LoadedExecutableGetExecutableArgs { loaded_executable: *mut PjrtLoadedExecutable, executable: *mut PjrtExecutable });

/// `PJRT_ExecuteOptions`.
#[repr(C)]
pub struct ExecuteOptions {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub send_callbacks: *mut *mut c_void,
    pub recv_callbacks: *mut *mut c_void,
    pub num_send_ops: usize,
    pub num_recv_ops: usize,
    pub launch_id: i32,
    pub non_donatable_input_indices: *const i64,
    pub num_non_donatable_input_indices: usize,
    pub context: *mut c_void,
    pub call_location: *const c_char,
    pub num_tasks: usize,
    pub task_ids: *mut i32,
    pub incarnation_ids: *mut i64,
    pub multi_slice_config: *mut c_void,
    pub use_major_to_minor_data_layout_for_callbacks: bool,
    pub hlo_output_callbacks: *mut c_void,
    pub num_hlo_output_callbacks: usize,
    pub custom_options: *const c_void,
    pub num_custom_options: usize,
}

impl ExecuteOptions {
    pub fn init() -> Self {
        // SAFETY: raw pointers, integers and a bool only.
        let mut value: Self = unsafe { std::mem::zeroed() };
        value.struct_size = size_of::<Self>();
        value
    }
}

args!(LoadedExecutableExecuteArgs {
    executable: *mut PjrtLoadedExecutable,
    options: *mut ExecuteOptions,
    argument_lists: *const *const *mut PjrtBuffer,
    num_devices: usize,
    num_args: usize,
    output_lists: *const *mut *mut PjrtBuffer,
    device_complete_events: *mut *mut PjrtEvent,
    execute_device: *mut PjrtDevice,
});

args!(BufferDestroyArgs { buffer: *mut PjrtBuffer });
args!(BufferElementTypeArgs { buffer: *mut PjrtBuffer, type_: i32 });
args!(BufferDimensionsArgs { buffer: *mut PjrtBuffer, dims: *const i64, num_dims: usize });
args!(BufferOnDeviceSizeInBytesArgs { buffer: *mut PjrtBuffer, on_device_size_in_bytes: usize });
args!(BufferDeleteArgs { buffer: *mut PjrtBuffer });
args!(BufferIsDeletedArgs { buffer: *mut PjrtBuffer, is_deleted: bool });
/// `PJRT_Buffer_MemoryLayout_Tiled`.
#[repr(C)]
pub struct MemoryLayoutTiled {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub minor_to_major: *const i64,
    pub minor_to_major_size: usize,
    pub tile_dims: *const i64,
    pub tile_dim_sizes: *const usize,
    pub num_tiles: usize,
}

/// `PJRT_Buffer_MemoryLayout_Strides`.
#[repr(C)]
pub struct MemoryLayoutStrides {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub byte_strides: *const i64,
    pub num_byte_strides: usize,
}

/// The union inside `PJRT_Buffer_MemoryLayout`.
#[repr(C)]
pub union MemoryLayoutUnion {
    pub tiled: std::mem::ManuallyDrop<MemoryLayoutTiled>,
    pub strides: std::mem::ManuallyDrop<MemoryLayoutStrides>,
}

/// `PJRT_Buffer_MemoryLayout_Type`.
pub mod layout_type {
    pub const TILED: i32 = 0;
    pub const STRIDES: i32 = 1;
}

/// `PJRT_Buffer_MemoryLayout`.
#[repr(C)]
pub struct MemoryLayout {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub layout: MemoryLayoutUnion,
    pub type_: i32,
}

args!(BufferToHostBufferArgs {
    src: *mut PjrtBuffer,
    host_layout: *mut c_void,
    dst: *mut c_void,
    dst_size: usize,
    event: *mut PjrtEvent,
});
args!(BufferReadyEventArgs { buffer: *mut PjrtBuffer, event: *mut PjrtEvent });

/// A table entry the caller never calls (kept so that the following fields stay at their offsets).
pub type RawFn = Option<unsafe extern "C" fn(*mut c_void) -> *mut PjrtError>;
/// A table entry with a known args struct that returns an error.
pub type Fun<A> = Option<unsafe extern "C" fn(*mut A) -> *mut PjrtError>;
/// A table entry with a known args struct that returns nothing.
pub type VoidFun<A> = Option<unsafe extern "C" fn(*mut A)>;

/// `PJRT_Api`: the function table of a plugin.
#[repr(C)]
pub struct PjrtApi {
    pub struct_size: usize,
    pub extension_start: *mut ExtensionBase,
    pub pjrt_api_version: ApiVersion,
    pub error_destroy: VoidFun<ErrorDestroyArgs>,
    pub error_message: VoidFun<ErrorMessageArgs>,
    pub error_get_code: Fun<ErrorGetCodeArgs>,
    pub plugin_initialize: Fun<PluginInitializeArgs>,
    pub plugin_attributes: RawFn,
    pub event_destroy: Fun<EventDestroyArgs>,
    pub event_is_ready: Fun<EventIsReadyArgs>,
    pub event_error: RawFn,
    pub event_await: Fun<EventAwaitArgs>,
    pub event_on_ready: RawFn,
    pub client_create: Fun<ClientCreateArgs>,
    pub client_destroy: Fun<ClientDestroyArgs>,
    pub client_platform_name: Fun<ClientPlatformNameArgs>,
    pub client_process_index: RawFn,
    pub client_platform_version: Fun<ClientPlatformVersionArgs>,
    pub client_devices: Fun<ClientDevicesArgs>,
    pub client_addressable_devices: Fun<ClientAddressableDevicesArgs>,
    pub client_lookup_device: RawFn,
    pub client_lookup_addressable_device: RawFn,
    pub client_addressable_memories: RawFn,
    pub client_compile: Fun<ClientCompileArgs>,
    pub client_default_device_assignment: RawFn,
    pub client_buffer_from_host_buffer: Fun<ClientBufferFromHostBufferArgs>,
    pub device_description_id: Fun<DeviceDescriptionIdArgs>,
    pub device_description_process_index: RawFn,
    pub device_description_attributes: RawFn,
    pub device_description_kind: Fun<DeviceDescriptionKindArgs>,
    pub device_description_debug_string: Fun<DeviceDescriptionDebugStringArgs>,
    pub device_description_to_string: Fun<DeviceDescriptionToStringArgs>,
    pub device_get_description: Fun<DeviceGetDescriptionArgs>,
    pub device_is_addressable: Fun<DeviceIsAddressableArgs>,
    pub device_local_hardware_id: Fun<DeviceLocalHardwareIdArgs>,
    pub device_addressable_memories: RawFn,
    pub device_default_memory: RawFn,
    pub device_memory_stats: Fun<DeviceMemoryStatsArgs>,
    pub memory_id: RawFn,
    pub memory_kind: RawFn,
    pub memory_debug_string: RawFn,
    pub memory_to_string: RawFn,
    pub memory_addressable_by_devices: RawFn,
    pub executable_destroy: Fun<ExecutableDestroyArgs>,
    pub executable_name: RawFn,
    pub executable_num_replicas: RawFn,
    pub executable_num_partitions: RawFn,
    pub executable_num_outputs: Fun<ExecutableNumOutputsArgs>,
    pub executable_size_of_generated_code_in_bytes: RawFn,
    pub executable_get_cost_analysis: RawFn,
    pub executable_output_memory_kinds: RawFn,
    pub executable_optimized_program: RawFn,
    pub executable_serialize: RawFn,
    pub loaded_executable_destroy: Fun<LoadedExecutableDestroyArgs>,
    pub loaded_executable_get_executable: Fun<LoadedExecutableGetExecutableArgs>,
    pub loaded_executable_addressable_devices: RawFn,
    pub loaded_executable_delete: RawFn,
    pub loaded_executable_is_deleted: RawFn,
    pub loaded_executable_execute: Fun<LoadedExecutableExecuteArgs>,
    pub executable_deserialize_and_load: RawFn,
    pub loaded_executable_fingerprint: RawFn,
    pub buffer_destroy: Fun<BufferDestroyArgs>,
    pub buffer_element_type: Fun<BufferElementTypeArgs>,
    pub buffer_dimensions: Fun<BufferDimensionsArgs>,
    pub buffer_unpadded_dimensions: RawFn,
    pub buffer_dynamic_dimension_indices: RawFn,
    pub buffer_get_memory_layout: RawFn,
    pub buffer_on_device_size_in_bytes: Fun<BufferOnDeviceSizeInBytesArgs>,
    pub buffer_device: RawFn,
    pub buffer_memory: RawFn,
    pub buffer_delete: Fun<BufferDeleteArgs>,
    pub buffer_is_deleted: Fun<BufferIsDeletedArgs>,
    pub buffer_copy_to_device: RawFn,
    pub buffer_to_host_buffer: Fun<BufferToHostBufferArgs>,
    pub buffer_is_on_cpu: RawFn,
    pub buffer_ready_event: Fun<BufferReadyEventArgs>,
    pub buffer_unsafe_pointer: RawFn,
    pub buffer_increase_external_reference_count: RawFn,
    pub buffer_decrease_external_reference_count: RawFn,
    pub buffer_opaque_device_memory_data_pointer: RawFn,
    pub copy_to_device_stream_destroy: RawFn,
    pub copy_to_device_stream_add_chunk: RawFn,
    pub copy_to_device_stream_total_bytes: RawFn,
    pub copy_to_device_stream_granule_size: RawFn,
    pub copy_to_device_stream_current_bytes: RawFn,
    pub topology_description_create: RawFn,
    pub topology_description_destroy: RawFn,
    pub topology_description_platform_name: RawFn,
    pub topology_description_platform_version: RawFn,
    pub topology_description_get_device_descriptions: RawFn,
    pub topology_description_serialize: RawFn,
    pub topology_description_attributes: RawFn,
    pub compile: RawFn,
    pub executable_output_element_types: RawFn,
    pub executable_output_dimensions: RawFn,
    pub buffer_copy_to_memory: RawFn,
    pub client_create_view_of_device_buffer: RawFn,
    pub executable_fingerprint: RawFn,
    pub client_topology_description: RawFn,
    pub executable_get_compiled_memory_stats: RawFn,
    pub memory_kind_id: RawFn,
    pub execute_context_create: RawFn,
    pub execute_context_destroy: RawFn,
    pub buffer_copy_raw_to_host: RawFn,
    pub async_host_to_device_transfer_manager_destroy: RawFn,
    pub async_host_to_device_transfer_manager_transfer_data: RawFn,
    pub client_create_buffers_for_async_host_to_device: RawFn,
    pub async_host_to_device_transfer_manager_retrieve_buffer: RawFn,
    pub async_host_to_device_transfer_manager_device: RawFn,
    pub async_host_to_device_transfer_manager_buffer_count: RawFn,
    pub async_host_to_device_transfer_manager_buffer_size: RawFn,
    pub async_host_to_device_transfer_manager_set_buffer_error: RawFn,
    pub async_host_to_device_transfer_manager_add_metadata: RawFn,
    pub client_dma_map: RawFn,
    pub client_dma_unmap: RawFn,
    pub client_create_uninitialized_buffer: RawFn,
    pub client_update_global_process_info: RawFn,
    pub topology_description_deserialize: RawFn,
    pub client_create_alias_buffer: RawFn,
    pub client_fulfill_alias_buffer: RawFn,
    pub loaded_executable_get_device_assignment: RawFn,
    pub client_create_error_buffer: RawFn,
    pub async_host_to_device_transfer_manager_transfer_literal: RawFn,
    pub buffer_copy_raw_to_host_future: RawFn,
    pub device_poison_execution: RawFn,
    pub device_create_async_tracking_event: RawFn,
    pub async_tracking_event_destroy: RawFn,
    pub executable_get_compile_options: RawFn,
    pub buffer_donate_with_control_dependency: RawFn,
    pub event_create: RawFn,
    pub event_set: RawFn,
    pub device_get_attributes: RawFn,
    pub client_load: RawFn,
    pub loaded_executable_addressable_device_logical_ids: RawFn,
    pub buffer_bitcast: RawFn,
    pub error_for_each_payload: RawFn,
    pub topology_description_fingerprint: RawFn,
    pub executable_parameter_memory_kinds: RawFn,
    pub device_clear_memory_stats: RawFn,
    pub topology_description_make_canonical_shape_for_memory_space: RawFn,
    pub topology_description_get_memory_space_kind_ids: RawFn,
}

impl PjrtApi {
    /// Number of bytes of the table this transcription knows (the plugin's `struct_size` says how much of it exists there).
    pub const KNOWN_SIZE: usize = size_of::<PjrtApi>();

    /// Whether the plugin's table includes the field that ends at byte `end_offset` of this transcription.
    pub fn has_through(&self, end_offset: usize) -> bool {
        self.struct_size >= end_offset
    }
}

/// `GetPjrtApi`, the one symbol a plugin exports.
pub type GetPjrtApiFn = unsafe extern "C" fn() -> *const PjrtApi;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_layout_matches_the_header() {
        // 3 header words (struct_size, extension_start, version: 8 + 8 + 4 + 4 + 8 + 8 bytes) and 138 pointers
        assert_eq!(size_of::<ApiVersion>(), 24);
        assert_eq!(size_of::<PjrtApi>(), 16 + 24 + 138 * 8);
    }

    #[test]
    fn args_structs_carry_their_size() {
        assert_eq!(
            ClientCreateArgs::init().struct_size,
            size_of::<ClientCreateArgs>()
        );
        // 2 header words, 9 fields of 8 bytes
        assert_eq!(size_of::<ClientCreateArgs>(), 16 + 9 * 8);
        // data, type + padding, dims, num_dims, strides, num_strides, semantics + padding, device, memory, layout, event, buffer
        assert_eq!(size_of::<ClientBufferFromHostBufferArgs>(), 16 + 13 * 8);
        assert_eq!(size_of::<LoadedExecutableExecuteArgs>(), 16 + 8 * 8);
        // header, the 56-byte tiled variant of the union, the type and its padding
        assert_eq!(size_of::<MemoryLayoutTiled>(), 56);
        assert_eq!(size_of::<MemoryLayout>(), 16 + 56 + 8);
    }
}
