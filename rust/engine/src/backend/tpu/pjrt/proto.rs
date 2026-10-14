//! Hand-written Protocol Buffers encoding of the `CompileOptionsProto` that `PJRT_Client_Compile` takes.
//!
//! The message layout is from `xla/pjrt/proto/compile_options.proto` and `xla/xla_data.proto` (OpenXLA, Apache-2.0):
//!
//! ```text
//! message CompileOptionsProto {
//!   ExecutableBuildOptionsProto executable_build_options = 3;
//!   bool compile_portable_executable = 4;
//!   PrecisionConfig.Precision matrix_unit_operand_precision = 10;
//! }
//! message ExecutableBuildOptionsProto {
//!   int64 device_ordinal = 1;
//!   int64 num_replicas = 4;
//!   int64 num_partitions = 5;
//!   DeviceAssignmentProto device_assignment = 9;
//! }
//! message DeviceAssignmentProto {
//!   int32 replica_count = 1;
//!   int32 computation_count = 2;
//!   message ComputationDevice { repeated int64 replica_device_ids = 1; }
//!   repeated ComputationDevice computation_devices = 3;
//! }
//! ```
//!
//! Only these fields are written; every other field keeps its proto3 default. Level of verification: L0 (the unit tests decode the
//! bytes again with a small reader; no plugin has seen them).

/// `PrecisionConfig.Precision` of the matrix unit operands (`xla_data.proto`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatrixPrecision {
    Default = 0,
    High = 1,
    Highest = 2,
}

/// What the compiler is told about the program besides the program itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileOptions {
    /// `id` of the device the single replica of the program runs on (`PJRT_DeviceDescription_Id`), `None` for the plugin's default.
    pub device_id: Option<i64>,
    pub num_replicas: i64,
    pub num_partitions: i64,
    /// A portable executable is not tied to a device and is placed by `execute_device` at run time.
    pub portable: bool,
    /// Default precision of matrix products whose program does not name one (TPU: bf16 passes of an f32 product).
    pub matrix_precision: Option<MatrixPrecision>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            device_id: None,
            num_replicas: 1,
            num_partitions: 1,
            portable: false,
            matrix_precision: None,
        }
    }
}

/// Append `value` as a base-128 varint.
pub fn put_varint(out: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Append a field key (`field << 3 | wire type`).
fn put_key(out: &mut Vec<u8>, field: u32, wire: u8) {
    put_varint(out, u64::from(field) << 3 | u64::from(wire));
}

/// Append a varint field (wire type 0). An int64 of -1 is the ten-byte two's complement varint, as protobuf writes it.
pub fn put_int(out: &mut Vec<u8>, field: u32, value: i64) {
    put_key(out, field, 0);
    put_varint(out, value as u64);
}

/// Append a length-delimited field (wire type 2).
pub fn put_bytes(out: &mut Vec<u8>, field: u32, payload: &[u8]) {
    put_key(out, field, 2);
    put_varint(out, payload.len() as u64);
    out.extend_from_slice(payload);
}

impl CompileOptions {
    /// Serialized `CompileOptionsProto`.
    pub fn encode(&self) -> Vec<u8> {
        let mut build = Vec::new();
        // -1: "use the device assignment", what the JAX client writes
        put_int(&mut build, 1, -1);
        put_int(&mut build, 4, self.num_replicas);
        put_int(&mut build, 5, self.num_partitions);
        if let Some(id) = self.device_id {
            let mut ids = Vec::new();
            put_varint(&mut ids, id as u64);
            let mut computation = Vec::new();
            put_bytes(&mut computation, 1, &ids); // packed repeated int64
            let mut assignment = Vec::new();
            put_int(&mut assignment, 1, self.num_replicas);
            put_int(&mut assignment, 2, self.num_partitions);
            put_bytes(&mut assignment, 3, &computation);
            put_bytes(&mut build, 9, &assignment);
        }
        let mut options = Vec::new();
        put_bytes(&mut options, 3, &build);
        if self.portable {
            put_int(&mut options, 4, 1);
        }
        if let Some(precision) = self.matrix_precision {
            put_int(&mut options, 10, precision as i64);
        }
        options
    }
}

/// A minimal reader for the tests: the `(field, wire type, value)` triples of a message; length-delimited values are returned as bytes.
#[cfg(test)]
pub(crate) fn decode(mut bytes: &[u8]) -> Vec<(u32, u8, Vec<u8>)> {
    fn varint(bytes: &mut &[u8]) -> u64 {
        let (mut value, mut shift) = (0u64, 0);
        loop {
            let byte = bytes[0];
            *bytes = &bytes[1..];
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return value;
            }
            shift += 7;
        }
    }
    let mut fields = Vec::new();
    while !bytes.is_empty() {
        let key = varint(&mut bytes);
        let (field, wire) = ((key >> 3) as u32, (key & 7) as u8);
        match wire {
            0 => {
                let value = varint(&mut bytes);
                fields.push((field, wire, value.to_le_bytes().to_vec()));
            }
            2 => {
                let len = varint(&mut bytes) as usize;
                fields.push((field, wire, bytes[..len].to_vec()));
                bytes = &bytes[len..];
            }
            other => panic!("unexpected wire type {other}"),
        }
    }
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn as_i64(bytes: &[u8]) -> i64 {
        i64::from_le_bytes(bytes.try_into().expect("eight bytes"))
    }

    #[test]
    fn varints() {
        let mut out = Vec::new();
        put_varint(&mut out, 0);
        put_varint(&mut out, 300);
        assert_eq!(out, [0x00, 0xac, 0x02]);
        let mut minus_one = Vec::new();
        put_int(&mut minus_one, 1, -1);
        assert_eq!(minus_one.len(), 1 + 10);
        assert_eq!(minus_one[0], 0x08);
    }

    #[test]
    fn default_options_name_one_replica_and_one_partition() {
        let bytes = CompileOptions::default().encode();
        let top = decode(&bytes);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 3);
        let build = decode(&top[0].2);
        let get = |field: u32| build.iter().find(|f| f.0 == field).map(|f| as_i64(&f.2));
        assert_eq!(get(1), Some(-1));
        assert_eq!(get(4), Some(1));
        assert_eq!(get(5), Some(1));
        assert_eq!(get(9), None);
    }

    #[test]
    fn device_assignment_and_flags() {
        let options = CompileOptions {
            device_id: Some(5),
            portable: true,
            matrix_precision: Some(MatrixPrecision::Highest),
            ..Default::default()
        };
        let top = decode(&options.encode());
        let field = |n: u32| top.iter().find(|f| f.0 == n);
        assert_eq!(as_i64(&field(4).expect("portable").2), 1);
        assert_eq!(as_i64(&field(10).expect("precision").2), 2);
        let build = decode(&field(3).expect("build options").2);
        let assignment = decode(&build.iter().find(|f| f.0 == 9).expect("assignment").2);
        assert_eq!(
            as_i64(&assignment.iter().find(|f| f.0 == 1).expect("replicas").2),
            1
        );
        assert_eq!(
            as_i64(
                &assignment
                    .iter()
                    .find(|f| f.0 == 2)
                    .expect("computations")
                    .2
            ),
            1
        );
        let computation = decode(
            &assignment
                .iter()
                .find(|f| f.0 == 3)
                .expect("computation devices")
                .2,
        );
        // one packed field holding the single varint 5
        assert_eq!(computation, vec![(1, 2, vec![5])]);
    }
}
