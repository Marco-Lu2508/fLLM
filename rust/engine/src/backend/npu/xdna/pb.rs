//! Minimal protobuf wire-format writer (package 11).
//!
//! The NPU studies need to emit two small protobuf messages without a protobuf dependency: an ONNX `ModelProto` (AMD XDNA, through
//! ONNX Runtime) and a Core ML `Model` with an ML Program (Apple Neural Engine). Only the encoder side is needed, and only the wire types
//! used by those schemas: varint, length-delimited and 32-bit fixed. No field is ever skipped by this writer: the callers decide which
//! proto3 default values to omit. Pure Rust, no hardware, compiled and unit-tested on every host.

/// A protobuf message under construction (the bytes of its fields in the order they were added).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Message {
    buf: Vec<u8>,
}

/// Wire type 0 (varint).
const WIRE_VARINT: u64 = 0;
/// Wire type 2 (length-delimited: strings, bytes, nested messages, packed repeated fields).
const WIRE_LEN: u64 = 2;
/// Wire type 5 (32-bit fixed: `float`, `fixed32`).
const WIRE_FIXED32: u64 = 5;

fn put_varint(buf: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        buf.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    buf.push(v as u8);
}

impl Message {
    pub fn new() -> Self {
        Self::default()
    }

    fn tag(&mut self, field: u32, wire: u64) {
        put_varint(&mut self.buf, (u64::from(field) << 3) | wire);
    }

    /// `uint32`/`uint64`/`bool`/enum field.
    pub fn varint(&mut self, field: u32, value: u64) -> &mut Self {
        self.tag(field, WIRE_VARINT);
        put_varint(&mut self.buf, value);
        self
    }

    /// `int32`/`int64` field (negative values use the 10-byte two's complement form, as protobuf specifies).
    pub fn int(&mut self, field: u32, value: i64) -> &mut Self {
        self.varint(field, value as u64)
    }

    pub fn boolean(&mut self, field: u32, value: bool) -> &mut Self {
        self.varint(field, u64::from(value))
    }

    /// `float` field.
    pub fn float(&mut self, field: u32, value: f32) -> &mut Self {
        self.tag(field, WIRE_FIXED32);
        self.buf.extend_from_slice(&value.to_le_bytes());
        self
    }

    /// `bytes` field.
    pub fn bytes(&mut self, field: u32, value: &[u8]) -> &mut Self {
        self.tag(field, WIRE_LEN);
        put_varint(&mut self.buf, value.len() as u64);
        self.buf.extend_from_slice(value);
        self
    }

    /// `string` field.
    pub fn string(&mut self, field: u32, value: &str) -> &mut Self {
        self.bytes(field, value.as_bytes())
    }

    /// Embedded message field.
    pub fn message(&mut self, field: u32, value: &Message) -> &mut Self {
        self.bytes(field, &value.buf)
    }

    /// `repeated int64` in packed form.
    pub fn packed_ints(&mut self, field: u32, values: &[i64]) -> &mut Self {
        let mut payload = Vec::new();
        for &v in values {
            put_varint(&mut payload, v as u64);
        }
        self.bytes(field, &payload)
    }

    /// `repeated float` in packed form.
    pub fn packed_floats(&mut self, field: u32, values: &[f32]) -> &mut Self {
        let mut payload = Vec::with_capacity(values.len() * 4);
        for v in values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        self.bytes(field, &payload)
    }

    /// One entry of a `map<string, Message>` field: an embedded message with the key as field 1 and the value as field 2.
    pub fn map_entry(&mut self, field: u32, key: &str, value: &Message) -> &mut Self {
        let mut entry = Message::new();
        entry.string(1, key).message(2, value);
        self.message(field, &entry)
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn len(&self) -> usize {
        self.buf.len()
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.buf
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_examples_from_the_protobuf_encoding_guide() {
        // field 1 = 150  ->  08 96 01
        let mut m = Message::new();
        m.varint(1, 150);
        assert_eq!(m.as_bytes(), &[0x08, 0x96, 0x01]);
        // 300 as a bare varint -> AC 02
        let mut buf = Vec::new();
        put_varint(&mut buf, 300);
        assert_eq!(buf, [0xAC, 0x02]);
    }

    #[test]
    fn string_and_nested_message() {
        // field 2 = "testing"  ->  12 07 74 65 73 74 69 6e 67
        let mut m = Message::new();
        m.string(2, "testing");
        assert_eq!(
            m.as_bytes(),
            &[0x12, 0x07, b't', b'e', b's', b't', b'i', b'n', b'g']
        );
        // field 3 = embedded message { field 1 = 150 }  ->  1a 03 08 96 01
        let mut inner = Message::new();
        inner.varint(1, 150);
        let mut outer = Message::new();
        outer.message(3, &inner);
        assert_eq!(outer.as_bytes(), &[0x1a, 0x03, 0x08, 0x96, 0x01]);
    }

    #[test]
    fn negative_int_takes_ten_bytes() {
        let mut m = Message::new();
        m.int(1, -1);
        assert_eq!(m.len(), 1 + 10);
        assert_eq!(
            &m.as_bytes()[1..],
            &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01]
        );
    }

    #[test]
    fn packed_fields_and_float() {
        let mut m = Message::new();
        m.packed_ints(1, &[3, 270, 86942]);
        // guide example (field 4 there): 22 06 03 8E 02 9E A7 05; here field 1 -> tag 0x0a
        assert_eq!(
            m.as_bytes(),
            &[0x0a, 0x06, 0x03, 0x8e, 0x02, 0x9e, 0xa7, 0x05]
        );
        let mut f = Message::new();
        f.float(2, 1.0);
        assert_eq!(f.as_bytes(), &[0x15, 0x00, 0x00, 0x80, 0x3f]);
    }

    #[test]
    fn map_entry_is_key_one_value_two() {
        let mut value = Message::new();
        value.varint(1, 7);
        let mut m = Message::new();
        m.map_entry(4, "k", &value);
        // 22 (field 4, len) 07 | 0a 01 'k' | 12 02 08 07
        assert_eq!(
            m.as_bytes(),
            &[0x22, 0x07, 0x0a, 0x01, b'k', 0x12, 0x02, 0x08, 0x07]
        );
    }
}
