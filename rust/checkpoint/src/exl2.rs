//! Checkpoint (unshuffled) EXL2 packing. See docs/exl-formats.md for provenance.
use crate::safetensors::{Payload, Tensor};
use half::f16;
use std::collections::HashMap;

#[derive(Clone, Debug)]
struct Group {
    bits: usize,
    packed_start: usize,
    input_start: usize,
}

#[derive(Clone, Debug)]
pub struct Exl2 {
    weight: Payload,
    scales: Payload,
    maxima: Payload,
    inverse: Vec<usize>,
    groups: Vec<Group>,
    pub(crate) input: usize,
    pub(crate) output: usize,
    elements: usize,
}

pub(crate) fn take(
    g: &mut HashMap<String, Tensor>,
    base: &str,
    key: &str,
) -> Result<Tensor, String> {
    g.remove(key)
        .ok_or_else(|| format!("EXL matrix {base}: missing tensor {base}.{key}"))
}

pub(crate) fn shape(t: &Tensor, dtype: &str, dims: &[usize]) -> Result<(), String> {
    let expected: Vec<u64> = dims.iter().map(|&n| n as u64).collect();
    let width = match dtype {
        "I16" | "F16" => 2,
        "I32" => 4,
        _ => return Err("unsupported EXL dtype".into()),
    };
    let bytes = dims
        .iter()
        .try_fold(width, |n: usize, &d| n.checked_mul(d))
        .ok_or_else(|| format!("tensor {}: shape overflow", t.name))?;
    if t.dtype != dtype || t.shape != expected || t.payload.as_slice().len() != bytes {
        return Err(format!(
            "tensor {}: expected {dtype}{expected:?}, got {}{:?}",
            t.name, t.dtype, t.shape
        ));
    }
    Ok(())
}

pub(crate) fn dim(t: &Tensor, i: usize) -> Result<usize, String> {
    t.shape
        .get(i)
        .copied()
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("tensor {}: invalid dimension {i}", t.name))
}

pub(crate) fn u16_at(data: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([data[2 * i], data[2 * i + 1]])
}
pub(crate) fn u32_at(data: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([
        data[4 * i],
        data[4 * i + 1],
        data[4 * i + 2],
        data[4 * i + 3],
    ])
}
pub(crate) fn half_round(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

pub(crate) fn split(
    all: Vec<Tensor>,
    declared: Option<bool>,
    map: &dyn Fn(&str) -> Result<Option<String>, String>,
) -> Result<crate::convert::PackedSplit, String> {
    let mut dense = Vec::new();
    let mut groups = std::collections::BTreeMap::<String, HashMap<String, Tensor>>::new();
    for t in all {
        let Some((base, suffix)) = t.name.rsplit_once('.') else {
            dense.push(t);
            continue;
        };
        if !matches!(
            suffix,
            "q_weight"
                | "q_scale"
                | "q_scale_max"
                | "q_groups"
                | "q_invperm"
                | "q_perm"
                | "trellis"
                | "suh"
                | "svh"
                | "su"
                | "sv"
                | "mcg"
                | "mul1"
                | "scale"
        ) {
            dense.push(t);
            continue;
        }
        let base = base.to_owned();
        let suffix = suffix.to_owned();
        if groups
            .entry(base.clone())
            .or_default()
            .insert(suffix.clone(), t)
            .is_some()
        {
            return Err(format!("EXL {base}: duplicate tensor {suffix}"));
        }
    }
    let mut packed = Vec::new();
    for (base, mut g) in groups {
        let is_exl3 = g.contains_key("trellis");
        if declared.is_some_and(|v| v != is_exl3) {
            return Err(format!(
                "EXL {base}: tensors disagree with quantization_config format"
            ));
        }
        let data = if is_exl3 {
            crate::packed_quant::PackedData::Exl3(crate::exl3::Exl3::read(&base, &mut g)?)
        } else {
            crate::packed_quant::PackedData::Exl2(Exl2::read(&base, &mut g)?)
        };
        let (input, output) = match &data {
            crate::packed_quant::PackedData::Exl2(t) => (t.input, t.output),
            crate::packed_quant::PackedData::Exl3(t) => (t.input, t.output),
            _ => return Err("internal EXL reader selection error".into()),
        };
        if !g.is_empty() {
            return Err(format!(
                "EXL {base}: unsupported components {:?}",
                g.keys().collect::<Vec<_>>()
            ));
        }
        let name = format!("{base}.weight");
        let mapped = map(&name)?
            .ok_or_else(|| format!("EXL tensor {name}: unsupported architecture mapping"))?;
        let payload = Payload::Owned(std::sync::Arc::new(Vec::new()), 0..0);
        packed.push((mapped, data, vec![output as u64, input as u64], payload));
    }
    if packed.is_empty() {
        return Err("EXL quantization_config has no supported matrix tensors".into());
    }
    Ok((dense, packed))
}

impl Exl2 {
    pub(crate) fn read(base: &str, g: &mut HashMap<String, Tensor>) -> Result<Self, String> {
        let weight = take(g, base, "q_weight")?;
        let scales = take(g, base, "q_scale")?;
        let maxima = take(g, base, "q_scale_max")?;
        let groups = take(g, base, "q_groups")?;
        let inv = take(g, base, "q_invperm")?;
        let input = dim(&inv, 0)?;
        let output = dim(&weight, 1)?;
        let packed_rows = dim(&weight, 0)?;
        let count = dim(&maxima, 0)?;
        let elements = input
            .checked_mul(output)
            .ok_or_else(|| format!("EXL2 {base}: matrix overflow"))?;
        shape(&weight, "I32", &[packed_rows, output])?;
        shape(&inv, "I32", &[input])?;
        shape(&maxima, "F16", &[count])?;
        shape(
            &groups,
            "I16",
            &[count.checked_mul(2).ok_or("EXL2 group count overflow")?],
        )?;
        if !output.is_multiple_of(8) {
            return Err(format!("EXL2 {base}: output size must be divisible by 8"));
        }
        shape(&scales, "I32", &[count, output / 8])?;
        let mut inverse = Vec::with_capacity(input);
        let mut seen = vec![false; input];
        for i in 0..input {
            let p = u32_at(inv.payload.as_slice(), i) as usize;
            if p >= input || seen[p] {
                return Err(format!("tensor {}: invalid permutation at {i}", inv.name));
            }
            seen[p] = true;
            inverse.push(p);
        }
        if let Some(perm) = g.remove("q_perm") {
            shape(&perm, "I32", &[input])?;
            for (original, &packed) in inverse.iter().enumerate() {
                if u32_at(perm.payload.as_slice(), packed) as usize != original {
                    return Err(format!(
                        "tensor {}: inconsistent with {}",
                        perm.name, inv.name
                    ));
                }
            }
        }
        let mut parsed = Vec::with_capacity(count);
        let mut rows = 0usize;
        for i in 0..count {
            let bits = u16_at(groups.payload.as_slice(), i * 2) as usize;
            let start = u16_at(groups.payload.as_slice(), i * 2 + 1) as usize;
            let end = if i + 1 == count {
                packed_rows
            } else {
                u16_at(groups.payload.as_slice(), i * 2 + 3) as usize
            };
            if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
                return Err(format!(
                    "tensor {}: unsupported group width {bits}",
                    groups.name
                ));
            }
            if (i == 0 && start != 0) || end <= start || end > packed_rows || rows >= input {
                return Err(format!(
                    "tensor {}: invalid packed row range in group {i}",
                    groups.name
                ));
            }
            let packed_bits = (end - start)
                .checked_mul(32)
                .ok_or("EXL2 group size overflow")?;
            if i + 1 < count && !packed_bits.is_multiple_of(bits) {
                return Err(format!(
                    "tensor {}: non-integral input group {i}",
                    groups.name
                ));
            }
            let n = (packed_bits / bits).min(input - rows);
            if n == 0 || (n * bits).div_ceil(32) != end - start {
                return Err(format!(
                    "tensor {}: group {i} exceeds input dimension",
                    groups.name
                ));
            }
            let max = f16::from_bits(u16_at(maxima.payload.as_slice(), i)).to_f32();
            if !max.is_finite() || max < 0.0 {
                return Err(format!(
                    "tensor {}: invalid scale in group {i}",
                    maxima.name
                ));
            }
            parsed.push(Group {
                bits,
                packed_start: start,
                input_start: rows,
            });
            rows += n;
        }
        if rows != input {
            return Err(format!(
                "tensor {}: groups cover {rows} inputs, expected {input}",
                groups.name
            ));
        }
        Ok(Self {
            weight: weight.payload,
            scales: scales.payload,
            maxima: maxima.payload,
            inverse,
            groups: parsed,
            input,
            output,
            elements,
        })
    }

    pub(crate) fn element_count(&self) -> usize {
        self.elements
    }

    pub(crate) fn decode_chunk(&self, start: usize, dst: &mut [f32]) -> Result<(), String> {
        if start
            .checked_add(dst.len())
            .is_none_or(|end| end > self.elements)
        {
            return Err("EXL2 decode range out of bounds".into());
        }
        for (i, value) in dst.iter_mut().enumerate() {
            let flat = start + i;
            let out = flat / self.input;
            let row = self.inverse[flat % self.input];
            let group = self.groups.partition_point(|g| g.input_start <= row) - 1;
            let g = &self.groups[group];
            let bit = (row - g.input_start) * g.bits;
            let word = g.packed_start + bit / 32;
            let shift = bit % 32;
            let mut code = u32_at(self.weight.as_slice(), word * self.output + out) >> shift;
            if shift + g.bits > 32 {
                code |=
                    u32_at(self.weight.as_slice(), (word + 1) * self.output + out) << (32 - shift);
            }
            code &= (1 << g.bits) - 1;
            let nibble = (u32_at(self.scales.as_slice(), group * (self.output / 8) + out / 8)
                >> (4 * (out % 8)))
                & 15;
            let max =
                half_round(f16::from_bits(u16_at(self.maxima.as_slice(), group)).to_f32() / 256.0);
            let scale = half_round(((nibble + 1) * (nibble + 1)) as f32 * max);
            *value = half_round((code as i32 - (1 << (g.bits - 1))) as f32 * scale);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    fn tensor(key: &str, dtype: &str, shape: Vec<u64>, data: Vec<u8>) -> Tensor {
        let len = data.len();
        Tensor {
            name: format!("matrix.{key}"),
            dtype: dtype.into(),
            shape,
            payload: Payload::Owned(Arc::new(data), 0..len),
        }
    }
    fn encoded() -> HashMap<String, Tensor> {
        let widths = [2usize, 3, 4, 5, 6, 8];
        let mut words = Vec::<u32>::new();
        let mut pairs = Vec::<u16>::new();
        for bits in widths {
            pairs.extend([bits as u16, (words.len() / 8) as u16]);
            let offset = words.len();
            words.resize(offset + bits * 8, 0);
            for row in 0..32 {
                for out in 0..8 {
                    let code = ((row + out) as u32) & ((1 << bits) - 1);
                    let bit = row * bits;
                    words[offset + (bit / 32) * 8 + out] |= code << (bit % 32);
                    if bit % 32 + bits > 32 {
                        words[offset + (bit / 32 + 1) * 8 + out] |= code >> (32 - bit % 32);
                    }
                }
            }
        }
        let rows = words.len() / 8;
        let tensors = [
            tensor(
                "q_weight",
                "I32",
                vec![rows as u64, 8],
                words.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            tensor(
                "q_groups",
                "I16",
                vec![12],
                pairs.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            tensor("q_scale", "I32", vec![6, 1], vec![0; 24]),
            tensor(
                "q_scale_max",
                "F16",
                vec![6],
                (0..6)
                    .flat_map(|_| f16::from_f32(256.0).to_bits().to_le_bytes())
                    .collect(),
            ),
            tensor(
                "q_invperm",
                "I32",
                vec![192],
                (0..192u32).rev().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        ];
        tensors
            .into_iter()
            .map(|t| (t.name.rsplit('.').next().unwrap().into(), t))
            .collect()
    }
    #[test]
    fn mixed_widths_cross_word_boundaries_and_inverse_permutation() {
        let reader = Exl2::read("matrix", &mut encoded()).unwrap();
        let mut decoded = vec![0.0; reader.element_count()];
        reader.decode_chunk(0, &mut decoded).unwrap();
        for out in 0..8 {
            for input in 0..192 {
                let row = 191 - input;
                let bits = [2, 3, 4, 5, 6, 8][row / 32];
                assert_eq!(
                    decoded[out * 192 + input],
                    (((row % 32 + out) & ((1 << bits) - 1)) as i32 - (1 << (bits - 1))) as f32
                );
            }
        }
        let mut slice = [0.0; 41];
        reader.decode_chunk(181, &mut slice).unwrap();
        assert_eq!(&slice, &decoded[181..222]);
        assert!(reader
            .decode_chunk(reader.element_count(), &mut slice)
            .is_err());
    }
    #[test]
    fn corrupt_permutation_and_missing_component_are_errors() {
        let mut g = encoded();
        g.remove("q_scale");
        assert!(Exl2::read("matrix", &mut g)
            .unwrap_err()
            .contains("matrix.q_scale"));
        let mut g = encoded();
        g.insert(
            "q_invperm".into(),
            tensor("q_invperm", "I32", vec![192], vec![0; 768]),
        );
        assert!(Exl2::read("matrix", &mut g)
            .unwrap_err()
            .contains("permutation"));
    }
}
