//! Pure description of a compute pipeline request: the kernel name plus its function-constant values.
//!
//! Function constants (`constant uint N [[function_constant(0)]]` in MSL) specialise a kernel at pipeline creation;
//! every distinct set of values is a distinct `MTLComputePipelineState`, cached under a [`PipelineKey`].

use std::fmt;

/// Value of one function constant. The variant fixes the Metal data type of the constant.
#[derive(Clone, Copy, Debug)]
pub enum ConstantValue {
    Bool(bool),
    I32(i32),
    U32(u32),
    F32(f32),
}

impl ConstantValue {
    fn tag(&self) -> u8 {
        match self {
            ConstantValue::Bool(_) => 0,
            ConstantValue::I32(_) => 1,
            ConstantValue::U32(_) => 2,
            ConstantValue::F32(_) => 3,
        }
    }

    /// The value as a 32-bit pattern (floats by their bits, so `NaN` keys are stable and `-0.0 != 0.0`).
    fn bits(&self) -> u32 {
        match *self {
            ConstantValue::Bool(v) => u32::from(v),
            ConstantValue::I32(v) => v as u32,
            ConstantValue::U32(v) => v,
            ConstantValue::F32(v) => v.to_bits(),
        }
    }
}

impl PartialEq for ConstantValue {
    fn eq(&self, other: &Self) -> bool {
        self.tag() == other.tag() && self.bits() == other.bits()
    }
}

impl Eq for ConstantValue {}

impl std::hash::Hash for ConstantValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tag().hash(state);
        self.bits().hash(state);
    }
}

impl fmt::Display for ConstantValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstantValue::Bool(v) => write!(f, "{v}"),
            ConstantValue::I32(v) => write!(f, "{v}i"),
            ConstantValue::U32(v) => write!(f, "{v}u"),
            ConstantValue::F32(v) => write!(f, "{v}f"),
        }
    }
}

/// One `[[function_constant(index)]]` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FunctionConstant {
    pub index: u32,
    pub value: ConstantValue,
}

impl FunctionConstant {
    pub fn bool(index: u32, value: bool) -> Self {
        Self {
            index,
            value: ConstantValue::Bool(value),
        }
    }
    pub fn i32(index: u32, value: i32) -> Self {
        Self {
            index,
            value: ConstantValue::I32(value),
        }
    }
    pub fn u32(index: u32, value: u32) -> Self {
        Self {
            index,
            value: ConstantValue::U32(value),
        }
    }
    pub fn f32(index: u32, value: f32) -> Self {
        Self {
            index,
            value: ConstantValue::F32(value),
        }
    }
}

/// Cache key of a pipeline: kernel name and constants sorted by index.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    function: String,
    constants: Vec<FunctionConstant>,
}

impl PipelineKey {
    /// Fails when two constants share an index (the request would be ambiguous).
    pub fn new(function: &str, constants: &[FunctionConstant]) -> Result<Self, String> {
        let mut constants = constants.to_vec();
        constants.sort_by_key(|c| c.index);
        if let Some(pair) = constants.windows(2).find(|w| w[0].index == w[1].index) {
            return Err(format!(
                "function constant {} of '{function}' is set twice",
                pair[0].index
            ));
        }
        Ok(Self {
            function: function.to_owned(),
            constants,
        })
    }

    pub fn function(&self) -> &str {
        &self.function
    }

    pub fn constants(&self) -> &[FunctionConstant] {
        &self.constants
    }
}

impl fmt::Display for PipelineKey {
    /// `name` or `name[0=4u,1=true]`, used for labels and error messages.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.function)?;
        if self.constants.is_empty() {
            return Ok(());
        }
        f.write_str("[")?;
        for (i, c) in self.constants.iter().enumerate() {
            if i > 0 {
                f.write_str(",")?;
            }
            write!(f, "{}={}", c.index, c.value)?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn order_of_constants_does_not_matter() {
        let a = PipelineKey::new(
            "k",
            &[FunctionConstant::u32(1, 8), FunctionConstant::bool(0, true)],
        )
        .unwrap();
        let b = PipelineKey::new(
            "k",
            &[FunctionConstant::bool(0, true), FunctionConstant::u32(1, 8)],
        )
        .unwrap();
        assert_eq!(a, b);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn values_and_types_distinguish_keys() {
        let u = PipelineKey::new("k", &[FunctionConstant::u32(0, 1)]).unwrap();
        let i = PipelineKey::new("k", &[FunctionConstant::i32(0, 1)]).unwrap();
        let other = PipelineKey::new("k", &[FunctionConstant::u32(0, 2)]).unwrap();
        let name = PipelineKey::new("j", &[FunctionConstant::u32(0, 1)]).unwrap();
        assert!(u != i && u != other && u != name);
        let nan_a = FunctionConstant::f32(0, f32::NAN);
        assert_eq!(nan_a, nan_a, "float constants compare by bits");
        assert_ne!(
            FunctionConstant::f32(0, 0.0),
            FunctionConstant::f32(0, -0.0)
        );
    }

    #[test]
    fn duplicate_index_is_rejected() {
        assert!(PipelineKey::new(
            "k",
            &[
                FunctionConstant::u32(3, 1),
                FunctionConstant::bool(3, false)
            ]
        )
        .is_err());
    }

    #[test]
    fn display_is_readable() {
        let key = PipelineKey::new(
            "gemv_q4_0",
            &[FunctionConstant::u32(0, 4), FunctionConstant::bool(1, true)],
        )
        .unwrap();
        assert_eq!(key.to_string(), "gemv_q4_0[0=4u,1=true]");
        assert_eq!(PipelineKey::new("plain", &[]).unwrap().to_string(), "plain");
    }
}
