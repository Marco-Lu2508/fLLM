//! Google TPU through PJRT / StableHLO: an emitter for StableHLO programs in MLIR text form.
//!
//! [`Func`] builds the body of one function. Every method that adds an operation checks the shapes and element types of its
//! operands (the rules of the StableHLO specification), allocates the SSA name of the result and records the operation in the generic
//! MLIR syntax (`"stablehlo.add"(%a, %b) : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>`), which every MLIR parser accepts and
//! which does not depend on the pretty-printing of single operations. A violated shape rule is a bug of the emitter and panics with the
//! operation and the shapes; the callers build programs from fixed shape keys, and the unit tests build every kind of program.
//!
//! - [`ops`]: composite operations on top of [`Func`] (softmax, RMSNorm, rotary embedding, linear layers, attention, …).
//! - [`validate`]: a text-level structure check of finished programs (bracket balance, SSA names, operand types).
//!
//! Level of verification: L0 (the programs were parsed and verified by an MLIR StableHLO parser on a developer machine; no TPU has
//! compiled or run them).

pub mod ops;
pub mod validate;

#[cfg(test)]
mod tests;

use std::fmt::{self, Write as _};

/// Element types of tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    Pred,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    F16,
    Bf16,
    F32,
}

impl DType {
    /// The MLIR spelling (`f32`, `bf16`, `i32`, `ui8`, `i1` for booleans).
    pub fn name(self) -> &'static str {
        match self {
            Self::Pred => "i1",
            Self::I8 => "i8",
            Self::U8 => "ui8",
            Self::I16 => "i16",
            Self::U16 => "ui16",
            Self::I32 => "i32",
            Self::U32 => "ui32",
            Self::I64 => "i64",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }

    /// Storage bytes of one element (a boolean takes a byte).
    pub fn bytes(self) -> usize {
        match self {
            Self::Pred | Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::Bf16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }

    pub fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F32)
    }

    pub fn is_int(self) -> bool {
        !self.is_float() && self != Self::Pred
    }

    /// Bit pattern of positive infinity in this float type.
    fn inf_bits(self) -> u32 {
        match self {
            Self::F32 => 0x7F80_0000,
            Self::F16 => 0x7C00,
            Self::Bf16 => 0x7F80,
            other => panic!("{other:?} has no infinity"),
        }
    }
}

/// The type of a tensor: dimensions and element type (`tensor<2x3xf32>`; a scalar is `tensor<f32>`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    pub dims: Vec<i64>,
    pub dtype: DType,
}

impl TensorType {
    pub fn new(dims: &[i64], dtype: DType) -> Self {
        Self {
            dims: dims.to_vec(),
            dtype,
        }
    }

    /// Number of elements.
    pub fn elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("tensor<")?;
        for d in &self.dims {
            write!(f, "{d}x")?;
        }
        write!(f, "{}>", self.dtype.name())
    }
}

/// An SSA value with its type.
#[derive(Debug, Clone, PartialEq)]
pub struct Value {
    pub name: String,
    pub ty: TensorType,
}

impl Value {
    pub fn dims(&self) -> &[i64] {
        &self.ty.dims
    }

    pub fn dtype(&self) -> DType {
        self.ty.dtype
    }
}

/// Decimal text of a float constant in the syntax MLIR requires (always with a decimal point or an exponent).
pub fn float_literal(v: f64) -> String {
    assert!(
        v.is_finite(),
        "float_literal: {v} is not finite (use infinity constants)"
    );
    if v != 0.0 && (v.abs() < 1e-5 || v.abs() >= 1e15) {
        let s = format!("{v:e}");
        return match s.split_once('e') {
            Some((mantissa, exponent)) if !mantissa.contains('.') => {
                format!("{mantissa}.0e{exponent}")
            }
            _ => s,
        };
    }
    let s = format!("{v}");
    if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

/// `[1, 2, 3]` for attribute lists.
fn list(values: &[i64]) -> String {
    let mut s = String::from("[");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{v}");
    }
    s.push(']');
    s
}

/// `array<i64: 1, 2>` for dense array attributes (`array<i64>` when empty).
fn array(values: &[i64]) -> String {
    if values.is_empty() {
        return "array<i64>".to_owned();
    }
    let mut s = String::from("array<i64: ");
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        let _ = write!(s, "{v}");
    }
    s.push('>');
    s
}

fn bad(op: &str, message: impl fmt::Display) -> ! {
    panic!("stablehlo emitter: {op}: {message}")
}

/// Combiner of a reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reducer {
    Sum,
    Max,
    Min,
}

impl Reducer {
    fn op(self) -> &'static str {
        match self {
            Self::Sum => "stablehlo.add",
            Self::Max => "stablehlo.maximum",
            Self::Min => "stablehlo.minimum",
        }
    }
}

/// Comparison directions of `compare`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl Cmp {
    fn name(self) -> &'static str {
        match self {
            Self::Eq => "EQ",
            Self::Ne => "NE",
            Self::Lt => "LT",
            Self::Le => "LE",
            Self::Gt => "GT",
            Self::Ge => "GE",
        }
    }
}

/// Precision of matrix products (`#stablehlo<precision …>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    Default,
    High,
    Highest,
}

impl Precision {
    fn name(self) -> &'static str {
        match self {
            Self::Default => "DEFAULT",
            Self::High => "HIGH",
            Self::Highest => "HIGHEST",
        }
    }
}

/// Dimension numbers of a `dot_general`.
#[derive(Debug, Clone, Default)]
pub struct DotDims {
    pub lhs_batch: Vec<i64>,
    pub rhs_batch: Vec<i64>,
    pub lhs_contract: Vec<i64>,
    pub rhs_contract: Vec<i64>,
}

/// One function of a module under construction.
pub struct Func {
    name: String,
    args: Vec<(Value, Option<String>)>,
    lines: Vec<String>,
    counter: usize,
}

impl Func {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            args: Vec::new(),
            lines: Vec::new(),
            counter: 0,
        }
    }

    fn fresh(&mut self) -> String {
        self.counter += 1;
        format!("%v{}", self.counter)
    }

    /// Add a parameter and return its value.
    pub fn param(&mut self, dims: &[i64], dtype: DType) -> Value {
        let value = Value {
            name: format!("%arg{}", self.args.len()),
            ty: TensorType::new(dims, dtype),
        };
        self.args.push((value.clone(), None));
        value
    }

    /// Attach an attribute (`tf.aliasing_output = 0 : i32`) to a parameter.
    pub fn set_param_attr(&mut self, param: &Value, attr: String) {
        match self.args.iter_mut().find(|(v, _)| v.name == param.name) {
            Some(entry) => entry.1 = Some(attr),
            None => bad(
                "set_param_attr",
                format!("{} is not a parameter", param.name),
            ),
        }
    }

    /// Number of parameters so far.
    pub fn param_count(&self) -> usize {
        self.args.len()
    }

    /// Record an operation in generic form and return its single result.
    pub fn op(
        &mut self,
        name: &str,
        operands: &[&Value],
        attrs: &str,
        result: TensorType,
    ) -> Value {
        let out = self.fresh();
        let operand_names: Vec<&str> = operands.iter().map(|v| v.name.as_str()).collect();
        let operand_types: Vec<String> = operands.iter().map(|v| v.ty.to_string()).collect();
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{attrs}}}")
        };
        self.lines.push(format!(
            "{out} = \"{name}\"({}){attrs} : ({}) -> {result}",
            operand_names.join(", "),
            operand_types.join(", ")
        ));
        Value {
            name: out,
            ty: result,
        }
    }

    // ----- constants -------------------------------------------------------------------------------------------------------------

    /// A constant of shape `dims` filled with `value` (a splat).
    pub fn constant(&mut self, value: f64, dims: &[i64], dtype: DType) -> Value {
        let ty = TensorType::new(dims, dtype);
        let literal = match dtype {
            DType::Pred => (if value != 0.0 { "true" } else { "false" }).to_owned(),
            d if d.is_float() => float_literal(value),
            _ => format!("{}", value as i64),
        };
        let out = self.fresh();
        self.lines.push(format!(
            "{out} = \"stablehlo.constant\"() {{value = dense<{literal}> : {ty}}} : () -> {ty}"
        ));
        Value { name: out, ty }
    }

    /// A splat constant holding positive or negative infinity.
    pub fn infinity(&mut self, negative: bool, dims: &[i64], dtype: DType) -> Value {
        let ty = TensorType::new(dims, dtype);
        let bits = dtype.inf_bits();
        let literal = match (dtype, negative) {
            (DType::F32, true) => format!("0x{:08X}", bits | 0x8000_0000),
            (DType::F32, false) => format!("0x{bits:08X}"),
            (_, true) => format!("0x{:04X}", bits | 0x8000),
            (_, false) => format!("0x{bits:04X}"),
        };
        let out = self.fresh();
        self.lines.push(format!(
            "{out} = \"stablehlo.constant\"() {{value = dense<{literal}> : {ty}}} : () -> {ty}"
        ));
        Value { name: out, ty }
    }

    /// `iota` along `dim` of an array of shape `dims`.
    pub fn iota(&mut self, dims: &[i64], dim: usize, dtype: DType) -> Value {
        assert!(dim < dims.len(), "iota: dimension {dim} of {dims:?}");
        let ty = TensorType::new(dims, dtype);
        let out = self.fresh();
        self.lines.push(format!(
            "{out} = \"stablehlo.iota\"() {{iota_dimension = {dim} : i64}} : () -> {ty}"
        ));
        Value { name: out, ty }
    }

    // ----- elementwise -----------------------------------------------------------------------------------------------------------

    pub fn unary(&mut self, op: &str, x: &Value) -> Value {
        self.op(&format!("stablehlo.{op}"), &[x], "", x.ty.clone())
    }

    pub fn neg(&mut self, x: &Value) -> Value {
        self.unary("negate", x)
    }

    pub fn abs(&mut self, x: &Value) -> Value {
        self.unary("abs", x)
    }

    pub fn exp(&mut self, x: &Value) -> Value {
        self.unary("exponential", x)
    }

    pub fn tanh(&mut self, x: &Value) -> Value {
        self.unary("tanh", x)
    }

    pub fn sqrt(&mut self, x: &Value) -> Value {
        self.unary("sqrt", x)
    }

    pub fn rsqrt(&mut self, x: &Value) -> Value {
        self.unary("rsqrt", x)
    }

    pub fn logistic(&mut self, x: &Value) -> Value {
        self.unary("logistic", x)
    }

    /// `x != x`: true where `x` is NaN.
    pub fn is_nan(&mut self, x: &Value) -> Value {
        self.compare(x, x, Cmp::Ne)
    }

    /// Logical negation of a boolean tensor.
    pub fn not(&mut self, x: &Value) -> Value {
        assert_eq!(x.dtype(), DType::Pred, "not: needs booleans");
        self.unary("not", x)
    }

    pub fn binary(&mut self, op: &str, a: &Value, b: &Value) -> Value {
        if a.ty != b.ty {
            bad(op, format!("operand types differ: {} and {}", a.ty, b.ty));
        }
        self.op(&format!("stablehlo.{op}"), &[a, b], "", a.ty.clone())
    }

    pub fn add(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("add", a, b)
    }

    pub fn sub(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("subtract", a, b)
    }

    pub fn mul(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("multiply", a, b)
    }

    pub fn div(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("divide", a, b)
    }

    pub fn max(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("maximum", a, b)
    }

    pub fn min(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("minimum", a, b)
    }

    pub fn and(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("and", a, b)
    }

    pub fn or(&mut self, a: &Value, b: &Value) -> Value {
        self.binary("or", a, b)
    }

    /// `a OP b` with a scalar constant `b` of the same type broadcast to the shape of `a`.
    pub fn add_const(&mut self, a: &Value, c: f64) -> Value {
        let k = self.constant(c, a.dims(), a.dtype());
        self.add(a, &k)
    }

    pub fn mul_const(&mut self, a: &Value, c: f64) -> Value {
        let k = self.constant(c, a.dims(), a.dtype());
        self.mul(a, &k)
    }

    pub fn compare(&mut self, a: &Value, b: &Value, dir: Cmp) -> Value {
        if a.ty != b.ty {
            bad(
                "compare",
                format!("operand types differ: {} and {}", a.ty, b.ty),
            );
        }
        let ty = TensorType::new(a.dims(), DType::Pred);
        self.op(
            "stablehlo.compare",
            &[a, b],
            &format!(
                "comparison_direction = #stablehlo<comparison_direction {}>",
                dir.name()
            ),
            ty,
        )
    }

    /// `select(pred, on_true, on_false)`; `pred` has the shape of the others (or is a scalar).
    pub fn select(&mut self, pred: &Value, on_true: &Value, on_false: &Value) -> Value {
        if on_true.ty != on_false.ty {
            bad(
                "select",
                format!("branch types differ: {} and {}", on_true.ty, on_false.ty),
            );
        }
        if pred.dtype() != DType::Pred || (pred.dims() != on_true.dims() && !pred.dims().is_empty())
        {
            bad(
                "select",
                format!("predicate {} does not fit {}", pred.ty, on_true.ty),
            );
        }
        self.op(
            "stablehlo.select",
            &[pred, on_true, on_false],
            "",
            on_true.ty.clone(),
        )
    }

    /// `clamp(lo, x, hi)` with scalar or same-shaped bounds.
    pub fn clamp(&mut self, lo: &Value, x: &Value, hi: &Value) -> Value {
        self.op("stablehlo.clamp", &[lo, x, hi], "", x.ty.clone())
    }

    /// Element type conversion (`stablehlo.convert`; float to integer truncates towards zero, float to float rounds to nearest even).
    pub fn convert(&mut self, x: &Value, dtype: DType) -> Value {
        if x.dtype() == dtype {
            return x.clone();
        }
        self.op(
            "stablehlo.convert",
            &[x],
            "",
            TensorType::new(x.dims(), dtype),
        )
    }

    /// Reinterpret the bits as another type. Same width: same shape. Wider to narrower: a trailing dimension of the ratio appears;
    /// narrower to wider: the trailing dimension (which must equal the ratio) disappears.
    pub fn bitcast_convert(&mut self, x: &Value, dtype: DType) -> Value {
        if x.dtype() == dtype {
            return x.clone();
        }
        let (from, to) = (x.dtype().bytes(), dtype.bytes());
        let mut dims = x.dims().to_vec();
        if from > to {
            dims.push((from / to) as i64);
        } else if from < to {
            let ratio = (to / from) as i64;
            if dims.last() != Some(&ratio) {
                bad(
                    "bitcast_convert",
                    format!("{} cannot be packed into {}", x.ty, dtype.name()),
                );
            }
            dims.pop();
        }
        self.op(
            "stablehlo.bitcast_convert",
            &[x],
            "",
            TensorType { dims, dtype },
        )
    }

    // ----- shapes ----------------------------------------------------------------------------------------------------------------

    pub fn reshape(&mut self, x: &Value, dims: &[i64]) -> Value {
        if x.dims() == dims {
            return x.clone();
        }
        let ty = TensorType::new(dims, x.dtype());
        if ty.elements() != x.ty.elements() {
            bad(
                "reshape",
                format!("{} has a different element count than {}", x.ty, ty),
            );
        }
        self.op("stablehlo.reshape", &[x], "", ty)
    }

    /// `broadcast_in_dim`: operand dimension `i` becomes dimension `map[i]` of `dims`.
    pub fn broadcast_in_dim(&mut self, x: &Value, dims: &[i64], map: &[i64]) -> Value {
        if map.len() != x.ty.rank() {
            bad(
                "broadcast_in_dim",
                format!("{} operand dimensions for map {map:?}", x.ty.rank()),
            );
        }
        for (i, &m) in map.iter().enumerate() {
            let target = *dims.get(m as usize).unwrap_or_else(|| {
                bad("broadcast_in_dim", format!("map {map:?} outside {dims:?}"))
            });
            if x.dims()[i] != target && x.dims()[i] != 1 {
                bad(
                    "broadcast_in_dim",
                    format!("dimension {i} of {} does not fit {dims:?}", x.ty),
                );
            }
        }
        self.op(
            "stablehlo.broadcast_in_dim",
            &[x],
            &format!("broadcast_dimensions = {}", array(map)),
            TensorType::new(dims, x.dtype()),
        )
    }

    /// Broadcast a scalar to `dims`.
    pub fn splat_of(&mut self, scalar: &Value, dims: &[i64]) -> Value {
        assert!(scalar.dims().is_empty(), "splat_of: needs a scalar");
        self.broadcast_in_dim(scalar, dims, &[])
    }

    /// Broadcast `x` (whose dimensions map onto `map`) to the shape `dims`.
    pub fn bcast(&mut self, x: &Value, dims: &[i64], map: &[i64]) -> Value {
        self.broadcast_in_dim(x, dims, map)
    }

    pub fn transpose(&mut self, x: &Value, perm: &[i64]) -> Value {
        if perm.len() != x.ty.rank() {
            bad("transpose", format!("permutation {perm:?} for {}", x.ty));
        }
        let dims: Vec<i64> = perm.iter().map(|&p| x.dims()[p as usize]).collect();
        self.op(
            "stablehlo.transpose",
            &[x],
            &format!("permutation = {}", array(perm)),
            TensorType {
                dims,
                dtype: x.dtype(),
            },
        )
    }

    /// Static slice `start[i]..limit[i]` of every dimension (stride 1).
    pub fn slice(&mut self, x: &Value, start: &[i64], limit: &[i64]) -> Value {
        if start.len() != x.ty.rank() || limit.len() != x.ty.rank() {
            bad("slice", format!("bounds {start:?}..{limit:?} for {}", x.ty));
        }
        let mut dims = Vec::new();
        for i in 0..start.len() {
            if start[i] < 0 || limit[i] > x.dims()[i] || start[i] > limit[i] {
                bad(
                    "slice",
                    format!("bounds {start:?}..{limit:?} outside {}", x.ty),
                );
            }
            dims.push(limit[i] - start[i]);
        }
        if dims == x.dims() {
            return x.clone();
        }
        let strides = vec![1i64; start.len()];
        self.op(
            "stablehlo.slice",
            &[x],
            &format!(
                "start_indices = {}, limit_indices = {}, strides = {}",
                array(start),
                array(limit),
                array(&strides)
            ),
            TensorType {
                dims,
                dtype: x.dtype(),
            },
        )
    }

    /// `dynamic_slice`: `sizes` elements from the scalar start indices (clamped so that the slice fits).
    pub fn dynamic_slice(&mut self, x: &Value, starts: &[&Value], sizes: &[i64]) -> Value {
        if starts.len() != x.ty.rank() || sizes.len() != x.ty.rank() {
            bad(
                "dynamic_slice",
                format!(
                    "{} start indices and sizes {sizes:?} for {}",
                    starts.len(),
                    x.ty
                ),
            );
        }
        if sizes.iter().zip(x.dims()).any(|(s, d)| s > d) {
            bad("dynamic_slice", format!("sizes {sizes:?} exceed {}", x.ty));
        }
        let mut operands: Vec<&Value> = vec![x];
        operands.extend_from_slice(starts);
        self.op(
            "stablehlo.dynamic_slice",
            &operands,
            &format!("slice_sizes = {}", array(sizes)),
            TensorType::new(sizes, x.dtype()),
        )
    }

    /// `dynamic_update_slice`: `update` replaces the part of `x` that starts at the scalar indices.
    pub fn dynamic_update_slice(&mut self, x: &Value, update: &Value, starts: &[&Value]) -> Value {
        if starts.len() != x.ty.rank()
            || update.ty.rank() != x.ty.rank()
            || update.dtype() != x.dtype()
        {
            bad(
                "dynamic_update_slice",
                format!(
                    "update {} into {} with {} indices",
                    update.ty,
                    x.ty,
                    starts.len()
                ),
            );
        }
        if update.dims().iter().zip(x.dims()).any(|(u, d)| u > d) {
            bad(
                "dynamic_update_slice",
                format!("update {} larger than {}", update.ty, x.ty),
            );
        }
        let mut operands: Vec<&Value> = vec![x, update];
        operands.extend_from_slice(starts);
        self.op(
            "stablehlo.dynamic_update_slice",
            &operands,
            "",
            x.ty.clone(),
        )
    }

    pub fn concatenate(&mut self, parts: &[&Value], dim: usize) -> Value {
        let first = parts
            .first()
            .unwrap_or_else(|| bad("concatenate", "no operands"));
        if parts.len() == 1 {
            return (*first).clone();
        }
        let mut dims = first.dims().to_vec();
        dims[dim] = 0;
        for p in parts {
            let mut other = p.dims().to_vec();
            if other.len() != dims.len() || p.dtype() != first.dtype() {
                bad("concatenate", format!("{} does not fit {}", p.ty, first.ty));
            }
            dims[dim] += other[dim];
            other[dim] = first.dims()[dim];
            if other != first.dims() {
                bad(
                    "concatenate",
                    format!("{} does not fit {} outside dimension {dim}", p.ty, first.ty),
                );
            }
        }
        self.op(
            "stablehlo.concatenate",
            parts,
            &format!("dimension = {dim} : i64"),
            TensorType {
                dims,
                dtype: first.dtype(),
            },
        )
    }

    // ----- reductions and products -----------------------------------------------------------------------------------------------

    /// Reduce `dims` of `x` with `reducer` (the identity is the neutral element of the combiner for the element type).
    pub fn reduce(&mut self, x: &Value, reducer: Reducer, dims: &[i64]) -> Value {
        let dtype = x.dtype();
        let init = match (reducer, dtype) {
            (Reducer::Sum, _) => self.constant(0.0, &[], dtype),
            (Reducer::Max, d) if d.is_float() => self.infinity(true, &[], d),
            (Reducer::Min, d) if d.is_float() => self.infinity(false, &[], d),
            (Reducer::Max, DType::I32) => self.constant(f64::from(i32::MIN), &[], dtype),
            (Reducer::Min, DType::I32) => self.constant(f64::from(i32::MAX), &[], dtype),
            (r, d) => bad("reduce", format!("{r:?} over {d:?} is not supported")),
        };
        let out_dims: Vec<i64> = x
            .dims()
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(&(*i as i64)))
            .map(|(_, &d)| d)
            .collect();
        let out_ty = TensorType {
            dims: out_dims,
            dtype,
        };
        let scalar = TensorType::new(&[], dtype);
        let (a, b, r) = (self.fresh(), self.fresh(), self.fresh());
        let out = self.fresh();
        // the region lines are indented so that the text stays readable; the validator treats them as one operation
        self.lines.push(format!(
            "{out} = \"stablehlo.reduce\"({}, {}) ({{",
            x.name, init.name
        ));
        self.lines
            .push(format!("^bb0({a}: {scalar}, {b}: {scalar}):"));
        self.lines.push(format!(
            "{r} = \"{}\"({a}, {b}) : ({scalar}, {scalar}) -> {scalar}",
            reducer.op()
        ));
        self.lines
            .push(format!("\"stablehlo.return\"({r}) : ({scalar}) -> ()"));
        self.lines.push(format!(
            "}}) {{dimensions = {}}} : ({}, {scalar}) -> {out_ty}",
            array(dims),
            x.ty
        ));
        Value {
            name: out,
            ty: out_ty,
        }
    }

    pub fn sum(&mut self, x: &Value, dims: &[i64]) -> Value {
        self.reduce(x, Reducer::Sum, dims)
    }

    /// Matrix product with batch and contracting dimensions. The result has the batch dimensions, then the free dimensions of `lhs`,
    /// then those of `rhs`. `out` is the element type of the result (it may be wider than the operands' type, for example `f32`
    /// from `bf16`).
    pub fn dot_general(
        &mut self,
        lhs: &Value,
        rhs: &Value,
        dims: &DotDims,
        out: DType,
        precision: Precision,
    ) -> Value {
        let mut result = Vec::new();
        for (&l, &r) in dims.lhs_batch.iter().zip(&dims.rhs_batch) {
            if lhs.dims()[l as usize] != rhs.dims()[r as usize] {
                bad(
                    "dot_general",
                    format!(
                        "batch dimension {l} of {} differs from {r} of {}",
                        lhs.ty, rhs.ty
                    ),
                );
            }
            result.push(lhs.dims()[l as usize]);
        }
        for (&l, &r) in dims.lhs_contract.iter().zip(&dims.rhs_contract) {
            if lhs.dims()[l as usize] != rhs.dims()[r as usize] {
                bad(
                    "dot_general",
                    format!(
                        "contracting dimension {l} of {} differs from {r} of {}",
                        lhs.ty, rhs.ty
                    ),
                );
            }
        }
        if lhs.dtype() != rhs.dtype() {
            bad(
                "dot_general",
                format!("operand types differ: {} and {}", lhs.ty, rhs.ty),
            );
        }
        for (i, &d) in lhs.dims().iter().enumerate() {
            if !dims.lhs_batch.contains(&(i as i64)) && !dims.lhs_contract.contains(&(i as i64)) {
                result.push(d);
            }
        }
        for (i, &d) in rhs.dims().iter().enumerate() {
            if !dims.rhs_batch.contains(&(i as i64)) && !dims.rhs_contract.contains(&(i as i64)) {
                result.push(d);
            }
        }
        let attrs = format!(
            "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = {}, rhs_batching_dimensions = {}, lhs_contracting_dimensions = {}, rhs_contracting_dimensions = {}>, precision_config = [#stablehlo<precision {p}>, #stablehlo<precision {p}>]",
            list(&dims.lhs_batch),
            list(&dims.rhs_batch),
            list(&dims.lhs_contract),
            list(&dims.rhs_contract),
            p = precision.name()
        );
        self.op(
            "stablehlo.dot_general",
            &[lhs, rhs],
            &attrs,
            TensorType {
                dims: result,
                dtype: out,
            },
        )
    }

    /// Rows of `table` (`[n, c]`) picked by `ids` (`[r]`, integers): `[r, c]`. Out-of-range ids are clamped by the hardware.
    pub fn gather_rows(&mut self, table: &Value, ids: &Value) -> Value {
        if table.ty.rank() != 2 || ids.ty.rank() != 1 || !ids.dtype().is_int() {
            bad(
                "gather_rows",
                format!("table {} and ids {}", table.ty, ids.ty),
            );
        }
        let (rows, cols) = (ids.dims()[0], table.dims()[1]);
        let indices = self.reshape(ids, &[rows, 1]);
        let attrs = format!(
            "dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = {}, indices_are_sorted = false",
            array(&[1, cols])
        );
        self.op(
            "stablehlo.gather",
            &[table, &indices],
            &attrs,
            TensorType::new(&[rows, cols], table.dtype()),
        )
    }

    /// `table` with row `ids[i]` replaced by `rows[i]` (`[r, c]`); with duplicate ids one of the rows wins unless `combine_add`, which
    /// adds all of them (`scatter` with an add combiner).
    pub fn scatter_rows(
        &mut self,
        table: &Value,
        ids: &Value,
        rows: &Value,
        combine_add: bool,
        unique: bool,
    ) -> Value {
        if table.ty.rank() != 2
            || ids.ty.rank() != 1
            || rows.ty.rank() != 2
            || rows.dtype() != table.dtype()
            || rows.dims()[1] != table.dims()[1]
            || rows.dims()[0] != ids.dims()[0]
        {
            bad(
                "scatter_rows",
                format!("table {}, ids {}, rows {}", table.ty, ids.ty, rows.ty),
            );
        }
        let indices = self.reshape(ids, &[ids.dims()[0], 1]);
        let scalar = TensorType::new(&[], table.dtype());
        let (a, b) = (self.fresh(), self.fresh());
        let out = self.fresh();
        self.lines.push(format!(
            "{out} = \"stablehlo.scatter\"({}, {}, {}) ({{",
            table.name, indices.name, rows.name
        ));
        self.lines
            .push(format!("^bb0({a}: {scalar}, {b}: {scalar}):"));
        if combine_add {
            let r = self.fresh();
            self.lines.push(format!(
                "{r} = \"stablehlo.add\"({a}, {b}) : ({scalar}, {scalar}) -> {scalar}"
            ));
            self.lines
                .push(format!("\"stablehlo.return\"({r}) : ({scalar}) -> ()"));
        } else {
            self.lines
                .push(format!("\"stablehlo.return\"({b}) : ({scalar}) -> ()"));
        }
        self.lines.push(format!(
            "}}) {{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, indices_are_sorted = false, unique_indices = {unique}}} : ({}, {}, {}) -> {}",
            table.ty, indices.ty, rows.ty, table.ty
        ));
        Value {
            name: out,
            ty: table.ty.clone(),
        }
    }

    // ----- output ----------------------------------------------------------------------------------------------------------------

    /// The function text with `results` returned (`func.func public @name(...) -> (...)`).
    pub fn finish(&self, results: &[&Value]) -> String {
        let mut text = String::new();
        let params: Vec<String> = self
            .args
            .iter()
            .map(|(v, attr)| match attr {
                Some(attr) => format!("{}: {} {{{attr}}}", v.name, v.ty),
                None => format!("{}: {}", v.name, v.ty),
            })
            .collect();
        let result_types: Vec<String> = results.iter().map(|v| v.ty.to_string()).collect();
        let _ = writeln!(
            text,
            "  func.func public @{}({}) -> ({}) {{",
            self.name,
            params.join(", "),
            result_types.join(", ")
        );
        for line in &self.lines {
            if line.starts_with("^bb") || line.starts_with("})") {
                let _ = writeln!(text, "    {line}");
            } else {
                let _ = writeln!(text, "      {line}");
            }
        }
        if results.is_empty() {
            text.push_str("    func.return\n");
        } else {
            let names: Vec<&str> = results.iter().map(|v| v.name.as_str()).collect();
            let _ = writeln!(
                text,
                "    func.return {} : {}",
                names.join(", "),
                result_types.join(", ")
            );
        }
        text.push_str("  }\n");
        text
    }
}

/// A module holding the function `main` built by `build` (the entry point PJRT compiles).
pub fn module(name: &str, main: &Func, results: &[&Value]) -> String {
    format!("module @{name} {{\n{}}}\n", main.finish(results))
}

/// Make `name` usable as an MLIR symbol (letters, digits and underscores).
pub fn symbol(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() || s.starts_with(|c: char| c.is_ascii_digit()) {
        format!("m_{s}")
    } else {
        s
    }
}
