//! A [`Plan`] couples the StableHLO program of one operation with the device arrays it reads and writes.
//!
//! The program text depends only on shapes and on the static options of the operation, never on the data of one call: window
//! offsets, positions, slot numbers, block tables and lengths travel in one small `i32` array (the "meta" argument, uploaded per
//! call). That makes the text a valid cache key: equal text, equal program.
//!
//! Using a plan takes two steps because the meta array must be complete before the first program line is written: collect the meta
//! values and bind the views with [`Meta`], then create the [`Plan`] and add inputs and outputs.
//!
//! Level of verification: L0.

use super::mem::{driver, Cell, Shared, View};
use crate::backend::tpu::pjrt::Buffer;
use crate::backend::tpu::stablehlo::{self, DType, Func, Value};
use crate::backend::BackendError;
use std::sync::Arc;

/// One argument of the program.
pub enum Arg {
    /// The `i32` meta array.
    Meta,
    /// A device array that is only read (weights, tables, a temporary).
    Buffer(Arc<Buffer>),
    /// The current buffer of a cell (read, and maybe donated to a result).
    Cell(Arc<Cell>),
}

/// Where a result of the program goes.
pub enum Sink {
    /// Replaces the buffer of the cell.
    Cell(Arc<Cell>),
    /// Handed back to the caller (to read on the host).
    Host,
}

/// A finished program with its arguments.
pub struct Finished {
    pub text: String,
    pub args: Vec<Arg>,
    pub sinks: Vec<Sink>,
    pub meta: Vec<i32>,
    /// The type of every result, in order.
    pub result_types: Vec<crate::backend::tpu::stablehlo::TensorType>,
}

/// A byte window of a cell, bound to the slot of the meta array that holds its start offset (in elements).
#[derive(Clone)]
pub struct Bound {
    pub view: View,
    /// Element type the cell stores (or the requested one while the cell has never been written).
    pub cell_dtype: DType,
    /// Meta slot of the start offset; `None` when the window is the whole array.
    pub slot: Option<usize>,
}

/// The `i32` values that travel with a program.
#[derive(Default)]
pub struct Meta {
    values: Vec<i32>,
}

impl Meta {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a value; returns its slot.
    pub fn push(&mut self, value: i64) -> Result<usize, BackendError> {
        let value = i32::try_from(value).map_err(|_| {
            BackendError::InvalidShape(format!(
                "{value} does not fit the 32-bit indices of the TPU backend"
            ))
        })?;
        self.values.push(value);
        Ok(self.values.len() - 1)
    }

    /// Append many values; returns the slot of the first.
    pub fn extend(&mut self, values: impl IntoIterator<Item = i64>) -> Result<usize, BackendError> {
        let start = self.values.len();
        for v in values {
            self.push(v)?;
        }
        Ok(start)
    }

    /// Append floats as their bit patterns; returns the slot of the first.
    pub fn extend_floats(&mut self, values: impl IntoIterator<Item = f32>) -> usize {
        let start = self.values.len();
        self.values
            .extend(values.into_iter().map(|v| v.to_bits() as i32));
        start
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Bind a window: a window that is not the whole array gets a meta slot for its start offset (in elements of `dtype`).
    pub fn bind(
        &mut self,
        shared: &Shared,
        view: &View,
        dtype: DType,
    ) -> Result<Bound, BackendError> {
        let _ = shared;
        let existing = view.cell.lock()?.dtype;
        let cell_dtype = existing.unwrap_or(dtype);
        if cell_dtype.bytes() != dtype.bytes() {
            return Err(BackendError::BufferError(format!(
                "a TPU buffer holding {cell_dtype:?} values is used as {dtype:?} (different element widths)"
            )));
        }
        let slot = if view.is_whole() {
            None
        } else {
            let width = cell_dtype.bytes();
            if !view.offset.is_multiple_of(width) || !view.size.is_multiple_of(width) {
                return Err(BackendError::BufferError(format!(
                    "buffer window {}+{} is not aligned to {width} bytes",
                    view.offset, view.size
                )));
            }
            Some(self.push((view.offset / width) as i64)?)
        };
        Ok(Bound {
            view: view.clone(),
            cell_dtype,
            slot,
        })
    }
}

/// The program under construction.
pub struct Plan<'a> {
    shared: &'a Shared,
    name: String,
    f: Func,
    params: Vec<Value>,
    args: Vec<Arg>,
    donated: Vec<Option<usize>>,
    results: Vec<Value>,
    sinks: Vec<Sink>,
    meta: Vec<i32>,
    meta_param: Option<Value>,
    scalars: Vec<(usize, Value)>,
}

impl<'a> Plan<'a> {
    pub fn new(shared: &'a Shared, name: &str, meta: Meta) -> Self {
        let mut f = Func::new("main");
        let mut params = Vec::new();
        let mut args = Vec::new();
        let mut donated = Vec::new();
        let meta_param = if meta.values.is_empty() {
            None
        } else {
            let p = f.param(&[meta.values.len() as i64], DType::I32);
            params.push(p.clone());
            args.push(Arg::Meta);
            donated.push(None);
            Some(p)
        };
        Self {
            shared,
            name: name.to_owned(),
            f,
            params,
            args,
            donated,
            results: Vec::new(),
            sinks: Vec::new(),
            meta: meta.values,
            meta_param,
            scalars: Vec::new(),
        }
    }

    pub fn shared(&self) -> &'a Shared {
        self.shared
    }

    /// The function the program body is written in.
    pub fn f(&mut self) -> &mut Func {
        &mut self.f
    }

    fn meta_value(&self) -> Value {
        self.meta_param
            .clone()
            .expect("this plan has no meta values")
    }

    /// Meta value `slot` as an `i32` scalar.
    pub fn meta_scalar(&mut self, slot: usize) -> Value {
        if let Some((_, v)) = self.scalars.iter().find(|(s, _)| *s == slot) {
            return v.clone();
        }
        let meta = self.meta_value();
        let one = self.f.slice(&meta, &[slot as i64], &[slot as i64 + 1]);
        let scalar = self.f.reshape(&one, &[]);
        self.scalars.push((slot, scalar.clone()));
        scalar
    }

    /// Meta values `start..start + len` as an `i32` vector.
    pub fn meta_ints(&mut self, start: usize, len: usize) -> Value {
        let meta = self.meta_value();
        self.f
            .slice(&meta, &[start as i64], &[(start + len) as i64])
    }

    /// Meta values `start..start + len` reinterpreted as `f32`.
    pub fn meta_floats(&mut self, start: usize, len: usize) -> Value {
        let ints = self.meta_ints(start, len);
        self.f.bitcast_convert(&ints, DType::F32)
    }

    /// The program parameter that carries `cell` (created on first use; a never-written cell gets its zero buffer first).
    fn cell_param(&mut self, cell: &Arc<Cell>, dtype: DType) -> Result<usize, BackendError> {
        for (i, arg) in self.args.iter().enumerate() {
            if let Arg::Cell(have) = arg {
                if Arc::ptr_eq(have, cell) {
                    return Ok(i);
                }
            }
        }
        self.shared.materialize(cell, dtype)?;
        let actual = cell.lock()?.dtype.unwrap_or(dtype);
        let (dims, dtype) = cell.signature(actual);
        let param = self.f.param(&dims, dtype);
        self.params.push(param);
        self.args.push(Arg::Cell(Arc::clone(cell)));
        self.donated.push(None);
        Ok(self.params.len() - 1)
    }

    /// `len` elements of type `dtype` from the start of the window, as a flat array.
    pub fn input(
        &mut self,
        bound: &Bound,
        dtype: DType,
        len: usize,
    ) -> Result<Value, BackendError> {
        if len * dtype.bytes() > bound.view.size {
            return Err(BackendError::InvalidShape(format!(
                "a read of {len} {dtype:?} values exceeds the {}-byte window",
                bound.view.size
            )));
        }
        let index = self.cell_param(&bound.view.cell, dtype)?;
        let raw = self.params[index].clone();
        if !bound.view.cell.is_flat() {
            return if bound.view.is_whole() && raw.ty.elements() == len {
                Ok(raw)
            } else {
                Err(BackendError::BufferError(
                    "only whole arrays of this kind can be read".into(),
                ))
            };
        }
        let total = raw.dims()[0] as usize;
        let window = if bound.slot.is_none() && len == total {
            raw
        } else {
            let start = match bound.slot {
                Some(slot) => self.meta_scalar(slot),
                None => self.f.constant(0.0, &[], DType::I32),
            };
            self.f.dynamic_slice(&raw, &[&start], &[len as i64])
        };
        Ok(self.f.bitcast_convert(&window, dtype))
    }

    /// The whole array of a shaped cell (a KV pool) as a read-only parameter of its own shape.
    pub fn pool(&mut self, view: &View) -> Result<Value, BackendError> {
        let dtype = view.cell.shaped.as_ref().map_or(DType::F32, |(_, d)| *d);
        let index = self.cell_param(&view.cell, dtype)?;
        Ok(self.params[index].clone())
    }

    /// A device array that is only read.
    pub fn buffer(&mut self, buffer: Arc<Buffer>) -> Value {
        let dtype = super::mem::dtype_of(buffer.element);
        let param = self.f.param(&buffer.dims, dtype);
        self.params.push(param.clone());
        self.args.push(Arg::Buffer(buffer));
        self.donated.push(None);
        param
    }

    /// Write the flat array `value` to the start of the window.
    pub fn output(&mut self, bound: &Bound, value: &Value) -> Result<(), BackendError> {
        if value.ty.rank() != 1 {
            return Err(BackendError::InvalidShape(format!(
                "an output must be flat, got {}",
                value.ty
            )));
        }
        let cell = &bound.view.cell;
        if !cell.is_flat() {
            return Err(BackendError::BufferError(
                "a window of this kind of array cannot be written".into(),
            ));
        }
        let bytes = value.ty.elements() * value.dtype().bytes();
        if bytes > bound.view.size {
            return Err(BackendError::InvalidShape(format!(
                "an output of {bytes} bytes does not fit the {}-byte window",
                bound.view.size
            )));
        }
        let stored = cell.lock()?.dtype;
        let target = stored.unwrap_or(value.dtype());
        if target.bytes() != value.dtype().bytes() {
            return Err(BackendError::BufferError(format!(
                "a TPU buffer holding {target:?} values is written with {:?} values",
                value.dtype()
            )));
        }
        let converted = self.f.bitcast_convert(value, target);
        let cell_elems = cell.bytes / target.bytes();
        if bound.view.is_whole() && converted.ty.elements() == cell_elems {
            self.results.push(converted);
            self.sinks.push(Sink::Cell(Arc::clone(cell)));
            return Ok(());
        }
        let index = self.cell_param(cell, target)?;
        let raw = self.params[index].clone();
        let start = match bound.slot {
            Some(slot) => self.meta_scalar(slot),
            None => self.f.constant(0.0, &[], DType::I32),
        };
        let updated = self.f.dynamic_update_slice(&raw, &converted, &[&start]);
        self.donated[index] = Some(self.results.len());
        self.results.push(updated);
        self.sinks.push(Sink::Cell(Arc::clone(cell)));
        Ok(())
    }

    /// The new contents of a whole shaped array (a KV pool); the old buffer is donated to the result.
    pub fn replace_pool(&mut self, view: &View, value: &Value) -> Result<(), BackendError> {
        let index = self.cell_param(&view.cell, DType::F16)?;
        if self.params[index].ty != value.ty {
            return Err(BackendError::InvalidShape(format!(
                "pool update {} does not match the pool {}",
                value.ty, self.params[index].ty
            )));
        }
        self.donated[index] = Some(self.results.len());
        self.results.push(value.clone());
        self.sinks.push(Sink::Cell(Arc::clone(&view.cell)));
        Ok(())
    }

    /// Return `value` to the caller.
    pub fn host_result(&mut self, value: &Value) {
        self.results.push(value.clone());
        self.sinks.push(Sink::Host);
    }

    /// Assemble the module text.
    pub fn finish(mut self) -> Result<Finished, BackendError> {
        if self.results.is_empty() {
            return Err(driver(format!("the program {} has no result", self.name)));
        }
        for (param, target) in self.params.iter().zip(&self.donated) {
            if let Some(result) = target {
                self.f
                    .set_param_attr(param, format!("tf.aliasing_output = {result} : i32"));
            }
        }
        let results: Vec<&Value> = self.results.iter().collect();
        let text = stablehlo::module(&stablehlo::symbol(&self.name), &self.f, &results);
        let result_types = self.results.iter().map(|v| v.ty.clone()).collect();
        Ok(Finished {
            text,
            args: self.args,
            sinks: self.sinks,
            meta: self.meta,
            result_types,
        })
    }
}
