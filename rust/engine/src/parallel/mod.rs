//! Splitting one model over several devices: tensor, pipeline and expert parallelism (`docs/multi-gpu.md`).
//!
//! * `shard`: how each stored weight layout is cut for a rank.
//! * `plan`: which splits a model supports, error messages naming the tensor and the sizes that work, memory per rank.
//! * `ctx`: the per-rank context the decoder consults (communicators, stage, shard geometry) and the step descriptor.
//! * `group`: the worker threads, one per rank, and the loader that builds the sharded decoders.
//! * `decoder_shard`: the part of the decoder that shards layers and runs the collectives (a child module of `rust_decoder`).
//! * `cluster_plan`: the planner for a node of identical large GPUs (TP x EP x PP x attention-DP, memory per rank, `fllm-rs plan-cluster`).
//! * `split`: the layer-split planner (which layers go to which device, memory and speed aware).
//! * `selftest`: the CPU simulation that proves multi-rank runs equal the single-device run (`fllm-rs selftest-parallel`).
pub mod cluster_plan;
pub mod config;
pub mod eplb;
pub mod ctx;
pub mod fixture;
pub mod group;
pub mod plan;
pub mod selftest;
pub mod shard;
pub mod split;

pub use config::{EpDispatch, ParallelConfig};
pub use ctx::{ParCtx, StepDesc};
pub use group::load_main;
