//! Harness of package p06c: the plan files exactly as they sit in the worktree, with the types they need from `crate::backend`.
#![allow(dead_code)]

pub mod backend {
    pub use fllm_engine::backend::*;
}

/// Stand-in of package 6b's `ops.rs` (same types as the local, uncommitted stand-in in the worktree).
pub mod ops {
    use std::sync::Arc;

    #[path = "../../../../../src/backend/metal/ops/moe_plan.rs"]
    pub mod moe_plan;
    #[path = "../../../../../src/backend/metal/ops/recurrent_plan.rs"]
    pub mod recurrent_plan;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Access {
        Read,
        Write,
        ReadWrite,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub enum SlotSource {
        Buffer,
        Host(Arc<Vec<u8>>),
        Scratch { id: u8, bytes: usize },
        Dummy,
        RunState(usize),
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct BufferSlot {
        pub index: u32,
        pub name: &'static str,
        pub access: Access,
        pub source: SlotSource,
    }

    #[derive(Clone, Debug, PartialEq)]
    pub struct OpPlan {
        pub kernel_name: String,
        pub constants: Vec<u8>,
        pub threadgroups: [u32; 3],
        pub threads_per_group: [u32; 3],
        pub slots: Vec<BufferSlot>,
    }
}
