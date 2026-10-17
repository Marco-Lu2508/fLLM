//! Starting a parallel group: one worker thread per rank (rank 0 is the caller's thread and owns the scheduler), the communicators,
//! and the loop the workers run: wait for rank 0's step descriptor, run the same step on their share of the model.
use super::config::{self, ParallelConfig};
use super::ctx::{ParCtx, StepDesc, COMM_ERROR};
use super::plan;
use crate::comm::fabric::{Fabric, LocalFabric};
use crate::comm::mesh::MeshComm;
use crate::comm::tcp::{Rendezvous, SocketFabric};
use crate::comm::{BackendKind, Communicator};
use crate::model::GgufModel;
use crate::rust_decoder::RustDecoder;
use std::sync::Arc;

/// `RustDecoder::load` for the main model: one rank of a parallel group when the command line asked for one.
pub fn load_main(model: &GgufModel, device: &str) -> Result<RustDecoder, String> {
    match config::global() {
        None => RustDecoder::load(model, device),
        Some(cfg) => launch(model, device, cfg),
    }
}

/// The ordinal-specific device string of local rank `local` (`cuda:1`), or `cpu`.
fn rank_device(backend: &str, cfg: &ParallelConfig, local: usize) -> String {
    if backend == "cpu" {
        "cpu".to_owned()
    } else {
        format!("{backend}:{}", cfg.device_of(local))
    }
}

/// The communicator that carries tensors between the devices of a group, on top of the host-memory group used for control.
fn data_plane(kind: BackendKind, host_group: &Arc<dyn Communicator>, backend: &str, ordinal: usize) -> Result<Arc<dyn Communicator>, String> {
    match kind {
        BackendKind::Local => Ok(host_group.clone()),
        BackendKind::Nccl | BackendKind::P2p => {
            #[cfg(all(feature = "cuda", not(feature = "hip")))]
            {
                if backend != "cuda" {
                    return Err(format!("--distributed-backend {} needs --device cuda (this run uses {backend}); use --distributed-backend local", kind.name()));
                }
                if kind == BackendKind::Nccl {
                    crate::comm::nccl::NcclComm::connect(host_group, ordinal).map(|c| c as Arc<dyn Communicator>)
                } else {
                    crate::comm::p2p::P2pComm::connect(host_group, ordinal).map(|c| c as Arc<dyn Communicator>)
                }
            }
            #[cfg(not(all(feature = "cuda", not(feature = "hip"))))]
            {
                let _ = (host_group, backend, ordinal);
                Err(format!("--distributed-backend {} needs a build with the cuda feature; use --distributed-backend local", kind.name()))
            }
        }
    }
}

/// Which communicator to use when `--distributed-backend` was not given.
fn default_backend(backend: &str) -> BackendKind {
    if backend != "cuda" {
        return BackendKind::Local;
    }
    #[cfg(all(feature = "cuda", not(feature = "hip")))]
    {
        match crate::comm::nccl::NcclLib::open() {
            Ok(_) => return BackendKind::Nccl,
            Err(why) => eprintln!("[fllm-rs] NCCL is not usable ({why}); using the peer-to-peer fallback for the tensor-parallel traffic"),
        }
    }
    BackendKind::P2p
}

fn make_ctx(
    cfg: &Arc<ParallelConfig>,
    world_rank: usize,
    local: usize,
    control: Arc<dyn Communicator>,
    backend: &str,
    kind: BackendKind,
) -> Result<Arc<ParCtx>, String> {
    let gpu = backend != "cpu";
    let ctx = ParCtx::new(cfg.clone(), world_rank, control, gpu, &|group| data_plane(kind, group, backend, cfg.device_of(local)))?;
    Ok(Arc::new(ctx))
}

/// The loop of ranks 1..n: run every step rank 0 announces until it announces the end.
pub(crate) fn worker_loop(decoder: &RustDecoder, ctx: &ParCtx) {
    let (key_len, value_len) = decoder.kv_buffer_lens();
    let mut key = vec![0u16; key_len];
    let mut value = vec![0u16; value_len];
    loop {
        let mut message = Vec::new();
        if let Err(e) = ctx.control.broadcast_bytes(&mut message, 0) {
            eprintln!("[fllm-rs] rank {}: lost the group: {e}", ctx.rank);
            ctx.control.abort();
            return;
        }
        let desc = match StepDesc::decode(&message) {
            Ok(Some(desc)) => desc,
            Ok(None) => return,
            Err(e) => {
                eprintln!("[fllm-rs] rank {}: unreadable step: {e}", ctx.rank);
                ctx.control.abort();
                return;
            }
        };
        let result = decoder.forward_with_hidden(
            &desc.ids,
            &desc.lengths,
            &desc.tables,
            &mut key,
            &mut value,
            &desc.logit_rows,
            desc.embeddings,
            &desc.overrides,
            None,
        );
        if let Err(e) = result {
            eprintln!("[fllm-rs] rank {}: step failed: {e}", ctx.rank);
            if e.starts_with(COMM_ERROR) {
                // the group is out of step: nobody can continue
                ctx.control.abort();
                return;
            }
        }
    }
}

/// Start the group described by `cfg` for `model`; returns rank 0's decoder. The other ranks live in their own threads until the
/// returned decoder is dropped.
pub fn launch(model: &GgufModel, device: &str, cfg: ParallelConfig) -> Result<RustDecoder, String> {
    cfg.validate()?;
    let chain = crate::devices::device_candidates(device)?;
    let (backend, first_ordinal) = chain.first().cloned().ok_or("no compute device")?;
    let mut cfg = cfg;
    if cfg.devices.is_empty() {
        cfg.devices = (first_ordinal..first_ordinal + cfg.local_world_size()).collect();
    }
    if backend != "cpu" && !std::env::var("FLLM_PARALLEL_GPU").is_ok_and(|v| v == "1") {
        return Err(format!(
            "tensor / pipeline parallelism runs on --device cpu in this version: the sharded decoder and the collectives are in place, but the {backend}              forward path does not call them yet (see docs/multi-gpu.md, \"Status\"). Set FLLM_PARALLEL_GPU=1 to try it anyway (untested)."
        ));
    }
    let kind = cfg.backend.unwrap_or_else(|| if backend == "cpu" { BackendKind::Local } else { default_backend(&backend) });
    let check = plan::check_model(model.mapping(), model.config(), &cfg, backend != "cpu")?;
    for note in &check.notes {
        eprintln!("[fllm-rs] note: {note}");
    }
    if crate::rust_decoder::clear_ffn_host_from() {
        eprintln!("[fllm-rs] --gpu-layers host offload is switched off with tensor / pipeline parallelism: every rank keeps its whole share on its device");
    }
    eprintln!("[fllm-rs] parallel group: {} on {} ({} traffic)", cfg.describe(), backend, kind.name());
    let cfg = Arc::new(cfg);
    let world = cfg.world_size();
    let local_count = cfg.local_world_size();

    // control plane: one endpoint per local rank
    let (fabrics, _rendezvous): (Vec<Arc<dyn Fabric>>, Option<Rendezvous>) = if cfg.rendezvous.nnodes <= 1 {
        (LocalFabric::create(world), None)
    } else {
        let rendezvous = Rendezvous::start(&cfg.rendezvous)?;
        let host = std::env::var("FLLM_NODE_ADDR").ok().unwrap_or_else(|| rendezvous.client.local_ip());
        let handles: Vec<_> = (0..local_count)
            .map(|local| {
                let client = rendezvous.client.clone();
                let host = host.clone();
                let rank = cfg.world_rank(local);
                std::thread::spawn(move || SocketFabric::connect(&client, "control", rank, world, &host))
            })
            .collect();
        let mut out = Vec::new();
        for handle in handles {
            out.push(handle.join().map_err(|_| "socket setup panicked".to_owned())??);
        }
        (out, Some(rendezvous))
    };
    let mut controls: Vec<Option<Arc<dyn Communicator>>> = if cfg.rendezvous.nnodes <= 1 {
        fabrics.into_iter().map(|f| Some(MeshComm::world(f, "local") as Arc<dyn Communicator>)).collect()
    } else {
        fabrics.into_iter().map(|f| Some(MeshComm::world(f, "socket") as Arc<dyn Communicator>)).collect()
    };

    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<(usize, Result<(), String>)>();
    let mut handles = Vec::new();
    for (local, slot) in controls.iter_mut().enumerate().skip(1) {
        let world_rank = cfg.world_rank(local);
        let control = slot.take().ok_or("control endpoint missing")?;
        let worker_model = model.duplicate()?;
        let (cfg, backend, tx) = (cfg.clone(), backend.clone(), ready_tx.clone());
        let device = rank_device(&backend, &cfg, local);
        let handle = std::thread::Builder::new()
            .name(format!("fllm-rank-{world_rank}"))
            .spawn(move || {
                let built = make_ctx(&cfg, world_rank, local, control.clone(), &backend, kind)
                    .and_then(|ctx| RustDecoder::load_sharded(&worker_model, &device, ctx.clone()).map(|d| (ctx, d)));
                match built {
                    Ok((ctx, decoder)) => {
                        let _ = tx.send((world_rank, Ok(())));
                        drop(tx);
                        worker_loop(&decoder, &ctx);
                    }
                    Err(e) => {
                        control.abort();
                        let _ = tx.send((world_rank, Err(e)));
                    }
                }
            })
            .map_err(|e| format!("cannot start rank {world_rank}: {e}"))?;
        handles.push(handle);
    }
    drop(ready_tx);

    let control0 = controls[0].take().ok_or("control endpoint missing")?;
    let rank0 = cfg.world_rank(0);
    let device0 = rank_device(&backend, &cfg, 0);
    let leader = make_ctx(&cfg, rank0, 0, control0.clone(), &backend, kind)
        .and_then(|ctx| RustDecoder::load_sharded(model, &device0, ctx).map_err(|e| format!("rank {rank0}: {e}")));
    let mut failure = leader.as_ref().err().cloned();
    if failure.is_some() {
        control0.abort();
    }
    // every worker reports once, after it loaded its share (or failed)
    while let Ok((rank, status)) = ready_rx.recv() {
        if let Err(e) = status {
            if failure.as_ref().is_none_or(|f| f.contains("aborted")) {
                failure = Some(format!("rank {rank}: {e}"));
            }
            control0.abort();
        }
    }
    if let Some(error) = failure {
        for h in handles {
            let _ = h.join();
        }
        return Err(error);
    }
    let decoder = leader?;
    decoder.attach_group_threads(handles);
    Ok(decoder)
}
