//! What the GPUs of this machine can do between each other, from CUDA's peer attributes (`nvidia-smi topo -m` without the tool).
use crate::comm::cuda_ffi::{check, device_count};
use std::ffi::c_int;

/// One line per device pair: `-` same device, `X` no peer access (traffic goes through host memory), otherwise the performance
/// rank CUDA reports (a lower number is a faster link: NVLink pairs show 0, PCIe pairs 1 or more), `a` marks native atomics.
pub fn report() -> Result<String, String> {
    let n = device_count()?;
    let mut out = format!("peer matrix of {n} device(s)\n     ");
    for j in 0..n {
        out += &format!("{j:>5}");
    }
    out.push('\n');
    for i in 0..n {
        out += &format!("{i:>4} ");
        for j in 0..n {
            if i == j {
                out += &format!("{:>5}", "-");
                continue;
            }
            let (mut can, mut rank, mut atomics): (c_int, c_int, c_int) = (0, 0, 0);
            // SAFETY: plain queries with valid out pointers.
            unsafe {
                check("cudaDeviceCanAccessPeer", crate::comm::cuda_ffi::fllm_comm_can_access_peer(i as c_int, j as c_int, &mut can))?;
                if can != 0 {
                    check("cudaDeviceGetP2PAttribute", crate::comm::cuda_ffi::fllm_comm_p2p_rank(i as c_int, j as c_int, &mut rank))?;
                    check("cudaDeviceGetP2PAttribute", crate::comm::cuda_ffi::fllm_comm_p2p_native_atomics(i as c_int, j as c_int, &mut atomics))?;
                }
            }
            out += &format!("{:>5}", if can == 0 { "X".to_owned() } else { format!("{rank}{}", if atomics != 0 { "a" } else { "" }) });
        }
        out.push('\n');
    }
    Ok(out)
}
