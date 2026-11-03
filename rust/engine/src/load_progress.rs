//! Progress of the weight upload to a device: a line every few seconds with the layer, the bytes copied and the free host memory, so that a
//! slow or stalled load (swapping on an integrated GPU, a slow disk, a shader compile) is visible in the log instead of silent.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::{Duration, Instant};

static UPLOADED: AtomicU64 = AtomicU64::new(0);
static RELEASED: AtomicU64 = AtomicU64::new(0);
static LAST_NOTE: Mutex<Option<(Instant, Instant)>> = Mutex::new(None);

/// Seconds between two progress lines.
const NOTE_EVERY: Duration = Duration::from_secs(5);

/// A new upload starts: the counters go back to zero.
pub fn start(backend: &str, layers: usize, shares_host_memory: bool) {
    UPLOADED.store(0, Relaxed);
    RELEASED.store(0, Relaxed);
    *LAST_NOTE.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some((Instant::now(), Instant::now()));
    eprintln!(
        "[fllm-rs] uploading the weights of {layers} layers to {backend} ({} MiB of host memory free{})",
        crate::devices::host_free_memory_bytes() >> 20,
        if shares_host_memory { "; the device shares the host memory: mapped pages are released after each weight" } else { "" }
    );
}

/// `bytes` of file data were copied to the device.
pub fn note_uploaded(bytes: usize) {
    UPLOADED.fetch_add(bytes as u64, Relaxed);
}

/// `bytes` of mapped file pages were released after their upload.
pub fn note_released(bytes: usize) {
    RELEASED.fetch_add(bytes as u64, Relaxed);
}

/// Bytes copied to the device and bytes of mapped pages released since `start`.
pub fn totals() -> (u64, u64) {
    (UPLOADED.load(Relaxed), RELEASED.load(Relaxed))
}

/// The progress line: `layer` is "layer 5/48" or the name of the weight being copied.
pub fn format_line(what: &str, uploaded: u64, released: u64, elapsed: Duration, free_host_mib: u64) -> String {
    let secs = elapsed.as_secs_f64().max(0.001);
    format!(
        "[fllm-rs] weights on the device: {what}, {:.2} GiB copied ({:.0} MiB/s), {:.2} GiB of mapped pages released, {free_host_mib} MiB of host memory free, {:.0} s",
        uploaded as f64 / (1u64 << 30) as f64,
        uploaded as f64 / (1u64 << 20) as f64 / secs,
        released as f64 / (1u64 << 30) as f64,
        secs
    )
}

fn note(what: &str, force: bool) {
    let mut slot = LAST_NOTE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((begin, last)) = slot.as_mut() else { return };
    if !force && last.elapsed() < NOTE_EVERY {
        return;
    }
    *last = Instant::now();
    let (uploaded, released) = totals();
    eprintln!("{}", format_line(what, uploaded, released, begin.elapsed(), crate::devices::host_free_memory_bytes() >> 20));
}

/// Layer `done` (1-based) of `total` is on the device: a line when the last one was five seconds ago, and at the end.
pub fn layer_done(done: usize, total: usize) {
    note(&format!("layer {done}/{total}"), done >= total);
}

/// The weight `name` is about to be copied: a line when the last one was five seconds ago (so a stall names the weight it stands on).
pub fn copying(name: &str) {
    note(&format!("copying {name}"), false);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_what_bytes_rate_and_free_memory() {
        let line = format_line("layer 5/48", 3 << 30, 2 << 30, Duration::from_secs(6), 1234);
        assert!(line.contains("layer 5/48"), "{line}");
        assert!(line.contains("3.00 GiB copied (512 MiB/s)"), "{line}");
        assert!(line.contains("2.00 GiB of mapped pages released"), "{line}");
        assert!(line.contains("1234 MiB of host memory free"), "{line}");
    }
}
