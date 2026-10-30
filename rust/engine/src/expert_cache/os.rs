//! The operating-system calls of the expert cache: read ahead, stop read-ahead, drop from memory, and ask whether pages are resident.
//!
//! Every function takes an address range of a memory-mapped model file (`ptr`, `len` in bytes). None of them changes the bytes that a
//! later read of the range returns: a dropped page is read from the file again, a prefetch only fills the page cache. Ranges are
//! rounded to whole pages: outward for read-ahead (a neighbouring page is read at most), inward for eviction (a page shared with a
//! neighbouring expert stays).
//!
//! | | read ahead | stop read-ahead | drop from memory | residency |
//! |---|---|---|---|---|
//! | Linux, Android | `madvise(MADV_WILLNEED)` | `MADV_RANDOM` | `MADV_PAGEOUT` (5.4+; older kernels: nothing) | `mincore` |
//! | macOS, BSD | `madvise(MADV_WILLNEED)` | `MADV_RANDOM` | nothing | `mincore` |
//! | Windows | `PrefetchVirtualMemory` | nothing | `VirtualUnlock` (leaves the working set; the page stays cached) | `QueryWorkingSetEx` (working set, not file cache) |

/// Pages of `len` bytes at `ptr`, rounded outward (`inward == false`) or inward to whole pages: `(first page address, bytes)`.
fn page_range(ptr: usize, len: usize, inward: bool) -> Option<(usize, usize)> {
    let page = page_size();
    let end = ptr.checked_add(len)?;
    let (start, end) = if inward { (ptr.checked_add(page - 1)? & !(page - 1), end & !(page - 1)) } else { (ptr & !(page - 1), end.checked_add(page - 1)? & !(page - 1)) };
    (end > start).then_some((start, end - start))
}

/// Bytes of the page cache that a fault on `ptr..ptr + len` could read in one piece, rounded to pages.
pub(super) fn pages_of(len: usize) -> usize {
    len.div_ceil(page_size())
}

#[cfg(unix)]
mod imp {
    use std::ffi::{c_int, c_void};

    extern "C" {
        fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        fn mincore(addr: *mut c_void, len: usize, vec: *mut u8) -> c_int;
        fn sysconf(name: c_int) -> i64;
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const SC_PAGESIZE: c_int = 30;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    const SC_PAGESIZE: c_int = 29;
    const MADV_RANDOM: c_int = 1;
    const MADV_WILLNEED: c_int = 3;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const MADV_PAGEOUT: c_int = 21;

    pub fn page_size() -> usize {
        static SIZE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *SIZE.get_or_init(|| {
            // SAFETY: sysconf only reads a system constant.
            let n = unsafe { sysconf(SC_PAGESIZE) };
            if n > 0 && (n as usize).is_power_of_two() {
                n as usize
            } else {
                4096
            }
        })
    }

    fn advise(ptr: usize, len: usize, advice: c_int) -> bool {
        // SAFETY: madvise on a mapped range only changes paging hints (or drops clean pages that are read again on the next use). A range
        // that is no longer mapped makes the call fail with ENOMEM; the cache keeps the mapping alive while it uses the range.
        unsafe { madvise(ptr as *mut c_void, len, advice) == 0 }
    }

    pub fn will_need(ptr: usize, len: usize) -> bool {
        match super::page_range(ptr, len, false) {
            Some((p, l)) => advise(p, l, MADV_WILLNEED),
            None => false,
        }
    }

    pub fn random_access(ptr: usize, len: usize) -> bool {
        match super::page_range(ptr, len, false) {
            Some((p, l)) => advise(p, l, MADV_RANDOM),
            None => false,
        }
    }

    pub fn evict(ptr: usize, len: usize) -> bool {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            match super::page_range(ptr, len, true) {
                Some((p, l)) => advise(p, l, MADV_PAGEOUT),
                None => false,
            }
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            let _ = (ptr, len);
            false
        }
    }

    /// Resident pages among the pages at the given addresses (`None` when the system call fails).
    pub fn resident_pages(addrs: &[usize]) -> Option<usize> {
        let page = page_size();
        let mut count = 0;
        for &a in addrs {
            let mut state = 0u8;
            // SAFETY: one page of a mapped range; `state` receives one byte.
            if unsafe { mincore((a & !(page - 1)) as *mut c_void, page, &mut state) } != 0 {
                return None;
            }
            count += usize::from(state & 1);
        }
        Some(count)
    }

    pub fn process_read_bytes() -> Option<u64> {
        #[cfg(any(target_os = "linux", target_os = "android"))]
        {
            let text = std::fs::read_to_string("/proc/self/io").ok()?;
            text.lines().find_map(|l| l.strip_prefix("read_bytes:")).and_then(|v| v.trim().parse().ok())
        }
        #[cfg(not(any(target_os = "linux", target_os = "android")))]
        {
            None
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    #[repr(C)]
    struct MemoryRangeEntry {
        virtual_address: *mut c_void,
        number_of_bytes: usize,
    }
    #[repr(C)]
    struct WorkingSetExInformation {
        virtual_address: *mut c_void,
        virtual_attributes: usize,
    }
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> *mut c_void;
        fn PrefetchVirtualMemory(process: *mut c_void, entries: usize, ranges: *mut MemoryRangeEntry, flags: u32) -> i32;
        fn VirtualUnlock(addr: *mut c_void, size: usize) -> i32;
        fn K32QueryWorkingSetEx(process: *mut c_void, info: *mut c_void, size: u32) -> i32;
    }

    pub fn page_size() -> usize {
        4096
    }

    pub fn will_need(ptr: usize, len: usize) -> bool {
        let Some((p, l)) = super::page_range(ptr, len, false) else { return false };
        let mut range = MemoryRangeEntry { virtual_address: p as *mut c_void, number_of_bytes: l };
        // SAFETY: one valid range entry; the call only asks the memory manager to bring the pages in.
        unsafe { PrefetchVirtualMemory(GetCurrentProcess(), 1, &mut range, 0) != 0 }
    }

    pub fn random_access(_ptr: usize, _len: usize) -> bool {
        false
    }

    pub fn evict(ptr: usize, len: usize) -> bool {
        let Some((p, l)) = super::page_range(ptr, len, true) else { return false };
        // SAFETY: on pages that are not locked this removes them from the working set (the call reports ERROR_NOT_LOCKED, which is the
        // expected outcome); the data stays in the file cache and is mapped again on the next access.
        unsafe { VirtualUnlock(p as *mut c_void, l) };
        true
    }

    pub fn resident_pages(addrs: &[usize]) -> Option<usize> {
        let mut info: Vec<WorkingSetExInformation> = addrs
            .iter()
            .map(|&a| WorkingSetExInformation { virtual_address: (a & !4095) as *mut c_void, virtual_attributes: 0 })
            .collect();
        if info.is_empty() {
            return Some(0);
        }
        // SAFETY: `info` is an array of PSAPI_WORKING_SET_EX_INFORMATION whose byte size is passed.
        let ok = unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), info.as_mut_ptr().cast(), (info.len() * std::mem::size_of::<WorkingSetExInformation>()) as u32) };
        (ok != 0).then(|| info.iter().filter(|i| i.virtual_attributes & 1 == 1).count())
    }

    pub fn process_read_bytes() -> Option<u64> {
        None
    }
}

#[cfg(not(any(unix, windows)))]
mod imp {
    pub fn page_size() -> usize {
        4096
    }
    pub fn will_need(_ptr: usize, _len: usize) -> bool {
        false
    }
    pub fn random_access(_ptr: usize, _len: usize) -> bool {
        false
    }
    pub fn evict(_ptr: usize, _len: usize) -> bool {
        false
    }
    pub fn resident_pages(_addrs: &[usize]) -> Option<usize> {
        None
    }
    pub fn process_read_bytes() -> Option<u64> {
        None
    }
}

pub(super) use imp::{evict, page_size, process_read_bytes, random_access, resident_pages, will_need};

/// Bytes the process has read from storage so far (Linux `/proc/self/io`; `None` elsewhere).
pub fn disk_read_bytes() -> Option<u64> {
    process_read_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_round_outward_for_read_ahead_and_inward_for_eviction() {
        let page = page_size();
        assert_eq!(page_range(page + 5, page, false), Some((page, 2 * page)));
        assert_eq!(page_range(page + 5, 3 * page, true), Some((2 * page, 2 * page)));
        // a range inside one page holds no whole page to drop
        assert_eq!(page_range(page + 5, 100, true), None);
        assert_eq!(page_range(usize::MAX - 10, 100, false), None);
    }

    #[test]
    fn page_size_is_a_power_of_two() {
        assert!(page_size().is_power_of_two() && page_size() >= 4096);
        assert_eq!(pages_of(1), 1);
        assert_eq!(pages_of(page_size() + 1), 2);
    }
}
