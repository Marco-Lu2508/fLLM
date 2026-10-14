//! Pure address arithmetic of the Metal buffer layer: page alignment, the split of a large host mapping (the memory
//! mapped GGUF file) into page-aligned no-copy windows, and the memory budget.
//!
//! `newBufferWithBytesNoCopy` needs a page-aligned start and a length that is a multiple of the page size, and a
//! buffer can be at most `maxBufferLength` bytes. A model file is bigger than that on small Macs and its tensors
//! start at arbitrary 32-byte offsets, so the mapping is covered by overlapping windows such that every item of at
//! most `max_item` bytes lies completely inside one window.

use std::sync::atomic::{AtomicU64, Ordering};

/// `value` rounded down to a multiple of `align` (a power of two).
pub fn round_down(value: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    value & !(align - 1)
}

/// `value` rounded up to a multiple of `align` (a power of two); `None` on overflow.
pub fn round_up(value: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    value.checked_add(align - 1).map(|v| v & !(align - 1))
}

/// A window of the aligned region, measured from the aligned base.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    /// Byte offset from `MappingPlan::aligned_base` (a multiple of the page size).
    pub start: usize,
    /// Length in bytes (a multiple of the page size, at most the buffer limit).
    pub len: usize,
}

impl Chunk {
    pub fn end(&self) -> usize {
        self.start + self.len
    }
}

/// How a host range is covered by page-aligned windows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappingPlan {
    /// Page size the plan was made for.
    pub page: usize,
    /// Address of the first window (the range start rounded down to a page).
    pub aligned_base: usize,
    /// Bytes between `aligned_base` and the requested start.
    pub lead: usize,
    /// Total covered length from `aligned_base` (a multiple of the page size).
    pub total: usize,
    /// Distance between the starts of two neighbouring windows.
    pub step: usize,
    pub chunks: Vec<Chunk>,
}

/// Cover `[base, base + len)` with windows of at most `max_buffer` bytes such that any item of at most `max_item`
/// bytes inside the range lies completely inside one window.
///
/// The windows reach up to the end of the last page of the range. Pages that hold any byte of the range are
/// mapped by definition, so a window never touches unmapped memory.
pub fn plan_mapping(
    base: usize,
    len: usize,
    page: usize,
    max_buffer: usize,
    max_item: usize,
) -> Result<MappingPlan, String> {
    if !page.is_power_of_two() {
        return Err(format!("page size {page} is not a power of two"));
    }
    if len == 0 {
        return Err("empty host range".to_owned());
    }
    let aligned_base = round_down(base, page);
    let lead = base - aligned_base;
    let total = lead
        .checked_add(len)
        .and_then(|v| round_up(v, page))
        .ok_or_else(|| "host range length overflows".to_owned())?;
    let window = round_down(max_buffer, page);
    if window == 0 {
        return Err(format!(
            "buffer limit {max_buffer} is smaller than one page ({page})"
        ));
    }
    if total <= window {
        return Ok(MappingPlan {
            page,
            aligned_base,
            lead,
            total,
            step: total,
            chunks: vec![Chunk {
                start: 0,
                len: total,
            }],
        });
    }
    let overlap =
        round_up(max_item.max(1), page).ok_or_else(|| "item length overflows".to_owned())?;
    if overlap >= window {
        return Err(format!(
            "an item of {max_item} bytes does not fit twice into the {max_buffer}-byte buffer limit, the mapping cannot be split"
        ));
    }
    let step = window - overlap;
    let mut chunks = Vec::new();
    let mut start = 0usize;
    loop {
        let chunk_len = window.min(total - start);
        chunks.push(Chunk {
            start,
            len: chunk_len,
        });
        if start + chunk_len >= total {
            break;
        }
        start += step;
    }
    Ok(MappingPlan {
        page,
        aligned_base,
        lead,
        total,
        step,
        chunks,
    })
}

impl MappingPlan {
    /// The window holding `[addr, addr + len)` and the offset of `addr` inside it. `None` if the range is not inside
    /// the planned region or longer than the overlap guaranteed at planning time.
    pub fn locate(&self, addr: usize, len: usize) -> Option<(usize, usize)> {
        let begin = addr.checked_sub(self.aligned_base)?;
        let end = begin.checked_add(len)?;
        if begin < self.lead || end > self.total {
            return None;
        }
        let guess = (begin / self.step).min(self.chunks.len() - 1);
        // the guess is exact for items within the planned maximum; neighbours cover rounding at the ends
        for index in [guess, guess.saturating_sub(1), guess + 1] {
            if let Some(chunk) = self.chunks.get(index) {
                if chunk.start <= begin && end <= chunk.end() {
                    return Some((index, begin - chunk.start));
                }
            }
        }
        None
    }
}

/// A byte budget shared between the allocator and the buffers it handed out (a buffer returns its bytes on drop).
#[derive(Debug)]
pub struct MemoryBudget {
    limit: u64,
    used: AtomicU64,
}

impl MemoryBudget {
    pub fn new(limit: u64) -> Self {
        Self {
            limit,
            used: AtomicU64::new(0),
        }
    }

    pub fn limit(&self) -> u64 {
        self.limit
    }

    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }

    pub fn available(&self) -> u64 {
        self.limit.saturating_sub(self.used())
    }

    /// Reserve `bytes`; `false` (and nothing reserved) if the limit would be exceeded.
    pub fn try_reserve(&self, bytes: u64) -> bool {
        let mut current = self.used.load(Ordering::Relaxed);
        loop {
            let next = match current.checked_add(bytes) {
                Some(next) if next <= self.limit => next,
                _ => return false,
            };
            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(actual) => current = actual,
            }
        }
    }

    pub fn release(&self, bytes: u64) {
        let mut current = self.used.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(bytes);
            match self.used.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAGE: usize = 16384;

    #[test]
    fn rounding() {
        assert_eq!(round_down(16385, PAGE), 16384);
        assert_eq!(round_up(16385, PAGE), Some(32768));
        assert_eq!(round_up(16384, PAGE), Some(16384));
        assert_eq!(round_up(usize::MAX, PAGE), None);
    }

    #[test]
    fn small_range_is_one_window() {
        let plan = plan_mapping(PAGE * 10 + 96, 5000, PAGE, 1 << 30, 4096).unwrap();
        assert_eq!(plan.lead, 96);
        assert_eq!(
            plan.chunks,
            vec![Chunk {
                start: 0,
                len: PAGE
            }]
        );
        assert_eq!(plan.locate(PAGE * 10 + 96 + 100, 200), Some((0, 196)));
        assert_eq!(
            plan.locate(PAGE * 10, 8),
            None,
            "before the requested start"
        );
        assert_eq!(plan.locate(PAGE * 11, 8), None, "past the covered pages");
    }

    #[test]
    fn large_range_is_split_with_overlap() {
        let max_buffer = PAGE * 8;
        let max_item = PAGE * 2 + 5;
        let plan = plan_mapping(PAGE * 3 + 32, PAGE * 100, PAGE, max_buffer, max_item).unwrap();
        assert!(plan.chunks.len() > 10);
        for chunk in &plan.chunks {
            assert_eq!(chunk.start % PAGE, 0);
            assert_eq!(chunk.len % PAGE, 0);
            assert!(chunk.len <= max_buffer);
        }
        assert_eq!(plan.chunks.last().unwrap().end(), plan.total);
        assert_eq!(plan.chunks[0].start, 0);
    }

    #[test]
    fn every_item_lies_in_a_window() {
        // deterministic pseudo-random items of up to max_item bytes anywhere in the range
        let mut state = 0x2545F491_4F6CDD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for round in 0..40 {
            let page = if round % 2 == 0 { 4096 } else { 16384 };
            let base = page * (1 + (next() as usize % 50)) + (next() as usize % page);
            let len = page * (5 + (next() as usize % 400)) + (next() as usize % page);
            let max_buffer = page * (8 + (next() as usize % 64));
            let max_item = 1 + (next() as usize % (max_buffer / 3));
            let plan = plan_mapping(base, len, page, max_buffer, max_item).unwrap();
            for _ in 0..300 {
                let item_len = 1 + (next() as usize % max_item.min(len));
                let offset = next() as usize % (len - item_len + 1);
                let located = plan.locate(base + offset, item_len);
                let (index, inner) = located
                    .unwrap_or_else(|| panic!("no window for offset {offset} len {item_len}"));
                let chunk = plan.chunks[index];
                assert!(inner + item_len <= chunk.len);
                assert_eq!(plan.aligned_base + chunk.start + inner, base + offset);
            }
        }
    }

    #[test]
    fn rejects_impossible_plans() {
        assert!(plan_mapping(0, 0, PAGE, PAGE * 4, 10).is_err());
        assert!(plan_mapping(0, PAGE * 100, PAGE, PAGE * 4, PAGE * 4).is_err());
        assert!(plan_mapping(0, PAGE * 100, PAGE, 100, 10).is_err());
        assert!(plan_mapping(0, 10, 3, PAGE, 10).is_err());
    }

    #[test]
    fn budget_reserves_and_releases() {
        let budget = MemoryBudget::new(100);
        assert!(budget.try_reserve(60));
        assert!(!budget.try_reserve(41));
        assert!(budget.try_reserve(40));
        assert_eq!(budget.available(), 0);
        budget.release(60);
        assert_eq!(budget.used(), 40);
        budget.release(1000);
        assert_eq!(budget.used(), 0);
    }
}
