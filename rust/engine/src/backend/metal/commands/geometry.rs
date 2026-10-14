//! Pure thread-grid helpers: rounding, dispatch validation against the pipeline and device limits, and the
//! threadgroup size heuristic for simple element-wise kernels.

/// `ceil(a / b)`; `b` must not be zero.
pub fn ceil_div(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Product of the three dimensions, `None` on overflow.
pub fn volume(dims: [usize; 3]) -> Option<usize> {
    dims[0].checked_mul(dims[1])?.checked_mul(dims[2])
}

/// Outcome of the validation of a dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Validated {
    /// Encode the dispatch.
    Run,
    /// A dimension of the grid is zero: nothing to do (not an error; the Metal validation layer would complain).
    Empty,
}

/// Check a dispatch against the limits before it reaches the driver (an invalid dispatch aborts the process
/// under Metal API validation and may hang the GPU otherwise).
///
/// `groups`: threadgroups per grid; `threads`: threads per threadgroup; `pipeline_max_threads`:
/// `maxTotalThreadsPerThreadgroup` of the pipeline; `device_max`: `maxThreadsPerThreadgroup` of the device.
pub fn validate_dispatch(
    groups: [usize; 3],
    threads: [usize; 3],
    pipeline_max_threads: usize,
    device_max: [usize; 3],
) -> Result<Validated, String> {
    if threads.contains(&0) {
        return Err(format!("threadgroup size {threads:?} has a zero dimension"));
    }
    let per_group =
        volume(threads).ok_or_else(|| format!("threadgroup size {threads:?} overflows"))?;
    if per_group > pipeline_max_threads {
        return Err(format!("threadgroup size {threads:?} = {per_group} threads exceeds the pipeline limit {pipeline_max_threads}"));
    }
    for axis in 0..3 {
        if device_max[axis] != 0 && threads[axis] > device_max[axis] {
            return Err(format!("threadgroup size {threads:?} exceeds the device limit {device_max:?} on axis {axis}"));
        }
    }
    if groups.contains(&0) {
        return Ok(Validated::Empty);
    }
    if volume(groups).is_none() {
        return Err(format!("grid {groups:?} overflows"));
    }
    Ok(Validated::Run)
}

/// Threads per threadgroup for a one-dimensional kernel over `items` elements: a multiple of the SIMD width, at
/// most `max_threads` (and at most 256: larger groups gain nothing for element-wise work), and not much larger
/// than `items`.
pub fn elementwise_threads(items: usize, simd_width: usize, max_threads: usize) -> usize {
    let simd = simd_width.max(1);
    let cap = max_threads.min(256).max(simd);
    let wanted = items.max(1).next_multiple_of(simd);
    wanted.min(cap / simd * simd)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounding_and_volume() {
        assert_eq!(ceil_div(10, 4), 3);
        assert_eq!(ceil_div(8, 4), 2);
        assert_eq!(volume([2, 3, 4]), Some(24));
        assert_eq!(volume([usize::MAX, 2, 1]), None);
    }

    #[test]
    fn validation() {
        let max = [1024, 1024, 64];
        assert_eq!(
            validate_dispatch([4, 1, 1], [256, 1, 1], 1024, max),
            Ok(Validated::Run)
        );
        assert_eq!(
            validate_dispatch([0, 1, 1], [256, 1, 1], 1024, max),
            Ok(Validated::Empty)
        );
        assert!(validate_dispatch([1, 1, 1], [0, 1, 1], 1024, max).is_err());
        assert!(
            validate_dispatch([1, 1, 1], [512, 4, 1], 1024, max).is_err(),
            "2048 > pipeline limit"
        );
        assert!(
            validate_dispatch([1, 1, 1], [1, 1, 128], 1024, max).is_err(),
            "z above the device limit"
        );
    }

    #[test]
    fn elementwise_group_size() {
        assert_eq!(elementwise_threads(1, 32, 1024), 32);
        assert_eq!(elementwise_threads(100, 32, 1024), 128);
        assert_eq!(elementwise_threads(1_000_000, 32, 1024), 256);
        assert_eq!(elementwise_threads(1_000_000, 32, 96), 96);
        assert_eq!(elementwise_threads(0, 32, 1024), 32);
    }
}
