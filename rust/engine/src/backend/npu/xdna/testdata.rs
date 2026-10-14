//! Deterministic test data shared by the NPU probes of package 11 (XDNA, Hexagon, Apple Neural Engine).

/// `len` pseudo-random values in [-1, 1) from an xorshift64 generator seeded with `seed` (must be non-zero), so that probes on different
/// machines work on identical data.
pub fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = if seed == 0 {
        0x9e37_79b9_7f4a_7c15
    } else {
        seed
    };
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// `y[m, n] = x[m, k] . w[k, n]` in f32 (row-major), the reference the probes compare against.
pub fn matmul_reference(x: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a = x[i * k + p];
            for j in 0..n {
                y[i * n + j] += a * w[p * n + j];
            }
        }
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pseudo_random_is_deterministic_and_in_range() {
        let a = pseudo_random(64, 7);
        assert_eq!(a, pseudo_random(64, 7));
        assert!(a.iter().all(|v| (-1.0..1.0).contains(v)));
        assert_ne!(a, pseudo_random(64, 8));
        assert_eq!(pseudo_random(4, 0).len(), 4);
    }

    #[test]
    fn matmul_reference_small_case() {
        // [1 2; 3 4] . [5 6; 7 8] = [19 22; 43 50]
        let y = matmul_reference(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
        assert_eq!(y, [19.0, 22.0, 43.0, 50.0]);
    }
}
