//! AMX tile primitives. The tile intrinsics of `core::arch` are unstable (`x86_amx_intrinsics`),
//! so every instruction is issued through inline assembly. The assembler accepts the mnemonics
//! without any target feature; the CPU must have AMX-TILE and the matching dot instruction, and on
//! Linux the process needs the tile-data permission (`ident::amx_ready`) before the first tile
//! instruction runs, otherwise it dies with SIGILL.
//!
//! Tile operations are a trait (`TileOps`) so the GEMM drivers in `amx_gemm` run unchanged on
//! `EmuTiles`, a scalar emulation of the instruction semantics used by the unit tests, and on
//! `RealTiles`, the hardware.
//!
//! Instruction semantics (palette 1, 8 tiles of up to 16 rows x 64 bytes):
//! - `tileloadd`/`tilestored`: row `r` of the tile <-> `ptr + r * stride`, `colsb` bytes per row.
//! - `tdpbssd C, A, B`: A is `M x K` bytes (row-major, `colsb = K`), B is `K/4` rows of `N`
//!   groups of 4 bytes (`colsb = 4 * N`), C is `M x N` dwords: `C[m][n] += sum_k A[m][k] *
//!   B[k / 4][n][k % 4]`. `tdpbusd`: A unsigned, B signed. `tdpbf16ps`: bf16 pairs, f32 C:
//!   `C[m][n] += sum_j A[m][2j] * B[j][n][0] + A[m][2j + 1] * B[j][n][1]`.

/// One tile of the configuration: `rows` (at most 16) and bytes per row `colsb` (at most 64).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileShape {
    pub rows: u8,
    pub colsb: u16,
}

/// The 64-byte `ldtilecfg` operand for palette 1.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub(crate) struct TileConfig {
    bytes: [u8; 64],
}

impl TileConfig {
    /// Palette 1 with the given shapes for tiles 0, 1, ...; unlisted tiles are left unconfigured
    /// (zero rows and columns).
    pub(crate) fn palette1(shapes: &[TileShape]) -> Self {
        assert!(shapes.len() <= 8);
        let mut bytes = [0u8; 64];
        bytes[0] = 1; // palette id
        for (i, shape) in shapes.iter().enumerate() {
            assert!(shape.rows <= 16 && shape.colsb <= 64);
            bytes[16 + 2 * i..18 + 2 * i].copy_from_slice(&shape.colsb.to_le_bytes());
            bytes[48 + i] = shape.rows;
        }
        Self { bytes }
    }

    #[cfg(test)]
    pub(crate) fn as_bytes(&self) -> &[u8; 64] {
        &self.bytes
    }
}

/// The tile instructions the GEMM drivers use. Tile numbers are const generics because the
/// instructions encode them.
pub(crate) trait TileOps {
    /// Load the configuration (`ldtilecfg`); tile data is zeroed.
    ///
    /// # Safety
    /// The CPU must support the tile instructions used and the process must hold the permission.
    unsafe fn configure(&mut self, config: &TileConfig);
    /// `tilerelease`: return the tile state to its initial state.
    ///
    /// # Safety
    /// As `configure`.
    unsafe fn release(&mut self);
    /// # Safety
    /// As `configure`; tile `T` must be configured.
    unsafe fn zero<const T: u8>(&mut self);
    /// Load tile `T` from `ptr` with a row stride of `stride` bytes.
    ///
    /// # Safety
    /// As `configure`; every row `ptr + r * stride` must be readable for the tile's `colsb`.
    unsafe fn load<const T: u8>(&mut self, ptr: *const u8, stride: usize);
    /// Store tile `T` to `ptr` with a row stride of `stride` bytes.
    ///
    /// # Safety
    /// As `configure`; every row `ptr + r * stride` must be writable for the tile's `colsb`.
    unsafe fn store<const T: u8>(&mut self, ptr: *mut u8, stride: usize);
    /// Signed by signed bytes.
    ///
    /// # Safety
    /// As `configure`; the three tiles must be configured with matching shapes.
    unsafe fn dpbssd<const C: u8, const A: u8, const B: u8>(&mut self);
    /// Unsigned (A) by signed (B) bytes.
    ///
    /// # Safety
    /// As `dpbssd`.
    unsafe fn dpbusd<const C: u8, const A: u8, const B: u8>(&mut self);
    /// bf16 pairs into f32.
    ///
    /// # Safety
    /// As `dpbssd`, and the CPU must support AMX-BF16.
    unsafe fn dpbf16ps<const C: u8, const A: u8, const B: u8>(&mut self);
}

/// The hardware.
#[derive(Default)]
pub(crate) struct RealTiles;

impl TileOps for RealTiles {
    #[inline(always)]
    unsafe fn configure(&mut self, config: &TileConfig) {
        std::arch::asm!(
            "ldtilecfg [{cfg}]",
            cfg = in(reg) config as *const TileConfig,
            options(nostack, readonly),
        );
    }
    #[inline(always)]
    unsafe fn release(&mut self) {
        std::arch::asm!("tilerelease", options(nostack));
    }
    #[inline(always)]
    unsafe fn zero<const T: u8>(&mut self) {
        std::arch::asm!("tilezero tmm{t}", t = const T, options(nostack));
    }
    #[inline(always)]
    unsafe fn load<const T: u8>(&mut self, ptr: *const u8, stride: usize) {
        std::arch::asm!(
            "tileloadd tmm{t}, [{p} + {s}*1]",
            t = const T,
            p = in(reg) ptr,
            s = in(reg) stride,
            options(nostack, readonly),
        );
    }
    #[inline(always)]
    unsafe fn store<const T: u8>(&mut self, ptr: *mut u8, stride: usize) {
        std::arch::asm!(
            "tilestored [{p} + {s}*1], tmm{t}",
            t = const T,
            p = in(reg) ptr,
            s = in(reg) stride,
            options(nostack),
        );
    }
    #[inline(always)]
    unsafe fn dpbssd<const C: u8, const A: u8, const B: u8>(&mut self) {
        std::arch::asm!(
            "tdpbssd tmm{c}, tmm{a}, tmm{b}",
            c = const C, a = const A, b = const B,
            options(nostack),
        );
    }
    #[inline(always)]
    unsafe fn dpbusd<const C: u8, const A: u8, const B: u8>(&mut self) {
        std::arch::asm!(
            "tdpbusd tmm{c}, tmm{a}, tmm{b}",
            c = const C, a = const A, b = const B,
            options(nostack),
        );
    }
    #[inline(always)]
    unsafe fn dpbf16ps<const C: u8, const A: u8, const B: u8>(&mut self) {
        std::arch::asm!(
            "tdpbf16ps tmm{c}, tmm{a}, tmm{b}",
            c = const C, a = const A, b = const B,
            options(nostack),
        );
    }
}

/// Scalar emulation of the instruction semantics above, for the unit tests. Unconfigured tiles
/// have zero rows; loads, stores and dot products only touch the configured shape, like the
/// hardware.
#[cfg(test)]
pub(crate) struct EmuTiles {
    shapes: [TileShape; 8],
    regs: Vec<[u8; 1024]>,
}

#[cfg(test)]
impl Default for EmuTiles {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
impl EmuTiles {
    pub(crate) fn new() -> Self {
        Self {
            shapes: [TileShape { rows: 0, colsb: 0 }; 8],
            regs: vec![[0; 1024]; 8],
        }
    }

    fn accumulate(&mut self, c: u8, a: u8, b: u8, kind: DotKind) {
        let (sc, sa, sb) = (
            self.shapes[c as usize],
            self.shapes[a as usize],
            self.shapes[b as usize],
        );
        let (m, n) = (sc.rows as usize, sc.colsb as usize / 4);
        // The hardware faults on mismatched shapes; the emulation asserts it.
        assert_eq!(sa.rows as usize, m, "A rows must equal C rows");
        assert_eq!(sb.colsb as usize, 4 * n, "B columns must equal C columns");
        let k = sa.colsb as usize;
        assert_eq!(sb.rows as usize * 4, k, "B rows must be K / 4");
        let (ra, rb) = (self.regs[a as usize], self.regs[b as usize]);
        let mut result = self.regs[c as usize];
        for i in 0..m {
            for j in 0..n {
                let at = (i * 16 + j) * 4;
                match kind {
                    DotKind::Int(a_signed) => {
                        let mut sum = i32::from_le_bytes(result[at..at + 4].try_into().unwrap());
                        for kk in 0..k {
                            let av = ra[i * 64 + kk];
                            let av = if a_signed {
                                i32::from(av as i8)
                            } else {
                                i32::from(av)
                            };
                            let bv = i32::from(rb[(kk / 4) * 64 + j * 4 + kk % 4] as i8);
                            sum = sum.wrapping_add(av * bv);
                        }
                        result[at..at + 4].copy_from_slice(&sum.to_le_bytes());
                    }
                    DotKind::Bf16 => {
                        let mut sum = f32::from_le_bytes(result[at..at + 4].try_into().unwrap());
                        let bf = |bytes: &[u8; 1024], off: usize| {
                            f32::from_bits(
                                u32::from(u16::from_le_bytes([bytes[off], bytes[off + 1]])) << 16,
                            )
                        };
                        for kk in 0..k / 4 {
                            for e in 0..2 {
                                let av = bf(&ra, i * 64 + kk * 4 + e * 2);
                                let bv = bf(&rb, kk * 64 + j * 4 + e * 2);
                                sum += av * bv;
                            }
                        }
                        result[at..at + 4].copy_from_slice(&sum.to_le_bytes());
                    }
                }
            }
        }
        self.regs[c as usize] = result;
    }
}

#[cfg(test)]
#[derive(Clone, Copy)]
enum DotKind {
    /// Integer; the flag tells whether A is signed (B always is).
    Int(bool),
    Bf16,
}

#[cfg(test)]
impl TileOps for EmuTiles {
    unsafe fn configure(&mut self, config: &TileConfig) {
        let bytes = config.as_bytes();
        assert_eq!(bytes[0], 1, "only palette 1 exists");
        for t in 0..8 {
            self.shapes[t] = TileShape {
                rows: bytes[48 + t],
                colsb: u16::from_le_bytes([bytes[16 + 2 * t], bytes[17 + 2 * t]]),
            };
            self.regs[t] = [0; 1024];
        }
    }
    unsafe fn release(&mut self) {
        self.shapes = [TileShape { rows: 0, colsb: 0 }; 8];
        self.regs.iter_mut().for_each(|r| *r = [0; 1024]);
    }
    unsafe fn zero<const T: u8>(&mut self) {
        self.regs[T as usize] = [0; 1024];
    }
    unsafe fn load<const T: u8>(&mut self, ptr: *const u8, stride: usize) {
        let shape = self.shapes[T as usize];
        assert!(shape.rows > 0, "tile {T} is not configured");
        let mut data = [0u8; 1024];
        for r in 0..shape.rows as usize {
            std::ptr::copy_nonoverlapping(
                ptr.add(r * stride),
                data.as_mut_ptr().add(r * 64),
                shape.colsb as usize,
            );
        }
        self.regs[T as usize] = data;
    }
    unsafe fn store<const T: u8>(&mut self, ptr: *mut u8, stride: usize) {
        let shape = self.shapes[T as usize];
        assert!(shape.rows > 0, "tile {T} is not configured");
        for r in 0..shape.rows as usize {
            std::ptr::copy_nonoverlapping(
                self.regs[T as usize].as_ptr().add(r * 64),
                ptr.add(r * stride),
                shape.colsb as usize,
            );
        }
    }
    unsafe fn dpbssd<const C: u8, const A: u8, const B: u8>(&mut self) {
        self.accumulate(C, A, B, DotKind::Int(true));
    }
    unsafe fn dpbusd<const C: u8, const A: u8, const B: u8>(&mut self) {
        self.accumulate(C, A, B, DotKind::Int(false));
    }
    unsafe fn dpbf16ps<const C: u8, const A: u8, const B: u8>(&mut self) {
        self.accumulate(C, A, B, DotKind::Bf16);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_one_config_has_the_documented_byte_layout() {
        let config = TileConfig::palette1(&[
            TileShape {
                rows: 16,
                colsb: 64,
            },
            TileShape { rows: 8, colsb: 32 },
        ]);
        let b = config.as_bytes();
        assert_eq!(b[0], 1);
        assert_eq!(&b[1..16], &[0u8; 15]);
        assert_eq!(u16::from_le_bytes([b[16], b[17]]), 64);
        assert_eq!(u16::from_le_bytes([b[18], b[19]]), 32);
        assert_eq!(b[48], 16);
        assert_eq!(b[49], 8);
        assert_eq!(b[50..64], [0u8; 14]);
        assert_eq!(std::mem::size_of::<TileConfig>(), 64);
        assert_eq!(std::mem::align_of::<TileConfig>(), 64);
    }

    #[test]
    fn emulated_int8_tile_dot_matches_the_definition() {
        // A: 4 rows x 8 bytes, B: 2 rows x 3 columns of 4 bytes, C: 4 x 3 dwords.
        let config = TileConfig::palette1(&[
            TileShape { rows: 4, colsb: 12 },
            TileShape { rows: 4, colsb: 8 },
            TileShape { rows: 2, colsb: 12 },
        ]);
        let a: Vec<u8> = (0..32).map(|i| (i * 7 + 3) as u8).collect();
        let b: Vec<u8> = (0..24).map(|i| (i * 13 + 5) as u8).collect();
        let mut c = [0i32; 12];
        let mut tiles = EmuTiles::new();
        unsafe {
            tiles.configure(&config);
            tiles.zero::<0>();
            tiles.load::<1>(a.as_ptr(), 8);
            tiles.load::<2>(b.as_ptr(), 12);
            tiles.dpbssd::<0, 1, 2>();
            tiles.store::<0>(c.as_mut_ptr().cast(), 12);
        }
        for m in 0..4 {
            for n in 0..3 {
                let mut want = 0i32;
                for k in 0..8 {
                    want += i32::from(a[m * 8 + k] as i8)
                        * i32::from(b[(k / 4) * 12 + n * 4 + k % 4] as i8);
                }
                assert_eq!(c[m * 3 + n], want, "({m},{n})");
            }
        }
        // The unsigned-A variant treats A as 0..=255.
        let mut c2 = [0i32; 12];
        unsafe {
            tiles.zero::<0>();
            tiles.dpbusd::<0, 1, 2>();
            tiles.store::<0>(c2.as_mut_ptr().cast(), 12);
        }
        let want = i32::from(a[0]) * i32::from(b[0] as i8)
            + i32::from(a[1]) * i32::from(b[1] as i8)
            + i32::from(a[2]) * i32::from(b[2] as i8)
            + i32::from(a[3]) * i32::from(b[3] as i8)
            + i32::from(a[4]) * i32::from(b[12] as i8)
            + i32::from(a[5]) * i32::from(b[13] as i8)
            + i32::from(a[6]) * i32::from(b[14] as i8)
            + i32::from(a[7]) * i32::from(b[15] as i8);
        assert_eq!(c2[0], want);
    }
}
