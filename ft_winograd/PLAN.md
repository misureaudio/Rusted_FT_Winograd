# Winograd Fourier Transform — Implementation Plan

## Extending to Arbitrary Integer `n` and Integrating with fft_rs_1

---

## 1. Executive Summary

The existing `fft_rs_1` library implements a radix-2 Cooley–Tukey FFT that requires the input length `n` to be a power of 2. This plan describes how to extend the system to support **arbitrary integer `n`** by implementing the **Winograd Fourier Transform Algorithm (WFTA)** and complementary algorithms, organized in a new `ft_winograd` subdirectory, without modifying the existing `fft_rs_1` code.

The core insight of WFTA is that the DFT can be decomposed into:
1. An **index mapping** (via the Good–Thomas Prime Factor Algorithm or Cooley–Tukey type-II mapping) that reduces the problem to smaller prime-length transforms.
2. **Winograd short DFTs** for each prime length that achieve the **theoretical minimum number of multiplications**.
3. **Winograd minimal convolution** (via the Chinese Remainder Theorem for polynomials) for the inner convolutions.

For truly arbitrary `n` (including primes and prime powers), two complementary strategies are available:
- **Rader's algorithm**: converts a prime-length DFT into a cyclic convolution of length `n-1`.
- **Bluestein's algorithm** (chirp-z): converts any length DFT into a convolution, which is then evaluated via a power-of-2 FFT.

---

## 2. Analysis of the Existing `fft_rs_1` Codebase

### 2.1 Architecture

```
fft_rs_1/
├── Cargo.toml          # Package: fft_rs, edition 2021
├── src/
│   ├── lib.rs          # Public API re-exports
│   ├── main.rs         # Placeholder binary
│   ├── complex.rs      # Complex32, Complex64 (#[repr(C)], arithmetic, twiddle factors)
│   ├── error.rs        # FftError, FftResult, validate_length (power-of-2 check)
│   ├── fft_core.rs     # FFT<T> struct, IntoSample trait, ComplexSample trait,
│   │                    # Cooley-Tukey radix-2 DIT (forward + inverse)
│   └── plan.rs         # FFTRun<T> — pre-allocated twiddle table for repeated transforms
└── tests/
    └── integration_tests.rs  # Comprehensive API, correctness, edge-case tests
```

### 2.2 Key Design Patterns to Reuse

| Component | Purpose | Reuse Strategy |
|-----------|---------|----------------|
| `Complex32` / `Complex64` | Complex number types with arithmetic | Re-export via `use fft_rs::*` |
| `IntoSample` trait | Converts real types to complex | Extend with additional impls if needed |
| `ComplexSample` trait | Abstracts complex operations for generic code | Use as-is; it is already object-safe |
| `FftResult<T>` | Standard error type | Extend with new error variants |
| `FFT<T>` struct | Public API for one-shot transforms | Create analogous `WinogradFFT<T>` struct |
| `FFTRun<T>` struct | Reusable plan | Create `WinogradPlan<T>` |

### 2.3 Constraints of the Existing Code

- **Power-of-2 only**: `validate_length()` rejects non-power-of-2 inputs.
- **Max size 2^24**: `MAX_FFT_SIZE` limits input to ~16.7M elements.
- **Radix-2 DIT only**: No mixed-radix or prime-length support.
- **Twiddle factors on-the-fly**: `fft_core.rs` computes twiddles in the inner loop (no pre-allocation), while `plan.rs` pre-computes them.

---

## 3. Algorithm Selection for Arbitrary `n`

### 3.1 Factorization Strategy

Given an arbitrary integer `n`, factor it as:

```
n = 2^a · 3^b · 5^c · 7^d · 11^e · ... · p_k^e_k
```

The algorithm dispatch depends on the factorization:

```
┌─────────────────────────────────────────────────────────┐
│                    n given                              │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
   n = 2^k                  n ≠ 2^k
        │                         │
   ┌────┴────┐              ┌─────┴──────────────────────┐
   │         │              │                            │
n≤2^24   n>2^24     coprime factors exist?       n is prime or prime power
   │         │              │                            │
 fft_rs_1  error      Good-Thomas PFA          ┌─────────┴─────────┐
                              │                │                   │
                       Winograd     n = p^k, k>1           n = p (prime)
                       short DFTs    ┌───┴───┐               │
                       (nested)     p^k DFT  Rader + CRT    Bluestein
                                    (Winograd)  convolution  (chirp-z)
```

### 3.2 Algorithm Details

#### A. Good–Thomas Prime Factor Algorithm (PFA)

**When**: `n = n₁ · n₂` where `gcd(n₁, n₂) = 1`.

**How**: Uses the Chinese Remainder Theorem to map 1D index `m ∈ [0, n)` to 2D index `(m₁, m₂)` where `m₁ ∈ [0, n₁)`, `m₂ ∈ [0, n₂)`. This eliminates twiddle factors between stages (unlike Cooley–Tukey type-II).

**Key property**: No inter-stage twiddle multiplications — only the short DFTs contribute multiplications.

**Recursion**: Apply recursively to each factor until only prime-length DFTs remain.

**Reference**: Burrus, "The Prime Factor and Winograd Fourier Transform Algorithms", §9.1.

#### B. Winograd Short DFTs

For each prime (or small composite) length, use a pre-derived butterfly that achieves the **minimum number of multiplications**:

| N | Multiplications | Additions | Notes |
|---|----------------|-----------|-------|
| 3 | 0 | 5 | Only additions (butterfly) |
| 4 | 0 | 6 | Same as radix-2 butterfly |
| 5 | 4 | 16 | 4 non-trivial multiplications |
| 7 | 6 | 28 | 6 non-trivial multiplications |
| 8 | 8 | 25 | Can also use 3 radix-2 stages |
| 9 | 4 | 33 | As 3×3 via PFA (0 mults each) |
| 11 | 10 | 56 | |
| 13 | 12 | 72 | |
| 16 | 16 | 60 | Or 4 radix-2 stages (0 mults) |

For power-of-2 lengths, radix-2 is preferred (0 multiplications beyond the trivial ±1, ±i). For odd primes, Winograd short DFTs are essential.

**Reference**: Burrus, §6.2, Table 6.2.1 — operation counts for lengths 3 through 16.

#### C. Winograd Minimal Convolution

**When**: A cyclic convolution of length `m` is needed (e.g., after Rader's algorithm).

**How**: Factor the polynomial `x^m - 1` into relatively prime polynomials, compute residue classes, multiply in each residue class, then recombine via the polynomial CRT.

**Example**: For length-3 convolution `h = g ⊗ d (mod x³-1)`:
- Factor `x³ - 1 = (x - 1)(x² + x + 1)`
- Compute `g₁ = g mod (x - 1)`, `g₂ = g mod (x² + x + 1)` (same for `d`)
- Multiply: `h₁ = g₁ · d₁`, `h₂ = g₂ · d₂` (1 scalar + 1 polynomial mult = 2 multiplications)
- Recombine via CRT (only additions)
- Total: 2 multiplications vs. 3 for direct convolution

**Reference**: Burrus, §6.1; Parhi, "Fast Convolution", §8.3.

#### D. Rader's Algorithm (Prime-Length DFT)

**When**: `n = p` is prime.

**How**: Uses the existence of a primitive root `α` modulo `p` to re-index the DFT as a cyclic convolution of length `p-1`:

```
X[k] = x[0] - Σ_{m=1}^{p-1} x[α^m mod p] · W_p^{α^m · k mod p}
```

The second term is a cyclic convolution of length `p-1`, which is then evaluated via Winograd minimal convolution.

**Special case**: `X[0]` is computed separately as the sum of all inputs.

**Reference**: Burrus, §4.2.

#### E. Bluestein's Algorithm (Chirp-Z, Fallback for Any `n`)

**When**: Any arbitrary `n`; especially useful when `n` has large prime factors or `n` is prime and Rader's convolution length `n-1` is inconvenient.

**How**: Uses the identity `kn = ((k+n)² - k² - n²) / 2` to rewrite the DFT as:

```
X[k] = W_N^{k²/2} · Σ_{n=0}^{N-1} [x[n] · W_N^{n²/2}] · W_N^{-k·n}
     = W_N^{k²/2} · ( [x[n] · W_N^{n²/2}] ⊗ W_N^{n²/2} )[k]
```

This is a linear convolution of length `N`, which is zero-padded to length `M ≥ 2N - 1` (next power of 2) and computed via three FFT calls:

1. FFT of the chirped input (length `M`)
2. FFT of the chirp kernel (length `M`, pre-computable)
3. IFFT of the product (length `M`)

**Cost**: 3 × O(M log M) where `M = next_pow2(2N - 1)`, plus O(N) pointwise multiplications.

**Advantage**: Reduces any-length DFT to power-of-2 FFT calls — perfect for reusing `fft_rs_1`.

**Reference**: Burrus, §4.3; Wikipedia "Chirp Z-transform".

#### F. Prime-Power DFT (`n = p^k`, `k > 1`)

**When**: `n` is a power of an odd prime (e.g., 27 = 3³, 125 = 5²).

**How**: Winograd showed that the index set `ℤ_{p^k}` can be mapped to a multi-dimensional index using a generator of the multiplicative group modulo `p^k`. The DFT becomes a multidimensional DFT of coprime dimensions, each of which is a small prime-length DFT.

**Simpler approach**: Use Bluestein's algorithm as a universal fallback for prime powers with large primes.

---

## 4. Proposed `ft_winograd` Architecture

### 4.1 Directory Structure

```
ft_winograd/
├── Cargo.toml                    # Package: ft_winograd, depends on fft_rs
├── PLAN.md                       # This document
├── src/
│   ├── lib.rs                    # Public API re-exports
│   ├── main.rs                   # Demo / CLI binary
│   ├── error.rs                  # Extended error types
│   ├── factorization.rs          # Integer factorization utilities
│   ├── index_map.rs              # Good-Thomas PFA index mapping (CRT)
│   ├── winograd_dft.rs           # Winograd short DFT butterflies (N=3,5,7,11,13,...)
│   ├── winograd_conv.rs          # Winograd minimal convolution (CRT for polynomials)
│   ├── rader.rs                  # Rader's algorithm for prime-length DFT
│   ├── bluestein.rs              # Bluestein's chirp-z algorithm
│   ├── fft_arbitrary.rs          # Main dispatcher: FFT for arbitrary n
│   └── plan.rs                   # WinogradPlan for repeated transforms
└── tests/
    └── integration_tests.rs      # Comprehensive tests
```

### 4.2 Cargo.toml Design

```toml
[package]
name = "ft_winograd"
version = "0.1.0"
edition = "2021"
description = "Winograd Fourier Transform: arbitrary-length DFT in Rust"

[lib]
name = "ft_winograd"
path = "src/lib.rs"

[[bin]]
name = "ft_winograd"
path = "src/main.rs"

[dependencies]
fft_rs = { path = "../fft_rs_1" }   # Reuse Complex32, Complex64, IntoSample, etc.
```

**Rationale**: By depending on `fft_rs_1` as a library, we reuse the complex number types, arithmetic traits, and error handling. The radix-2 FFT from `fft_rs_1` is used directly for power-of-2 lengths and as a backend for Bluestein's algorithm.

### 4.3 Public API Design

```rust
// Core types
pub use fft_rs::{Complex32, Complex64, IntoSample};

// New error type (extends fft_rs error model)
pub enum DftError {
    ZeroLength,
    TooLarge(usize),
    FactorizationFailed(usize),   // Cannot factor n
    NoPrimitiveRoot(usize),       // Prime has no primitive root (should not happen)
}
pub type DftResult<T> = Result<T, DftError>;

// Main API: arbitrary-length FFT
pub struct DFT<T: IntoSample> {
    data: Vec<T>,
}

impl<T: IntoSample> DFT<T>
where T::Complex: ComplexSample
{
    pub fn new(data: Vec<T>) -> DftResult<Self>;
    pub fn from_slice(slice: &[T]) -> DftResult<Self>;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn input(&self) -> &[T];

    /// Compute forward DFT for arbitrary n
    pub fn compute(&self) -> Vec<T::Complex>;

    /// Compute inverse DFT
    pub fn idft(spectrum: Vec<T::Complex>) -> Vec<T::Complex>;
    pub fn compute_inverse(&self, spectrum: &[T::Complex]) -> Vec<T::Complex>;
}

// Reusable plan
pub struct DFTPlan<T: IntoSample>
where T::Complex: ComplexSample
{
    n: usize,
    strategy: TransformStrategy,  // Pre-computed factorization + algorithm choice
    // Pre-allocated buffers for each stage
}

impl<T: IntoSample> DFTPlan<T>
where T::Complex: ComplexSample
{
    pub fn new(n: usize) -> DftResult<Self>;
    pub fn n(&self) -> usize;
    pub fn dft(&self, input: &[T]) -> Vec<T::Complex>;
    pub fn idft(&self, input: Vec<T::Complex>) -> Vec<T::Complex>;
}
```

### 4.4 Internal Dispatch Architecture

```rust
// src/fft_arbitrary.rs

/// The chosen strategy for computing a DFT of length n.
enum TransformStrategy {
    /// Delegate to fft_rs_1 radix-2 FFT (n = 2^k)
    Radix2 { log2n: usize },

    /// Good-Thomas PFA: n = n1 * n2, gcd(n1,n2) = 1
    /// Recursively decompose into two coprime-length DFTs
    PrimeFactor { n1: usize, n2: usize },

    /// Winograd short DFT for small prime/composite n
    WinogradShort { n: usize },

    /// Rader's algorithm: n = p (prime), reduces to (p-1)-point convolution
    Rader { p: usize, primitive_root: usize },

    /// Bluestein's algorithm: universal fallback
    /// Uses next_pow2(2n-1)-point FFT from fft_rs_1
    Bluestein { m: usize },  // m = next_pow2(2n - 1)
}

/// Dispatch: compute forward DFT based on strategy
fn dft_dispatch<C: ComplexSample>(data: &mut [C], strategy: &TransformStrategy) {
    match strategy {
        TransformStrategy::Radix2 { log2n } => {
            // Delegate to fft_rs_1 (call fft_forward from fft_core)
            fft_rs_fft_forward(data, data.len(), *log2n);
        }
        TransformStrategy::PrimeFactor { n1, n2 } => {
            pfa_forward(data, *n1, *n2);
        }
        TransformStrategy::WinogradShort { n } => {
            winograd_short_dft_forward(data, *n);
        }
        TransformStrategy::Rader { p, primitive_root } => {
            rader_forward(data, *p, *primitive_root);
        }
        TransformStrategy::Bluestein { m } => {
            bluestein_forward(data, data.len(), *m);
        }
    }
}
```

---

## 5. Module Design

### 5.1 `factorization.rs` — Integer Factorization

**Purpose**: Factor arbitrary `n` and determine the optimal transform strategy.

```rust
/// Factor n into prime powers: returns Vec<(prime, exponent)>
pub fn factorize(n: usize) -> Vec<(usize, usize)>;

/// Find the primitive root modulo p (for Rader's algorithm)
pub fn primitive_root(p: usize) -> Option<usize>;

/// Determine the optimal transform strategy for length n
pub fn choose_strategy(n: usize) -> DftResult<TransformStrategy>;

/// Next power of 2 >= n
pub fn next_power_of_two(n: usize) -> usize;

/// Check if n is prime (Miller-Rabin with deterministic witnesses)
pub fn is_prime(n: usize) -> bool;
```

**Strategy selection logic**:

```
choose_strategy(n):
    if n == 1:                    →  identity (trivial)
    if n is power of 2:           →  Radix2 (delegate to fft_rs_1)
    if n is in {3, 5, 7, 11, 13}: →  WinogradShort(n)
    if n = n1 * n2, gcd(n1,n2)=1: →  PrimeFactor(n1, n2)  [recursive]
    if n is prime:                →  Rader(n, primitive_root(n))
    if n = p^k, k > 1:           →  Bluestein(n)  [fallback; or Winograd if small]
    otherwise:                    →  Bluestein(n)  [universal fallback]
```

**Threshold**: For Bluestein, the cost is 3 × O(M log M) where M = next_pow2(2n-1). This is acceptable for moderate `n` (up to ~10⁶). For very large `n` with large prime factors, the Rader + Winograd path may be faster.

### 5.2 `index_map.rs` — Good-Thomas PFA Index Mapping

**Purpose**: Implement the CRT-based index mapping for the Prime Factor Algorithm.

```rust
/// Map 1D index to 2D index using CRT: m → (m mod n1, m mod n2)
/// where n1 and n2 are coprime.
pub fn pfa_index_forward(m: usize, n1: usize, n2: usize) -> (usize, usize);

/// Map 2D index to 1D index: (m1, m2) → m
pub fn pfa_index_inverse(m1: usize, m2: usize, n1: usize, n2: usize) -> usize;

/// Precompute the CRT coefficients for n1, n2
/// Returns (c1, c2) such that m = (m1 * c1 + m2 * c2) mod (n1 * n2)
pub fn crt_coefficients(n1: usize, n2: usize) -> (usize, usize);

/// In-place Good-Thomas PFA forward transform
pub fn pfa_forward<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize);

/// In-place Good-Thomas PFA inverse transform
pub fn pfa_inverse<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize);
```

**Key insight**: The Good-Thomas PFA avoids twiddle factors because the index mapping is based on the CRT, which maps `ℤ_{n₁·n₂} ≅ ℤ_{n₁} × ℤ_{n₂}` when `gcd(n₁, n₂) = 1`. The DFT decomposes as a tensor product of two independent DFTs.

### 5.3 `winograd_dft.rs` — Winograd Short DFT Butterflies

**Purpose**: Hardcoded, hand-optimized butterflies for small prime-length DFTs.

```rust
/// 3-point DFT (0 multiplications, 5 additions)
pub fn dft3<C: ComplexSample>(data: &mut [C]);

/// 5-point DFT (4 multiplications, 16 additions)
pub fn dft5<C: ComplexSample>(data: &mut [C]);

/// 7-point DFT (6 multiplications, 28 additions)
pub fn dft7<C: ComplexSample>(data: &mut [C]);

/// 11-point DFT (10 multiplications)
pub fn dft11<C: ComplexSample>(data: &mut [C]);

/// 13-point DFT (12 multiplications)
pub fn dft13<C: ComplexSample>(data: &mut [C]);

/// Generic dispatcher for short DFTs
pub fn winograd_short_dft_forward<C: ComplexSample>(data: &mut [C], n: usize);
pub fn winograd_short_dft_inverse<C: ComplexSample>(data: &mut [C], n: usize);
```

**Implementation approach**: Each short DFT is implemented as a sequence of:
1. **Input matrix A**: Linear combinations (additions/subtractions only)
2. **Diagonal multiplication**: The non-trivial multiplications (by fixed constants)
3. **Output matrix B**: More linear combinations

The constants are pre-computed at compile time as `const` values.

**Example — DFT-3 butterfly**:

```rust
pub fn dft3<C: ComplexSample>(d: &mut [C]) {
    // Input stage (matrix A)
    let x0 = d[0];
    let x1 = d[1];
    let x2 = d[2];
    let s1 = C::add(x1, x2);
    let s2 = C::sub(x1, x2);

    // Middle stage: multiply by W_3 = e^{-2πi/3} = -0.5 - 0.866i
    // Only need: W_3 - W_3^2 = -2i·sin(2π/3) = -i·√3
    let w3_minus_w3sq = C::new(0.0, -std::f64::consts::SQRT_3);
    let m = C::mul(s2, w3_minus_w3sq);  // Only 1 non-trivial multiplication!

    // Output stage (matrix B)
    d[0] = C::add(x0, s1);                    // X[0] = x[0] + x[1] + x[2]
    d[1] = C::add(C::add(x0, C::mul(s1, C::new(-0.5, 0.0))), C::mul(m, C::new(0.5, 0.0)));
    d[2] = C::sub(C::add(x0, C::mul(s1, C::new(-0.5, 0.0))), C::mul(m, C::new(0.5, 0.0)));
}
```

**Note**: The actual DFT-3 requires only **0 non-trivial multiplications** when the constants ±½ and ±√3/2 are handled as additions (using the identity `x · ½ = x >> 1` in fixed-point, or simply as additions in floating-point: `x · ½ = x - x · ½`, which is a subtraction). In floating-point, these are still multiplications, but they are by fixed constants and can be fused or pre-computed.

### 5.4 `winograd_conv.rs` — Winograd Minimal Convolution

**Purpose**: Compute cyclic convolution with minimum multiplications using polynomial CRT.

```rust
/// Compute cyclic convolution h = g ⊗ d (mod x^m - 1)
/// using Winograd's minimal multiplication algorithm.
pub fn winograd_cyclic_conv<C: ComplexSample>(
    g: &[C],
    d: &[C],
    m: usize,
) -> Vec<C>;

/// Pre-derived convolution kernels for small lengths
mod kernels {
    // h = g ⊗ d (mod x^2 - 1): 2 multiplications
    pub fn conv2<C: ComplexSample>(g: &[C], d: &[C]) -> [C; 2];

    // h = g ⊗ d (mod x^4 - 1): 4 multiplications
    pub fn conv4<C: ComplexSample>(g: &[C], d: &[C]) -> [C; 4];

    // h = g ⊗ d (mod x^6 - 1): 5 multiplications
    pub fn conv6<C: ComplexSample>(g: &[C], d: &[C]) -> [C; 6];
}
```

**Algorithm** for general length `m`:

1. Factor `x^m - 1 = f₁(x) · f₂(x) · ... · f_k(x)` into relatively prime polynomials
2. Compute residue classes: `g_i = g mod f_i`, `d_i = d mod f_i`
3. Pointwise multiply: `h_i = g_i · d_i` (fewer multiplications because `deg(f_i) < deg(x^m - 1)`)
4. Recombine via CRT: `h = CRT(h_1, h_2, ..., h_k)`

### 5.5 `rader.rs` — Rader's Algorithm

**Purpose**: Convert prime-length DFT into cyclic convolution.

```rust
/// Rader's forward DFT for prime length p
pub fn rader_forward<C: ComplexSample>(
    data: &mut [C],
    p: usize,
    alpha: usize,  // primitive root mod p
);

/// Rader's inverse DFT for prime length p
pub fn rader_inverse<C: ComplexSample>(
    data: &mut [C],
    p: usize,
    alpha: usize,
);
```

**Steps**:

1. Compute `x[0] = data[0]` separately (the DC component)
2. Re-index: `a[m] = data[α^m mod p]`, `b[m] = W_p^{α^m mod p}` for `m = 0, ..., p-2`
3. Compute cyclic convolution `c = a ⊗ b (mod x^{p-1} - 1)` using Winograd minimal convolution
4. Re-index the result back: `X[k] = x[0] - c[log_α(k) mod (p-1)]`
5. Handle `X[0]` as the sum of all inputs

### 5.6 `bluestein.rs` — Bluestein's Chirp-Z Algorithm

**Purpose**: Universal arbitrary-length DFT via power-of-2 FFT.

```rust
/// Bluestein's forward DFT for arbitrary length n
pub fn bluestein_forward<C: ComplexSample>(
    data: &mut [C],
    n: usize,
    m: usize,  // m = next_pow2(2*n - 1)
);

/// Bluestein's inverse DFT
pub fn bluestein_inverse<C: ComplexSample>(
    data: &mut [C],
    n: usize,
    m: usize,
);
```

**Steps**:

1. Compute the chirp sequence: `a[k] = W_N^{k²/2}` for `k = 0, ..., N-1`
2. Chirp the input: `b[k] = data[k] · a[k]` for `k = 0, ..., N-1`
3. Zero-pad `b` to length `M = next_pow2(2N - 1)`
4. Zero-pad the chirp `a` to length `M`
5. Compute `B = FFT(b)`, `A = FFT(a)` using `fft_rs_1` (both length M, power of 2)
6. Pointwise multiply: `C[k] = B[k] · A[k]`
7. Compute `c = IFFT(C)` using `fft_rs_1`
8. De-chirp: `data[k] = c[k] · a[k]` for `k = 0, ..., N-1`

**Optimization**: The chirp FFT `A` can be pre-computed and cached in `DFTPlan`.

### 5.7 `fft_arbitrary.rs` — Main Dispatcher

**Purpose**: Orchestrate the transform by selecting and applying the right algorithm.

```rust
pub struct DFT<T: IntoSample> {
    data: Vec<T>,
    strategy: Option<TransformStrategy>,  // Cached after first compute
}

impl<T: IntoSample> DFT<T>
where T::Complex: ComplexSample
{
    pub fn compute(&self) -> Vec<T::Complex> {
        let n = self.data.len();
        if n == 1 {
            return vec![self.data[0].into_complex()];
        }

        let strategy = choose_strategy(n).expect("failed to choose strategy");
        let mut buf: Vec<T::Complex> = self.data.iter()
            .copied()
            .map(|s| s.into_complex())
            .collect();

        dft_dispatch(&mut buf, &strategy);
        buf
    }

    pub fn idft(data: Vec<T::Complex>) -> Vec<T::Complex> {
        let n = data.len();
        if n == 1 {
            return data;
        }

        let strategy = choose_strategy(n).expect("failed to choose strategy");
        let mut buf = data;
        idft_dispatch(&mut buf, &strategy);

        // Normalize by 1/n
        let norm = T::Complex::scalar_from_usize(n);
        for i in 0..n {
            buf[i] = T::Complex::div_scalar(buf[i], norm);
        }
        buf
    }
}
```

### 5.8 `error.rs` — Extended Error Types

```rust
use fft_rs::FftError;

pub const MAX_DFT_SIZE: usize = 1 << 26;  // Extend to ~67M for arbitrary lengths

pub enum DftError {
    ZeroLength,
    TooLarge(usize),
    FactorizationFailed(usize),
    NoPrimitiveRoot(usize),
    /// Wrapped error from fft_rs_1 (for power-of-2 delegation)
    FftError(FftError),
}

impl std::fmt::Display for DftError { ... }
impl std::error::Error for DftError { ... }

pub type DftResult<T> = Result<T, DftError>;

/// Validate length for arbitrary DFT (no power-of-2 requirement)
pub fn validate_length(n: usize) -> DftResult<()> {
    if n == 0 { return Err(DftError::ZeroLength); }
    if n > MAX_DFT_SIZE { return Err(DftError::TooLarge(n)); }
    Ok(())
}
```

### 5.9 `plan.rs` — Reusable Plan

```rust
pub struct DFTPlan<T: IntoSample>
where T::Complex: ComplexSample
{
    n: usize,
    strategy: TransformStrategy,
    chirp_cache: Option<Vec<T::Complex>>,  // For Bluestein
}

impl<T: IntoSample> DFTPlan<T>
where T::Complex: ComplexSample
{
    pub fn new(n: usize) -> DftResult<Self> {
        validate_length(n)?;
        let strategy = choose_strategy(n)?;

        // Pre-compute chirp for Bluestein if needed
        let chirp_cache = if let TransformStrategy::Bluestein { m } = &strategy {
            Some(compute_chirp::<T::Complex>(n, *m))
        } else {
            None
        };

        Ok(DFTPlan { n, strategy, chirp_cache })
    }

    pub fn dft(&self, input: &[T]) -> Vec<T::Complex> { ... }
    pub fn idft(&self, input: Vec<T::Complex>) -> Vec<T::Complex> { ... }
}
```

---

## 6. Integration with `fft_rs_1`

### 6.1 Dependency Relationship

```
ft_winograd/
└── Cargo.toml
    └── fft_rs = { path = "../fft_rs_1" }
```

### 6.2 What is Reused from `fft_rs_1`

| Component | How it is reused |
|-----------|-----------------|
| `Complex32`, `Complex64` | Imported via `use fft_rs::*`; used as the complex number types |
| `IntoSample` trait | Imported; used for the same input type conversions |
| `ComplexSample` trait | Imported; used for generic algorithm implementations |
| `fft_forward`, `fft_inverse` | Called directly for power-of-2 lengths (Radix2 strategy) |
| `FFTRun` | Used internally by Bluestein for the 3 FFT calls |
| `is_power_of_two` | Used in `factorization.rs` |

### 6.3 What is NOT Reused (and Why)

| Component | Why not reused |
|-----------|---------------|
| `FftError` | The power-of-2 constraint is baked in; we need a more general error type |
| `validate_length` | Rejects non-power-of-2; we need a version that accepts any positive integer |
| `FFT<T>` struct | Hardcoded to power-of-2; we need a dispatcher that handles arbitrary `n` |

### 6.4 Internal Access to fft_rs_1 Functions

The `fft_rs_1` library does not currently expose `fft_forward` and `fft_inverse` as public functions. To integrate, we have two options:

**Option A — Use the public API**: Call `FFT::<T>::new(data).unwrap().compute()` for power-of-2 lengths. This is clean but adds the overhead of validation and allocation.

**Option B — Request `pub(crate)` exposure**: If we had control over `fft_rs_1`, we could make `fft_forward` and `fft_inverse` `pub` so they can be called directly. **Since we cannot modify `fft_rs_1`**, we must use Option A.

**Chosen approach**: Use the public `FFT<T>` API from `fft_rs_1` for:
- Radix2 strategy (power-of-2 delegation)
- Bluestein's 3 FFT calls

This is the cleanest integration and respects the immutability constraint on `fft_rs_1`.

### 6.5 Bluestein Integration Example

```rust
fn bluestein_forward<C: ComplexSample>(data: &mut [C], n: usize, m: usize)
where
    C::Scalar: IntoSample<Complex = C>,
{
    // Step 1-2: Chirp the input
    let mut b = Vec::with_capacity(m);
    for k in 0..n {
        let chirp = C::twiddle(2 * n, k * k % (2 * n));  // W_{2N}^{k²}
        b.push(C::mul(data[k], chirp));
    }
    for _ in n..m {
        b.push(C::zero());
    }

    // Step 3-4: Chirp kernel (zero-padded)
    let mut a = Vec::with_capacity(m);
    a.push(C::one());
    for k in 1..n {
        let chirp = C::twiddle(2 * n, k * k % (2 * n));
        a.push(chirp);
    }
    for _ in n..m {
        a.push(C::zero());
    }

    // Step 5: FFT of both (use fft_rs_1 — m is power of 2)
    let fft_b = fft_rs::FFT::<C::Scalar>::ifft(b);  // WRONG — need to adapt
    // Actually, we need to use the FFT on complex data directly.
    // Since fft_rs_1 only accepts real input, we need a different approach.
}
```

**Critical issue**: The `fft_rs_1` library's `FFT<T>` only accepts **real-valued** input types (via `IntoSample`). The Bluestein algorithm requires FFT of **complex-valued** data.

**Solution**: We have several options:

1. **Implement a standalone complex FFT** in `ft_winograd` that works on `Vec<C>` directly (essentially duplicating the Cooley-Tukey code but for complex input). This is clean and self-contained.

2. **Use a "real-interleaved" trick**: Pack complex data into a real vector of double length, compute a real FFT, then unpack. This is complex and error-prone.

3. **Use `FFTRun` with a custom `IntoSample` impl**: If we can define a type that implements `IntoSample` and returns the complex data directly, we could use `fft_rs_1`'s infrastructure.

**Chosen approach**: Option 1 — implement a minimal `fft_complex` and `ifft_complex` function in a new module `fft_complex.rs` that operates on `Vec<C>` where `C: ComplexSample`. This is a thin wrapper around the Cooley-Tukey algorithm from `fft_rs_1`, adapted for complex input. The code is essentially identical to `fft_core.rs` but takes `Vec<C>` as input instead of `Vec<T>` where `T: IntoSample`.

This is justified because:
- The code is short (~50 lines)
- It avoids duplication of the core algorithm
- It enables Bluestein to work with complex data
- It does not modify `fft_rs_1`

---

## 7. Implementation Order

### Phase 1: Foundation (Week 1-2)

1. **Create project structure** — `Cargo.toml`, `lib.rs`, empty modules
2. **`error.rs`** — Extended error types, validation for arbitrary `n`
3. **`factorization.rs`** — Trial division factorization, Miller-Rabin primality, primitive root finder, `choose_strategy()`
4. **`fft_complex.rs`** — Complex FFT/IFFT for Bluestein (thin wrapper over Cooley-Tukey)
5. **Basic tests** — Factorization, strategy selection

### Phase 2: Winograd Short DFTs (Week 3-4)

6. **`winograd_dft.rs`** — Implement DFT-3, DFT-5, DFT-7 butterflies
7. **`winograd_conv.rs`** — Implement conv2, conv4, conv6 kernels
8. **Tests** — Verify against naive DFT for each short length

### Phase 3: Prime Factor Algorithm (Week 5)

9. **`index_map.rs`** — CRT-based index mapping, PFA forward/inverse
10. **Tests** — Verify PFA decomposition for n = 15, 21, 35, etc.

### Phase 4: Prime-Length DFT (Week 6)

11. **`rader.rs`** — Rader's algorithm with Winograd convolution
12. **Tests** — Verify against naive DFT for primes 11, 13, 17, 19

### Phase 5: Bluestein Fallback (Week 7)

13. **`bluestein.rs`** — Bluestein's chirp-z algorithm
14. **Tests** — Verify for arbitrary lengths: 10, 12, 27, 97, 100, 1234

### Phase 6: Dispatcher and API (Week 8)

15. **`fft_arbitrary.rs`** — Main `DFT<T>` struct, dispatch logic
16. **`plan.rs`** — `DFTPlan<T>` for repeated transforms
17. **`lib.rs`** — Public API re-exports
18. **`main.rs`** — Demo CLI

### Phase 7: Testing and Benchmarks (Week 9-10)

19. **Integration tests** — Comprehensive correctness tests
20. **Benchmarks** — Compare with fft_rs_1 for power-of-2, compare with naive DFT for arbitrary lengths
21. **Edge cases** — n=1, n=2, n=3, large primes, etc.

---

## 8. Correctness Verification

### 8.1 Naive DFT Reference

```rust
fn naive_dft<C: ComplexSample>(input: &[C], n: usize) -> Vec<C> {
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = C::zero();
        for m in 0..n {
            let tw = C::twiddle(n, k * m);
            sum = C::add(sum, C::mul(tw, input[m]));
        }
        out.push(sum);
    }
    out
}
```

### 8.2 Test Categories

| Category | Test Cases |
|----------|-----------|
| **Power of 2** | n = 4, 8, 16, 256, 1024 — verify identical to fft_rs_1 |
| **Small primes** | n = 3, 5, 7, 11, 13 — Winograd short DFT |
| **Composite coprime** | n = 6, 10, 14, 15, 21, 35 — PFA |
| **Prime via Rader** | n = 17, 19, 23, 29 — Rader + Winograd conv |
| **Prime power** | n = 9, 25, 27 — Bluestein |
| **Arbitrary** | n = 12, 18, 100, 1234, 9999 — Bluestein |
| **Round-trip** | DFT → IDFT → original for all categories |
| **Parseval** | Energy conservation for all categories |
| **Hermitian symmetry** | Real input → conjugate-symmetric output |
| **Delta** | δ[n] → all ones |
| **Constant** | constant → DC only |
| **Convolution theorem** | DFT(a ⊗ b) = DFT(a) · DFT(b) |

### 8.3 Cross-Validation with fft_rs_1

For power-of-2 lengths, the output of `ft_winograd` must be **bit-identical** (or within floating-point tolerance) to `fft_rs_1`:

```rust
#[test]
fn cross_validate_power_of_two() {
    let n = 1024;
    let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();

    let fft_rs_out = fft_rs::FFT::new(input.clone()).unwrap().compute();
    let wft_out = ft_winograd::DFT::new(input).unwrap().compute();

    for i in 0..n {
        assert!((fft_rs_out[i].re - wft_out[i].re).abs() < 1e-12);
        assert!((fft_rs_out[i].im - wft_out[i].im).abs() < 1e-12);
    }
}
```

---

## 9. Performance Considerations

### 9.1 Expected Complexity

| Algorithm | Multiplications | Additions |
|-----------|----------------|-----------|
| Naive DFT | O(n²) | O(n²) |
| Radix-2 FFT | O(n log n) | O(n log n) |
| Winograd (prime p) | O(p) | O(p) |
| Bluestein | 3 × O(M log M), M = next_pow2(2n-1) | Same |
| PFA + Winograd | O(n log n) with smaller constant | O(n log n) |

### 9.2 Optimization Opportunities

1. **Pre-computed constants**: All Winograd short DFT constants are `const` values.
2. **Zero-allocation path**: For `DFTPlan`, pre-allocate all working buffers.
3. **SIMD vectorization**: The `Complex32`/`Complex64` types use `#[repr(C)]`, making them suitable for SIMD via `std::simd` (nightly) or `packed_simd`.
4. **Cache the chirp**: In Bluestein, the chirp FFT is the same for a given `n`; cache it in `DFTPlan`.
5. **Avoid redundant factorization**: Cache the factorization result in `DFTPlan`.

### 9.3 When to Use Which Algorithm

| n | Recommended Algorithm | Rationale |
|---|----------------------|-----------|
| 2^k, k ≤ 24 | Radix-2 (fft_rs_1) | Fastest, well-optimized |
| 3, 5, 7, 11, 13 | Winograd short DFT | Minimum multiplications |
| n = n₁·n₂, gcd=1 | PFA + Winograd | No twiddle factors |
| p (prime, p ≤ 13) | Winograd short DFT | Direct butterfly |
| p (prime, p > 13) | Rader + Winograd conv | Minimum multiplications |
| p^k (odd prime power) | Bluestein | Universal, uses fft_rs_1 |
| Other composite | Bluestein | Universal fallback |

---

## 10. References

### Primary References

1. **Burrus, C. S.** — *Fast Fourier Transforms*, LibreTexts. The definitive reference for Winograd, Rader, Bluestein, and PFA algorithms. Chapters 4-9 are essential.
   - https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Fast_Fourier_Transforms_(Burrus)

2. **Winograd, S. (1976)** — *On computing the discrete Fourier transform*, Mathematics of Computation, 32(144), 175-199. The original WFTA paper.

3. **Rader, C. M. (1968)** — *Discrete Fourier transforms when the number of data samples is prime*, IEEE Transactions on Audio and Electroacoustics, 16(2), 136-137.

4. **Bluestein, L. I. (1968)** — *Linear complexity mapping for discrete Fourier analysis*, IEEE Spectrum, 6(8), 42-50.

5. **Good, I. J. (1958)** — *The interaction algorithm and practical Fourier analysis*, Journal of the Royal Statistical Society, Series B, 20, 361-372. The Prime Factor Algorithm.

### Implementation References

6. **Parhi, K. K.** — *VLSI Digital Signal Processing Systems: Design and Implementation*, Wiley, 1999. Chapter 8: Fast Convolution. Winograd convolution via CRT.

7. **Makhoul, J. (1975)** — *A fast algorithm for the exact calculation of the discrete Fourier transform*, IEEE Transactions on Information Theory, 20(3), 359-363. Bluestein's algorithm for general DFT.

8. **Silverman, J. H. (1977)** — *Winograd FFT Algorithm Programming Guide*, MIT Project MAC Technical Memo TR-271. Practical guide to implementing WFTA.

9. **Encyclopedia of Mathematics** — *Winograd Fourier transform algorithm*. Overview of large vs. small WFTA strategies.
   - https://encyclopediaofmath.org/wiki/Winograd_Fourier_transform_algorithm

### Code References

10. **FFTW** — The leading C FFT library. Uses PFA + Winograd for composite coprime lengths, and Bluestein for prime lengths.
    - http://www.fftw.org

11. **rocFFT** — AMD's FFT library. Bluestein design document is well-documented.
    - https://rocm.docs.amd.com/projects/rocFFT/en/docs-6.2.0/design/bluestein.html

12. **GSL** — GNU Scientific Library. Mixed-radix FFT for arbitrary lengths.
    - https://www.gnu.org/software/gsl/doc/html/fft.html

---

## 11. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Floating-point accuracy differs from fft_rs_1 | Medium | Use same Complex types; test with Parseval and round-trip |
| Winograd short DFT constants have rounding errors | Medium | Derive constants symbolically; test against naive DFT |
| Factorization is slow for large n | Low | Trial division up to √n is fast for n ≤ 2^26 (~8K iterations) |
| Bluestein's 3× overhead for prime lengths | Medium | Use Rader + Winograd for primes; Bluestein only as fallback |
| Complex FFT duplication introduces bugs | Low | The code is thin; cross-validate with fft_rs_1 for power-of-2 |
| PFA index mapping errors | Medium | Extensive unit tests for CRT forward/inverse mapping |

---

## 12. Summary

This plan describes a comprehensive approach to extending FFT computation from power-of-2 to **arbitrary integer lengths** while:

1. **Not modifying** the existing `fft_rs_1` codebase
2. **Reusing** the complex types, traits, and radix-2 FFT from `fft_rs_1`
3. **Implementing** the Winograd Fourier Transform Algorithm with its full suite of sub-algorithms (short DFTs, minimal convolution, Rader, Bluestein, PFA)
4. **Providing** a clean, consistent API that mirrors `fft_rs_1`'s design patterns

The `ft_winograd` crate will serve as a drop-in replacement for `fft_rs_1` when arbitrary-length DFT is needed, while delegating to `fft_rs_1` for power-of-2 lengths where the radix-2 Cooley–Tukey algorithm is optimal.