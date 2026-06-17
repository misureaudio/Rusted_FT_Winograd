# Winograd Fourier Transform — Implementation Plan (REVISED)

## Extending to Arbitrary Integer `n` and Integrating with fft_rs_1

> **Revision Note**: This is a revised version of the original plan, incorporating critical feedback from an expert review (PLAN_review_Gemini3_1.md). The four major changes are:

> 1. **Rader's convolution strategy** — replaced Winograd minimal convolution with Convolution Theorem + radix-2 FFT for large primes.
> 2. **Elimination of `fft_complex.rs`** — replaced with a single `IntoSample` impl for `Complex64`/`Complex32`.
> 3. **Configurable strategy thresholds** — Bluestein may dominate PFA for N > 1000 due to cache locality.
> 4. **Bluestein chirp cache memory documentation** — explicit warning about O(M) memory for large N.

---

## 1. Executive Summary

The existing `fft_rs_1` library implements a radix-2 Cooley–Tukey FFT that requires the input length `n` to be a power of 2. This plan describes how to extend the system to support **arbitrary integer `n`** by implementing the **Winograd Fourier Transform Algorithm (WFTA)** and complementary algorithms, organized in a new `ft_winograd` subdirectory, without modifying the existing `fft_rs_1` code.

The core insight of WFTA is that the DFT can be decomposed into:

1. An **index mapping** (via the Good–Thomas Prime Factor Algorithm) that reduces the problem to smaller prime-length transforms.
2. **Winograd short DFTs** for each small prime length that achieve the **theoretical minimum number of multiplications**.
3. **Winograd minimal convolution** for small cyclic convolutions arising from Rader's algorithm.

For truly arbitrary `n` (including primes and prime powers), two complementary strategies are available:

- **Rader's algorithm**: converts a prime-length DFT into a cyclic convolution of length `n-1`, evaluated via the **Convolution Theorem** (FFT → multiply → IFFT) for large primes, or Winograd minimal convolution for small primes.
- **Bluestein's algorithm** (chirp-z): universal fallback that converts any-length DFT into a convolution evaluated via three power-of-2 FFT calls.

**Key architectural principle**: Bluestein's algorithm is the workhorse for large non-power-of-2 lengths. Its 3× O(M log M) cost has very low constants because it delegates to the cache-friendly, contiguous radix-2 backend in `fft_rs_1`. The PFA, while theoretically elegant, has stride-jumping memory access patterns that cause cache misses on modern CPUs.

---

## 2. Analysis of the Existing `fft_rs_1` Codebase

### 2.1 Architecture

```text
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
| `IntoSample` trait | Converts real types to complex | **EXTEND** with impl for `Complex32` and `Complex64` (pass-through) |
| `ComplexSample` trait | Abstracts complex operations for generic code | Use as-is; already object-safe |
| `FftResult<T>` | Standard error type | Extend with new error variants |
| `FFT<T>` struct | Public API for one-shot transforms | **USE DIRECTLY** for power-of-2 and Bluestein FFT calls |
| `FFTRun<T>` struct | Reusable plan | Use internally by Bluestein for the 3 FFT calls |

### 2.3 Constraints of the Existing Code

- **Power-of-2 only**: `validate_length()` rejects non-power-of-2 inputs.
- **Max size 2^24**: `MAX_FFT_SIZE` limits input to ~16.7M elements.
- **Radix-2 DIT only**: No mixed-radix or prime-length support.
- **Real-valued input only**: `FFT<T>` requires `T: IntoSample`, which is only implemented for `i32`, `i64`, `f32`, `f64`.
- **Twiddle factors on-the-fly**: `fft_core.rs` computes twiddles in the inner loop; `plan.rs` pre-computes them.

---

## 3. Algorithm Selection for Arbitrary `n`

### 3.1 Factorization Strategy

Given an arbitrary integer `n`, factor it as:

```text
n = 2^a · 3^b · 5^c · 7^d · 11^e · ... · p_k^e_k
```

The algorithm dispatch depends on the factorization and on **configurable thresholds**:

```
┌─────────────────────────────────────────────────────────────────┐
│                          n given                                │
└────────────────────────┬────────────────────────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
       n = 2^k                 n ≠ 2^k
            │                         │
       ┌────┴────┐          ┌────────┴────────────────────────────┐
       │         │          │                                     │
   n≤2^24   n>2^24   n ≤ THRESHOLD_PFA   n > THRESHOLD_PFA
       │         │          │                      │
 fft_rs_1  error   n in {3,5,7,11,13}?          Bluestein
                         │           (universal —
                    ┌────┴────┐         cache-friendly,
                    │         │         low constants)
               Winograd   n = n1*n2,       │
                Short     gcd=1            │
               DFT(n)      │               │
                         PFA +            │
                      Winograd Short     │
                      DFTs (nested)      │
                                         │
                              p (prime) ≤ THRESHOLD_RADER?
                                         │
                                    ┌────┴────┐
                                    │         │
                                  Yes        No
                                    │         │
                              Rader +    Bluestein
                         Winograd conv   (for large
                         (small p-1)     primes — use
                                         Convolution Thm)
```

**Configurable thresholds** (default values, adjustable post-benchmarking):

```rust
/// Maximum n at which PFA + Winograd is preferred over Bluestein.
/// Default: 500. Above this, Bluestein's cache-friendly radix-2 delegation
/// is typically faster despite the 3× FFT overhead.
const THRESHOLD_PFA: usize = 500;

/// Maximum prime p at which Rader + Winograd convolution is preferred
/// over Bluestein. Above this, the convolution length (p-1) is too large
/// for hand-written Winograd kernels; use Bluestein instead.
const THRESHOLD_RADER: usize = 13;
```

### 3.2 Algorithm Details

#### A. Good–Thomas Prime Factor Algorithm (PFA)

**When**: `n = n₁ · n₂` where `gcd(n₁, n₂) = 1`, **and** `n ≤ THRESHOLD_PFA`.

**How**: Uses the Chinese Remainder Theorem to map 1D index `m ∈ [0, n)` to 2D index `(m₁, m₂)` where `m₁ ∈ [0, n₁)`, `m₂ ∈ [0, n₂)`. This eliminates twiddle factors between stages (unlike Cooley–Tukey type-II).

**Key property**: No inter-stage twiddle multiplications — only the short DFTs contribute multiplications.

**Recursion**: Apply recursively to each factor until only prime-length DFTs remain.

**⚠️ Cache consideration**: The PFA stride-jumping access pattern causes L1/L2 cache misses on modern CPUs. For `n > THRESHOLD_PFA`, Bluestein (which uses contiguous memory) is typically faster in practice despite the theoretical 3× FFT overhead.

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

**When**: A cyclic convolution of **small** length `m` is needed (e.g., after Rader's algorithm for small primes where `p-1` is small).

**How**: Factor the polynomial `x^m - 1` into relatively prime polynomials, compute residue classes, multiply in each residue class, then recombine via the polynomial CRT.

**Example**: For length-3 convolution `h = g ⊗ d (mod x³-1)`:

- Factor `x³ - 1 = (x - 1)(x² + x + 1)`
- Compute `g₁ = g mod (x - 1)`, `g₂ = g mod (x² + x + 1)` (same for `d`)
- Multiply: `h₁ = g₁ · d₁`, `h₂ = g₂ · d₂` (1 scalar + 1 polynomial mult = 2 multiplications)
- Recombine via CRT (only additions)
- Total: 2 multiplications vs. 3 for direct convolution

**⚠️ Limitation**: Hard-written Winograd convolution kernels are only practical for small `m` (up to about 12). For larger lengths, use the Convolution Theorem (FFT-based) instead.

**Reference**: Burrus, §6.1; Parhi, "Fast Convolution", §8.3.

#### D. Rader's Algorithm (Prime-Length DFT) — REVISED

**When**: `n = p` is prime, **and** `p ≤ THRESHOLD_RADER` (default: 13).

**How**: Uses the existence of a primitive root `α` modulo `p` to re-index the DFT as a cyclic convolution of length `p-1`:

```text
X[k] = x[0] - Σ_{m=1}^{p-1} x[α^m mod p] · W_p^{α^m · k mod p}
```

**CRITICAL CHANGE from original plan**: The convolution of length `p-1` is evaluated as follows:

| Convolution length `p-1` | Evaluation method |
|--------------------------|-------------------|
| `p-1 ≤ 12` (i.e., `p ≤ 13`) | Winograd minimal convolution (hand-written kernels) |
| `p-1 > 12` (i.e., `p > 13`) | **Convolution Theorem**: zero-pad to `M = next_pow2(p-1)`, then FFT → multiply → IFFT via `fft_rs_1` |

**Why this change?** For large primes (e.g., p = 97, so p-1 = 96), writing a hand-derived Winograd convolution kernel is impractical — the polynomial factorization of `x⁹⁶ - 1` requires dozens of residue classes. The Convolution Theorem approach is simpler, more general, and delegates to the optimized radix-2 backend.

**Special case**: `X[0]` is computed separately as the sum of all inputs.

**Reference**: Burrus, §4.2.

#### E. Bluestein's Algorithm (Chirp-Z, Universal Fallback)

**When**: Any arbitrary `n`; the preferred strategy for:

- `n > THRESHOLD_PFA` (cache-friendly radix-2 delegation beats PFA)
- Prime `p > THRESHOLD_RADER` (simpler than Rader + Convolution Theorem)
- Prime powers `p^k` (no PFA decomposition possible)
- Composite with no coprime factorization available

**How**: Uses the identity `kn = ((k+n)² - k² - n²) / 2` to rewrite the DFT as:

```text
X[k] = W_N^{k²/2} · Σ_{n=0}^{N-1} [x[n] · W_N^{n²/2}] · W_N^{-k·n}
     = W_N^{k²/2} · ( [x[n] · W_N^{n²/2}] ⊗ W_N^{n²/2} )[k]
```

This is a linear convolution of length `N`, which is zero-padded to length `M ≥ 2N - 1` (next power of 2) and computed via three FFT calls:

1. FFT of the chirped input (length `M`)
2. FFT of the chirp kernel (length `M`, pre-computable)
3. IFFT of the product (length `M`)

**Cost**: 3 × O(M log M) where `M = next_pow2(2N - 1)`, plus O(N) pointwise multiplications.

**Advantage**: Reduces any-length DFT to power-of-2 FFT calls — perfect for reusing `fft_rs_1`. The radix-2 FFT has excellent cache locality and low constants, making Bluestein competitive even for moderate `N`.

**⚠️ Memory consideration**: For `N = 8,000,000`, `M = 16,777,216`, the chirp FFT cache requires a `Vec<Complex64>` of length M, which is **~268 MB of RAM**. This is documented in the API.

**Reference**: Burrus, §4.3; Wikipedia "Chirp Z-transform".

#### F. Prime-Power DFT (`n = p^k`, `k > 1`)

**When**: `n` is a power of an odd prime (e.g., 27 = 3³, 125 = 5²).

**How**: Use Bluestein's algorithm as the universal fallback. For small prime powers (e.g., n = 9 = 3²), a Winograd short DFT may be available, but Bluestein is simpler and equally correct.

---

## 4. Proposed `ft_winograd` Architecture

### 4.1 Directory Structure

```text
ft_winograd/
├── Cargo.toml                    # Package: ft_winograd, depends on fft_rs
├── PLAN.md                       # This document (revised)
├── src/
│   ├── lib.rs                    # Public API re-exports + IntoSample impl for Complex32/64
│   ├── main.rs                   # Demo / CLI binary
│   ├── error.rs                  # Extended error types
│   ├── factorization.rs          # Integer factorization + strategy selection
│   ├── index_map.rs              # Good-Thomas PFA index mapping (CRT)
│   ├── winograd_dft.rs           # Winograd short DFT butterflies (N=3,5,7,11,13,...)
│   ├── winograd_conv.rs          # Winograd minimal convolution (small lengths)
│   ├── rader.rs                  # Rader's algorithm for prime-length DFT
│   ├── bluestein.rs              # Bluestein's chirp-z algorithm
│   ├── fft_arbitrary.rs          # Main dispatcher: FFT for arbitrary n
│   └── plan.rs                   # DFTPlan for repeated transforms
└── tests/
    └── integration_tests.rs      # Comprehensive tests
```

**NOTICE**: The original plan included `fft_complex.rs` for a standalone complex FFT. **This module is eliminated** by the `IntoSample` pass-through trick (see Section 6.5).

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

### 4.3 Public API Design

```rust
// Core types
pub use fft_rs::{Complex32, Complex64, IntoSample};

// New error type (extends fft_rs error model)
pub enum DftError {
    ZeroLength,
    TooLarge(usize),
    FactorizationFailed(usize),
    NoPrimitiveRoot(usize),
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
    strategy: TransformStrategy,
    chirp_cache: Option<Vec<T::Complex>>,  // For Bluestein
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

/// Configurable thresholds (adjustable post-benchmarking).
const THRESHOLD_PFA: usize = 500;
const THRESHOLD_RADER: usize = 13;

/// The chosen strategy for computing a DFT of length n.
enum TransformStrategy {
    /// Delegate to fft_rs_1 radix-2 FFT (n = 2^k)
    Radix2 { log2n: usize },

    /// Good-Thomas PFA: n = n1 * n2, gcd(n1,n2) = 1
    PrimeFactor { n1: usize, n2: usize },

    /// Winograd short DFT for small prime/composite n
    WinogradShort { n: usize },

    /// Rader's algorithm: n = p (prime, p ≤ THRESHOLD_RADER)
    Rader { p: usize, primitive_root: usize },

    /// Bluestein's algorithm: universal fallback
    Bluestein { m: usize },  // m = next_pow2(2n - 1)
}
```

---

## 5. Module Design

### 5.1 `factorization.rs` — Integer Factorization and Strategy Selection

**Purpose**: Factor arbitrary `n` and determine the optimal transform strategy.

```rust
/// Factor n into prime powers: returns Vec<(prime, exponent)>
pub fn factorize(n: usize) -> Vec<(usize, usize)>;

/// Find the primitive root modulo p (for Rader's algorithm)
pub fn primitive_root(p: usize) -> Option<usize>;

/// Determine the optimal transform strategy for length n.
/// Uses configurable thresholds THRESHOLD_PFA and THRESHOLD_RADER.
pub fn choose_strategy(n: usize) -> DftResult<TransformStrategy>;

/// Next power of 2 >= n
pub fn next_power_of_two(n: usize) -> usize;

/// Check if n is prime (Miller-Rabin with deterministic witnesses)
pub fn is_prime(n: usize) -> bool;
```

**Strategy selection logic** (REVISED with thresholds):

```text
choose_strategy(n):
    if n == 1:                    →  identity (trivial)
    if n is power of 2:           →  Radix2 (delegate to fft_rs_1)
    if n is in {3, 5, 7, 11, 13}: →  WinogradShort(n)
    if n is prime:
        if n ≤ THRESHOLD_RADER:   →  Rader(n, primitive_root(n))
        else:                      →  Bluestein(n)  [was: Rader for all primes]
    if n = p^k, k > 1:           →  Bluestein(n)  [prime power]
    if n ≤ THRESHOLD_PFA:        →  try PFA decomposition
        if n = n1 * n2, gcd=1:   →  PrimeFactor(n1, n2)  [recursive]
        else:                     →  Bluestein(n)
    otherwise:                    →  Bluestein(n)  [was: try PFA; now: Bluestein]
```

**Key change from original plan**: For `n > THRESHOLD_PFA`, skip the PFA and go directly to Bluestein. The PFA's stride-jumping memory pattern causes cache misses that outweigh the theoretical multiplication savings.

### 5.2 `index_map.rs` — Good-Thomas PFA Index Mapping

**Purpose**: Implement the CRT-based index mapping for the Prime Factor Algorithm.

```rust
/// Map 1D index to 2D index using CRT: m → (m mod n1, m mod n2)
pub fn pfa_index_forward(m: usize, n1: usize, n2: usize) -> (usize, usize);

/// Map 2D index to 1D index: (m1, m2) → m
pub fn pfa_index_inverse(m1: usize, m2: usize, n1: usize, n2: usize) -> usize;

/// Precompute the CRT coefficients for n1, n2
pub fn crt_coefficients(n1: usize, n2: usize) -> (usize, usize);

/// In-place Good-Thomas PFA forward transform
pub fn pfa_forward<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize);

/// In-place Good-Thomas PFA inverse transform
pub fn pfa_inverse<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize);
```

### 5.3 `winograd_dft.rs` — Winograd Short DFT Butterflies

**Purpose**: Hardcoded, hand-optimized butterflies for small prime-length DFTs.

```rust
pub fn dft3<C: ComplexSample>(data: &mut [C]);
pub fn dft5<C: ComplexSample>(data: &mut [C]);
pub fn dft7<C: ComplexSample>(data: &mut [C]);
pub fn dft11<C: ComplexSample>(data: &mut [C]);
pub fn dft13<C: ComplexSample>(data: &mut [C]);

pub fn winograd_short_dft_forward<C: ComplexSample>(data: &mut [C], n: usize);
pub fn winograd_short_dft_inverse<C: ComplexSample>(data: &mut [C], n: usize);
```

### 5.4 `winograd_conv.rs` — Winograd Minimal Convolution (Small Lengths Only)

**Purpose**: Compute cyclic convolution with minimum multiplications for **small** lengths (m ≤ 12).

```rust
/// Compute cyclic convolution for small lengths using Winograd CRT.
/// For m > 12, falls back to FFT-based convolution.
pub fn winograd_cyclic_conv<C: ComplexSample>(
    g: &[C],
    d: &[C],
    m: usize,
) -> Vec<C>;

/// FFT-based cyclic convolution for larger lengths.
/// Zero-pads to M = next_pow2(m), then FFT → multiply → IFFT.
pub fn fft_cyclic_conv<C: ComplexSample>(
    g: &[C],
    d: &[C],
    m: usize,
) -> Vec<C>;

/// Pre-derived kernels for small lengths
mod kernels {
    pub fn conv2<C: ComplexSample>(g: &[C], d: &[C]) -> [C; 2];
    pub fn conv4<C: ComplexSample>(g: &[C], d: &[C]) -> [C; 4];
    pub fn conv6<C: ComplexSample>(g: &[C], d: &[C]) -> [C; 6];
}
```

### 5.5 `rader.rs` — Rader's Algorithm (REVISED)

**Purpose**: Convert prime-length DFT into cyclic convolution, evaluated via the appropriate method.

```rust
/// Rader's forward DFT for prime length p.
/// Uses Winograd convolution for p ≤ THRESHOLD_RADER (small p-1),
/// and FFT-based convolution for larger primes.
pub fn rader_forward<C: ComplexSample>(
    data: &mut [C],
    p: usize,
    alpha: usize,  // primitive root mod p
);

pub fn rader_inverse<C: ComplexSample>(
    data: &mut [C],
    p: usize,
    alpha: usize,
);
```

**Steps** (REVISED):

1. Compute `x[0] = data[0]` separately (the DC component)
2. Re-index: `a[m] = data[α^m mod p]`, `b[m] = W_p^{α^m mod p}` for `m = 0, ..., p-2`
3. **Choose convolution method**:
   - If `p-1 ≤ 12`: Winograd minimal convolution
   - Otherwise: FFT-based convolution (zero-pad to `M = next_pow2(p-1)`, FFT → multiply → IFFT)
4. Re-index the result back
5. Handle `X[0]` as the sum of all inputs

### 5.6 `bluestein.rs` — Bluestein's Chirp-Z Algorithm

**Purpose**: Universal arbitrary-length DFT via power-of-2 FFT.

```rust
/// Bluestein's forward DFT for arbitrary length n.
/// Memory: requires O(M) where M = next_pow2(2n-1).
/// For n = 8,000,000, M = 16,777,216 → ~268 MB for Complex64.
pub fn bluestein_forward<C: ComplexSample>(
    data: &mut [C],
    n: usize,
    m: usize,  // m = next_pow2(2*n - 1)
);

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
5. Compute `B = FFT(b)`, `A = FFT(a)` using `fft_rs_1` (both length M, power of 2) — **see Section 6.5 for how this works with complex data**
6. Pointwise multiply: `C[k] = B[k] · A[k]`
7. Compute `c = IFFT(C)` using `fft_rs_1`
8. De-chirp: `data[k] = c[k] · a[k]` for `k = 0, ..., N-1`

**Optimization**: The chirp FFT `A` can be pre-computed and cached in `DFTPlan`.

**⚠️ Memory footprint documentation**:

```rust
/// Compute the memory required for a Bluestein DFT of length n.
/// Returns M = next_pow2(2n-1) and the approximate byte count
/// for the chirp cache (M * sizeof(Complex64) = M * 16 bytes).
pub fn bluestein_memory_estimate(n: usize) -> (usize, usize);
```

### 5.7 `fft_arbitrary.rs` — Main Dispatcher

**Purpose**: Orchestrate the transform by selecting and applying the right algorithm.

```rust
pub struct DFT<T: IntoSample> {
    data: Vec<T>,
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
        if n == 1 { return data; }

        let strategy = choose_strategy(n).expect("failed to choose strategy");
        let mut buf = data;
        idft_dispatch(&mut buf, &strategy);

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
    FftError(FftError),
}

pub type DftResult<T> = Result<T, DftError>;

pub fn validate_length(n: usize) -> DftResult<()> {
    if n == 0 { return Err(DftError::ZeroLength); }
    if n > MAX_DFT_SIZE { return Err(DftError::TooLarge(n)); }
    Ok(())
}
```

### 5.9 `plan.rs` — Reusable Plan (with Memory Warning)

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
    /// Create a new DFTPlan for `n` samples.
    ///
    /// ⚠️ Memory warning: for Bluestein strategy, the chirp cache
    /// requires M = next_pow2(2n-1) complex elements. For n = 8,000,000,
    /// this is ~268 MB with Complex64. Consider using smaller `n`
    /// or avoiding the plan for very large arbitrary lengths.
    pub fn new(n: usize) -> DftResult<Self> {
        validate_length(n)?;
        let strategy = choose_strategy(n)?;

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

```text
ft_winograd/
└── Cargo.toml
    └── fft_rs = { path = "../fft_rs_1" }
```

### 6.2 What is Reused from `fft_rs_1`

| Component | How it is reused |
|-----------|-----------------|
| `Complex32`, `Complex64` | Imported via `use fft_rs::*` |
| `IntoSample` trait | Imported; **EXTENDED** with impls for `Complex32` and `Complex64` |
| `ComplexSample` trait | Imported; used for generic algorithm implementations |
| `FFT<T>` | **Used directly** for power-of-2 delegation and Bluestein's 3 FFT calls |
| `FFTRun<T>` | Used internally by Bluestein for efficient repeated FFT calls |
| `is_power_of_two` | Used in `factorization.rs` |

### 6.3 What is NOT Reused (and Why)

| Component | Why not reused |
|-----------|---------------|
| `FftError` | The power-of-2 constraint is baked in; we need a more general error type |
| `validate_length` | Rejects non-power-of-2; we need a version that accepts any positive integer |
| `FFT<T>` struct | Hardcoded to power-of-2; we need a dispatcher that handles arbitrary `n` |

### 6.4 Internal Access to fft_rs_1 Functions

**Chosen approach**: Use the public `FFT<T>` API from `fft_rs_1` for:

- Radix2 strategy (power-of-2 delegation)
- Bluestein's 3 FFT calls
- FFT-based cyclic convolution (Rader for large primes)

This is the cleanest integration and respects the immutability constraint on `fft_rs_1`.

### 6.5 The `IntoSample` Pass-Through Trick — **ELIMINATES `fft_complex.rs`**

**The problem**: `fft_rs_1`'s `FFT<T>` only accepts real-valued input types (via `IntoSample`, implemented for `i32`, `i64`, `f32`, `f64`). The Bluestein algorithm requires FFT of **complex-valued** data.

**The original plan's solution**: Write `fft_complex.rs` — a standalone complex FFT duplicating the Cooley-Tukey logic. ❌

**The revised solution** (from review): Implement `IntoSample` for `Complex32` and `Complex64` as a pass-through:

```rust
// Inside ft_winograd/src/lib.rs

use fft_rs::{Complex32, Complex64, IntoSample};

/// "A Complex64 is already a complex sample — just pass it through."
impl IntoSample for Complex64 {
    type Complex = Complex64;
    #[inline]
    fn into_complex(self) -> Complex64 {
        self  // Pass-through — no conversion needed!
    }
}

/// Same for Complex32
impl IntoSample for Complex32 {
    type Complex = Complex32;
    #[inline]
    fn into_complex(self) -> Complex32 {
        self
    }
}
```

**Why this works**: With this implementation, `fft_rs::FFT::<Complex64>::new(vec_of_complex64).unwrap().compute()` now compiles and works correctly. The `IntoSample::into_complex()` method is a no-op pass-through, so the radix-2 Cooley-Tukey algorithm processes the complex data directly.

**Benefits**:
- **Zero code duplication** — no need for `fft_complex.rs`
- **Uses the same optimized radix-2 backend** as `fft_rs_1`
- **Clean and idiomatic Rust** — leveraging the trait system
- **No modification to `fft_rs_1`** — the impl lives in `ft_winograd`

**Bluestein integration with this trick**:

```rust
fn bluestein_forward(data: &mut [Complex64], n: usize, m: usize) {
    // ... chirp the input, zero-pad to length m ...
    let b: Vec<Complex64> = /* chirped + zero-padded */;
    let a: Vec<Complex64> = /* chirp kernel + zero-padded */;

    // m is a power of 2 — this now works because Complex64: IntoSample!
    let fft_b = fft_rs::FFT::<Complex64>::new(b).unwrap().compute();
    let fft_a = fft_rs::FFT::<Complex64>::new(a).unwrap().compute();

    // Pointwise multiply
    let product: Vec<Complex64> = fft_b.iter().zip(fft_a.iter())
        .map(|(&b, &a)| b * a).collect();

    // IFFT
    let result = fft_rs::FFT::<Complex64>::ifft(product);

    // De-chirp
    for k in 0..n {
        data[k] = result[k] * chirp[k];
    }
}
```

---

## 7. Implementation Order

### Phase 1: Foundation (Week 1-2)

1. **Create project structure** — `Cargo.toml`, `lib.rs`, empty modules
2. **`error.rs`** — Extended error types, validation for arbitrary `n`
3. **`lib.rs`** — `IntoSample` impl for `Complex32` and `Complex64` (the pass-through trick)
4. **`factorization.rs`** — Trial division, Miller-Rabin, primitive root, `choose_strategy()` with configurable thresholds
5. **Basic tests** — Factorization, strategy selection, `IntoSample` pass-through

### Phase 2: Winograd Short DFTs (Week 3-4)

6. **`winograd_dft.rs`** — Implement DFT-3, DFT-5, DFT-7 butterflies
7. **`winograd_conv.rs`** — Implement conv2, conv4, conv6 kernels + FFT-based convolution fallback
8. **Tests** — Verify against naive DFT for each short length

### Phase 3: Bluestein Fallback (Week 5) ⬅ MOVED UP

9. **`bluestein.rs`** — Bluestein's chirp-z algorithm (now simpler: no `fft_complex.rs` needed)
10. **Tests** — Verify for arbitrary lengths: 10, 12, 27, 97, 100, 1234

**Rationale for moving Phase 3 before Phase 4**: Bluestein is the universal fallback and is now simpler (no complex FFT to write). Getting it working early provides a working arbitrary-length DFT for testing and benchmarking, and serves as a correctness reference for the more complex algorithms.

### Phase 4: Prime Factor Algorithm (Week 6)

11. **`index_map.rs`** — CRT-based index mapping, PFA forward/inverse
12. **Tests** — Verify PFA decomposition for n = 15, 21, 35, etc.

### Phase 5: Prime-Length DFT (Week 7)

13. **`rader.rs`** — Rader's algorithm with convolution method selection
14. **Tests** — Verify against naive DFT for primes 11, 13, 17, 19, 97

### Phase 6: Dispatcher and API (Week 8)

15. **`fft_arbitrary.rs`** — Main `DFT<T>` struct, dispatch logic
16. **`plan.rs`** — `DFTPlan<T>` for repeated transforms (with memory warnings)
17. **`lib.rs`** — Public API re-exports
18. **`main.rs`** — Demo CLI

### Phase 7: Testing and Benchmarks (Week 9-10)

19. **Integration tests** — Comprehensive correctness tests
20. **Benchmarks** — Compare PFA vs. Bluestein at various sizes to tune thresholds
21. **Threshold tuning** — Adjust `THRESHOLD_PFA` and `THRESHOLD_RADER` based on benchmarks
22. **Edge cases** — n=1, n=2, n=3, large primes, memory stress for Bluestein

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
| **Prime via Rader** | n = 17, 19, 23, 29 — Rader + FFT-based convolution |
| **Prime via Bluestein** | n = 97, 101, 127 — Bluestein (large primes) |
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

### 9.2 Why Bluestein Dominates for Large n

The Bluestein algorithm's 3× FFT overhead is deceptive because:

1. **Cache-friendly**: The radix-2 FFT accesses memory in contiguous, strided patterns that are L1/L2 cache-friendly.
2. **Low constants**: The Cooley-Tukey radix-2 butterfly is one of the simplest operations in FFT — just 1 complex multiply and 2 complex adds per butterfly.
3. **SIMD-friendly**: The contiguous memory layout enables auto-vectorization.

In contrast, the PFA's CRT index mapping causes **stride-jumping** access patterns:

```
// PFA access pattern for n = 35 (n1=5, n2=7):
// Index m maps to (m mod 5, m mod 7)
// m=0 → (0,0), m=1 → (1,1), m=2 → (2,2), m=3 → (3,3), m=4 → (4,4)
// m=5 → (0,5), m=6 → (1,6), m=7 → (2,0), m=8 → (3,1), m=9 → (4,2)
```

This stride-jumping causes cache misses that can outweigh the multiplication savings, especially for `n > 500`.

### 9.3 Configurable Thresholds

```rust
// src/factorization.rs

/// Maximum n at which PFA + Winograd is preferred over Bluestein.
/// Default: 500. Tune this based on benchmarks for your target hardware.
#[cfg(feature = "tune-thresholds")]
pub const THRESHOLD_PFA: usize = 500;
#[cfg(not(feature = "tune-thresholds"))]
const THRESHOLD_PFA: usize = 500;

/// Maximum prime p at which Rader + Winograd convolution is preferred
/// over Bluestein. Default: 13 (the largest prime with p-1 ≤ 12).
const THRESHOLD_RADER: usize = 13;
```

### 9.4 When to Use Which Algorithm (REVISED)

| n | Recommended Algorithm | Rationale |
|---|----------------------|-----------|
| 2^k, k ≤ 24 | Radix-2 (fft_rs_1) | Fastest, well-optimized |
| 3, 5, 7, 11, 13 | Winograd short DFT | Minimum multiplications |
| n = n₁·n₂, gcd=1, **n ≤ 500** | PFA + Winograd | No twiddle factors, small enough for cache |
| p (prime, p ≤ 13) | Rader + Winograd conv | Small p-1, hand-written kernels available |
| p (prime, p > 13) | **Bluestein** | Simpler than Rader + FFT-conv; cache-friendly |
| p^k (odd prime power) | Bluestein | Universal, uses fft_rs_1 |
| n > 500, not power of 2 | **Bluestein** | Cache-friendly beats PFA stride-jumping |
| Other composite | Bluestein | Universal fallback |

---

## 10. References

### Primary References

1. **Burrus, C. S.** — *Fast Fourier Transforms*, LibreTexts. The definitive reference for Winograd, Rader, Bluestein, and PFA algorithms. Chapters 4-9 are essential.
   - https://eng.libretexts.org/Bookshelves/Electrical_Engineering/Signal_Processing_and_Modeling/Fast_Fourier_Transforms_(Burrus)

2. **Winograd, S. (1976)** — *On computing the discrete Fourier transform*, Mathematics of Computation, 32(144), 175-199.

3. **Rader, C. M. (1968)** — *Discrete Fourier transforms when the number of data samples is prime*, IEEE Transactions on Audio and Electroacoustics, 16(2), 136-137.

4. **Bluestein, L. I. (1968)** — *Linear complexity mapping for discrete Fourier analysis*, IEEE Spectrum, 6(8), 42-50.

5. **Good, I. J. (1958)** — *The interaction algorithm and practical Fourier analysis*, JRSS Series B, 20, 361-372.

### Implementation References

6. **Parhi, K. K.** — *VLSI Digital Signal Processing Systems*, Wiley, 1999. Chapter 8: Fast Convolution.

7. **Makhoul, J. (1975)** — *A fast algorithm for the exact calculation of the discrete Fourier transform*, IEEE IT, 20(3), 359-363.

8. **Silverman, J. H. (1977)** — *Winograd FFT Algorithm Programming Guide*, MIT TR-271.

9. **Encyclopedia of Mathematics** — *Winograd Fourier transform algorithm*.
   - https://encyclopediaofmath.org/wiki/Winograd_Fourier_transform_algorithm

### Code References

10. **FFTW** — Uses PFA + Winograd for composite coprime lengths, and Bluestein for prime lengths.
    - http://www.fftw.org

11. **rocFFT** — AMD's FFT library with well-documented Bluestein design.
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
| Bluestein's 3× overhead for prime lengths | **Low** (revised) | Bluestein's low constants make it competitive; benchmark to tune thresholds |
| **Bluestein chirp cache uses too much memory** | **Medium** (new) | Document memory footprint; provide `bluestein_memory_estimate()` |
| PFA index mapping errors | Medium | Extensive unit tests for CRT forward/inverse mapping |
| **Thresholds need post-benchmark tuning** | **Low** (new) | Make thresholds configurable constants; add `tune-thresholds` feature flag |

---

## 12. Summary of Changes from Original Plan

| # | Change | Impact |
|---|--------|--------|
| 1 | **Rader + Winograd → Rader + FFT-conv for large primes** | Eliminates impractical hand-written convolution kernels for large p-1 |
| 2 | **`IntoSample` pass-through for Complex32/64** | Eliminates `fft_complex.rs`; zero code duplication; idiomatic Rust |
| 3 | **Configurable thresholds (THRESHOLD_PFA, THRESHOLD_RADER)** | Allows post-benchmark tuning; Bluestein dominates for n > 500 |
| 4 | **Bluestein memory documentation** | Users aware of O(M) memory for large N |
| 5 | **Phase order: Bluestein before PFA** | Earlier working arbitrary-length DFT; simpler to test |
| 6 | **FFT-based cyclic convolution** | General fallback for Rader when p-1 > 12 |

---

## 13. Final Summary

This revised plan describes a comprehensive approach to extending FFT computation from power-of-2 to **arbitrary integer lengths** while:

1. **Not modifying** the existing `fft_rs_1` codebase
2. **Reusing** the complex types, traits, and radix-2 FFT from `fft_rs_1` — including the elegant `IntoSample` pass-through trick that eliminates the need for a separate complex FFT
3. **Implementing** the Winograd Fourier Transform Algorithm with its full suite of sub-algorithms (short DFTs, minimal convolution for small lengths, Rader with FFT-based convolution fallback, Bluestein as universal fallback, PFA for small coprime composites)
4. **Providing** configurable thresholds that can be tuned post-benchmarking based on actual hardware performance
5. **Documenting** the memory footprint of Bluestein's chirp cache so users can make informed decisions

The `ft_winograd` crate will serve as a drop-in replacement for `fft_rs_1` when arbitrary-length DFT is needed, while delegating to `fft_rs_1` for power-of-2 lengths where the radix-2 Cooley–Tukey algorithm is optimal.
