# ft_winograd — API User Manual

## **Winograd Fourier Transform: arbitrary-length DFT in Rust**

`ft_winograd` extends the radix-2 `fft_rs` library with algorithms that support **any positive integer length** `n`, not just powers of 2. It automatically selects the most efficient algorithm based on the input length.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Quick Start](#2-quick-start)
3. [Core API: `DFT<T>`](#3-core-api-dftt)
4. [Reusable Plans: `DFTPlan<T>`](#4-reusable-plans-dftplant)
5. [Supported Input Types](#5-supported-input-types)
6. [Error Handling](#6-error-handling)
7. [Strategy Selection](#7-strategy-selection)
8. [Low-Level Modules](#8-low-level-modules)
9. [Practical Use Cases](#9-practical-use-cases)
10. [Performance Tips](#10-performance-tips)

---

## 1. Installation

Add `ft_winograd` to your `Cargo.toml`:

```toml
[dependencies]
ft_winograd = "0.1"
```

---

## 2. Quick Start

```rust
use ft_winograd::DFT;

// Arbitrary length — no power-of-2 requirement
let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];  // n = 5
let dft = DFT::new(input).unwrap();
let spectrum = dft.compute();
```

---

## 3. Core API: `DFT<T>`

`DFT<T>` is the primary entry point for one-shot DFT computations. It accepts `i32`, `i64`, `f32`, `f64`, and complex types.

### 3.1 Construction

```rust
/// Create from an owned Vec (consumes the vector)
let dft = DFT::new(vec![1.0f64, 2.0, 3.0]).unwrap();

/// Create by cloning a slice (does not consume)
let dft = DFT::from_slice(&[1.0f64, 2.0, 3.0]).unwrap();
```

Both methods return `DftResult<Self>`. They fail with `DftError::ZeroLength` if the input is empty, or `DftError::TooLarge` if the length exceeds `MAX_DFT_SIZE` (2²⁶ ≈ 67 million).

### 3.2 Forward DFT

```rust
let spectrum: Vec<Complex64> = dft.compute();
```

Returns the forward DFT as a `Vec<T::Complex>`. For real-valued input (e.g., `f64`), the imaginary part of each output element encodes the quadrature component.

### 3.3 Inverse DFT

```rust
/// Class method — create the time-domain signal from a spectrum
let time_domain: Vec<Complex64> = DFT::<f64>::idft(spectrum);

/// Instance method — same as above
let time_domain = dft.compute_inverse(&spectrum);
```

The result is normalized by 1/n (standard convention).

### 3.4 Inspection

```rust
let n: usize = dft.len();          // number of samples
let empty: bool = dft.is_empty();  // always false for valid DFT
let data: &[f64] = dft.input();    // reference to original input
```

### 3.5 Complete Example: Forward + Inverse Roundtrip

```rust
use ft_winograd::DFT;

let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
let dft = DFT::new(input.clone()).unwrap();

// Forward
let spectrum = dft.compute();

// Inverse
let recovered = DFT::<f64>::idft(spectrum);

// Verify: recovered ≈ input
for (original, r) in input.iter().zip(recovered.iter()) {
    assert!((r.re - *original).abs() < 1e-10);
}
```

---

## 4. Reusable Plans: `DFTPlan<T>`

When computing multiple DFTs of the **same length**, use `DFTPlan<T>` to amortize the cost of chirp sequence pre-computation (Bluestein's algorithm).

### 4.1 Construction

```rust
use ft_winograd::DFTPlan;

let plan = DFTPlan::<f64>::new(97).unwrap();
assert_eq!(plan.n(), 97);
```

For Bluestein-length inputs, the chirp kernel is pre-computed at plan creation time, saving one FFT per subsequent transform.

### 4.2 Forward / Inverse

```rust
let spectrum = plan.dft(&input);       // forward DFT (takes a slice)
let recovered = plan.idft(spectrum);   // inverse DFT (consumes the vector)
```

### 4.3 Strategy Inspection

```rust
use ft_winograd::factorization::TransformStrategy;

let strategy = plan.strategy();
match strategy {
    TransformStrategy::Bluestein { m } => println!("Bluestein, FFT size = {}", m),
    TransformStrategy::Radix2 { log2n } => println!("Radix-2, log2 = {}", log2n),
    TransformStrategy::WinogradShort { n } => println!("Winograd, n = {}", n),
    TransformStrategy::Rader { p, primitive_root } => {
        println!("Rader, p = {}, alpha = {}", p, primitive_root)
    }
    TransformStrategy::PrimeFactor { n1, n2 } => println!("PFA, {} × {}", n1, n2),
}
```

### 4.4 Complete Example: Batch Spectral Analysis

```rust
use ft_winograd::DFTPlan;

// Create plan once for length 97 (Bluestein will pre-compute chirp)
let plan = DFTPlan::<f64>::new(97).unwrap();

let frames: Vec<Vec<f64>> = load_audio_frames(97); // 1000 frames

let spectra: Vec<Vec<Complex64>> = frames.iter()
    .map(|frame| plan.dft(frame))
    .collect();
```

---

## 5. Supported Input Types

| Input Type | Description | Complex Output Type |
|---|---|---|
| `i32` | 32-bit signed integer | `Complex64` |
| `i64` | 64-bit signed integer | `Complex64` |
| `f32` | 32-bit float | `Complex64` |
| `f64` | 64-bit float | `Complex64` |
| `Complex32` | 32-bit complex (via `C32` wrapper) | `Complex32` |
| `Complex64` | 64-bit complex (via `C64` wrapper) | `Complex64` |

For complex input, wrap the data using the provided helpers:

```rust
use ft_winograd::{wrap_c64, C64, DFT};
use ft_winograd::Complex64;

let data: Vec<Complex64> = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
let wrapped = wrap_c64(&data);
let dft = DFT::new(wrapped).unwrap();
let spectrum = dft.compute();
```

---

## 6. Error Handling

`ft_winograd` uses the `DftResult<T>` type alias for fallible operations:

```rust
pub type DftResult<T> = Result<T, DftError>;
```

### 6.1 Error Variants

| Variant | Meaning |
|---|---|
| `DftError::ZeroLength` | Input length is 0 |
| `DftError::TooLarge(n)` | Length exceeds `MAX_DFT_SIZE` (2²⁶) |
| `DftError::FactorizationFailed(n)` | Integer factorization failed |
| `DftError::NoPrimitiveRoot(p)` | No primitive root for prime p (theoretically impossible) |
| `DftError::FftError(e)` | Wrapped error from `fft_rs` backend |

### 6.2 Handling Errors

```rust
use ft_winograd::{DFT, DftError};

match DFT::<f64>::new(vec![]) {
    Ok(_) => println!("OK"),
    Err(DftError::ZeroLength) => println!("Cannot DFT an empty signal"),
    Err(DftError::TooLarge(n)) => println!("Signal too large: {}", n),
    Err(e) => println!("Error: {}", e),
}
```

### 6.3 Validation

```rust
use ft_winograd::validate_length;

assert!(validate_length(1000).is_ok());
assert!(validate_length(0).is_err());
```

---

## 7. Strategy Selection

The library automatically chooses the most efficient algorithm. The strategy is determined by `choose_strategy(n)`:

| Condition | Strategy | Description |
|---|---|---|
| `n = 2^k` | **Radix-2** | Standard Cooley-Tukey FFT |
| `n ∈ {3, 5, 7, 11, 13}` | **Winograd Short DFT** | Minimum-multiplication butterfly |
| `n = p` (prime, `p ≤ 13`) | **Rader's Algorithm** | Prime-length → cyclic convolution |
| `n = n₁ × n₂`, `gcd(n₁,n₂) = 1`, `n ≤ 500` | **PFA (Prime Factor Algorithm)** | Coprime decomposition, no twiddles |
| Everything else | **Bluestein** | Universal fallback via chirp-z |

### 7.1 Inspecting Strategy Programmatically

```rust
use ft_winograd::factorization::{choose_strategy, TransformStrategy};

for n in [3, 5, 10, 15, 97, 1024] {
    let s = choose_strategy(n).unwrap();
    match s {
        TransformStrategy::Radix2 { log2n } => println!("{} → Radix2(log2={})", n, log2n),
        TransformStrategy::WinogradShort { n: wn } => println!("{} → Winograd({})", n, wn),
        TransformStrategy::Rader { p, primitive_root } => {
            println!("{} → Rader(p={}, α={})", n, p, primitive_root)
        }
        TransformStrategy::PrimeFactor { n1, n2 } => println!("{} → PFA({}×{})", n, n1, n2),
        TransformStrategy::Bluestein { m } => println!("{} → Bluestein(M={})", n, m),
    }
}
```

Output:

```text
3 → Winograd(3)
5 → Winograd(5)
10 → Bluestein(M=16)
15 → PFA(3×5)
97 → Bluestein(M=196608)
1024 → Radix2(log2=10)
```

---

## 8. Low-Level Modules

For advanced use cases, the library exposes individual algorithm modules.

### 8.1 `winograd_dft` — Short DFT Butterflies

```rust
use ft_winograd::winograd_dft::{dft3, dft5, dft7, naive_dft};

let mut data = vec![1.0f64, 2.0, 3.0];
dft3(&mut data);  // in-place DFT-3

// Verification against naive O(n²)
let naive = naive_dft(&vec![1.0f64, 2.0, 3.0]);
```

### 8.2 `bluestein` — Chirp-Z Algorithm

```rust
use ft_winograd::bluestein::{bluestein_forward, bluestein_inverse, bluestein_memory_estimate};
use ft_winograd::Complex64;

let mut data: Vec<Complex64> = (0..97).map(|i| Complex64::new(i as f64, 0.0)).collect();
bluestein_forward(&mut data);

// Memory estimate before allocation
let (fft_size, bytes) = bluestein_memory_estimate(1_000_000);
println!("FFT size: {}, Memory: {} bytes", fft_size, bytes);
```

### 8.3 `rader` — Prime-Length DFT

```rust
use ft_winograd::rader::{rader_forward, rader_inverse};

let mut data: Vec<Complex64> = vec![
    Complex64::new(1.0, 0.0),
    Complex64::new(2.0, 0.0),
    Complex64::new(3.0, 0.0),
];
rader_forward(&mut data, 3);
rader_inverse(&mut data, 3);
```

### 8.4 `index_map` — PFA Index Mapping

```rust
use ft_winograd::index_map::{pfa_index_forward, pfa_index_inverse, pfa_forward, pfa_inverse};

// 1D → 2D index mapping
let (m1, m2) = pfa_index_forward(7, 3, 5);   // 7 → (1, 2)
let m = pfa_index_inverse(m1, m2, 3, 5);      // (1, 2) → 7
assert_eq!(m, 7);

// In-place permutation
let mut data: Vec<Complex64> = (0..15).map(|i| Complex64::new(i as f64, 0.0)).collect();
pfa_forward(&mut data, 3, 5);
pfa_inverse(&mut data, 3, 5);
```

### 8.5 `winograd_conv` — Cyclic Convolution

```rust
use ft_winograd::winograd_conv::{winograd_cyclic_conv, fft_cyclic_conv, conv2, conv4};
use ft_winograd::Complex64;

let g = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
let d = vec![Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)];
let h = conv2(&g, &d);  // length-2 cyclic convolution
```

### 8.6 `factorization` — Integer Factorization

```rust
use ft_winograd::factorization::{factorize, is_prime, is_power_of_two, next_power_of_two, primitive_root};

assert_eq!(factorize(60), vec![(2, 2), (3, 1), (5, 1)]);
assert!(is_prime(97));
assert!(is_power_of_two(1024));
assert_eq!(next_power_of_two(15), 16);
assert_eq!(primitive_root(5), Some(2));
```

---

## 9. Practical Use Cases

### 9.1 Audio Spectrogram (Non-Power-of-2 Frame Size)

```rust
use ft_winograd::DFT;

/// Compute a spectrogram from an audio signal using 513-sample frames
/// (a common size for 22050 Hz audio at ~43 ms window length).
fn spectrogram(signal: &[f64], frame_size: usize, hop: usize) -> Vec<Vec<Complex64>> {
    let dft = DFT::<f64>::new(vec![0.0; frame_size]).unwrap();

    let mut frames = Vec::new();
    for start in (0..signal.len()).step_by(hop) {
        let end = (start + frame_size).min(signal.len());
        let mut frame = signal[start..end].to_vec();
        frame.resize(frame_size, 0.0);
        let dft = DFT::new(frame).unwrap();
        frames.push(dft.compute());
    }
    frames
}

let audio = vec![0.0f64; 44100]; // 1 second at 44.1 kHz
let spec = spectrogram(&audio, 513, 256);
println!("Spectrogram shape: {} frames × {} bins", spec.len(), spec[0].len());
```

### 9.2 Polynomial Multiplication via FFT (Arbitrary Degree)

```rust
use ft_winograd::DFT;

/// Multiply two polynomials represented as coefficient vectors.
/// P(x) = [1, 2, 3] means 1 + 2x + 3x²
fn polynomial_multiply(p: &[f64], q: &[f64]) -> Vec<f64> {
    let result_len = p.len() + q.len() - 1;
    let fft_len = result_len.next_power_of_two();

    let mut p_padded = p.to_vec();
    let mut q_padded = q.to_vec();
    p_padded.resize(fft_len, 0.0);
    q_padded.resize(fft_len, 0.0);

    let p_fft = DFT::new(p_padded).unwrap().compute();
    let q_fft = DFT::new(q_padded).unwrap().compute();

    let mut product = Vec::with_capacity(fft_len);
    for i in 0..fft_len {
        product.push(p_fft[i] * q_fft[i]);
    }

    let result = DFT::<f64>::idft(product);
    let mut real_result = result[..result_len]
        .iter()
        .map(|c| c.re.round())
        .collect::<Vec<_>>();

    // For integer coefficients, round to nearest integer
    real_result
}

// (1 + 2x + 3x²) × (4 + 5x) = 4 + 13x + 22x² + 15x³
let p = vec![1.0, 2.0, 3.0];
let q = vec![4.0, 5.0];
println!("{:?}", polynomial_multiply(&p, &q));
// Output: [4.0, 13.0, 22.0, 15.0]
```

### 9.3 Cyclostationary Signal Analysis (Prime-Length Analysis)

```rust
use ft_winograd::DFTPlan;

/// Analyze a signal at a prime length to avoid spectral leakage
/// from periodic boundary conditions.
fn prime_length_analysis(signal: &[f64], prime: usize) -> Vec<Complex64> {
    let plan = DFTPlan::<f64>::new(prime).unwrap();
    let chunk = &signal[..prime];
    plan.dft(chunk)
}

// Use length 97 (prime) for analyzing a 100-sample signal
let signal: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
let spectrum = prime_length_analysis(&signal, 97);
println!("DC component: {:?}", spectrum[0]);
```

### 9.4 Image Block DCT Approximation via DFT

```rust
use ft_winograd::DFT;

/// Approximate the DCT-II of a 1D signal using a DFT of length 2n.
/// This is a standard technique: embed the signal in a 2n-length array,
/// compute the DFT, and extract the even-indexed components.
fn dct2_via_dft(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    let mut extended = vec![0.0f64; 2 * n];

    // Create the even-symmetric extension
    for i in 0..n {
        extended[i] = signal[i];
        extended[2 * n - 1 - i] = signal[i];
    }

    let dft = DFT::new(extended).unwrap().compute();

    // Extract even-indexed real parts (DCT-II)
    (0..n)
        .map(|k| dft[2 * k].re)
        .collect()
}

let signal = vec![1.0f64, 2.0, 3.0, 4.0];
let dct = dct2_via_dft(&signal);
println!("DCT coefficients: {:?}", dct);
```

### 9.5 Batch Processing with Plans

```rust
use ft_winograd::DFTPlan;

/// Efficiently compute DFTs for many signals of the same arbitrary length.
fn batch_dft(signals: &[Vec<f64>], length: usize) -> Vec<Vec<Complex64>> {
    let plan = DFTPlan::<f64>::new(length).unwrap();

    signals.iter()
        .map(|sig| plan.dft(sig))
        .collect()
}

let signals = vec![
    vec![1.0, 2.0, 3.0, 4.0, 5.0],
    vec![5.0, 4.0, 3.0, 2.0, 1.0],
];
let spectra = batch_dft(&signals, 5);
```

---

## 10. Performance Tips

1. **Use `DFTPlan` for repeated transforms.** When computing many DFTs of the same length, the plan amortizes chirp pre-computation for Bluestein-length inputs.

2. **Prefer lengths with small prime factors.** A length like 1024 (pure power of 2) or 30 (2×3×5) is much faster than a large prime like 997 because the library can use Radix-2 or PFA instead of Bluestein.

3. **Be aware of memory for Bluestein.** Bluestein's algorithm requires `M = next_pow2(2n - 1)` complex elements. For n = 1,000,000, this is approximately 16 MB for `Complex64`. Use `bluestein_memory_estimate()` to check before allocating.

4. **Use `f64` for precision.** The internal algorithms use `f64` arithmetic. Input types are converted to complex and processed as `Complex64`.

5. **The maximum DFT size is 2²⁶ (≈67M).** Attempting larger lengths will return `DftError::TooLarge`.

---

## Appendix: Algorithm Reference

| Algorithm | Length Condition | Multiplication Complexity |
|---|---|---|
| Radix-2 Cooley-Tukey | n = 2^k | O(n log n) |
| Winograd Short DFT | n ∈ {3, 5, 7, 11, 13} | Minimum for that n |
| Rader's Algorithm | n = p (prime, p ≤ 13) | O(p log p) via convolution |
| Good-Thomas PFA | n = n₁×n₂, gcd(n₁,n₂)=1 | O(n log n), no twiddles |
| Bluestein (Chirp-Z) | Any n | O(n log n) via 3× FFT |
