//! Bluestein's Chirp-Z algorithm for arbitrary-length DFT.
//!
//! Bluestein's algorithm converts any-length DFT into a convolution
//! evaluated via three power-of-2 FFT calls. It is the universal fallback
//! for arbitrary-length DFT computation.
//!
//! Uses the identity: `kn = ((k+n)² - k² - n²) / 2` to rewrite the DFT as:
//! ```text
//! X[k] = W_N^{k²/2} · ( [x[n] · W_N^{n²/2}] ⊗ W_N^{n²/2} )[k]
//! ```
//!
//! Memory: requires O(M) where M = next_pow2(2n-1). For n = 8,000,000,
//! M = 16,777,216 → ~268 MB for Complex64.

use fft_rs_ma::{Complex64, FFT};
use crate::C64;
use crate::factorization::next_power_of_two;

/// Compute the memory required for a Bluestein DFT of length n.
/// Returns M = next_pow2(2n-1) and the approximate byte count
/// for the chirp cache (M * sizeof(Complex64) = M * 16 bytes).
pub fn bluestein_memory_estimate(n: usize) -> (usize, usize) {
    let m = next_power_of_two(2 * n - 1);
    let bytes = m * std::mem::size_of::<Complex64>();
    (m, bytes)
}

/// Bluestein's forward DFT for arbitrary length n.
///
/// Uses the identity: kn = ((k+n)² - k² - n²) / 2
/// Derivation:
///   W_N^{kn} = exp(-2πi*kn/N) = a[k+n] * a[k]^{-1} * a[n]^{-1}
///   where a[m] = exp(-πi*m²/N) = W_N^{m²/2}
/// Therefore:
///   X[k] = a[k]^{-1} * sum_n (x[n] * a[n]^{-1}) * a[k+n]
/// This is a cross-correlation: b ⊛ a where b[n] = x[n]*a[n]^{-1}
/// Computed via convolution: (b_rev * a)[k+N-1]
///
/// Memory: requires O(M) where M = next_pow2(2n-1).
pub fn bluestein_forward(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let m = next_power_of_two(2 * n - 1);

    // Step 1: Compute the chirp sequence: a[m] = exp(-πi*m²/N)
    let chirp: Vec<Complex64> = (0..m)
        .map(|k| {
            let angle = -std::f64::consts::PI * (k as f64) * (k as f64) / (n as f64);
            let (sin, cos) = angle.sin_cos();
            Complex64::new(cos, sin)
        })
        .collect();

    // Step 2: b[n] = x[n] * a[n]^{-1} = x[n] * conj(a[n]) for n = 0..N-1
    let mut b = vec![Complex64::zero(); m];
    for k in 0..n {
        let c = chirp[k];
        b[k] = data[k] * Complex64::new(c.re, -c.im);
    }

    // Step 3: Reverse b: b_rev[n] = b[N-1-n]
    let mut b_rev = vec![Complex64::zero(); m];
    for k in 0..n {
        b_rev[k] = b[n - 1 - k];
    }

    // Step 4: FFT of b_rev (length m, power of 2)
    let b_wrapped: Vec<C64> = b_rev.iter().map(|&c| C64(c)).collect();
    let fft_b = FFT::<C64>::new(b_wrapped).unwrap().compute();

    // Step 5: FFT of chirp kernel (length m, power of 2)
    let a_wrapped: Vec<C64> = chirp.iter().map(|&c| C64(c)).collect();
    let fft_a = FFT::<C64>::new(a_wrapped).unwrap().compute();

    // Step 6: Pointwise multiply: C[k] = B[k] * A[k]
    let mut product = Vec::with_capacity(m);
    for k in 0..m {
        product.push(fft_b[k] * fft_a[k]);
    }

    // Step 7: IFFT of the product → linear convolution
    let c = FFT::<C64>::ifft(product);

    // Step 8: Extract X[k] = a[k]^{-1} * c[k + N - 1] for k = 0..N-1
    for k in 0..n {
        let chirp_k = chirp[k];
        data[k] = Complex64::new(chirp_k.re, -chirp_k.im) * c[k + n - 1];
    }
}

/// Bluestein's inverse DFT for arbitrary length n.
/// Same structure as forward but with conjugate chirp: a[m] = exp(+πi*m²/N)
pub fn bluestein_inverse(data: &mut [Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let m = next_power_of_two(2 * n - 1);

    // Chirp: a[m] = exp(+πi*m²/N) — conjugate of forward
    let chirp: Vec<Complex64> = (0..m)
        .map(|k| {
            let angle = std::f64::consts::PI * (k as f64) * (k as f64) / (n as f64);
            let (sin, cos) = angle.sin_cos();
            Complex64::new(cos, sin)
        })
        .collect();

    // Step 2: b[n] = X[n] * conj(a[n])
    let mut b = vec![Complex64::zero(); m];
    for k in 0..n {
        let c = chirp[k];
        b[k] = data[k] * Complex64::new(c.re, -c.im);
    }

    // Step 3: Reverse b
    let mut b_rev = vec![Complex64::zero(); m];
    for k in 0..n {
        b_rev[k] = b[n - 1 - k];
    }

    // Step 4: FFT of b_rev
    let b_wrapped: Vec<C64> = b_rev.iter().map(|&c| C64(c)).collect();
    let fft_b = FFT::<C64>::new(b_wrapped).unwrap().compute();

    // Step 5: FFT of chirp kernel
    let a_wrapped: Vec<C64> = chirp.iter().map(|&c| C64(c)).collect();
    let fft_a = FFT::<C64>::new(a_wrapped).unwrap().compute();

    // Step 6: Pointwise multiply
    let mut product = Vec::with_capacity(m);
    for k in 0..m {
        product.push(fft_b[k] * fft_a[k]);
    }

    // Step 7: IFFT → linear convolution
    let c = FFT::<C64>::ifft(product);

    // Step 8: Extract: x[k] = conj(a[k]) * c[k + N - 1] / N
    let norm = n as f64;
    for k in 0..n {
        let chirp_k = chirp[k];
        data[k] = Complex64::new(chirp_k.re, -chirp_k.im) * c[k + n - 1] / norm;
    }
}

/// Precompute the chirp sequence for Bluestein's forward transform.
/// Can be cached for repeated transforms of the same length.
pub fn compute_chirp_forward(n: usize, m: usize) -> Vec<Complex64> {
    (0..m)
        .map(|k| {
            let k2 = (k as f64) * (k as f64) / (2.0 * n as f64);
            let angle = -2.0 * std::f64::consts::PI * k2;
            let (sin, cos) = angle.sin_cos();
            Complex64::new(cos, sin)
        })
        .collect()
}

/// Precompute the chirp sequence for Bluestein's inverse transform.
pub fn compute_chirp_inverse(n: usize, m: usize) -> Vec<Complex64> {
    (0..m)
        .map(|k| {
            let k2 = (k as f64) * (k as f64) / (2.0 * n as f64);
            let angle = 2.0 * std::f64::consts::PI * k2;
            let (sin, cos) = angle.sin_cos();
            Complex64::new(cos, sin)
        })
        .collect()
}

/// Bluestein forward using a pre-computed chirp (for DFTPlan).
/// chirp[m] = exp(-πi*m²/N) — the same chirp as bluestein_forward computes.
pub fn bluestein_forward_with_chirp(data: &mut [Complex64], chirp: &[Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let m = chirp.len();

    // b[n] = x[n] * conj(a[n])
    let mut b = vec![Complex64::zero(); m];
    for k in 0..n {
        let c = chirp[k];
        b[k] = data[k] * Complex64::new(c.re, -c.im);
    }

    // Reverse b
    let mut b_rev = vec![Complex64::zero(); m];
    for k in 0..n {
        b_rev[k] = b[n - 1 - k];
    }

    // FFT of b_rev
    let b_wrapped: Vec<C64> = b_rev.iter().map(|&c| C64(c)).collect();
    let fft_b = FFT::<C64>::new(b_wrapped).unwrap().compute();

    // FFT of chirp kernel
    let a_wrapped: Vec<C64> = chirp.iter().map(|&c| C64(c)).collect();
    let fft_a = FFT::<C64>::new(a_wrapped).unwrap().compute();

    // Pointwise multiply
    let mut product = Vec::with_capacity(m);
    for k in 0..m {
        product.push(fft_b[k] * fft_a[k]);
    }

    // IFFT → linear convolution
    let c = FFT::<C64>::ifft(product);

    // Extract X[k] = conj(a[k]) * c[k + N - 1]
    for k in 0..n {
        let chirp_k = chirp[k];
        data[k] = Complex64::new(chirp_k.re, -chirp_k.im) * c[k + n - 1];
    }
}

/// Bluestein inverse using a pre-computed chirp.
/// chirp[m] = exp(+πi*m²/N) — conjugate of forward chirp.
pub fn bluestein_inverse_with_chirp(data: &mut [Complex64], chirp: &[Complex64]) {
    let n = data.len();
    if n <= 1 {
        return;
    }

    let m = chirp.len();

    // b[n] = X[n] * conj(a[n])
    let mut b = vec![Complex64::zero(); m];
    for k in 0..n {
        let c = chirp[k];
        b[k] = data[k] * Complex64::new(c.re, -c.im);
    }

    // Reverse b
    let mut b_rev = vec![Complex64::zero(); m];
    for k in 0..n {
        b_rev[k] = b[n - 1 - k];
    }

    // FFT of b_rev
    let b_wrapped: Vec<C64> = b_rev.iter().map(|&c| C64(c)).collect();
    let fft_b = FFT::<C64>::new(b_wrapped).unwrap().compute();

    // FFT of chirp kernel
    let a_wrapped: Vec<C64> = chirp.iter().map(|&c| C64(c)).collect();
    let fft_a = FFT::<C64>::new(a_wrapped).unwrap().compute();

    // Pointwise multiply
    let mut product = Vec::with_capacity(m);
    for k in 0..m {
        product.push(fft_b[k] * fft_a[k]);
    }

    // IFFT → linear convolution
    let c = FFT::<C64>::ifft(product);

    // Extract x[k] = conj(a[k]) * c[k + N - 1] / N
    let norm = n as f64;
    for k in 0..n {
        let chirp_k = chirp[k];
        data[k] = Complex64::new(chirp_k.re, -chirp_k.im) * c[k + n - 1] / norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::winograd_dft::naive_dft;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn test_bluestein_vs_naive_n5() {
        let data: Vec<Complex64> = (0..5)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();
        let naive = naive_dft(&data);
        let mut blue = data.clone();
        bluestein_forward(&mut blue);
        for i in 0..5 {
            assert!(approx_eq(blue[i], naive[i], 1e-10), "bluestein n=5 mismatch at {}", i);
        }
    }

    #[test]
    fn test_bluestein_vs_naive_n10() {
        let data: Vec<Complex64> = (0..10)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), 0.0))
            .collect();
        let naive = naive_dft(&data);
        let mut blue = data.clone();
        bluestein_forward(&mut blue);
        for i in 0..10 {
            assert!(approx_eq(blue[i], naive[i], 1e-10), "bluestein n=10 mismatch at {}", i);
        }
    }

    #[test]
    fn test_bluestein_vs_naive_n97() {
        let data: Vec<Complex64> = (0..97)
            .map(|i| Complex64::new((i as f64 * 0.05).cos(), (i as f64 * 0.1).sin()))
            .collect();
        let naive = naive_dft(&data);
        let mut blue = data.clone();
        bluestein_forward(&mut blue);
        for i in 0..97 {
            assert!(approx_eq(blue[i], naive[i], 1e-10), "bluestein n=97 mismatch at {}", i);
        }
    }

    #[test]
    fn test_bluestein_vs_naive_n1234() {
        let data: Vec<Complex64> = (0..1234)
            .map(|i| Complex64::new((i as f64 * 0.001).sin(), (i as f64 * 0.003).cos()))
            .collect();
        let naive = naive_dft(&data);
        let mut blue = data.clone();
        bluestein_forward(&mut blue);
        for i in 0..1234 {
            assert!(approx_eq(blue[i], naive[i], 1e-8), "bluestein n=1234 mismatch at {}", i);
        }
    }

    #[test]
    fn test_bluestein_roundtrip_n10() {
        let data: Vec<Complex64> = (0..10)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()))
            .collect();
        let mut buf = data.clone();
        bluestein_forward(&mut buf);
        bluestein_inverse(&mut buf);
        for i in 0..10 {
            assert!(
                approx_eq(buf[i], data[i], 1e-10),
                "bluestein roundtrip n=10 mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_bluestein_roundtrip_n97() {
        let data: Vec<Complex64> = (0..97)
            .map(|i| Complex64::new((i as f64 * 0.05).sin(), 0.0))
            .collect();
        let mut buf = data.clone();
        bluestein_forward(&mut buf);
        bluestein_inverse(&mut buf);
        for i in 0..97 {
            assert!(
                approx_eq(buf[i], data[i], 1e-10),
                "bluestein roundtrip n=97 mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_bluestein_memory_estimate() {
        let (m, bytes) = bluestein_memory_estimate(1000);
        assert_eq!(m, 2048); // next_pow2(1999) = 2048
        assert_eq!(bytes, 2048 * 16); // Complex64 = 16 bytes
    }
}