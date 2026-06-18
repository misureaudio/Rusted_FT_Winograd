//! Rader's algorithm for prime-length DFT.
//!
//! Rader's algorithm converts a prime-length DFT into a cyclic convolution
//! of length `p-1` using a primitive root modulo `p`.
//!
//! For small primes (p ≤ 13), the convolution is evaluated using Winograd
//! minimal convolution kernels. For larger primes, FFT-based convolution
//! is used via the Convolution Theorem.

use fft_rs_ma::Complex64;
use crate::factorization::primitive_root;
use crate::winograd_conv::winograd_cyclic_conv;

/// Rader's forward DFT for prime length p.
///
/// Uses Winograd convolution for p ≤ 13 (small p-1),
/// and FFT-based convolution for larger primes.
///
/// The key identity: X[α^j] = x[0] + sum_{m=0}^{p-2} x[α^m] * W_p^{α^{j+m}}
/// This is a cross-correlation, computed as (a ⊗ b_rev) where b[m] = W_p^{α^m}
pub fn rader_forward(data: &mut [Complex64], p: usize) {
    assert_eq!(data.len(), p, "data length must equal p");
    assert!(p >= 2, "p must be at least 2");

    let alpha = primitive_root(p).expect("no primitive root for p");

    if p == 2 {
        // Trivial: DFT of length 2
        let x0 = data[0];
        let x1 = data[1];
        data[0] = x0 + x1;
        data[1] = x0 - x1;
        return;
    }

    // Step 1: Save x[0]
    let x0 = data[0];

    // Step 2: Re-index: a[m] = data[α^m] for m = 0, ..., p-2
    let mut a = Vec::with_capacity(p - 1);
    let mut b = Vec::with_capacity(p - 1);
    let mut pow_alpha = 1usize;

    for _m in 0..(p - 1) {
        a.push(data[pow_alpha]);
        b.push(Complex64::twiddle(p, pow_alpha));
        pow_alpha = (pow_alpha * alpha) % p;
    }

    // Step 3: We need result[j] = sum_m a[m] * b[(j+m) mod N] (cross-correlation)
    // This equals (a_rev ⊗ b)[j] where a_rev[m] = a[(-m) mod N]
    let conv_len = p - 1;
    let mut a_rev = vec![Complex64::zero(); conv_len];
    for m in 0..conv_len {
        a_rev[m] = a[(conv_len - m) % conv_len];
    }

    // Step 4: Cyclic convolution of length p-1
    let conv_result = if conv_len <= 12 {
        winograd_cyclic_conv(&a_rev, &b, conv_len)
    } else {
        crate::winograd_conv::fft_cyclic_conv(&a_rev, &b, conv_len)
    };

    // Step 5: X[α^j] = x[0] + conv_result[j]
    let mut result = vec![Complex64::zero(); p];

    // X[0] = sum of all inputs
    let mut x0_sum = x0;
    for m in 1..p {
        x0_sum = x0_sum + data[m];
    }
    result[0] = x0_sum;

    pow_alpha = 1usize;
    for m in 0..(p - 1) {
        let k = pow_alpha;
        result[k] = x0 + conv_result[m];
        pow_alpha = (pow_alpha * alpha) % p;
    }

    data.copy_from_slice(&result);
}

/// Rader's inverse DFT for prime length p.
pub fn rader_inverse(data: &mut [Complex64], p: usize) {
    assert_eq!(data.len(), p, "data length must equal p");
    assert!(p >= 2, "p must be at least 2");

    if p == 2 {
        // Inverse DFT of length 2
        let x0 = data[0];
        let x1 = data[1];
        data[0] = (x0 + x1) / 2.0;
        data[1] = (x0 - x1) / 2.0;
        return;
    }

    let alpha = primitive_root(p).expect("no primitive root for p");

    // Step 1: Save X[0]
    let x0 = data[0];

    // Step 2: Re-index: a[m] = X[α^m] for m = 0, ..., p-2
    let mut a = Vec::with_capacity(p - 1);
    let mut pow_alpha = 1usize;

    for _m in 0..(p - 1) {
        a.push(data[pow_alpha]);
        pow_alpha = (pow_alpha * alpha) % p;
    }

    // Step 3: b[m] = W_p^{-α^m} (conjugate twiddle for inverse)
    let mut b = Vec::with_capacity(p - 1);
    pow_alpha = 1usize;
    for _m in 0..(p - 1) {
        b.push(Complex64::twiddle_inverse(p, pow_alpha));
        pow_alpha = (pow_alpha * alpha) % p;
    }

    // Step 4: Cross-correlation via (a_rev ⊗ b)
    let conv_len = p - 1;
    let mut a_rev = vec![Complex64::zero(); conv_len];
    for m in 0..conv_len {
        a_rev[m] = a[(conv_len - m) % conv_len];
    }

    // Step 5: Cyclic convolution
    let conv_result = if conv_len <= 12 {
        winograd_cyclic_conv(&a_rev, &b, conv_len)
    } else {
        crate::winograd_conv::fft_cyclic_conv(&a_rev, &b, conv_len)
    };

    // Step 6: x[α^j] = (X[0] + conv_result[j]) / p
    let norm = p as f64;
    let mut result = vec![Complex64::zero(); p];

    pow_alpha = 1usize;
    for m in 0..(p - 1) {
        let n = pow_alpha;
        result[n] = (x0 + conv_result[m]) / norm;
        pow_alpha = (pow_alpha * alpha) % p;
    }

    // x[0] = (1/p) * sum_{k=0}^{p-1} X[k]
    let mut x0_sum = x0;
    for k in 1..p {
        x0_sum = x0_sum + data[k];
    }
    result[0] = x0_sum / norm;

    data.copy_from_slice(&result);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::winograd_dft::naive_dft;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn test_rader_vs_naive_p3() {
        let data: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
        ];
        let naive = naive_dft(&data);
        let mut rader = data.clone();
        rader_forward(&mut rader, 3);
        for i in 0..3 {
            assert!(approx_eq(rader[i], naive[i], 1e-10), "rader p=3 mismatch at {}", i);
        }
    }

    #[test]
    fn test_rader_vs_naive_p5() {
        let data: Vec<Complex64> = (0..5)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();
        let naive = naive_dft(&data);
        let mut rader = data.clone();
        rader_forward(&mut rader, 5);
        for i in 0..5 {
            assert!(approx_eq(rader[i], naive[i], 1e-10), "rader p=5 mismatch at {}", i);
        }
    }

    #[test]
    fn test_rader_vs_naive_p7() {
        let data: Vec<Complex64> = (0..7)
            .map(|i| Complex64::new((i as f64 * 0.5).cos(), (i as f64 * 0.2).sin()))
            .collect();
        let naive = naive_dft(&data);
        let mut rader = data.clone();
        rader_forward(&mut rader, 7);
        for i in 0..7 {
            assert!(approx_eq(rader[i], naive[i], 1e-10), "rader p=7 mismatch at {}", i);
        }
    }

    #[test]
    fn test_rader_vs_naive_p11() {
        let data: Vec<Complex64> = (0..11)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), 0.0))
            .collect();
        let naive = naive_dft(&data);
        let mut rader = data.clone();
        rader_forward(&mut rader, 11);
        for i in 0..11 {
            assert!(approx_eq(rader[i], naive[i], 1e-10), "rader p=11 mismatch at {}", i);
        }
    }

    #[test]
    fn test_rader_vs_naive_p13() {
        let data: Vec<Complex64> = (0..13)
            .map(|i| Complex64::new((i as f64 * 0.2).cos(), (i as f64 * 0.1).sin()))
            .collect();
        let naive = naive_dft(&data);
        let mut rader = data.clone();
        rader_forward(&mut rader, 13);
        for i in 0..13 {
            assert!(approx_eq(rader[i], naive[i], 1e-10), "rader p=13 mismatch at {}", i);
        }
    }

    #[test]
    fn test_rader_roundtrip_p7() {
        let data: Vec<Complex64> = (0..7)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let mut buf = data.clone();
        rader_forward(&mut buf, 7);
        rader_inverse(&mut buf, 7);
        for i in 0..7 {
            assert!(
                approx_eq(buf[i], data[i], 1e-10),
                "rader roundtrip p=7 mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_rader_roundtrip_p11() {
        let data: Vec<Complex64> = (0..11)
            .map(|i| Complex64::new((i as f64 * 0.1).cos(), 0.0))
            .collect();
        let mut buf = data.clone();
        rader_forward(&mut buf, 11);
        rader_inverse(&mut buf, 11);
        for i in 0..11 {
            assert!(
                approx_eq(buf[i], data[i], 1e-10),
                "rader roundtrip p=11 mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_rader_p2() {
        let data: Vec<Complex64> = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        let naive = naive_dft(&data);
        let mut rader = data.clone();
        rader_forward(&mut rader, 2);
        for i in 0..2 {
            assert!(approx_eq(rader[i], naive[i], 1e-10));
        }
    }
}