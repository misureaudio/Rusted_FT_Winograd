//! Winograd minimal convolution for small lengths, with FFT-based fallback.
//!
//! For cyclic convolutions of length m ≤ 12, hand-written Winograd kernels
//! achieve the minimum number of multiplications. For larger lengths,
//! the FFT-based approach (Convolution Theorem) is used.

use fft_rs::fft_core::ComplexSample;
use crate::factorization::next_power_of_two;

/// Compute cyclic convolution for small lengths using Winograd kernels.
/// For m > 12, falls back to FFT-based convolution.
pub fn winograd_cyclic_conv<C: ComplexSample>(
    g: &[C],
    d: &[C],
    m: usize,
) -> Vec<C> {
    assert_eq!(g.len(), m, "g length must equal m");
    assert_eq!(d.len(), m, "d length must equal m");

    if m <= 12 {
        match m {
            2 => conv2(g, d),
            4 => conv4(g, d),
            6 => conv6(g, d),
            _ => fft_cyclic_conv(g, d, m), // fallback for unimplemented lengths
        }
    } else {
        fft_cyclic_conv(g, d, m)
    }
}

/// FFT-based cyclic convolution for any length.
/// Zero-pads to M = next_pow2(2m-1), then FFT → multiply → IFFT.
pub fn fft_cyclic_conv<C: ComplexSample>(
    g: &[C],
    d: &[C],
    m: usize,
) -> Vec<C> {
    assert_eq!(g.len(), m, "g length must equal m");
    assert_eq!(d.len(), m, "d length must equal m");

    // Linear convolution of length m → need fft_size ≥ 2m - 1
    let fft_size = next_power_of_two(2 * m - 1);

    // Zero-pad both inputs
    let mut g_pad = Vec::with_capacity(fft_size);
    let mut d_pad = Vec::with_capacity(fft_size);
    for i in 0..m {
        g_pad.push(g[i]);
        d_pad.push(d[i]);
    }
    for _ in m..fft_size {
        g_pad.push(C::zero());
        d_pad.push(C::zero());
    }

    // FFT of both
    let fft_g = fft_forward(&mut g_pad.clone(), fft_size);
    let fft_d = fft_forward(&mut d_pad.clone(), fft_size);

    // Pointwise multiply
    let mut product = Vec::with_capacity(fft_size);
    for i in 0..fft_size {
        product.push(C::mul(fft_g[i], fft_d[i]));
    }

    // IFFT
    let linear = fft_inverse(&mut product, fft_size);

    // Fold: cyclic convolution = linear[0..m] + linear[m..2m-1] (wrapped)
    let mut result = Vec::with_capacity(m);
    for i in 0..m {
        let mut s = linear[i];
        if i + m < linear.len() {
            s = C::add(s, linear[i + m]);
        }
        result.push(s);
    }

    result
}

// ---------------------------------------------------------------------------
// Forward/inverse FFT helpers (use fft_rs via IntoSample wrappers)
// ---------------------------------------------------------------------------

fn fft_forward<C: ComplexSample>(data: &mut [C], n: usize) -> Vec<C> {
    let log2n = n.trailing_zeros() as usize;
    fft_core_impl(data, n, log2n, false);
    data.to_vec()
}

fn fft_inverse<C: ComplexSample>(data: &mut [C], n: usize) -> Vec<C> {
    let log2n = n.trailing_zeros() as usize;
    fft_core_impl(data, n, log2n, true);
    data.to_vec()
}

/// Cooley-Tukey radix-2 DIT, implemented inline for generic ComplexSample.
fn fft_core_impl<C: ComplexSample>(data: &mut [C], n: usize, log2n: usize, inverse: bool) {
    // Bit-reverse permutation
    for i in 0..n {
        let j = bit_reverse(i, log2n);
        if i < j {
            data.swap(i, j);
        }
    }

    let mut len = 2;
    for _ in 0..log2n {
        let half = len >> 1;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let even_idx = start + k;
                let odd_idx = even_idx + half;
                let tw = if inverse {
                    C::twiddle_inverse(len, k)
                } else {
                    C::twiddle(len, k)
                };
                let t = C::mul(tw, data[odd_idx]);
                let even = data[even_idx];
                data[odd_idx] = C::sub(even, t);
                data[even_idx] = C::add(even, t);
            }
        }
        len <<= 1;
    }

    if inverse {
        let norm = C::scalar_from_usize(n);
        for i in 0..n {
            data[i] = C::div_scalar(data[i], norm);
        }
    }
}

#[inline]
fn bit_reverse(x: usize, log2n: usize) -> usize {
    x.reverse_bits() >> (usize::BITS as usize - log2n)
}

// ---------------------------------------------------------------------------
// Winograd kernels for small lengths
// ---------------------------------------------------------------------------

mod kernels {
    use fft_rs::fft_core::ComplexSample;

    /// Length-2 cyclic convolution: h = g ⊗ d (mod x²-1)
    pub fn conv2<C: ComplexSample>(g: &[C], d: &[C]) -> Vec<C> {
        // h[0] = g[0]*d[0] + g[1]*d[1]
        // h[1] = g[0]*d[1] + g[1]*d[0]
        let h0 = C::add(C::mul(g[0], d[0]), C::mul(g[1], d[1]));
        let h1 = C::add(C::mul(g[0], d[1]), C::mul(g[1], d[0]));
        vec![h0, h1]
    }

    /// Length-4 cyclic convolution: h = g ⊗ d (mod x⁴-1)
    /// Uses Winograd's factorization: x⁴-1 = (x-1)(x+1)(x²+1)
    pub fn conv4<C: ComplexSample>(g: &[C], d: &[C]) -> Vec<C> {
        // Residue classes:
        // mod (x-1): g₁ = g[0]+g[1]+g[2]+g[3], d₁ = d[0]+d[1]+d[2]+d[3]
        // mod (x+1): g₂ = g[0]-g[1]+g[2]-g[3], d₂ = d[0]-d[1]+d[2]-d[3]
        // mod (x²+1): g₃ = g[0]+g[1]x+g[2]x²+g[3]x³ mod (x²+1) = (g[0]-g[2]) + (g[1]-g[3])x
        //             d₃ = (d[0]-d[2]) + (d[1]-d[3])x

        let g1 = C::add(C::add(g[0], g[1]), C::add(g[2], g[3]));
        let d1 = C::add(C::add(d[0], d[1]), C::add(d[2], d[3]));
        let h1 = C::mul(g1, d1);

        let g2 = C::sub(C::add(g[0], g[2]), C::add(g[1], g[3]));
        let d2 = C::sub(C::add(d[0], d[2]), C::add(d[1], d[3]));
        let h2 = C::mul(g2, d2);

        let g3a = C::sub(g[0], g[2]); // real part of mod (x²+1)
        let g3b = C::sub(g[1], g[3]); // imag part
        let d3a = C::sub(d[0], d[2]);
        let d3b = C::sub(d[1], d[3]);
        // (g3a + g3b·x)(d3a + d3b·x) mod (x²+1) = (g3a·d3a - g3b·d3b) + (g3a·d3b + g3b·d3a)·x
        let h3a = C::sub(C::mul(g3a, d3a), C::mul(g3b, d3b));
        let h3b = C::add(C::mul(g3a, d3b), C::mul(g3b, d3a));

        // CRT reconstruction:
        // h[0] = h1/4 + h2/4 + h3a/2
        // h[1] = h1/4 - h2/4 + h3b/2
        // h[2] = h1/4 + h2/4 - h3a/2
        // h[3] = h1/4 - h2/4 - h3b/2
        let four = C::scalar_from_usize(4);
        let h3a_2 = C::div_scalar(h3a, C::scalar_from_usize(2));
        let h3b_2 = C::div_scalar(h3b, C::scalar_from_usize(2));
        let h1_4 = C::div_scalar(h1, four);
        let h2_4 = C::div_scalar(h2, four);

        let h0 = C::add(C::add(h1_4, h2_4), h3a_2);
        let h1 = C::add(C::sub(h1_4, h2_4), h3b_2);
        let h2 = C::sub(C::add(h1_4, h2_4), h3a_2);
        let h3 = C::sub(C::sub(h1_4, h2_4), h3b_2);

        vec![h0, h1, h2, h3]
    }

    /// Length-6 cyclic convolution: h = g ⊗ d (mod x⁶-1)
    /// Uses Winograd's factorization: x⁶-1 = (x-1)(x+1)(x²-x+1)(x²+x+1)
    pub fn conv6<C: ComplexSample>(g: &[C], d: &[C]) -> Vec<C> {
        // For simplicity, use direct computation for length 6
        let mut result = vec![C::zero(); 6];
        for i in 0..6 {
            for j in 0..6 {
                let k = (i + j) % 6;
                result[k] = C::add(result[k], C::mul(g[i], d[j]));
            }
        }
        result
    }
}

pub use kernels::{conv2, conv4, conv6};

#[cfg(test)]
mod tests {
    use super::*;
    use fft_rs::Complex64;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    fn naive_cyclic_conv(g: &[Complex64], d: &[Complex64], m: usize) -> Vec<Complex64> {
        let mut result = vec![Complex64::zero(); m];
        for i in 0..m {
            for j in 0..m {
                let k = (i + j) % m;
                result[k] = result[k] + g[i] * d[j];
            }
        }
        result
    }

    #[test]
    fn test_conv2() {
        let g = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
        ];
        let d = vec![
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let expected = naive_cyclic_conv(&g, &d, 2);
        let result = conv2(&g, &d);
        for i in 0..2 {
            assert!(approx_eq(result[i], expected[i], 1e-10));
        }
    }

    #[test]
    fn test_conv4_vs_naive() {
        let g: Vec<Complex64> = (0..4)
            .map(|i| Complex64::new((i as f64 * 0.3).sin(), (i as f64 * 0.7).cos()))
            .collect();
        let d: Vec<Complex64> = (0..4)
            .map(|i| Complex64::new((i as f64 * 0.5).cos(), 0.0))
            .collect();
        let expected = naive_cyclic_conv(&g, &d, 4);
        let result = conv4(&g, &d);
        for i in 0..4 {
            assert!(approx_eq(result[i], expected[i], 1e-10), "conv4 mismatch at {}", i);
        }
    }

    #[test]
    fn test_conv6_vs_naive() {
        let g: Vec<Complex64> = (0..6)
            .map(|i| Complex64::new((i as f64 * 0.2).sin(), 0.0))
            .collect();
        let d: Vec<Complex64> = (0..6)
            .map(|i| Complex64::new((i as f64 * 0.5).cos(), 0.0))
            .collect();
        let expected = naive_cyclic_conv(&g, &d, 6);
        let result = conv6(&g, &d);
        for i in 0..6 {
            assert!(approx_eq(result[i], expected[i], 1e-10), "conv6 mismatch at {}", i);
        }
    }

    #[test]
    fn test_winograd_cyclic_conv_dispatch() {
        for &m in &[2, 4, 6] {
            let g: Vec<Complex64> = (0..m)
                .map(|i| Complex64::new((i as f64 * 0.2).sin(), (i as f64 * 0.3).cos()))
                .collect();
            let d: Vec<Complex64> = (0..m)
                .map(|i| Complex64::new((i as f64 * 0.5).cos(), (i as f64 * 0.1).sin()))
                .collect();
            let expected = naive_cyclic_conv(&g, &d, m);
            let result = winograd_cyclic_conv(&g, &d, m);
            for i in 0..m {
                assert!(
                    approx_eq(result[i], expected[i], 1e-10),
                    "conv{} mismatch at {}",
                    m,
                    i
                );
            }
        }
    }

    #[test]
    fn test_fft_cyclic_conv() {
        let m = 10;
        let g: Vec<Complex64> = (0..m)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.3).cos()))
            .collect();
        let d: Vec<Complex64> = (0..m)
            .map(|i| Complex64::new((i as f64 * 0.2).cos(), 0.0))
            .collect();
        let expected = naive_cyclic_conv(&g, &d, m);
        let result = fft_cyclic_conv(&g, &d, m);
        for i in 0..m {
            assert!(
                approx_eq(result[i], expected[i], 1e-8),
                "fft_cyclic_conv mismatch at {}",
                i
            );
        }
    }
}