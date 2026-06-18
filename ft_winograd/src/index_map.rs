//! Good–Thomas Prime Factor Algorithm (PFA) index mapping.
//!
//! The PFA uses the Chinese Remainder Theorem to map 1D index `m ∈ [0, n)` to
//! 2D index `(m₁, m₂)` where `m₁ ∈ [0, n₁)`, `m₂ ∈ [0, n₂)` and `gcd(n₁, n₂) = 1`.
//! This eliminates twiddle factors between stages.

use fft_rs_ma::fft_core::ComplexSample;

/// Compute gcd using Euclid's algorithm.
#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Extended Euclidean algorithm: returns (g, x, y) such that a*x + b*y = g = gcd(a,b).
fn extended_gcd(a: usize, b: usize) -> (usize, i64, i64) {
    let (mut old_r, mut r) = (a as i64, b as i64);
    let (mut old_s, mut s) = (1i64, 0i64);
    let (mut old_t, mut t) = (0i64, 1i64);

    while r != 0 {
        let q = old_r / r;
        let tmp_r = old_r - q * r;
        let tmp_s = old_s - q * s;
        let tmp_t = old_t - q * t;
        old_r = r; r = tmp_r;
        old_s = s; s = tmp_s;
        old_t = t; t = tmp_t;
    }

    (old_r as usize, old_s, old_t)
}

/// Precompute the CRT coefficients for n1, n2.
///
/// Returns (c1, c2) such that:
/// - c1 ≡ 1 (mod n1), c1 ≡ 0 (mod n2)
/// - c2 ≡ 0 (mod n1), c2 ≡ 1 (mod n2)
///
/// Then m = (m1 * c1 + m2 * c2) mod (n1 * n2)
pub fn crt_coefficients(n1: usize, n2: usize) -> (usize, usize) {
    assert_eq!(gcd(n1, n2), 1, "n1 and n2 must be coprime");

    let n = n1 * n2;
    let (_, x, y) = extended_gcd(n1, n2);

    // c1 ≡ 1 (mod n1), c1 ≡ 0 (mod n2)
    // c1 = y * n2 (since y*n2 ≡ 1 (mod n1) from extended_gcd)
    let c1 = ((y % n1 as i64 + n1 as i64) % n1 as i64 * n2 as i64) as usize % n;

    // c2 ≡ 0 (mod n1), c2 ≡ 1 (mod n2)
    // c2 = x * n1 (since x*n1 ≡ 1 (mod n2))
    let c2 = ((x % n2 as i64 + n2 as i64) % n2 as i64 * n1 as i64) as usize % n;

    (c1, c2)
}

/// Map 1D index to 2D index using CRT: m → (m mod n1, m mod n2).
#[inline]
pub fn pfa_index_forward(m: usize, n1: usize, n2: usize) -> (usize, usize) {
    (m % n1, m % n2)
}

/// Map 2D index to 1D index: (m1, m2) → m using precomputed CRT coefficients.
#[inline]
pub fn pfa_index_inverse(m1: usize, m2: usize, n1: usize, n2: usize) -> usize {
    let (c1, c2) = crt_coefficients(n1, n2);
    let n = n1 * n2;
    (m1 * c1 + m2 * c2) % n
}

/// In-place Good-Thomas PFA forward transform.
///
/// Rearranges data from 1D layout to 2D layout (row-major):
/// data[m1 * n2 + m2] = original[pfa_index_inverse(m1, m2, n1, n2)]
pub fn pfa_forward<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize) {
    assert_eq!(data.len(), n1 * n2, "data length must equal n1 * n2");
    assert_eq!(gcd(n1, n2), 1, "n1 and n2 must be coprime");

    let mut temp = data.to_vec();

    for m1 in 0..n1 {
        for m2 in 0..n2 {
            let m = pfa_index_inverse(m1, m2, n1, n2);
            temp[m1 * n2 + m2] = data[m];
        }
    }

    data.copy_from_slice(&temp);
}

/// In-place Good-Thomas PFA inverse transform.
///
/// Rearranges data from 2D layout back to 1D layout:
/// data[pfa_index_inverse(m1, m2, n1, n2)] = temp[m1 * n2 + m2]
pub fn pfa_inverse<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize) {
    assert_eq!(data.len(), n1 * n2, "data length must equal n1 * n2");
    assert_eq!(gcd(n1, n2), 1, "n1 and n2 must be coprime");

    let mut temp = data.to_vec();

    for m1 in 0..n1 {
        for m2 in 0..n2 {
            let m = pfa_index_inverse(m1, m2, n1, n2);
            temp[m] = data[m1 * n2 + m2];
        }
    }

    data.copy_from_slice(&temp);
}

/// Compute the full PFA forward DFT for n = n1 * n2 (coprime).
///
/// Uses CRT to map indices: m -> (m mod n1, m mod n2).
/// The DFT factors into 2D DFT with modified twiddles:
/// W_{n1}^{c1_bar * k1 * m1} * W_{n2}^{c2_bar * k2 * m2}
/// where c1_bar = c1/n2 and c2_bar = c2/n1.
///
/// Data layout after pfa_forward: data[m1 * n2 + m2] = original[index(m1,m2)]
pub fn pfa_dft_forward<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize,
    _short_dft: &dyn Fn(&mut [C], usize))
{
    assert_eq!(data.len(), n1 * n2, "data length must equal n1 * n2");
    assert_eq!(gcd(n1, n2), 1, "n1 and n2 must be coprime");

    let (c1, c2) = crt_coefficients(n1, n2);
    let c1_bar = c1 / n2;
    let c2_bar = c2 / n1;

    // Step 1: PFA forward permutation
    pfa_forward(data, n1, n2);

    // Step 2: Inner DFT along n2 dimension with modified twiddle
    // Y[m1, k2] = sum_{m2} W_{n2}^{c2_bar * k2 * m2} * x[m1, m2]
    for m1 in 0..n1 {
        let row = &data[m1 * n2..(m1 + 1) * n2];
        let mut out = vec![C::zero(); n2];
        for k2 in 0..n2 {
            let mut sum = C::zero();
            for m2 in 0..n2 {
                let tw = C::twiddle(n2, (c2_bar.wrapping_mul(k2).wrapping_rem(n2))
                                     .wrapping_mul(m2).wrapping_rem(n2));
                sum = C::add(sum, C::mul(row[m2], tw));
            }
            out[k2] = sum;
        }
        data[m1 * n2..(m1 + 1) * n2].copy_from_slice(&out);
    }

    // Step 3: Outer DFT along n1 dimension with modified twiddle
    // X[k1, k2] = sum_{m1} W_{n1}^{c1_bar * k1 * m1} * Y[m1, k2]
    for k2 in 0..n2 {
        let mut col = Vec::with_capacity(n1);
        for m1 in 0..n1 {
            col.push(data[m1 * n2 + k2]);
        }
        let mut out = vec![C::zero(); n1];
        for k1 in 0..n1 {
            let mut sum = C::zero();
            for m1 in 0..n1 {
                let tw = C::twiddle(n1, (c1_bar.wrapping_mul(k1).wrapping_rem(n1))
                                     .wrapping_mul(m1).wrapping_rem(n1));
                sum = C::add(sum, C::mul(col[m1], tw));
            }
            out[k1] = sum;
        }
        for m1 in 0..n1 {
            data[m1 * n2 + k2] = out[m1];
        }
    }

    // Step 4: PFA inverse permutation
    pfa_inverse(data, n1, n2);
}

/// Compute the full PFA inverse DFT for n = n1 * n2 (coprime).
pub fn pfa_dft_inverse<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize,
    _short_idft: &dyn Fn(&mut [C], usize))
{
    assert_eq!(data.len(), n1 * n2, "data length must equal n1 * n2");
    assert_eq!(gcd(n1, n2), 1, "n1 and n2 must be coprime");

    let (c1, c2) = crt_coefficients(n1, n2);
    let c1_bar = c1 / n2;
    let c2_bar = c2 / n1;

    // Step 1: PFA forward permutation
    pfa_forward(data, n1, n2);

    // Step 2: Inner IDFT along n2 dimension with modified twiddle
    for m1 in 0..n1 {
        let row = &data[m1 * n2..(m1 + 1) * n2];
        let mut out = vec![C::zero(); n2];
        for k2 in 0..n2 {
            let mut sum = C::zero();
            for m2 in 0..n2 {
                let tw = C::twiddle_inverse(n2, (c2_bar.wrapping_mul(k2).wrapping_rem(n2))
                                             .wrapping_mul(m2).wrapping_rem(n2));
                sum = C::add(sum, C::mul(row[m2], tw));
            }
            out[k2] = sum;
        }
        data[m1 * n2..(m1 + 1) * n2].copy_from_slice(&out);
    }

    // Step 3: Outer IDFT along n1 dimension with modified twiddle
    for k2 in 0..n2 {
        let mut col = Vec::with_capacity(n1);
        for m1 in 0..n1 {
            col.push(data[m1 * n2 + k2]);
        }
        let mut out = vec![C::zero(); n1];
        for k1 in 0..n1 {
            let mut sum = C::zero();
            for m1 in 0..n1 {
                let tw = C::twiddle_inverse(n1, (c1_bar.wrapping_mul(k1).wrapping_rem(n1))
                                             .wrapping_mul(m1).wrapping_rem(n1));
                sum = C::add(sum, C::mul(col[m1], tw));
            }
            out[k1] = sum;
        }
        for m1 in 0..n1 {
            data[m1 * n2 + k2] = out[m1];
        }
    }

    // Step 4: PFA inverse permutation
    pfa_inverse(data, n1, n2);

    // Step 5: Normalize by 1/n
    let n = n1 * n2;
    let norm = C::scalar_from_usize(n);
    for i in 0..n {
        data[i] = C::div_scalar(data[i], norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fft_rs_ma::Complex64;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    fn naive_dft(input: &[Complex64]) -> Vec<Complex64> {
        let n = input.len();
        let mut out = Vec::with_capacity(n);
        for k in 0..n {
            let mut sum = Complex64::zero();
            for m in 0..n {
                let tw = Complex64::twiddle(n, (k * m) % n);
                sum = sum + tw * input[m];
            }
            out.push(sum);
        }
        out
    }

    #[test]
    fn test_crt_coefficients() {
        let (c1, c2) = crt_coefficients(3, 5);
        // c1 ≡ 1 (mod 3), c1 ≡ 0 (mod 5)
        assert_eq!(c1 % 3, 1);
        assert_eq!(c2 % 3, 0);
        assert_eq!(c1 % 5, 0);
        assert_eq!(c2 % 5, 1);
    }

    #[test]
    fn test_pfa_index_roundtrip() {
        let (_c1, _c2) = crt_coefficients(3, 5);
        let n = 15;
        for m in 0..n {
            let (m1, m2) = pfa_index_forward(m, 3, 5);
            let m_back = pfa_index_inverse(m1, m2, 3, 5);
            assert_eq!(m_back, m, "roundtrip failed for m={}", m);
        }
    }

    #[test]
    fn test_pfa_forward_inverse_roundtrip() {
        let data: Vec<Complex64> = (0..15)
            .map(|i| Complex64::new(i as f64, 0.0))
            .collect();
        let mut buf = data.clone();
        pfa_forward(&mut buf, 3, 5);
        pfa_inverse(&mut buf, 3, 5);
        for i in 0..15 {
            assert!(approx_eq(buf[i], data[i], 1e-10));
        }
    }

    #[test]
    fn test_pfa_dft_n15_vs_naive() {
        let data: Vec<Complex64> = (0..15)
            .map(|i| Complex64::new((i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()))
            .collect();
        let naive = naive_dft(&data);
        let mut pfa_data = data.clone();

        pfa_dft_forward(&mut pfa_data, 3, 5,
            &|buf, n| crate::winograd_dft::winograd_short_dft_forward(buf, n));

        for i in 0..15 {
            assert!(approx_eq(pfa_data[i], naive[i], 1e-10), "PFA n=15 mismatch at {}", i);
        }
    }

    #[test]
    fn test_pfa_dft_n21_vs_naive() {
        let data: Vec<Complex64> = (0..21)
            .map(|i| Complex64::new((i as f64 * 0.1).cos(), 0.0))
            .collect();
        let naive = naive_dft(&data);
        let mut pfa_data = data.clone();

        pfa_dft_forward(&mut pfa_data, 3, 7,
            &|buf, n| crate::winograd_dft::winograd_short_dft_forward(buf, n));

        for i in 0..21 {
            assert!(approx_eq(pfa_data[i], naive[i], 1e-10), "PFA n=21 mismatch at {}", i);
        }
    }
}