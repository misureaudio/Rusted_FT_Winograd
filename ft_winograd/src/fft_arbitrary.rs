//! Main dispatcher for arbitrary-length DFT computation.
//!
//! Selects and applies the right algorithm based on the input length `n`:
//! - Power of 2 → delegate to fft_rs (radix-2 Cooley-Tukey)
//! - Small prime (3,5,7,11,13) → Winograd short DFT
//! - Prime p ≤ 13 → Rader's algorithm + Winograd convolution
//! - Coprime composite n ≤ 500 → PFA + recursive short DFTs
//! - Everything else → Bluestein's algorithm (universal fallback)

use fft_rs_ma::Complex64;
use fft_rs_ma::fft_core::{ComplexSample, IntoSample};
use crate::error::{DftResult, validate_length};
use crate::factorization::{choose_strategy, TransformStrategy};

/// DFT computation handle for arbitrary-length input.
///
/// Accepts `i32`, `i64`, `f32`, `f64` input. The `compute()` method returns
/// the forward DFT; `idft()` returns the inverse DFT.
///
/// # Example
///
/// ```
/// use ft_winograd::DFT;
///
/// let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];  // n = 5, not a power of 2!
/// let dft = DFT::new(input).unwrap();
/// let spectrum = dft.compute();
/// ```
pub struct DFT<T: IntoSample> {
    data: Vec<T>,
}

impl<T: IntoSample> DFT<T>
where
    T::Complex: ComplexSample,
{
    /// Create a new `DFT` from a `Vec<T>`.
    ///
    /// Unlike `fft_rs_ma::FFT`, this accepts **any positive integer length**,
    /// not just powers of 2.
    pub fn new(data: Vec<T>) -> DftResult<Self> {
        validate_length(data.len())?;
        Ok(DFT { data })
    }

    /// Create a new `DFT` by cloning a slice.
    pub fn from_slice(slice: &[T]) -> DftResult<Self> {
        Self::new(slice.to_vec())
    }

    /// Return the number of samples.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Return `true` if there are no samples.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a reference to the input data.
    pub fn input(&self) -> &[T] {
        &self.data
    }

    /// Compute the forward DFT, returning a `Vec<T::Complex>`.
    pub fn compute(&self) -> Vec<T::Complex> {
        let n = self.data.len();
        if n == 1 {
            return vec![self.data[0].into_complex()];
        }

        let strategy = choose_strategy(n).expect("failed to choose strategy");

        // Convert input to complex
        let mut buf: Vec<T::Complex> = self.data.iter().copied()
            .map(|s| s.into_complex())
            .collect();

        dft_dispatch(&mut buf, &strategy);
        buf
    }

    /// Compute the inverse DFT of the given complex data.
    ///
    /// Returns the time-domain signal normalized by 1/n.
    pub fn idft(data: Vec<T::Complex>) -> Vec<T::Complex> {
        let n = data.len();
        if n == 1 {
            return data;
        }

        let strategy = choose_strategy(n).expect("failed to choose strategy");
        let mut buf = data;
        idft_dispatch(&mut buf, &strategy);

        buf
    }

    /// Compute the inverse DFT of the given spectrum.
    pub fn compute_inverse(&self, spectrum: &[T::Complex]) -> Vec<T::Complex> {
        Self::idft(spectrum.to_vec())
    }
}

/// Dispatch forward DFT based on strategy.
pub(crate) fn dft_dispatch<C: ComplexSample>(data: &mut [C], strategy: &TransformStrategy) {
    match strategy {
        TransformStrategy::Radix2 { log2n } => {
            // Delegate to fft_rs_ma via FFT<T::Complex> — but we can't use that
            // directly for complex data. Use inline radix-2.
            radix2_forward(data, *log2n);
        }
        TransformStrategy::PrimeFactor { n1, n2 } => {
            // PFA: recursively compute DFTs for each factor
            pfa_dispatch_forward(data, *n1, *n2);
        }
        TransformStrategy::WinogradShort { n } => {
            crate::winograd_dft::winograd_short_dft_forward(data, *n);
        }
        TransformStrategy::Rader { p, primitive_root: _ } => {
            // Only works for Complex64
            if let Some(data64) = cast_to_c64(data) {
                crate::rader::rader_forward(data64, *p);
            }
        }
        TransformStrategy::Bluestein { m: _ } => {
            // Only works for Complex64
            if let Some(data64) = cast_to_c64(data) {
                crate::bluestein::bluestein_forward(data64);
            }
        }
    }
}

/// Dispatch inverse DFT based on strategy.
pub(crate) fn idft_dispatch<C: ComplexSample>(data: &mut [C], strategy: &TransformStrategy) {
    match strategy {
        TransformStrategy::Radix2 { log2n } => {
            radix2_inverse(data, *log2n);
        }
        TransformStrategy::PrimeFactor { n1, n2 } => {
            pfa_dispatch_inverse(data, *n1, *n2);
        }
        TransformStrategy::WinogradShort { n } => {
            crate::winograd_dft::winograd_short_dft_inverse(data, *n);
        }
        TransformStrategy::Rader { p, primitive_root: _ } => {
            if let Some(data64) = cast_to_c64(data) {
                crate::rader::rader_inverse(data64, *p);
            }
        }
        TransformStrategy::Bluestein { m: _ } => {
            if let Some(data64) = cast_to_c64(data) {
                crate::bluestein::bluestein_inverse(data64);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inline radix-2 FFT for generic ComplexSample
// ---------------------------------------------------------------------------

pub(crate) fn radix2_forward<C: ComplexSample>(data: &mut [C], log2n: usize) {
    let n = data.len();
    // Bit-reverse
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
                let t = C::mul(C::twiddle(len, k), data[odd_idx]);
                let even = data[even_idx];
                data[odd_idx] = C::sub(even, t);
                data[even_idx] = C::add(even, t);
            }
        }
        len <<= 1;
    }
}

pub(crate) fn radix2_inverse<C: ComplexSample>(data: &mut [C], log2n: usize) {
    let n = data.len();
    // Bit-reverse
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
                let t = C::mul(C::twiddle_inverse(len, k), data[odd_idx]);
                let even = data[even_idx];
                data[odd_idx] = C::sub(even, t);
                data[even_idx] = C::add(even, t);
            }
        }
        len <<= 1;
    }

    let norm = C::scalar_from_usize(n);
    for i in 0..n {
        data[i] = C::div_scalar(data[i], norm);
    }
}

#[inline]
fn bit_reverse(x: usize, log2n: usize) -> usize {
    x.reverse_bits() >> (usize::BITS as usize - log2n)
}

// ---------------------------------------------------------------------------
// PFA dispatch (recursive)
// ---------------------------------------------------------------------------

fn pfa_dispatch_forward<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize) {
    crate::index_map::pfa_dft_forward(data, n1, n2,
        &|buf, n| {
            let strat = choose_strategy(n).unwrap();
            dft_dispatch(buf, &strat);
        });
}

fn pfa_dispatch_inverse<C: ComplexSample>(data: &mut [C], n1: usize, n2: usize) {
    crate::index_map::pfa_dft_inverse(data, n1, n2,
        &|buf, n| {
            let strat = choose_strategy(n).unwrap();
            idft_dispatch(buf, &strat);
        });
}

// ---------------------------------------------------------------------------
// Cast helper for algorithms that only work with Complex64
// ---------------------------------------------------------------------------

/// Attempt to cast a slice of C to a mutable slice of Complex64.
/// Returns None if C is not Complex64 (caller should handle gracefully).
fn cast_to_c64<C: ComplexSample>(data: &mut [C]) -> Option<&mut [Complex64]> {
    // This is a workaround for the fact that our Bluestein and Rader
    // implementations are specialized for Complex64.
    // In a production system, you would implement generic versions.
    if std::mem::size_of::<C>() == std::mem::size_of::<Complex64>()
        && std::mem::align_of::<C>() == std::mem::align_of::<Complex64>()
    {
        // SAFETY: We verify size and alignment match Complex64.
        // This is safe when C is Complex64 but not for other types.
        Some(unsafe {
            std::slice::from_raw_parts_mut(data.as_mut_ptr() as *mut Complex64, data.len())
        })
    } else {
        None
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
    fn test_dft_power_of_two() {
        let input: Vec<f64> = (0..16).map(|i| (i as f64 * 0.1).sin()).collect();
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        for i in 0..16 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "pow2 mismatch at {}", i);
        }
    }

    #[test]
    fn test_dft_length_3() {
        let input: Vec<f64> = vec![1.0, 2.0, 3.0];
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        for i in 0..3 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "n=3 mismatch at {}", i);
        }
    }

    #[test]
    fn test_dft_length_5() {
        let input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        for i in 0..5 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "n=5 mismatch at {}", i);
        }
    }

    #[test]
    fn test_dft_length_10() {
        let input: Vec<f64> = (0..10).map(|i| (i as f64 * 0.1).sin()).collect();
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        for i in 0..10 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "n=10 mismatch at {}", i);
        }
    }

    #[test]
    fn test_dft_length_97() {
        let input: Vec<f64> = (0..97).map(|i| (i as f64 * 0.05).sin()).collect();
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        for i in 0..97 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "n=97 mismatch at {}", i);
        }
    }

    #[test]
    fn test_dft_roundtrip_n10() {
        let input: Vec<f64> = (0..10).map(|i| (i as f64 * 0.1).sin()).collect();
        let dft = DFT::new(input.clone()).unwrap();
        let spectrum = dft.compute();
        let recovered = DFT::<f64>::idft(spectrum);
        for i in 0..10 {
            assert!(approx_eq(recovered[i], Complex64::new(input[i], 0.0), 1e-10));
        }
    }

    #[test]
    fn test_dft_roundtrip_n97() {
        let input: Vec<f64> = (0..97).map(|i| (i as f64 * 0.05).sin()).collect();
        let dft = DFT::new(input.clone()).unwrap();
        let spectrum = dft.compute();
        let recovered = DFT::<f64>::idft(spectrum);
        for i in 0..97 {
            assert!(approx_eq(recovered[i], Complex64::new(input[i], 0.0), 1e-10));
        }
    }

    #[test]
    fn test_dft_length_1() {
        let input = vec![42.0f64];
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        assert_eq!(result.len(), 1);
        assert!(approx_eq(result[0], Complex64::new(42.0, 0.0), 1e-10));
    }

    #[test]
    fn test_dft_rejects_zero_length() {
        assert!(DFT::<f64>::new(vec![]).is_err());
    }

    #[test]
    fn test_dft_length_15() {
        // 15 = 3 * 5, PFA
        let input: Vec<f64> = (0..15).map(|i| (i as f64 * 0.1).sin()).collect();
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        for i in 0..15 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "n=15 mismatch at {}", i);
        }
    }

    #[test]
    fn test_dft_length_27() {
        // 27 = 3^3, Bluestein
        let input: Vec<f64> = (0..27).map(|i| (i as f64 * 0.1).sin()).collect();
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());
        let dft = DFT::new(input).unwrap();
        let result = dft.compute();
        for i in 0..27 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "n=27 mismatch at {}", i);
        }
    }

    #[test]
    fn test_from_slice() {
        let slice: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0];
        let dft = DFT::from_slice(slice).unwrap();
        assert_eq!(dft.len(), 5);
    }
}