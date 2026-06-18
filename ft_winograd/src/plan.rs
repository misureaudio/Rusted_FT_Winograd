//! Reusable DFT plan for repeated transforms of the same length.
//!
//! Pre-computes chirp sequences for Bluestein's algorithm, avoiding
//! redundant FFTs of the chirp kernel on subsequent calls.

use fft_rs_ma::Complex64;
use fft_rs_ma::fft_core::ComplexSample;
use fft_rs_ma::IntoSample;
use crate::error::{DftResult, validate_length};
use crate::factorization::{choose_strategy, TransformStrategy};
use std::marker::PhantomData;

/// A pre-computed plan for repeated DFTs of the same length.
///
/// ⚠️ Memory warning: for Bluestein strategy, the chirp cache
/// requires M = next_pow2(2n-1) complex elements. For n = 8,000,000,
/// this is ~268 MB with Complex64. Consider using smaller `n`
/// or avoiding the plan for very large arbitrary lengths.
pub struct DFTPlan<T: IntoSample>
where
    T::Complex: ComplexSample,
{
    n: usize,
    strategy: TransformStrategy,
    chirp_cache_forward: Option<Vec<Complex64>>,
    chirp_cache_inverse: Option<Vec<Complex64>>,
    _marker: PhantomData<T>,
}

impl<T: IntoSample> DFTPlan<T>
where
    T::Complex: ComplexSample,
{
    /// Create a new DFTPlan for `n` samples.
    ///
    /// For Bluestein strategy, pre-computes the chirp sequences which
    /// saves one FFT per subsequent transform.
    pub fn new(n: usize) -> DftResult<Self> {
        validate_length(n)?;
        let strategy = choose_strategy(n)?;

        let (chirp_cache_forward, chirp_cache_inverse) =
            if let TransformStrategy::Bluestein { m } = &strategy {
                (
                    Some(crate::bluestein::compute_chirp_forward(n, *m)),
                    Some(crate::bluestein::compute_chirp_inverse(n, *m)),
                )
            } else {
                (None, None)
            };

        Ok(DFTPlan {
            n,
            strategy,
            chirp_cache_forward,
            chirp_cache_inverse,
            _marker: PhantomData,
        })
    }

    /// Return the DFT length this plan was created for.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// Return the transform strategy for this plan.
    #[inline]
    pub fn strategy(&self) -> &TransformStrategy {
        &self.strategy
    }

    /// Compute the forward DFT of the given input slice.
    ///
    /// For Bluestein strategy, uses the pre-computed chirp cache.
    pub fn dft(&self, input: &[T]) -> Vec<T::Complex> {
        assert_eq!(input.len(), self.n, "input length must match plan length");

        if self.n == 1 {
            return vec![input[0].into_complex()];
        }

        let mut buf: Vec<T::Complex> = input.iter().copied()
            .map(|s| s.into_complex())
            .collect();

        self.dft_dispatch(&mut buf);
        buf
    }

    /// Compute the inverse DFT of the given spectrum.
    ///
    /// For Bluestein strategy, uses the pre-computed chirp cache.
    pub fn idft(&self, input: Vec<T::Complex>) -> Vec<T::Complex> {
        assert_eq!(input.len(), self.n, "input length must match plan length");

        if self.n == 1 {
            return input;
        }

        let mut buf = input;
        self.idft_dispatch(&mut buf);
        buf
    }

    fn dft_dispatch<C: ComplexSample>(&self, data: &mut [C]) {
        match &self.strategy {
            TransformStrategy::Radix2 { log2n } => {
                crate::fft_arbitrary::radix2_forward(data, *log2n);
            }
            TransformStrategy::PrimeFactor { n1, n2 } => {
                crate::index_map::pfa_dft_forward(data, *n1, *n2,
                    &|buf, n| {
                        let strat = choose_strategy(n).unwrap();
                        // Note: doesn't use plan — for full plan support,
                        // the PFA would need its own nested plan.
                        crate::fft_arbitrary::dft_dispatch(buf, &strat);
                    });
            }
            TransformStrategy::WinogradShort { n } => {
                crate::winograd_dft::winograd_short_dft_forward(data, *n);
            }
            TransformStrategy::Rader { p, primitive_root: _ } => {
                if let Some(data64) = cast_to_c64(data) {
                    crate::rader::rader_forward(data64, *p);
                }
            }
            TransformStrategy::Bluestein { m: _ } => {
                if let Some(data64) = cast_to_c64(data) {
                    if let Some(ref chirp) = self.chirp_cache_forward {
                        crate::bluestein::bluestein_forward_with_chirp(data64, chirp);
                    } else {
                        crate::bluestein::bluestein_forward(data64);
                    }
                }
            }
        }
    }

    fn idft_dispatch<C: ComplexSample>(&self, data: &mut [C]) {
        match &self.strategy {
            TransformStrategy::Radix2 { log2n } => {
                crate::fft_arbitrary::radix2_inverse(data, *log2n);
            }
            TransformStrategy::PrimeFactor { n1, n2 } => {
                crate::index_map::pfa_dft_inverse(data, *n1, *n2,
                    &|buf, n| {
                        let strat = choose_strategy(n).unwrap();
                        crate::fft_arbitrary::idft_dispatch(buf, &strat);
                    });
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
                    if let Some(ref chirp) = self.chirp_cache_inverse {
                        crate::bluestein::bluestein_inverse_with_chirp(data64, chirp);
                    } else {
                        crate::bluestein::bluestein_inverse(data64);
                    }
                }
            }
        }
    }
}

fn cast_to_c64<C: ComplexSample>(data: &mut [C]) -> Option<&mut [Complex64]> {
    if std::mem::size_of::<C>() == std::mem::size_of::<Complex64>()
        && std::mem::align_of::<C>() == std::mem::align_of::<Complex64>()
    {
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
    use fft_rs_ma::Complex64;
    use crate::winograd_dft::naive_dft;

    fn approx_eq(a: Complex64, b: Complex64, eps: f64) -> bool {
        (a.re - b.re).abs() < eps && (a.im - b.im).abs() < eps
    }

    #[test]
    fn test_plan_dft_n10() {
        let input: Vec<f64> = (0..10).map(|i| (i as f64 * 0.1).sin()).collect();
        let plan = DFTPlan::<f64>::new(10).unwrap();
        assert_eq!(plan.n(), 10);

        let result = plan.dft(&input);
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());

        for i in 0..10 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "plan n=10 mismatch at {}", i);
        }
    }

    #[test]
    fn test_plan_roundtrip_n10() {
        let input: Vec<f64> = (0..10).map(|i| (i as f64 * 0.1).sin()).collect();
        let plan = DFTPlan::<f64>::new(10).unwrap();

        let spectrum = plan.dft(&input);
        let recovered = plan.idft(spectrum);

        for i in 0..10 {
            assert!(
                approx_eq(recovered[i], Complex64::new(input[i], 0.0), 1e-10),
                "plan roundtrip n=10 mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_plan_bluestein_n97() {
        let input: Vec<f64> = (0..97).map(|i| (i as f64 * 0.05).sin()).collect();
        let plan = DFTPlan::<f64>::new(97).unwrap();
        assert!(matches!(plan.strategy(), TransformStrategy::Bluestein { .. }));

        let result = plan.dft(&input);
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());

        for i in 0..97 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "plan bluestein n=97 mismatch at {}", i);
        }
    }

    #[test]
    fn test_plan_bluestein_roundtrip_n97() {
        let input: Vec<f64> = (0..97).map(|i| (i as f64 * 0.05).sin()).collect();
        let plan = DFTPlan::<f64>::new(97).unwrap();

        let spectrum = plan.dft(&input);
        let recovered = plan.idft(spectrum);

        for i in 0..97 {
            assert!(
                approx_eq(recovered[i], Complex64::new(input[i], 0.0), 1e-10),
                "plan bluestein roundtrip n=97 mismatch at {}",
                i
            );
        }
    }

    #[test]
    fn test_plan_power_of_two_n32() {
        let input: Vec<f64> = (0..32).map(|i| (i as f64 * 0.1).sin()).collect();
        let plan = DFTPlan::<f64>::new(32).unwrap();

        let result = plan.dft(&input);
        let naive = naive_dft(&input.iter().map(|&x| Complex64::new(x, 0.0)).collect::<Vec<_>>());

        for i in 0..32 {
            assert!(approx_eq(result[i], naive[i], 1e-10), "plan pow2 n=32 mismatch at {}", i);
        }
    }
}