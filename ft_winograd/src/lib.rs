//! # ft_winograd
//!
//! Winograd Fourier Transform: arbitrary-length DFT in Rust.
//!
//! Extends the radix-2 `fft_rs` library with algorithms that support
//! **any positive integer length** `n`, not just powers of 2.
//!
//! ## Algorithms Implemented
//!
//! - **Winograd Short DFT** — minimum-multiplication butterflies for
//!   small prime/composite lengths (3, 5, 7, 11, 13, …).
//! - **Good–Thomas Prime Factor Algorithm (PFA)** — decomposes a
//!   coprime-factor DFT into independent smaller DFTs with no twiddle factors.
//! - **Rader's Algorithm** — converts a prime-length DFT into a cyclic
//!   convolution of length `p-1`.
//! - **Bluestein's Algorithm (Chirp-Z)** — universal fallback that reduces
//!   any-length DFT to three power-of-2 FFTs.
//!
//! ## Quick Start
//!
//! ```
//! use ft_winograd::DFT;
//!
//! // Arbitrary length — no power-of-2 requirement
//! let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];  // n = 5
//! let dft = DFT::new(input).unwrap();
//! let spectrum = dft.compute();
//! ```
//!
//! ## Supported Input Types
//!
//! The same types as `fft_rs`: `i32`, `i64`, `f32`, `f64`, plus
//! `Complex32` and `Complex64` directly (via newtype wrappers).

// we like mathematical loops
#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod factorization;
pub mod winograd_dft;
pub mod winograd_conv;
pub mod bluestein;
pub mod index_map;
pub mod rader;
pub mod fft_arbitrary;
pub mod plan;

pub use error::{DftError, DftResult, validate_length};
pub use fft_arbitrary::DFT;
pub use plan::DFTPlan;

// Re-export fft_rs_ma types for convenience
pub use fft_rs_ma::{Complex32, Complex64};


// ---------------------------------------------------------------------------
// Newtype wrappers for Complex32/64 to use with fft_rs_ma::FFT
// ---------------------------------------------------------------------------
//
// The orphan rule prevents us from implementing fft_rs_ma::IntoSample for
// fft_rs_ma::Complex32/64. Instead, we use newtype wrappers that implement
// IntoSample and delegate to the inner complex type.

use fft_rs_ma::IntoSample;

/// Newtype wrapper around Complex64 that implements IntoSample.
///
/// This allows passing complex-valued data directly to fft_rs_ma::FFT,
/// which is needed for Bluestein's algorithm and FFT-based convolution.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct C64(pub Complex64);

impl IntoSample for C64 {
    type Complex = Complex64;
    #[inline]
    fn into_complex(self) -> Complex64 {
        self.0 // Pass-through — no conversion needed!
    }
}

/// Newtype wrapper around Complex32 that implements IntoSample.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct C32(pub Complex32);

impl IntoSample for C32 {
    type Complex = Complex32;
    #[inline]
    fn into_complex(self) -> Complex32 {
        self.0
    }
}

/// Convert a slice of Complex64 to a Vec of C64 wrappers.
#[inline]
pub fn wrap_c64(data: &[Complex64]) -> Vec<C64> {
    data.iter().map(|&c| C64(c)).collect()
}

/// Convert a slice of Complex32 to a Vec of C32 wrappers.
#[inline]
pub fn wrap_c32(data: &[Complex32]) -> Vec<C32> {
    data.iter().map(|&c| C32(c)).collect()
}