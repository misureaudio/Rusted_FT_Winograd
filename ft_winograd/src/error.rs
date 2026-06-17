//! Error types for arbitrary-length DFT computation.

use std::fmt;
use fft_rs::FftError;

/// Maximum DFT size for arbitrary-length transforms.
/// Extended beyond fft_rs's 2^24 to support ~67M samples,
/// since Bluestein's algorithm pads to the next power of 2.
pub const MAX_DFT_SIZE: usize = 1 << 26;

/// Errors that can occur during arbitrary-length DFT computation.
#[derive(Debug)]
pub enum DftError {
    /// Input length is 0.
    ZeroLength,
    /// Input length exceeds `MAX_DFT_SIZE`.
    TooLarge(usize),
    /// Integer factorization failed for the given length.
    FactorizationFailed(usize),
    /// No primitive root found for the given prime (should not happen mathematically).
    NoPrimitiveRoot(usize),
    /// Wrapped error from the fft_rs backend (power-of-2 FFT).
    FftError(FftError),
}

impl fmt::Display for DftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DftError::ZeroLength => write!(f, "DFT length must be positive (got 0)"),
            DftError::TooLarge(n) => write!(
                f,
                "DFT length {} exceeds maximum {}",
                n, MAX_DFT_SIZE
            ),
            DftError::FactorizationFailed(n) => write!(
                f,
                "Failed to factorize DFT length {}",
                n
            ),
            DftError::NoPrimitiveRoot(p) => write!(
                f,
                "No primitive root found for prime {} (mathematically impossible)",
                p
            ),
            DftError::FftError(e) => write!(f, "fft_rs error: {}", e),
        }
    }
}

impl std::error::Error for DftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DftError::FftError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<FftError> for DftError {
    fn from(e: FftError) -> Self {
        DftError::FftError(e)
    }
}

/// Result type for arbitrary-length DFT operations.
pub type DftResult<T> = Result<T, DftError>;

/// Validate length for arbitrary DFT (accepts any positive integer, not just power of 2).
pub fn validate_length(n: usize) -> DftResult<()> {
    if n == 0 {
        return Err(DftError::ZeroLength);
    }
    if n > MAX_DFT_SIZE {
        return Err(DftError::TooLarge(n));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_length_zero() {
        assert!(matches!(validate_length(0), Err(DftError::ZeroLength)));
    }

    #[test]
    fn test_validate_length_too_large() {
        assert!(matches!(
            validate_length(MAX_DFT_SIZE + 1),
            Err(DftError::TooLarge(_))
        ));
    }

    #[test]
    fn test_validate_length_ok_power_of_two() {
        assert!(validate_length(1).is_ok());
        assert!(validate_length(1024).is_ok());
    }

    #[test]
    fn test_validate_length_ok_arbitrary() {
        assert!(validate_length(3).is_ok());
        assert!(validate_length(1234).is_ok());
        assert!(validate_length(99999).is_ok());
    }

    #[test]
    fn test_display_errors() {
        assert!(DftError::ZeroLength.to_string().contains("0"));
        assert!(DftError::TooLarge(42).to_string().contains("42"));
    }
}