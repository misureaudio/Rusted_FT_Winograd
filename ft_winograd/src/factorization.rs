//! Integer factorization and strategy selection for arbitrary-length DFT.

use crate::error::{DftError, DftResult};

/// Maximum n at which PFA + Winograd is preferred over Bluestein.
const THRESHOLD_PFA: usize = 500;

/// Maximum prime p at which Rader + Winograd convolution is preferred
/// over Bluestein.
const THRESHOLD_RADER: usize = 13;

/// The chosen strategy for computing a DFT of length n.
#[derive(Debug, Clone)]
pub enum TransformStrategy {
    /// Delegate to fft_rs radix-2 FFT (n = 2^k).
    Radix2 { log2n: usize },
    /// Good-Thomas PFA: n = n1 * n2, gcd(n1, n2) = 1.
    PrimeFactor { n1: usize, n2: usize },
    /// Winograd short DFT for small prime/composite n.
    WinogradShort { n: usize },
    /// Rader's algorithm: n = p (prime, p ≤ THRESHOLD_RADER).
    Rader { p: usize, primitive_root: usize },
    /// Bluestein's algorithm: universal fallback.
    Bluestein { m: usize },
}

/// Factor n into prime powers: returns Vec<(prime, exponent)> in ascending order.
pub fn factorize(n: usize) -> Vec<(usize, usize)> {
    let mut factors = Vec::new();
    let mut d = 2usize;
    let mut remaining = n;

    while d * d <= remaining {
        if remaining.is_multiple_of(d) {
            let mut exp = 0;
            while remaining.is_multiple_of(d) {
                remaining /= d;
                exp += 1;
            }
            factors.push((d, exp));
        }
        d += 1;
    }

    if remaining > 1 {
        factors.push((remaining, 1));
    }

    factors
}

/// Check if n is a power of 2.
#[inline]
pub fn is_power_of_two(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

/// Next power of 2 ≥ n.
#[inline]
pub fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    (1usize).checked_shl(64 - n.leading_zeros()).unwrap_or(1)
}

/// Miller-Rabin primality test with deterministic witnesses for usize range.
pub fn is_prime(n: usize) -> bool {
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n.is_multiple_of(2) || n.is_multiple_of(3) {
        return false;
    }

    for &p in &[5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n == p {
            return true;
        }
        if n.is_multiple_of(p) {
            return false;
        }
    }

    let mut d = n - 1;
    let mut s = 0usize;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    let witnesses: &[usize] = &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    for &a in witnesses {
        if a >= n {
            continue;
        }
        if !miller_rabin_pass(a as u64, d as u64, s, n as u64) {
            return false;
        }
    }
    true
}

fn miller_rabin_pass(a: u64, d: u64, s: usize, n: u64) -> bool {
    let mut x = mod_pow(a, d, n);
    if x == 1 || x == n - 1 {
        return true;
    }
    // We already checked x = a^d; now square up to s-1 more times
    for _ in 1..s {
        x = mod_mul(x, x, n);
        if x == n - 1 {
            return true;
        }
    }
    false
}

fn mod_pow(base: u64, mut exp: u64, modu: u64) -> u64 {
    let mut result: u128 = 1;
    let base = base as u128;
    let modu = modu as u128;
    let mut b = base % modu;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % modu;
        }
        b = (b * b) % modu;
        exp >>= 1;
    }
    result as u64
}

fn mod_mul(a: u64, b: u64, modu: u64) -> u64 {
    (((a as u128) * (b as u128)) % (modu as u128)) as u64
}

/// Find the smallest primitive root modulo p, where p is prime.
pub fn primitive_root(p: usize) -> Option<usize> {
    if p < 2 || !is_prime(p) {
        return None;
    }
    if p == 2 {
        return Some(1);
    }

    let phi = p - 1;
    let prime_factors_of_phi = factorize(phi)
        .into_iter()
        .map(|(q, _)| q)
        .collect::<Vec<_>>();

    for g in 2..p {
        let mut ok = true;
        for &q in &prime_factors_of_phi {
            if mod_pow(g as u64, (phi / q) as u64, p as u64) == 1 {
                ok = false;
                break;
            }
        }
        if ok {
            return Some(g);
        }
    }
    None
}

/// Determine the optimal transform strategy for length n.
pub fn choose_strategy(n: usize) -> DftResult<TransformStrategy> {
    if n == 1 {
        return Ok(TransformStrategy::Radix2 { log2n: 0 });
    }

    if is_power_of_two(n) {
        return Ok(TransformStrategy::Radix2 {
            log2n: n.trailing_zeros() as usize,
        });
    }

    if [3, 5, 7, 11, 13].contains(&n) {
        return Ok(TransformStrategy::WinogradShort { n });
    }

    if is_prime(n) {
        if n <= THRESHOLD_RADER {
            let alpha = primitive_root(n)
                .ok_or(DftError::NoPrimitiveRoot(n))?;
            return Ok(TransformStrategy::Rader { p: n, primitive_root: alpha });
        } else {
            let m = next_power_of_two(2 * n - 1);
            return Ok(TransformStrategy::Bluestein { m });
        }
    }

    if n <= THRESHOLD_PFA {
        if let Some((n1, n2)) = find_coprime_factors(n) {
            return Ok(TransformStrategy::PrimeFactor { n1, n2 });
        }
    }

    let m = next_power_of_two(2 * n - 1);
    Ok(TransformStrategy::Bluestein { m })
}

fn find_coprime_factors(n: usize) -> Option<(usize, usize)> {
    let factors = factorize(n);
    if factors.len() < 2 {
        return None;
    }

    // Split factors into two coprime groups
    let mut n1 = 1usize;
    for (i, &(p, exp)) in factors.iter().enumerate() {
        if i == 0 {
            n1 = p.pow(exp as u32);
        } else {
            break;
        }
    }
    let n2 = n / n1;

    if n2 == 1 {
        return None; // All same prime — prime power
    }

    if gcd(n1, n2) == 1 {
        return Some((n1, n2));
    }

    None
}

#[inline]
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize_primes() {
        assert_eq!(factorize(2), vec![(2, 1)]);
        assert_eq!(factorize(7), vec![(7, 1)]);
    }

    #[test]
    fn test_factorize_composites() {
        assert_eq!(factorize(6), vec![(2, 1), (3, 1)]);
        assert_eq!(factorize(60), vec![(2, 2), (3, 1), (5, 1)]);
    }

    #[test]
    fn test_is_prime() {
        assert!(is_prime(97));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(is_prime(5));
        assert!(is_prime(7));
        assert!(is_prime(11));
        assert!(is_prime(13));
        assert!(is_prime(17));
        assert!(!is_prime(100));
    }

    #[test]
    fn test_primitive_root() {
        assert_eq!(primitive_root(3), Some(2));
        assert_eq!(primitive_root(5), Some(2));
    }

    #[test]
    fn test_choose_strategy_power_of_two() {
        let s = choose_strategy(1024).unwrap();
        assert!(matches!(s, TransformStrategy::Radix2 { log2n: 10 }));
    }

    #[test]
    fn test_choose_strategy_small_prime() {
        let s = choose_strategy(5).unwrap();
        assert!(matches!(s, TransformStrategy::WinogradShort { n: 5 }));
    }

    #[test]
    fn test_choose_strategy_large_prime() {
        let s = choose_strategy(97).unwrap();
        assert!(matches!(s, TransformStrategy::Bluestein { .. }));
    }

    #[test]
    fn test_choose_strategy_coprime() {
        let s = choose_strategy(15).unwrap();
        assert!(matches!(s, TransformStrategy::PrimeFactor { .. }));
    }
}