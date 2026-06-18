//! Demo CLI for ft_winograd — arbitrary-length DFT computation.
//!
//! Usage:
//!   ft_winograd <length> [seed]
//!
//! Computes the DFT of a random signal of the given length using
//! the Winograd/Bluestein/Rader algorithm dispatch, then verifies
//! the result against a naive O(n²) DFT.

// we like mathematical loops
#![allow(clippy::needless_range_loop)]

use ft_winograd::DFT;
use fft_rs_ma::Complex64;

fn naive_dft(input: &[f64]) -> Vec<Complex64> {
    let n = input.len();
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let mut sum = Complex64::zero();
        for m in 0..n {
            let tw = Complex64::twiddle(n, (k * m) % n);
            sum = sum + tw * Complex64::new(input[m], 0.0);
        }
        out.push(sum);
    }
    out
}

fn max_error(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(&x, &y)| {
            let dr = (x.re - y.re).abs();
            let di = (x.im - y.im).abs();
            if dr > di { dr } else { di }
        })
        .fold(0.0f64, f64::max)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let n: usize = if args.len() > 1 {
        args[1].parse().unwrap_or_else(|_| {
            eprintln!("Error: invalid length '{}'", args[1]);
            std::process::exit(1);
        })
    } else {
        eprintln!("Usage: ft_winograd <length> [seed]");
        eprintln!("  Computes DFT of a random signal of given length");
        eprintln!("  and verifies against naive O(n²) DFT.");
        std::process::exit(1);
    };

    let seed: u64 = if args.len() > 2 {
        args[2].parse().unwrap_or(42)
    } else {
        42
    };

    if n == 0 {
        eprintln!("Error: length must be positive");
        std::process::exit(1);
    }

    if n > 10000 {
        eprintln!("Warning: naive verification for n={} may be slow", n);
    }

    // Generate deterministic random signal
    let mut rng = seed;
    let signal: Vec<f64> = (0..n).map(|_| {
        // Simple LCG random number generator
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
    }).collect();

    println!("ft_winograd demo — arbitrary-length DFT");
    println!("========================================");
    println!("  Length:    {}", n);
    println!("  Seed:      {}", seed);

    // Compute DFT using our library
    let dft = DFT::new(signal.clone()).expect("failed to create DFT");
    let start = std::time::Instant::now();
    let result = dft.compute();
    let elapsed = start.elapsed();

    println!("  DFT time:  {:.3} ms", elapsed.as_secs_f64() * 1000.0);

    // Verify against naive DFT (only for reasonable sizes)
    if n <= 1000 {
        let naive_result = naive_dft(&signal);
        let err = max_error(&result, &naive_result);
        println!("  Max err:   {:.2e} (vs naive DFT)", err);

        if err < 1e-8 {
            println!("  Status:    ✓ PASS");
        } else {
            println!("  Status:    ✗ FAIL (error too large)");
            std::process::exit(1);
        }
    } else {
        println!("  Verify:    skipped (n > 1000)");
    }

    // Round-trip test
    let start = std::time::Instant::now();
    let recovered = DFT::<f64>::idft(result.clone());
    let roundtrip_elapsed = start.elapsed();

    let roundtrip_err = signal.iter().zip(recovered.iter())
        .map(|(&s, &r)| (s - r.re).abs())
        .fold(0.0f64, f64::max);

    println!("  Roundtrip: {:.3} ms, max err = {:.2e}",
        roundtrip_elapsed.as_secs_f64() * 1000.0, roundtrip_err);

    if roundtrip_err < 1e-8 {
        println!("  Status:    ✓ PASS");
    } else {
        println!("  Status:    ✗ FAIL (roundtrip error too large)");
        std::process::exit(1);
    }

    // Show first few frequency bins
    println!("\nFirst 5 frequency bins:");
    for k in 0..std::cmp::min(5, n) {
        println!("  X[{}] = {:12.6} + {:12.6}i", k, result[k].re, result[k].im);
    }
}