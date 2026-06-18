# Question

Determine whether the FT code in .\ft_winograd uses the code in .\fft_rs_1 to treat cases where samples vactor's length is a power of 2 or it uses instead a self contained version of Cooley & Tukey

The FT code in `.\ft_winograd` uses a **self-contained Cooley-Tukey radix-2 implementation** for power-of-2 cases, NOT the `fft_rs_1` library.

Here is the detailed breakdown:

## 1. Power-of-2 case (primary dispatch): SELF-CONTAINED

In `fft_arbitrary.rs`, lines 109-115, when the input length is a power of 2, the `TransformStrategy::Radix2` arm is matched:

```rust
TransformStrategy::Radix2 { log2n } => {
    // Delegate to fft_rs via FFT<T::Complex> — but we can't use that
    // directly for complex data. Use inline radix-2.
    radix2_forward(data, *log2n);
}
```

The comment on line 112-113 explicitly states: *"Delegate to fft_rs via FFT<T::Complex> — but we can't use that directly for complex data. Use inline radix-2."* — meaning the original intent was to use `fft_rs`, but they could not due to type constraints (the `fft_rs::FFT` requires `IntoSample`, and `Complex64` doesn't implement it directly due to Rust's orphan rule). Instead, they implemented their own Cooley-Tukey at lines 167-192:

```rust
pub(crate) fn radix2_forward<C: ComplexSample>(data: &mut [C], log2n: usize) {
    let n = data.len();
    // Bit-reverse
    for i in 0..n {
        let j = bit_reverse(i, log2n);
        if i < j { data.swap(i, j); }
    }
    let mut len = 2;
    for _ in 0..log2n {
        let half = len >> 1;
        for start in (0..n).step_by(len) {
            for k in 0..half {
                let t = C::mul(C::twiddle(len, k), data[odd_idx]);
                // ... butterfly operations
            }
        }
        len <<= 1;
    }
}
```

This is a classic radix-2 decimation-in-time Cooley-Tukey with bit-reversal permutation.

## 2. Bluestein's algorithm (internal power-of-2 FFT): USES fft_rs_1

In `bluestein.rs`, the Bluestein algorithm (which reduces arbitrary-length DFT to convolution via power-of-2 FFTs) **does** use `fft_rs::FFT` directly for its three internal FFT calls (lines 72, 76, 85):

```rust
use fft_rs::{Complex64, FFT};
// ...
let fft_b = FFT::<C64>::new(b_wrapped).unwrap().compute();   // line 72
let fft_a = FFT::<C64>::new(a_wrapped).unwrap().compute();   // line 76
let c = FFT::<C64>::ifft(product);                            // line 85
```

This works because Bluestein wraps `Complex64` in the `C64` newtype (which implements `IntoSample`), allowing it to use `fft_rs::FFT`.

## Summary

| Use Case | Implementation | Source |
|---|---|---|
| Direct power-of-2 DFT | **Self-contained Cooley-Tukey** | `fft_arbitrary.rs:radix2_forward/inverse` |
| Bluestein's internal FFT calls | **fft_rs_1** (via `FFT::<C64>`) | `bluestein.rs` |

The `fft_rs_1` dependency is used for: (1) re-exporting `Complex32`/`Complex64` types, (2) the `IntoSample`/`ComplexSample` traits, and (3) the `FFT` struct used internally by Bluestein. However, the **main dispatch path for power-of-2 inputs uses the self-contained radix-2 Cooley-Tukey**, not `fft_rs_1`.
