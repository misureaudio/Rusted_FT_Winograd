# Plan for Fixing Cargo Warnings in ft_winograd

I have analyzed all 22 warnings across 6 source files. They fall into 5 categories. Below is a detailed plan for each, with the reasoning for why some cannot be fixed "blindly."

## ---

### Category 1: Unused Imports (7 warnings)

These require careful analysis because some imports appear unused at the top level but are actually needed by `#[cfg(test)]` modules via `use super::*`.

| File | Line | Import | Fix |
|------|------|--------|-----|
| `winograd_dft.rs` | 7 | `fft_rs::Complex64` | **Move into test module.** `Complex64` is used only in the `#[cfg(test)]` module (via `use super::*`). Remove from top-level; add `use fft_rs::Complex64;` inside the test module. |
| `fft_arbitrary.rs` | 10 | `fft_rs::FFT` | **Remove from import.** `FFT` is not used anywhere in this file. Change to `use fft_rs::Complex64;`. |
| `fft_arbitrary.rs` | 12 | `crate::error::DftError` | **Remove from import.** `DftError` is not used directly; only `DftResult` and `validate_length` are used. Change to `use crate::error::{DftResult, validate_length};`. |
| `bluestein.rs` | 16 | `fft_rs::fft_core::ComplexSample` | **Remove entirely.** All functions in this file are specialized for `Complex64`; `ComplexSample` is never referenced. |
| `rader.rs` | 11 | `fft_rs::fft_core::ComplexSample` | **Remove entirely.** Same reasoning — all functions work with `Complex64` only. |

**Why not blind:** Simply removing `Complex64` from `winograd_dft.rs` would break the test module. The import must be moved, not deleted.

---

### Category 2: Unused `mut` (1 warning)

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `factorization.rs` | 122 | `mut base` in `mod_pow` | **Remove `mut` from `base`.** The parameter is immediately shadowed by `let base = base as u128` on line 124, so the `mut` on the parameter is never used. Change signature from `fn mod_pow(mut base: u64, mut exp: u64, modu: u64)` to `fn mod_pow(base: u64, mut exp: u64, modu: u64)`. |

---

### Category 3: Unused Variables (8 warnings)

These are variables that are computed but never consumed. Each requires different treatment:

| File | Line | Variable | Fix |
|------|------|----------|-----|
| `winograd_dft.rs` | 27 | `s2` | **Remove the line.** The dft3 function computes `s2 = x1 - x2` as part of the Winograd decomposition, but the actual implementation uses `C::twiddle` (lines 51-58) instead. The variable is a leftover from the intended Winograd path that was never completed. |
| `winograd_dft.rs` | 46-47 | `neg_half`, `half_sqrt3` | **Remove both lines.** These are placeholder variables for the intended Winograd DFT-3 optimization (see comment on line 47: "placeholder — need sqrt3"). Since the implementation uses `C::twiddle` instead, they are dead code. Also remove `s1` on line 26 if it becomes unused after removing `s2` — **but** `s1` IS used on line 30 (`data[0] = C::add(x0, s1)`), so only `s2` is removed. |
| `index_map.rs` | 85 | `n` in `pfa_forward` | **Remove the line.** `let n = n1 * n2` is computed but never used. The function already has `data.len()` available if needed. |
| `index_map.rs` | 106 | `n` in `pfa_inverse` | **Remove the line.** Same reasoning. |
| `index_map.rs` | 144 | `n` in `pfa_dft_forward` | **Remove the line.** Same reasoning. |
| `index_map.rs` | 292 | `c1`, `c2` in test | **Prefix with underscore.** Change to `let (_c1, _c2) = crt_coefficients(3, 5);` — the test is verifying the function runs without panicking, not using the return values. |

---

### Category 4: Dead Code (1 warning)

| File | Line | Issue | Fix |
|------|------|-------|-----|
| `index_map.rs` | 120 | `mod_inverse` function | **Remove the function.** It is defined but never called anywhere in the crate. The `crt_coefficients` function uses `extended_gcd` directly and does not depend on `mod_inverse`. |

---

### Category 5: Non-Snake-Case Variables (5 warnings)

**These cannot be fixed blindly.** Simply renaming `M` → `m` or `N` → `n` would shadow existing parameters with the same lowercase name, causing subtle bugs.

| File | Line | Variable | Fix |
|------|------|----------|-----|
| `winograd_conv.rs` | 43 | `M` | **Rename to `fft_size`.** The parameter `m` already exists (the convolution length), so `M` (the padded FFT size) must have a distinct descriptive name. `fft_size` or `padded_length` are appropriate. |
| `rader.rs` | 53 | `N` | **Rename to `conv_len`.** The value `p - 1` represents the cyclic convolution length in Rader's algorithm. The lowercase `n` is already used as a loop variable in other parts of the file. |
| `rader.rs` | 103 | `X0` | **Rename to `x0`.** This is the frequency-domain value `X[0]`. Since we're in the inverse DFT context, a more descriptive name would be `x_freq_0` or simply `x0` (matching the convention used for `x0` in the forward function at line 38). |
| `rader.rs` | 123 | `N` | **Rename to `conv_len`.** Same as line 53 — the cyclic convolution length `p - 1`. |

**Why not blind:** In `winograd_conv.rs`, `M` and `m` coexist as different values. Renaming `M` to `m` would create a shadowing bug. In `rader.rs`, `N` appears in two separate functions, and `n` is already used as a loop variable at line 142. Consistent, descriptive names are required.

---

### Summary of Changes by File

1. **`winograd_dft.rs`** — Move `Complex64` import into test module; remove unused `s2`, `neg_half`, `half_sqrt3` variables
2. **`fft_arbitrary.rs`** — Remove `FFT` and `DftError` from imports
3. **`bluestein.rs`** — Remove `ComplexSample` import
4. **`rader.rs`** — Remove `ComplexSample` import; rename `N` → `conv_len` (2 places); rename `X0` → `x0`
5. **`factorization.rs`** — Remove `mut` from `base` parameter in `mod_pow`
6. **`index_map.rs`** — Remove 3 unused `let n = ...` lines; remove `mod_inverse` function; prefix `c1`, `c2` with underscores in test

### Verification

After applying all fixes, run `cargo test` to confirm:

- Zero warnings
- All existing tests still pass (especially the PFA tests in `index_map.rs` and the Bluestein/Rader tests)
