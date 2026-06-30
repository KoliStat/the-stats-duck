# `std::uniform_int_distribution` isn't portable — seeded `bootstrap` wasn't reproducible across platforms

*June 2026 (v0.7), found while hardening the `r*` sampler tests for the 0.6.0 release.*

`bootstrap(value, statistic, n_iters, seed)` takes a `seed` precisely so results
are reproducible. They were — *on one platform*. The same seed produced a
**different** resample stream on libstdc++ (Linux / MinGW), libc++, and MSVC,
because the resampling drew indices through `std::uniform_int_distribution`.

## TL;DR

1. **The C++ standard pins the *engine* bit-for-bit, but not the
   *distributions*.** `std::mt19937_64` yields an identical sequence everywhere;
   `std::uniform_int_distribution<size_t>(0, n-1)` does **not** — each standard
   library implements it differently, so `dist(rng)` returns different indices
   per platform from the same engine state. Same for `uniform_real_distribution`
   (see [[2026-06-volatile-and-rng-bias]], which hit its MSVC tail bias).
2. **Consequence:** `bootstrap(x, 'mean', 2000, 1)` returned one resample stream
   on the MSVC dev box and a different one on a Linux CI runner. Per-platform
   determinism (the `same seed → identical list` test) held; cross-platform
   reproducibility — the actual promise of `seed` — did not. The round-to-1
   value-convergence tests were also tuned to MSVC and could round differently
   elsewhere.
3. **Fix:** draw indices from the engine directly and bound them with an
   unbiased rejection step — no distribution object. Reproducible on every
   platform we ship.

## Why it's invisible until it isn't

`uniform_int_distribution` is *deterministic given a platform*, so every
single-platform test passes: the determinism check (`f(seed) = f(seed)`) compares
two calls in the same process, and the convergence checks (`round(mean, 1) = 5.5`)
land in the middle of their rounding bucket regardless of the exact stream. The
only observable that crosses platforms is the *exact* resample list — which no
test compares across platforms, and which a user only notices when their seeded
analysis won't reproduce on a colleague's machine.

## The fix

```cpp
// std::uniform_int_distribution is NOT portable across standard libraries, which
// would make seeded bootstraps irreproducible across the platforms we ship.
// mt19937_64 itself is standard-specified, so draw from it directly and reject
// the modulo-biased tail.
static inline size_t BoundedIndex(std::mt19937_64 &rng, size_t n) {
	const uint64_t range  = static_cast<uint64_t>(n);
	const uint64_t reject = (0ULL - range) % range; // 2^64 mod range
	uint64_t r;
	do { r = rng(); } while (r < reject);
	return static_cast<size_t>(r % range);
}
```

Rejection removes modulo bias without needing a 128-bit multiply (MSVC lacks
`__uint128_t`), and `reject` is `2^64 mod range`, so the discarded tail is at most
`range/2^64` of draws — effectively never taken for realistic `n`.

## The general rule

For anything that must reproduce across platforms, **consume the engine, not the
distribution.** `std::mt19937` / `mt19937_64` are portable; `uniform_int`,
`uniform_real`, `normal_distribution`, `shuffle`, etc. are not — their output is
implementation-defined and has changed even between versions of the same library.
The `r*` functions already follow this (manual `2^-53` scaling); `bootstrap` now
does too.
