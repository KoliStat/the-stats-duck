# A `pinv` rank tolerance silently changed a degrees-of-freedom count

*October 2026, in the whole-branch review of `lin_hyp` (#20).*

`lin_hyp` tests L·β = 0 against an already-computed fit. Its joint test on DS1
reported `df_num` 2 and F 132.5324744. Multiply one predictor by 1000 and
divide the other by 1000 — the same model in different units — and the same
hypothesis reported `df_num` 1 and F 9.4662513, with a p-value of 0.0132 in
place of 2.1e-07. Nothing in the output said anything was wrong.

## TL;DR

1. `lin_hyp` inverts C = L·V·Lᵀ with `linalg::pinv`, which decides the
   numerical rank by comparing each singular value against `1e-12 · σ_max` — a
   tolerance relative to the **largest** singular value.
2. C's diagonal holds the constraints' variances. So when two constraints have
   variances more than about 1e12 apart — standard errors more than about 1e6
   apart — a genuinely full-rank C is read as rank-deficient. A real direction
   is dropped from the Wald statistic, and both `df_num` and F come back wrong.
3. A change of units alone reaches that. "One predictor in thousands, another
   as a proportion" is an ordinary modelling choice.
4. Fix: equilibrate C by its diagonal before the rank decision. With
   D = diag(1/√C_ii), use C⁻ = D·(D·C·D)⁻·D. D·C·D is a correlation matrix, so
   the rank decision is made against singular values that are all O(1), and the
   identity is exact when C has full rank, so nothing already correct moves.
5. The design spec mandated the unconditioned call, in as many words: "C is
   inverted with `pinv`, always, on one code path." It said nothing about
   conditioning. That is where the bug came from, so the spec was corrected
   along with the code.

A single-row hypothesis is exactly invariant, so this only affected q ≥ 2 —
which is the function's headline case, including a Type III test of a
multi-column term.

## How it was found

A probe through `lm_fit` into `lin_hyp` on DS1, with
`L = [[0,1,0],[0,0,1]]` — the joint test of both slopes. `lm_fit`'s own model F
is invariant to rescaling and acts as the control:

| model | `lm_fit` model F | `lin_hyp` `df_num` | `lin_hyp` F | `f_p_value` |
|---|---|---|---|---|
| `y ~ x1 + x2` | 132.53247 | 2 | 132.53247 | 2.1e-07 |
| `y ~ x1*1e3 + x2*1e-3` | 132.53247 | 1 | 9.4662513 | 0.0132 |
| `y ~ x1*1e6 + x2` | 132.53247 | 1 | 9.4662513 | 0.0132 |

The control held in every row, so the fit was fine and the hypothesis layer was
not. The order of proof after that:

1. **The rank, not the arithmetic.** `df_num` moved from 2 to 1, and F moved
   with it. A rounding problem does not change an integer.
2. **The tolerance, not the matrix.** C is 2×2 and well conditioned in relative
   terms: as a correlation matrix its singular values are 1.0554 and 0.9446.
   What was 1e12 apart was the *scale* of its two diagonal entries, and
   `pinv`'s threshold is a single absolute number, `1e-12 · σ_max`, applied to
   both directions at once. Rescaling x1 by 1e3 divides C(0,0) by 1e6;
   rescaling x2 by 1e-3 multiplies C(1,1) by 1e6. The singular values go from
   0.10124 and 0.016918 to 1.0118e5 and 1.6929e-08, a ratio of 1.67e-13 — under
   the 1e-12 threshold, so the second direction is discarded.
3. **The invariance is an oracle fact, not our preference.** statsmodels'
   `wald_test` was run on all three rescaled fits and reported `df_num` 2 and
   F 132.532474 for each. `test/cpp/gen_lin_hyp_fixtures.py` now asserts that
   on every run, so the claim is checked rather than remembered.
4. **The branch's own tests already asserted the invariant, and could not see
   it break.** `test/sql/lin_hyp.test` pins that the joint F equals `lm_fit`'s
   model F — an identity that only holds for well-scaled predictors, so the
   existing test passed while the function was wrong for anything else.

## What to do instead

Equilibrate before the rank decision, and symmetrize in the same pass:

```cpp
std::vector<double> d(q, 1.0);
for (std::size_t i = 0; i < q; i++) {
    if (C(i, i) > 0.0) {
        d[i] = 1.0 / std::sqrt(C(i, i));
    }
}
linalg::Mat Cs(q, q);
for (std::size_t i = 0; i < q; i++) {
    for (std::size_t j = 0; j < q; j++) {
        Cs(i, j) = 0.5 * (C(i, j) + C(j, i)) * d[i] * d[j];
    }
}
// theta is scaled to match, and W = theta_s^T * pinv(Cs) * theta_s.
```

Three details that are easy to get wrong:

- **A zero diagonal entry must keep working.** A zero row of `L` legitimately
  gives C_ii = 0, and that whole row and column of C are zero. Leaving the
  scale factor at 1 there is right: the direction carries no information and
  `pinv` drops it, which is what `estimable = false` for a fully zero `L`
  depends on.
- **The reported `std_error` reads the unscaled C_ii.** √C_ii is the standard
  error of that constraint and must stay in the caller's units. Only the rank
  decision and the quadratic form move to the scaled matrix.
- **Symmetrizing is free and worth doing.** The matrix product leaves a ~1e-17
  asymmetry behind even for a symmetric V, and a quadratic form only has a
  meaning for the symmetric part. It also makes a hand-built asymmetric `cov`
  give one deterministic answer instead of whichever half the product favoured.

Equilibration leaves the correct cases bit-identical in practice. All five
cases the review listed as "must not move" were re-run and did not:

| case | `df_num` | F, before and after |
|---|---|---|
| joint test, as given | 2 | 132.5324744 |
| exactly redundant `[[0,1,-1],[0,-1,1]]` | 1 | 81.19250894 |
| a zero row | 1 | 249.4063533 |
| q > k, four rows spanning two dimensions | 2 | 132.5324744 |
| nearly redundant, ε = 1e-6 | 1 | 81.19246442 |

## The general rule

**A rank tolerance relative to the largest singular value is a scale decision,
not a numerical one, when the matrix diagonal carries variances.** Any kernel
in this repo that calls `linalg::pinv` on a covariance — or on a sandwich of
one — needs the same equilibration. `emmeans` (#21) reaches for exactly this
`pinv` on exactly this kind of matrix, which is why this note exists.

`linalg::pinv` was deliberately left alone. Its tolerance is correct for a
general matrix and other callers rely on it; the conditioning belongs to the
caller that knows what its matrix means.

## See also

- `src/lin_hyp.cpp` — the equilibration, with the reason in the comment.
- `notes/plans/2026-10-14-lin-hyp-design.md` — the Numerics section, corrected.
- `docs/kernel_api.md` — the `lin_hyp.hpp` conventions, which now carry the rule.
- `test/sql/lin_hyp.test` and `test/cpp/test_lin_hyp.cpp` — the invariance tests.
