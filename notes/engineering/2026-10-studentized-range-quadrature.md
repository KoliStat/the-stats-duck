# A quadrature that passed its whole golden grid and was still wrong

*October 2026, building the studentized range distribution for `emmeans` (#21).*

`StudentizedRangeCDF` is a double integral:

```
P(Q <= q; k, nu) = integral f_nu(s) * W(q*s; k) ds
W(w; k)          = k * integral phi(z) * [Phi(z) - Phi(z-w)]^(k-1) dz
```

Both are fixed-node panelled Gauss-Legendre. The first version used 10 nodes on
10 panels for each. It passed all 224 scipy goldens, all four identities, and
agreed with the closed form at k = 2 to 1.2e-15. It was also wrong by 1.8e-9 at
k = 20, which is 170 times the error it should have had and would have been
invisible without going looking for it.

## TL;DR

1. **The k = 2 case is no guide to the k = 20 case.** The inner integrand
   carries the power k − 1. At k = 2 it is a plain bell of width 1. At k = 20 it
   is a bump of width 1/sqrt(19), about 0.22, and panels 1.6 wide do not resolve
   it. The closed form that exists at k = 2 therefore certifies nothing about
   the corner that matters.
2. **A golden grid that passes is not evidence that the error is small.** The
   band was 1e-8 and the grid passed at 6.5e-10. That is inside the band and 650
   times worse than the same cost can buy.
3. **For an entire integrand, raise the node count, not the panel count.** 48
   nodes on 2 panels and 10 nodes on 10 panels cost the same 96 evaluations. The
   first is accurate to 2.3e-15 and the second to 1.8e-9. Gauss-Legendre
   converges faster than any power of n for a function analytic in a
   neighbourhood of the panel, and both of these integrands are entire in their
   integration variable.
4. **Find out whose error you are looking at before you stop tuning.** The error
   against scipy bottomed out at 9.9e-13 and stopped moving when nodes were
   added. It is tempting to call that the oracle's noise. It was not: an
   independent 30-digit mpmath computation agreed with scipy to 5e-15. The
   9.9e-13 was the deliberate 1e-12 tail mass cut from the outer integration
   range, which was then confirmed by varying it.
5. **A density with a non-integer power has a branch point, and a fixed-node
   rule converges only algebraically against one.** `s^(df-1)` is a polynomial
   for an integer df and the integrand is entire. For any other df the branch
   point at s = 0 sits against the lower end of the integration range, and the
   convergence rate drops to about n^-(2·df). This is a real, documented
   limitation of the shipped function, not a bug that got fixed.

## How the k = 20 error was found

Not by the golden grid, which passed. By asking the inner integral alone to
report its own error, two ways:

- against the closed form at k = 2: `W(w; 2) = 2*Phi(w/sqrt(2)) - 1`, worst
  1.2e-15 over w from 0.05 to 18. Reassuring and useless.
- against a deliberately over-resolved copy of itself, 24 nodes on 96 panels,
  over every k on the grid: worst 1.8e-9, at k = 20 and w = 2.25.

The second check is the one that found it. Self-convergence against an
over-resolved copy of the same rule needs no oracle, works at every parameter
value, and is the only check in this whole exercise that located the defect. The
over-resolved copy was itself validated against the k = 2 closed form (1.7e-15)
and against a third resolution (1.8e-15), so it was not just agreeing with
itself.

The lesson generalises past this function: when a numerical routine has a
special case with a closed form, that case is the easiest one and usually the
least representative. Sweep the whole parameter grid against a finer version of
the same rule.

## The node counts the sweep chose

Inner integral, worst error over k in {2,…,20} and w from 0.05 to 21, against
24 nodes on 96 panels:

| nodes x panels | points | worst error |
|---|---|---|
| 10 x 10 | 100 | 1.8e-09 |
| 24 x 4 | 96 | 1.1e-11 |
| 32 x 3 | 96 | 8.0e-12 |
| 40 x 2 | 80 | 2.9e-12 |
| 48 x 2 | 96 | 2.3e-15 |

Outer integral, worst error over the 224-case golden grid, inner fixed at 48 x 2:

| nodes x panels | W calls | worst error |
|---|---|---|
| 20 x 2 | 40 | 2.7e-08 |
| 12 x 4 | 48 | 3.3e-08 |
| 24 x 2 | 48 | 2.1e-10 |
| 28 x 2 | 56 | 9.9e-13 |
| 10 x 10 | 100 | 9.9e-13 |

Both tables say the same thing: at a fixed cost, nodes beat panels here. Note
also that 12 x 4 is worse than 24 x 2 at the identical cost.

## The limitation that was not fixed

The density of s = sqrt(chi-squared_df / df) carries `s^(df-1)`. The shipped
accuracy, measured against scipy at k = 2:

| df | 0.5 | 0.75 | 1.5 | 2.5 | 3.5 | 4.5 | integer, any |
|---|---|---|---|---|---|---|---|
| worst error | 1.2e-02 | 1.6e-03 | 2.0e-05 | 1.0e-07 | 6.4e-10 | 5.2e-12 | 1e-12 |

Every integer df is exact, which is why a 224-case grid of integer df passes at
1e-12 and says nothing about the rest. Three things were tried and did not work:

- **More nodes.** The rate against an endpoint branch point is about n^-(2·df),
  so df = 1.5 would need roughly 350 nodes for 1e-8.
- **More panels.** The error falls like P^-(df), so df = 1.5 would need about 160
  panels.
- **Integrating in u = log s**, which removes the branch point because the
  integrand becomes a composition of exponentials. It is worse: the range in u
  is 20 to 50 wide for a small df, since the lower 1e-12 quantile of s is 1e-8
  at df = 1.5 and 1e-24 at df = 0.5, and the resolution goes.

What would work is a graded mesh from the singularity up to the bulk, with a
bounded ratio per panel so that each panel sees the same self-similar piece of
`s^(df-1)`. That needs about 8 to 12 extra panels, which is 3x to 5x the cost of
the whole function, and the panel count then depends on df. A Gauss-Jacobi rule
carrying `s^(df-1)` in the weight is the textbook answer, but its nodes depend on
df and so cannot be cached by a compile-time constant the way the Legendre rules
are.

Either is a different quadrature rather than a tuning change, so the limitation
is documented on `StudentizedRangeCDF` with its numbers, `test_tukey` prints the
deviation on every run and asserts what holds regardless (a value in [0, 1],
increasing in q, invertible by the quantile), and a df that is not an integer and
is below 4 is known to be less accurate than the band. Anyone adding
Satterthwaite or Kenward-Roger degrees of freedom to a caller of `ptukey` should
read that comment first: those df are not integers.

## Two smaller things worth keeping

**Normalising by the quadrature of the same density.** The outer integral is
computed as a ratio: the W-weighted quadrature over the numerator, the same rule
over the density alone in the denominator. Three things fall out. Any constant
factor in the density cancels, so the normalising constant — which is a
difference of two terms of size 1e11 at df = 1e10 and loses eleven digits — is
never computed. The leading quadrature error of the density cancels too. And a q
so large that W is 1 at every node returns exactly 1.0, because the numerator and
the denominator are then the same sum in the same order.

**`GammaP` stops after 200 terms, and a chi-squared near its mode needs about
sqrt(df/2) of them.** `ChiSquareQuantile` is therefore not trustworthy for a
large df, and nothing says so. Measured against scipy's `chi2.ppf`, its 1e-12
quantile is exact to 5.2e-15 relative at df = 2000 and 5.2e-14 at 1e4, then
drifts to 6.7e-07 at 1e5 and 2.7e-05 at 1e6. The studentized range switches to a
normal approximation for its integration range above df = 2000 for that reason.
Any other caller of `ChiSquareQuantile` with a df in the tens of thousands has
the same problem.
