#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

// Centralized PDF / CDF / quantile implementations for the distributions used
// throughout stats_duck. Header-only so callers pay no link cost and the
// compiler can inline across the aggregate hot path.
//
// Numerical references:
//   - Numerical Recipes in C, 3rd ed., ch. 6 (gamma/beta functions and their
//     incomplete companions).
//   - P. J. Acklam, "An algorithm for computing the inverse normal
//     cumulative distribution function" (Acklam's rational approximation).
//   - M. Abramowitz and I. A. Stegun, Handbook of Mathematical Functions,
//     §8 (Legendre polynomials) and §25.4 (Gauss-Legendre quadrature): the
//     node and weight formulas used by the studentized range section.
//   - W. M. Copenhaver and B. Holland, "Computation of the distribution of
//     the maximum studentized range statistic with application to multiple
//     significance testing of simple effects", Journal of Statistical
//     Computation and Simulation 30 (1988), 1-15: the two-integral form of
//     the studentized range distribution. The mathematics only. R's ptukey
//     is GPL and no part of it was read, ported or paraphrased here.

namespace stats_duck {

#ifndef STATS_DUCK_PI
#define STATS_DUCK_PI 3.14159265358979323846
#endif

// ── Gamma / Beta helpers ────────────────────────────────────────────────────

//! Log of the Beta function: log(B(a, b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
inline double LogBeta(double a, double b) {
	return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

//! Lower regularized incomplete gamma function P(a, x) via its series expansion.
//! Used when x < a + 1 (series converges quickly in that region).
inline double GammaPSeries(double a, double x) {
	const int max_iter = 200;
	const double eps = std::numeric_limits<double>::epsilon();
	if (x == 0.0) {
		return 0.0;
	}
	double ap = a;
	double del = 1.0 / a;
	double sum = del;
	for (int n = 1; n < max_iter; n++) {
		ap += 1.0;
		del *= x / ap;
		sum += del;
		if (std::abs(del) < std::abs(sum) * eps) {
			break;
		}
	}
	double log_prefix = -x + a * std::log(x) - std::lgamma(a);
	return sum * std::exp(log_prefix);
}

//! Upper regularized incomplete gamma function Q(a, x) via continued fraction.
//! Used when x >= a + 1 (CF converges quickly there).
inline double GammaQContinuedFraction(double a, double x) {
	const int max_iter = 200;
	const double eps = std::numeric_limits<double>::epsilon();
	const double tiny = 1e-30;

	// Modified Lentz's algorithm.
	double b = x + 1.0 - a;
	double c = 1.0 / tiny;
	double d = 1.0 / b;
	double h = d;
	for (int i = 1; i <= max_iter; i++) {
		double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
		b += 2.0;
		d = an * d + b;
		if (std::abs(d) < tiny) {
			d = tiny;
		}
		c = b + an / c;
		if (std::abs(c) < tiny) {
			c = tiny;
		}
		d = 1.0 / d;
		double del = d * c;
		h *= del;
		if (std::abs(del - 1.0) < eps) {
			break;
		}
	}
	double log_prefix = -x + a * std::log(x) - std::lgamma(a);
	return std::exp(log_prefix) * h;
}

//! Regularized lower incomplete gamma P(a, x). Dispatches between series and CF.
inline double GammaP(double a, double x) {
	if (a <= 0.0) {
		throw std::invalid_argument("a must be > 0");
	}
	if (x < 0.0) {
		throw std::invalid_argument("x must be >= 0");
	}
	if (x == 0.0) {
		return 0.0;
	}
	if (x < a + 1.0) {
		return GammaPSeries(a, x);
	}
	return 1.0 - GammaQContinuedFraction(a, x);
}

//! Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x).
inline double GammaQ(double a, double x) {
	if (a <= 0.0) {
		throw std::invalid_argument("a must be > 0");
	}
	if (x < 0.0) {
		throw std::invalid_argument("x must be >= 0");
	}
	if (x == 0.0) {
		return 1.0;
	}
	if (x < a + 1.0) {
		return 1.0 - GammaPSeries(a, x);
	}
	return GammaQContinuedFraction(a, x);
}

//! Regularized incomplete beta function I_x(a, b) via continued fraction (Lentz).
inline double BetaIncomplete(double a, double b, double x) {
	if (x < 0.0 || x > 1.0) {
		throw std::invalid_argument("x must be in [0, 1]");
	}
	if (x == 0.0) {
		return 0.0;
	}
	if (x == 1.0) {
		return 1.0;
	}

	// Symmetry relation for better CF convergence when x is large.
	if (x > (a + 1.0) / (a + b + 2.0)) {
		return 1.0 - BetaIncomplete(b, a, 1.0 - x);
	}

	double log_prefix = a * std::log(x) + b * std::log(1.0 - x) - LogBeta(a, b) - std::log(a);

	const double eps = std::numeric_limits<double>::epsilon();
	const double tiny = 1e-30;
	const int max_iter = 200;

	double f = 1.0;
	double c = 1.0;
	double d = 1.0 - (a + b) * x / (a + 1.0);
	if (std::abs(d) < tiny) {
		d = tiny;
	}
	d = 1.0 / d;
	f = d;

	for (int m = 1; m <= max_iter; m++) {
		double m_d = static_cast<double>(m);
		double numerator = m_d * (b - m_d) * x / ((a + 2.0 * m_d - 1.0) * (a + 2.0 * m_d));

		d = 1.0 + numerator * d;
		if (std::abs(d) < tiny) {
			d = tiny;
		}
		c = 1.0 + numerator / c;
		if (std::abs(c) < tiny) {
			c = tiny;
		}
		d = 1.0 / d;
		f *= c * d;

		numerator = -((a + m_d) * (a + b + m_d) * x) / ((a + 2.0 * m_d) * (a + 2.0 * m_d + 1.0));

		d = 1.0 + numerator * d;
		if (std::abs(d) < tiny) {
			d = tiny;
		}
		c = 1.0 + numerator / c;
		if (std::abs(c) < tiny) {
			c = tiny;
		}
		d = 1.0 / d;
		double delta = c * d;
		f *= delta;

		if (std::abs(delta - 1.0) < eps) {
			break;
		}
	}

	return std::exp(log_prefix) * f;
}

// ── Normal distribution ─────────────────────────────────────────────────────

inline double NormalPDF(double x, double mean = 0.0, double sd = 1.0) {
	if (sd <= 0.0) {
		throw std::invalid_argument("sd must be > 0");
	}
	double z = (x - mean) / sd;
	return std::exp(-0.5 * z * z) / (sd * std::sqrt(2.0 * STATS_DUCK_PI));
}

inline double NormalCDF(double x, double mean = 0.0, double sd = 1.0) {
	if (sd <= 0.0) {
		throw std::invalid_argument("sd must be > 0");
	}
	double z = (x - mean) / sd;
	return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}

//! Quantile of the normal distribution via Acklam's rational approximation.
//! Accurate to ~1e-9 in the tails; constant-time.
inline double NormalQuantile(double p, double mean = 0.0, double sd = 1.0) {
	if (sd <= 0.0) {
		throw std::invalid_argument("sd must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return -std::numeric_limits<double>::infinity();
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}

	static const double a_coef[] = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
	                                 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
	static const double b_coef[] = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
	                                 6.680131188771972e+01, -1.328068155288572e+01};
	static const double c_coef[] = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
	                                -2.549732539343734e+00, 4.374664141464968e+00,  2.938163982698783e+00};
	static const double d_coef[] = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
	                                3.754408661907416e+00};

	const double p_low = 0.02425;
	const double p_high = 1.0 - p_low;

	double q, r, z;
	if (p < p_low) {
		q = std::sqrt(-2.0 * std::log(p));
		z = (((((c_coef[0] * q + c_coef[1]) * q + c_coef[2]) * q + c_coef[3]) * q + c_coef[4]) * q + c_coef[5]) /
		    ((((d_coef[0] * q + d_coef[1]) * q + d_coef[2]) * q + d_coef[3]) * q + 1.0);
	} else if (p <= p_high) {
		q = p - 0.5;
		r = q * q;
		z = (((((a_coef[0] * r + a_coef[1]) * r + a_coef[2]) * r + a_coef[3]) * r + a_coef[4]) * r + a_coef[5]) * q /
		    (((((b_coef[0] * r + b_coef[1]) * r + b_coef[2]) * r + b_coef[3]) * r + b_coef[4]) * r + 1.0);
	} else {
		q = std::sqrt(-2.0 * std::log(1.0 - p));
		z = -(((((c_coef[0] * q + c_coef[1]) * q + c_coef[2]) * q + c_coef[3]) * q + c_coef[4]) * q + c_coef[5]) /
		    ((((d_coef[0] * q + d_coef[1]) * q + d_coef[2]) * q + d_coef[3]) * q + 1.0);
	}
	return mean + sd * z;
}

// ── Student's t distribution ────────────────────────────────────────────────

inline double StudentTPDF(double x, double df) {
	if (df <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	double log_pdf = std::lgamma(0.5 * (df + 1.0)) - std::lgamma(0.5 * df) - 0.5 * std::log(df * STATS_DUCK_PI) -
	                 0.5 * (df + 1.0) * std::log(1.0 + x * x / df);
	return std::exp(log_pdf);
}

inline double StudentTCDF(double t, double df) {
	if (df <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	double x = df / (df + t * t);
	double ibeta = BetaIncomplete(df / 2.0, 0.5, x);
	if (t >= 0.0) {
		return 1.0 - 0.5 * ibeta;
	}
	return 0.5 * ibeta;
}

//! Quantile (inverse CDF) of Student's t via bisection.
inline double StudentTQuantile(double p, double df) {
	if (df <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	if (p <= 0.0 || p >= 1.0) {
		if (p == 0.0) {
			return -std::numeric_limits<double>::infinity();
		}
		if (p == 1.0) {
			return std::numeric_limits<double>::infinity();
		}
		throw std::invalid_argument("p must be in (0, 1)");
	}

	double lo = -1000.0;
	double hi = 1000.0;
	while (StudentTCDF(lo, df) > p) {
		lo *= 2.0;
	}
	while (StudentTCDF(hi, df) < p) {
		hi *= 2.0;
	}
	for (int i = 0; i < 100; i++) {
		double mid = 0.5 * (lo + hi);
		if (StudentTCDF(mid, df) < p) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5 * (lo + hi);
}

// ── Chi-square distribution ─────────────────────────────────────────────────

inline double ChiSquarePDF(double x, double df) {
	if (df <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	if (x < 0.0) {
		return 0.0;
	}
	if (x == 0.0) {
		if (df < 2.0) {
			return std::numeric_limits<double>::infinity();
		}
		if (df == 2.0) {
			return 0.5;
		}
		return 0.0;
	}
	double k = df / 2.0;
	double log_pdf = (k - 1.0) * std::log(x) - 0.5 * x - k * std::log(2.0) - std::lgamma(k);
	return std::exp(log_pdf);
}

inline double ChiSquareCDF(double x, double df) {
	if (df <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	return GammaP(df / 2.0, x / 2.0);
}

inline double ChiSquareQuantile(double p, double df) {
	if (df <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}

	double lo = 0.0;
	double hi = std::max(df * 4.0, 10.0);
	while (ChiSquareCDF(hi, df) < p) {
		hi *= 2.0;
	}
	for (int i = 0; i < 100; i++) {
		double mid = 0.5 * (lo + hi);
		if (ChiSquareCDF(mid, df) < p) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5 * (lo + hi);
}

// ── F distribution ──────────────────────────────────────────────────────────

inline double FPDF(double x, double df1, double df2) {
	if (df1 <= 0.0 || df2 <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	double log_pdf = 0.5 * df1 * std::log(df1) + 0.5 * df2 * std::log(df2) + (0.5 * df1 - 1.0) * std::log(x) -
	                 0.5 * (df1 + df2) * std::log(df2 + df1 * x) - LogBeta(0.5 * df1, 0.5 * df2);
	return std::exp(log_pdf);
}

inline double FCDF(double x, double df1, double df2) {
	if (df1 <= 0.0 || df2 <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	double z = df2 / (df2 + df1 * x);
	return 1.0 - BetaIncomplete(df2 / 2.0, df1 / 2.0, z);
}

inline double FQuantile(double p, double df1, double df2) {
	if (df1 <= 0.0 || df2 <= 0.0) {
		throw std::invalid_argument("df must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}

	double lo = 0.0;
	double hi = 20.0;
	while (FCDF(hi, df1, df2) < p) {
		hi *= 2.0;
	}
	for (int i = 0; i < 100; i++) {
		double mid = 0.5 * (lo + hi);
		if (FCDF(mid, df1, df2) < p) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5 * (lo + hi);
}

// ── Gamma distribution ──────────────────────────────────────────────────────
// Parametrization: shape α > 0, rate β > 0 (matching R's default rate form;
// scale = 1/rate). PDF = β^α / Γ(α) · x^(α-1) · e^(-βx) on x ≥ 0.

inline double GammaPDF(double x, double shape, double rate = 1.0) {
	if (shape <= 0.0 || rate <= 0.0) {
		throw std::invalid_argument("shape and rate must be > 0");
	}
	if (x < 0.0) {
		return 0.0;
	}
	if (x == 0.0) {
		if (shape < 1.0) {
			return std::numeric_limits<double>::infinity();
		}
		if (shape == 1.0) {
			return rate;
		}
		return 0.0;
	}
	double log_pdf = shape * std::log(rate) + (shape - 1.0) * std::log(x) - rate * x -
	                 std::lgamma(shape);
	return std::exp(log_pdf);
}

inline double GammaCDF(double x, double shape, double rate = 1.0) {
	if (shape <= 0.0 || rate <= 0.0) {
		throw std::invalid_argument("shape and rate must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	return GammaP(shape, rate * x);
}

inline double GammaQuantile(double p, double shape, double rate = 1.0) {
	if (shape <= 0.0 || rate <= 0.0) {
		throw std::invalid_argument("shape and rate must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}

	// Bracket via the mean (shape/rate); expand until CDF(hi) ≥ p.
	double lo = 0.0;
	double hi = std::max(shape / rate * 2.0, 1.0 / rate);
	while (GammaCDF(hi, shape, rate) < p) {
		hi *= 2.0;
	}
	for (int i = 0; i < 100; i++) {
		double mid = 0.5 * (lo + hi);
		if (GammaCDF(mid, shape, rate) < p) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5 * (lo + hi);
}

// ── Beta distribution ───────────────────────────────────────────────────────
// Shape parameters α, β > 0, support [0, 1]. PDF = x^(α-1)(1-x)^(β-1) / B(α, β).

inline double BetaPDF(double x, double alpha, double beta) {
	if (alpha <= 0.0 || beta <= 0.0) {
		throw std::invalid_argument("alpha and beta must be > 0");
	}
	if (x < 0.0 || x > 1.0) {
		return 0.0;
	}
	if (x == 0.0) {
		if (alpha < 1.0) {
			return std::numeric_limits<double>::infinity();
		}
		if (alpha == 1.0) {
			return beta;
		}
		return 0.0;
	}
	if (x == 1.0) {
		if (beta < 1.0) {
			return std::numeric_limits<double>::infinity();
		}
		if (beta == 1.0) {
			return alpha;
		}
		return 0.0;
	}
	double log_pdf =
	    (alpha - 1.0) * std::log(x) + (beta - 1.0) * std::log(1.0 - x) - LogBeta(alpha, beta);
	return std::exp(log_pdf);
}

inline double BetaCDF(double x, double alpha, double beta) {
	if (alpha <= 0.0 || beta <= 0.0) {
		throw std::invalid_argument("alpha and beta must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	if (x >= 1.0) {
		return 1.0;
	}
	return BetaIncomplete(alpha, beta, x);
}

inline double BetaQuantile(double p, double alpha, double beta) {
	if (alpha <= 0.0 || beta <= 0.0) {
		throw std::invalid_argument("alpha and beta must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return 1.0;
	}

	// Bisect on [0, 1] — support is bounded, no need to expand.
	double lo = 0.0;
	double hi = 1.0;
	for (int i = 0; i < 100; i++) {
		double mid = 0.5 * (lo + hi);
		if (BetaCDF(mid, alpha, beta) < p) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5 * (lo + hi);
}

// ── Exponential distribution ────────────────────────────────────────────────
// Rate λ > 0. PDF = λ · e^(-λx) on x ≥ 0. Special case of Gamma(1, λ) but
// worth a closed-form path because the quantile is analytic.

inline double ExponentialPDF(double x, double rate = 1.0) {
	if (rate <= 0.0) {
		throw std::invalid_argument("rate must be > 0");
	}
	if (x < 0.0) {
		return 0.0;
	}
	return rate * std::exp(-rate * x);
}

inline double ExponentialCDF(double x, double rate = 1.0) {
	if (rate <= 0.0) {
		throw std::invalid_argument("rate must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	return 1.0 - std::exp(-rate * x);
}

inline double ExponentialQuantile(double p, double rate = 1.0) {
	if (rate <= 0.0) {
		throw std::invalid_argument("rate must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}
	return -std::log(1.0 - p) / rate;
}

// ── Weibull distribution ────────────────────────────────────────────────────
// Shape k > 0, scale λ > 0. Support x ≥ 0. PDF = (k/λ)·(x/λ)^(k-1)·e^(-(x/λ)^k).
// CDF and quantile are both closed-form via log/exp. Special case k=1 → Exp(1/λ).
// R parameterisation: dweibull(x, shape, scale = 1).

inline double WeibullPDF(double x, double shape, double scale = 1.0) {
	if (shape <= 0.0 || scale <= 0.0) {
		throw std::invalid_argument("shape and scale must be > 0");
	}
	if (x < 0.0) {
		return 0.0;
	}
	if (x == 0.0) {
		// Limit: shape > 1 → 0; shape = 1 → 1/scale; shape < 1 → +inf.
		if (shape > 1.0) {
			return 0.0;
		}
		if (shape == 1.0) {
			return 1.0 / scale;
		}
		return std::numeric_limits<double>::infinity();
	}
	double ratio = x / scale;
	return (shape / scale) * std::pow(ratio, shape - 1.0) * std::exp(-std::pow(ratio, shape));
}

inline double WeibullCDF(double x, double shape, double scale = 1.0) {
	if (shape <= 0.0 || scale <= 0.0) {
		throw std::invalid_argument("shape and scale must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	return 1.0 - std::exp(-std::pow(x / scale, shape));
}

inline double WeibullQuantile(double p, double shape, double scale = 1.0) {
	if (shape <= 0.0 || scale <= 0.0) {
		throw std::invalid_argument("shape and scale must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}
	return scale * std::pow(-std::log(1.0 - p), 1.0 / shape);
}

// ── Log-normal distribution ─────────────────────────────────────────────────
// Parameters meanlog (μ) and sdlog (σ > 0); X = exp(μ + σZ) where Z ~ N(0,1).
// Support x > 0. R parameterisation: dlnorm(x, meanlog = 0, sdlog = 1).

inline double LogNormalPDF(double x, double meanlog = 0.0, double sdlog = 1.0) {
	if (sdlog <= 0.0) {
		throw std::invalid_argument("sdlog must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	double z = (std::log(x) - meanlog) / sdlog;
	return std::exp(-0.5 * z * z) / (x * sdlog * std::sqrt(2.0 * STATS_DUCK_PI));
}

inline double LogNormalCDF(double x, double meanlog = 0.0, double sdlog = 1.0) {
	if (sdlog <= 0.0) {
		throw std::invalid_argument("sdlog must be > 0");
	}
	if (x <= 0.0) {
		return 0.0;
	}
	return NormalCDF(std::log(x), meanlog, sdlog);
}

inline double LogNormalQuantile(double p, double meanlog = 0.0, double sdlog = 1.0) {
	if (sdlog <= 0.0) {
		throw std::invalid_argument("sdlog must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}
	return std::exp(NormalQuantile(p, meanlog, sdlog));
}

// ── Poisson distribution ────────────────────────────────────────────────────
// Discrete; rate λ > 0. PMF on k = 0, 1, 2, …  CDF: P(X ≤ k) = Q(k+1, λ) via
// the regularized upper incomplete gamma, exploiting the identity
//   Σ_{i=0..k} e^-λ λ^i / i! = Q(k+1, λ)  (Numerical Recipes §6.2).
// Quantile is monotone integer search seeded from the normal approximation
// k ≈ λ + Φ^-1(p)·sqrt(λ).

inline double PoissonPMF(double k, double lambda) {
	if (lambda <= 0.0) {
		throw std::invalid_argument("lambda must be > 0");
	}
	if (k < 0.0) {
		return 0.0;
	}
	double k_int = std::floor(k);
	if (k_int != k) {
		// Non-integer k → PMF is 0 (matches R's dpois(non-int) → 0 with warning).
		return 0.0;
	}
	// e^(-λ) · λ^k / k!  via logs to avoid overflow for moderate λ.
	double log_pmf = -lambda + k_int * std::log(lambda) - std::lgamma(k_int + 1.0);
	return std::exp(log_pmf);
}

inline double PoissonCDF(double k, double lambda) {
	if (lambda <= 0.0) {
		throw std::invalid_argument("lambda must be > 0");
	}
	if (k < 0.0) {
		return 0.0;
	}
	double k_int = std::floor(k);
	// P(X ≤ k) = Q(k+1, λ).
	return GammaQ(k_int + 1.0, lambda);
}

inline double PoissonQuantile(double p, double lambda) {
	if (lambda <= 0.0) {
		throw std::invalid_argument("lambda must be > 0");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}
	// Normal approximation seed; clamp to ≥ 0.
	double seed = lambda + NormalQuantile(p) * std::sqrt(lambda);
	double k = std::floor(std::max(0.0, seed));
	double cdf = PoissonCDF(k, lambda);
	// March in the direction of p.
	if (cdf < p) {
		while (cdf < p) {
			k += 1.0;
			cdf = PoissonCDF(k, lambda);
		}
		return k;
	}
	while (k > 0.0 && PoissonCDF(k - 1.0, lambda) >= p) {
		k -= 1.0;
	}
	return k;
}

// ── Negative Binomial distribution ──────────────────────────────────────────
// R parameterisation: dnbinom(x, size, prob). size > 0 is the target number of
// successes (not required to be integer — Gamma-Poisson mixture admits any
// positive real), prob ∈ (0, 1] is the per-trial success probability. Support
// is k = 0, 1, 2, …  CDF closed-form identity:
//   P(X ≤ k) = I_prob(size, k + 1)
// where I_x(a, b) is the regularized incomplete beta function (see
// BetaIncomplete). Quantile is monotone integer search seeded from the normal
// approximation.

inline double NegBinomPMF(double k, double size, double prob) {
	if (size <= 0.0 || prob <= 0.0 || prob > 1.0) {
		throw std::invalid_argument("size must be > 0 and prob in (0, 1]");
	}
	if (k < 0.0) {
		return 0.0;
	}
	double k_int = std::floor(k);
	if (k_int != k) {
		// Non-integer k → 0 (matches R's warn-and-zero convention).
		return 0.0;
	}
	// log PMF = lgamma(k + size) - lgamma(size) - lgamma(k + 1)
	//          + size·log(prob) + k·log(1 - prob)
	double log_pmf = std::lgamma(k_int + size) - std::lgamma(size) - std::lgamma(k_int + 1.0) +
	                 size * std::log(prob) + k_int * std::log(1.0 - prob);
	return std::exp(log_pmf);
}

inline double NegBinomCDF(double k, double size, double prob) {
	if (size <= 0.0 || prob <= 0.0 || prob > 1.0) {
		throw std::invalid_argument("size must be > 0 and prob in (0, 1]");
	}
	if (k < 0.0) {
		return 0.0;
	}
	double k_int = std::floor(k);
	return BetaIncomplete(size, k_int + 1.0, prob);
}

inline double NegBinomQuantile(double p, double size, double prob) {
	if (size <= 0.0 || prob <= 0.0 || prob > 1.0) {
		throw std::invalid_argument("size must be > 0 and prob in (0, 1]");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}
	// Normal approximation seed: mean = size·(1-prob)/prob, variance same / prob.
	double mean = size * (1.0 - prob) / prob;
	double var = mean / prob;
	double seed = mean + NormalQuantile(p) * std::sqrt(var);
	double k = std::floor(std::max(0.0, seed));
	double cdf = NegBinomCDF(k, size, prob);
	if (cdf < p) {
		while (cdf < p) {
			k += 1.0;
			cdf = NegBinomCDF(k, size, prob);
		}
		return k;
	}
	while (k > 0.0 && NegBinomCDF(k - 1.0, size, prob) >= p) {
		k -= 1.0;
	}
	return k;
}

// ── Hypergeometric distribution ─────────────────────────────────────────────
// R parameterisation: dhyper(x, m, n, k). m = total successes in population,
// n = total failures, k = number drawn without replacement. Population size is
// N = m + n. PMF support: max(0, k - n) ≤ x ≤ min(m, k). No closed-form CDF;
// we sum the PMF over the support up to floor(x). All three parameters are
// taken as real-valued (matches R's permissive signature) but are interpreted
// as floor()-truncated counts; non-integer x in the PMF / quantile yields 0
// to match R's warn-and-zero.

namespace detail {
inline double LogChoose(double a, double b) {
	// log C(a, b) = lgamma(a+1) - lgamma(b+1) - lgamma(a-b+1)
	return std::lgamma(a + 1.0) - std::lgamma(b + 1.0) - std::lgamma(a - b + 1.0);
}
} // namespace detail

inline double HyperGeomPMF(double x, double m, double n, double k) {
	if (m < 0.0 || n < 0.0 || k < 0.0 || k > m + n) {
		throw std::invalid_argument("hyper: m, n, k must be >= 0 and k <= m + n");
	}
	double x_int = std::floor(x);
	if (x_int != x) {
		return 0.0;
	}
	double lo = std::max(0.0, k - n);
	double hi = std::min(m, k);
	if (x_int < lo || x_int > hi) {
		return 0.0;
	}
	double log_pmf =
	    detail::LogChoose(m, x_int) + detail::LogChoose(n, k - x_int) - detail::LogChoose(m + n, k);
	return std::exp(log_pmf);
}

inline double HyperGeomCDF(double q, double m, double n, double k) {
	if (m < 0.0 || n < 0.0 || k < 0.0 || k > m + n) {
		throw std::invalid_argument("hyper: m, n, k must be >= 0 and k <= m + n");
	}
	double q_int = std::floor(q);
	double lo = std::max(0.0, k - n);
	double hi = std::min(m, k);
	if (q_int < lo) {
		return 0.0;
	}
	if (q_int >= hi) {
		return 1.0;
	}
	// Linear sum from lo. Support is bounded by min(m, k) so this is fine for
	// the population sizes biostat / sampling-without-replacement workloads
	// actually use. A normal-approximation fast path would help for huge m, k.
	double cdf = 0.0;
	for (double i = lo; i <= q_int; i += 1.0) {
		cdf += HyperGeomPMF(i, m, n, k);
	}
	if (cdf > 1.0) {
		cdf = 1.0; // clamp tiny rounding overshoots
	}
	return cdf;
}

inline double HyperGeomQuantile(double p, double m, double n, double k) {
	if (m < 0.0 || n < 0.0 || k < 0.0 || k > m + n) {
		throw std::invalid_argument("hyper: m, n, k must be >= 0 and k <= m + n");
	}
	if (p < 0.0 || p > 1.0) {
		throw std::invalid_argument("p must be in [0, 1]");
	}
	double lo = std::max(0.0, k - n);
	double hi = std::min(m, k);
	if (p == 0.0) {
		return lo;
	}
	if (p == 1.0) {
		return hi;
	}
	double cdf = 0.0;
	for (double x = lo; x <= hi; x += 1.0) {
		cdf += HyperGeomPMF(x, m, n, k);
		if (cdf >= p) {
			return x;
		}
	}
	return hi;
}

// ── Studentized range distribution ──────────────────────────────────────────
// Q = (max - min) / S for k independent standard normals and an independent S
// with df degrees of freedom, S^2 ~ chi-squared_df / df. This is the
// distribution behind Tukey's multiplicity adjustment.
//
//   P(Q <= q; k, nu) = integral over s > 0 of  f_nu(s) * W(q*s; k) ds
//   W(w; k)          = k * integral over z of  phi(z) * [Phi(z) - Phi(z-w)]^(k-1) dz
//
// nu in the formulas is the df argument. f_nu is the density of
// s = sqrt(chi-squared_nu / nu) and W is the CDF of the range of k standard
// normals. The inner form counts which of the k values is the largest (the
// factor k), puts it at z, and requires the other k - 1 to lie within w below
// it.
//
// Both integrals are panelled Gauss-Legendre with fixed node and panel counts,
// and the quantile is bisection with a fixed step count, so identical inputs
// give identical results on every platform, as docs/fitter_conventions.md
// requires. There is no adaptive stopping rule anywhere in this section.
//
// Unlike the rest of this header these two functions return NaN for an
// argument outside the domain instead of throwing. They are called per row
// from emmeans and from the ptukey / qtukey SQL scalars, where a NaN travels
// with the row and an exception would abort the query.
//
// The mathematics is the published two-integral form (Copenhaver and Holland,
// 1988; see the reference block at the top of this file). R's ptukey is GPL
// and no part of it was read, ported or paraphrased.

namespace detail {

// 1 / sqrt(2) and 1 / sqrt(2*pi), written out so the inner loop does not
// recompute them.
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;

//! n-point Gauss-Legendre nodes and weights on [-1, 1].
struct GaussLegendreRule {
	std::vector<double> x;
	std::vector<double> w;
};

//! Nodes and weights for the n-point rule, by Newton iteration on P_n.
//!
//! The nodes are the roots of the Legendre polynomial P_n. P_n and P_{n-1}
//! come from the three-term recurrence j*P_j(z) = (2j-1)*z*P_{j-1}(z) -
//! (j-1)*P_{j-2}(z), the derivative from (z^2 - 1)*P_n'(z) = n*(z*P_n(z) -
//! P_{n-1}(z)), and the weight at a root from w = 2 / ((1 - z^2) * P_n'(z)^2)
//! (Abramowitz and Stegun, §8 and §25.4). The starting guess
//! cos(pi*(i + 0.75)/(n + 0.5)) sits within one root spacing of the i-th root,
//! counting down from +1.
//!
//! The iteration count is fixed rather than tested for convergence: Newton
//! doubles the number of correct digits per step, so five steps already land
//! on the root from that guess and the remaining ones leave it where it is.
//! A fixed count cannot stop at a different place on a different platform.
//!
//! The two rules this file uses were checked against numpy's leggauss, which
//! finds the same nodes as eigenvalues of a tridiagonal matrix rather than by
//! Newton: the 28-node and 48-node rules agree with it to 1.1e-16 on the nodes
//! and 4.4e-15 on the weights, and each rule's weights sum to 2 to within
//! 9e-16.
inline GaussLegendreRule ComputeGaussLegendreRule(int n) {
	const int kNewtonSteps = 8;
	GaussLegendreRule rule;
	rule.x.assign(static_cast<std::size_t>(n), 0.0);
	rule.w.assign(static_cast<std::size_t>(n), 0.0);
	for (int i = 0; i < n; i++) {
		double z = std::cos(STATS_DUCK_PI * (static_cast<double>(i) + 0.75) /
		                    (static_cast<double>(n) + 0.5));
		double deriv = 1.0;
		for (int step = 0; step < kNewtonSteps; step++) {
			double p_prev = 1.0; // P_0
			double p_cur = z;    // P_1
			for (int j = 2; j <= n; j++) {
				const double jd = static_cast<double>(j);
				const double p_next = ((2.0 * jd - 1.0) * z * p_cur - (jd - 1.0) * p_prev) / jd;
				p_prev = p_cur;
				p_cur = p_next;
			}
			deriv = static_cast<double>(n) * (z * p_cur - p_prev) / (z * z - 1.0);
			z -= p_cur / deriv;
		}
		rule.x[static_cast<std::size_t>(i)] = z;
		rule.w[static_cast<std::size_t>(i)] = 2.0 / ((1.0 - z * z) * deriv * deriv);
	}
	return rule;
}

//! The cached rule for a node count fixed at compile time.
//!
//! One function-local static per node count, which C++11 and later initialize
//! once and thread-safely. The cache is keyed by the template argument and not
//! by a map: a std::string-keyed unordered_map with the default hash pulls in
//! a libc++ symbol duckdb-wasm does not export, and this header is compiled
//! into the wasm build. See notes/engineering/ and the stats_duck WASM notes.
template <int kNodes>
inline const GaussLegendreRule &GaussLegendreNodes() {
	static const GaussLegendreRule rule = ComputeGaussLegendreRule(kNodes);
	return rule;
}

//! The kNodes-point rule applied on each of `panels` equal panels of [a, b].
//!
//! Panels matter because one high-order rule over a wide interval has to fit a
//! single polynomial to the whole integrand, while panels only have to fit it
//! locally. The loop order is fixed, panels then nodes ascending, so the sum
//! is reproducible. An empty or inverted interval integrates to 0.
template <int kNodes, typename Integrand>
inline double PanelledGaussLegendre(double a, double b, int panels, Integrand f) {
	if (!(b > a) || panels < 1) {
		return 0.0;
	}
	const GaussLegendreRule &rule = GaussLegendreNodes<kNodes>();
	const double width = (b - a) / static_cast<double>(panels);
	const double half = 0.5 * width;
	double total = 0.0;
	for (int panel = 0; panel < panels; panel++) {
		const double mid = a + (static_cast<double>(panel) + 0.5) * width;
		double sum = 0.0;
		for (int i = 0; i < kNodes; i++) {
			sum += rule.w[static_cast<std::size_t>(i)] * f(mid + half * rule.x[static_cast<std::size_t>(i)]);
		}
		total += half * sum;
	}
	return total;
}

//! The w beyond which the range CDF of k standard normals is 1.0 exactly.
//!
//! The range exceeds w only if the largest of the k values is above w/2 or the
//! smallest is below -w/2, so
//!   1 - W(w; k) <= 2*k*(1 - Phi(w/2)) <= 2*k*exp(-w*w/8),
//! and that bound reaches 1e-17 at w = 2*sqrt(2*log(2e17*k)). 1e-17 is below
//! half the spacing of doubles at 1, which is 1.1e-16, so 1.0 is the correctly
//! rounded answer there rather than a clamp. The point is 18.0 at k = 2 and
//! 20.9 at k = 1e6: it grows with k, slowly.
inline double NormalRangeFlatPoint(double k) {
	return 2.0 * std::sqrt(2.0 * std::log(2.0e17 * k));
}

// ---- The tuned constants of the studentized range quadrature.
//
// The node and panel counts were chosen by sweeping them against the whole
// scipy fixture grid in test/cpp/tukey_fixtures.hpp and against the identities
// in test/cpp/test_tukey.cpp, then stopping at the point where the measured
// error stops improving. Each constant carries its own evidence below.
//
// Two facts guided the sweep. The integrands are entire functions of the
// integration variable, so a high-order rule on a few wide panels beats a
// low-order rule on many narrow ones: the inner integral at 48 nodes on 2
// panels is 2.3e-15, while 10 nodes on 10 panels, the same cost, is 1.8e-9.
// And the k = 2 case is no guide to k = 20: the inner integrand carries the
// power k - 1, which at k = 20 is a bump of width 1/sqrt(19), so the error at
// k = 20 was 1.8e-9 while the k = 2 closed form agreed to 1.2e-15.

//! Inner integral: panels and nodes per panel over [-8, 8] in z.
//!
//! 2 panels of 48 nodes. Worst error 2.3e-15 against a 96-panel 24-node
//! reference, over k in {2,3,4,5,6,10,20} and w from 0.05 to 21 in steps of
//! 0.05, which is the whole range the outer integral asks for. That reference
//! agrees with the k = 2 closed form 2*Phi(w/sqrt(2)) - 1 to 1.7e-15, so 2.3e-15
//! is the floor of the measurement and not a residual. Dropping to 40 nodes
//! gives 2.9e-12 and to 36 nodes 1.3e-10, both still inside the band; 48 is
//! kept because it costs 20% more than 40 and removes the error entirely.
constexpr int kRangePanels = 2;
constexpr int kRangeNodes = 48;

//! The truncation point of the inner integral.
//!
//! The integrand is bounded by phi(z), whose mass beyond |z| = 8 is below
//! 1.3e-15, so truncating there costs less than k * 1.3e-15 — six orders below
//! the 1e-8 band this function is held to at k = 20.
constexpr double kRangeZLimit = 8.0;

//! Outer integral: panels and nodes per panel in s.
//!
//! 2 panels of 28 nodes, on each side of the break point below. Worst error
//! 9.9e-13 over the whole 224-case fixture grid, which is where that error
//! stops falling: 32 nodes, 10 panels of 10 nodes and 4 panels of 20 nodes all
//! give the same 9.9e-13, because what is left is the tail mass below, not this
//! rule. Dropping to 24 nodes gives 2.1e-10 and to 20 nodes 2.7e-08, which is
//! over the band, so the margin here is two nodes wide and that is why 28 is
//! not cut further.
constexpr int kStudentizedPanels = 2;
constexpr int kStudentizedNodes = 28;

//! The tail mass left outside the outer integration range at each end.
//!
//! This, and not the quadrature, is what the 9.9e-13 above is made of: cutting
//! it to 1e-13 moves the worst df = 1 case from 9.8e-13 to 9.8e-14 and the
//! whole grid to 5.2e-13, in step with the change. It is left at 1e-12 because
//! that is four orders inside the band and the chi-squared quantile that
//! supplies the range is itself measured at this level, not below it.
constexpr double kStudentizedTailMass = 1e-12;

//! Degrees of freedom at or above which the outer integral is replaced by its
//! limit W(q; k), the normal-range probability.
//!
//! The jump at the switch, measured at the last df below this cutoff over k in
//! {2,3,6,20} and q in [1, 8], is 1.877e-10. That is the number the cutoff was
//! chosen against: the target was below 1e-9. The gap falls off as C/df, which
//! the measurement bears out at 1.876e-06 for df = 1e6, 1.876e-07 at 1e7,
//! 1.876e-08 at 1e8 and 1.877e-09 at 1e9, always at k = 20 and q = 4, so a
//! cutoff of 1e9 would have left 1.9e-09, above the target.
//!
//! Replacing the integral this late asks the quadrature to stay accurate at
//! df = 1e10, which is why the density below is written in shape form. Two
//! checks that it is: at df = 1e6 and 1e8 the k = 2 value agrees with scipy's
//! own t CDF to 1.7e-15 and 2.8e-14, and the gap keeps following C/df to the
//! cutoff, which it would not if the quadrature were breaking down.
constexpr double kStudentizedNormalLimit = 1e10;

//! Degrees of freedom above which the integration range in s stops coming from
//! ChiSquareQuantile.
//!
//! Measured against scipy's chi2.ppf, ChiSquareQuantile at 1e-12 is exact to
//! 5.2e-15 relative at df = 2000 and 5.2e-14 at df = 1e4, then drifts: 6.7e-07
//! at 1e5 and 2.7e-05 at 1e6. The switch sits at 2000, well before the drift
//! matters, and the measured jump in the CDF across it is 6.2e-14 at k = 20.
constexpr double kStudentizedChiSquareLimit = 2000.0;

//! The integration range in s = sqrt(chi-squared_df / df).
//!
//! Up to kStudentizedChiSquareLimit it comes from the chi-squared quantile
//! already in this header, at kStudentizedTailMass and its complement: the
//! mass outside is below 2e-12, which is under the 1e-8 band.
//!
//! That quantile rests on GammaP, whose series and continued fraction both
//! stop after 200 terms while near the mode of a chi-squared both need about
//! sqrt(df/2) of them, so for large df it silently returns a range that is too
//! narrow. Above the limit the range therefore comes from the normal
//! approximation to s, which is where s is heading anyway: mean 1 - 1/(4*df),
//! standard deviation 1/sqrt(2*df). Eight standard deviations leave less than
//! 1e-15 of mass outside, which is less than the quantile range leaves, so the
//! switch never narrows the range.
inline void StudentizedRangeSpread(double df, double &s_lo, double &s_hi) {
	if (df <= kStudentizedChiSquareLimit) {
		s_lo = std::sqrt(ChiSquareQuantile(kStudentizedTailMass, df) / df);
		s_hi = std::sqrt(ChiSquareQuantile(1.0 - kStudentizedTailMass, df) / df);
		return;
	}
	const double sd = 1.0 / std::sqrt(2.0 * df);
	const double center = 1.0 - 1.0 / (4.0 * df);
	s_lo = std::max(0.0, center - 8.0 * sd);
	s_hi = center + 8.0 * sd;
}

//! One abscissa of the inner integral, with everything that does not depend on
//! w worked out in advance.
struct RangeGridPoint {
	double z;        //!< the abscissa
	double weight;   //!< the quadrature weight, times exp(-z*z/2), times the panel half width
	double erfc_pos; //!< erfc(z / sqrt(2))
	double erfc_neg; //!< erfc(-z / sqrt(2)), which is 2*Phi(z)
};

//! The fixed abscissae of the inner integral.
//!
//! The layout is the same panelled Gauss-Legendre layout PanelledGaussLegendre
//! walks, in the same order, so this is the same rule. What it adds is that
//! phi(z) and the two erfc values at z do not depend on w, and the outer
//! integral calls the inner one about 56 times at different w, so precomputing
//! them halves the erfc calls. erfc is about 80% of the run time of this
//! distribution, so that is worth the table.
//!
//! A function-local static, which C++11 and later initialize once and
//! thread-safely. Keyed by nothing, because the layout is fixed at compile
//! time; a std::string-keyed unordered_map with the default hash would break
//! the wasm build, which does not export the libc++ symbol it needs.
inline const std::vector<RangeGridPoint> &RangeGrid() {
	static const std::vector<RangeGridPoint> grid = []() {
		const GaussLegendreRule &rule = GaussLegendreNodes<kRangeNodes>();
		const double width = 2.0 * kRangeZLimit / static_cast<double>(kRangePanels);
		const double half = 0.5 * width;
		std::vector<RangeGridPoint> points;
		points.reserve(static_cast<std::size_t>(kRangePanels) * kRangeNodes);
		for (int panel = 0; panel < kRangePanels; panel++) {
			const double mid = -kRangeZLimit + (static_cast<double>(panel) + 0.5) * width;
			for (int i = 0; i < kRangeNodes; i++) {
				const std::size_t idx = static_cast<std::size_t>(i);
				const double z = mid + half * rule.x[idx];
				points.push_back({z, half * rule.w[idx] * std::exp(-0.5 * z * z),
				                  std::erfc(z * kInvSqrt2), std::erfc(-z * kInvSqrt2)});
			}
		}
		return points;
	}();
	return grid;
}

} // namespace detail

//! W(w; k): the probability that the range of k independent standard normals
//! is at most w. w <= 0 gives 0, w at or beyond the flat point gives exactly 1,
//! and k < 2 or any NaN argument gives NaN.
//!
//! This is the inner integral of StudentizedRangeCDF below, and on its own it
//! is that distribution's limit as df grows, which is how the test pins it
//! against NormalCDF at k = 2. It sits next to the other CDFs rather than in
//! detail for the same reason GammaPSeries does: it is a helper with a name
//! worth reading and a domain worth stating.
inline double NormalRangeCDF(double w, double k) {
	if (std::isnan(w) || std::isnan(k)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (!(k >= 2.0)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (!(w > 0.0)) {
		return 0.0;
	}
	if (w >= detail::NormalRangeFlatPoint(k)) {
		return 1.0;
	}
	// k / sqrt(2*pi) * integral over [-8, 8] of
	//   exp(-z*z/2) * [Phi(z) - Phi(z-w)]^(k-1) dz,
	// with the phi factor and the two erfc values at each z taken from the
	// precomputed grid. Panels then nodes ascending, a fixed order, so the sum
	// is reproducible.
	const double km1 = k - 1.0;
	double sum = 0.0;
	for (const detail::RangeGridPoint &pt : detail::RangeGrid()) {
		// Phi(z) - Phi(z-w), in whichever form does not cancel. The difference
		// is raised to the power k - 1, so a relative error in it comes out
		// multiplied by k - 1, and subtracting two values of 0.5*(1 + erf(.))
		// loses digits when both are near 1. With a = (z-w)/sqrt(2) at or above
		// 0 both erfc values are small and the difference of them is clean;
		// below 0 both are near 2, and the erfc values at the negated arguments
		// are the clean pair.
		const double a = (pt.z - w) * detail::kInvSqrt2;
		const double inside =
		    a >= 0.0 ? 0.5 * (std::erfc(a) - pt.erfc_pos) : 0.5 * (pt.erfc_neg - std::erfc(-a));
		sum += pt.weight * std::pow(inside, km1);
	}
	const double p = k * detail::kInvSqrt2Pi * sum;
	return p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
}

//! P(Q <= q; k, df) for the studentized range of k means on df degrees of
//! freedom. q <= 0 gives 0, k < 2 or df <= 0 or any NaN argument gives NaN.
//!
//! Accuracy, measured against scipy.stats.studentized_range: 9.9e-13 over the
//! whole committed grid, which is every integer df from 1 to 1000 and k from 2
//! to 20, and 5e-12 or better for any real df of 4 or more.
//!
//! k is a count of means, so it is not held to a band beyond the grid's 20. It
//! stays sensible to about 1e6; above that [Phi(z) - Phi(z-w)]^(k-1) underflows
//! to zero at every node and the CDF reports 0.
//!
//! A df that is not an integer and is below 4 is less accurate than that, and
//! the error grows as df falls: 6.4e-10 at df = 3.5, 1.0e-07 at 2.5, 2.0e-05 at
//! 1.5 and 1.2e-02 at 0.5, worst at k = 2 and halfway between integers. The
//! reason is the s^(df-1) factor in the density below. For an integer df that
//! factor is a polynomial and the integrand is entire, which is why the whole
//! committed grid sits at 1e-12; for any other df it has a branch point at
//! s = 0, which lands against the lower end of the integration range, and a
//! fixed-node Gauss-Legendre rule converges only algebraically against a
//! branch point at the edge of a panel. Raising the node or panel count does
//! not fix it: the rate is about n^-(2*df) there, so df = 1.5 would need about
//! 350 nodes. Fixing it needs a graded mesh or a Gauss-Jacobi rule carrying
//! s^(df-1) in the weight, which is a different quadrature.
inline double StudentizedRangeCDF(double q, double k, double df) {
	if (std::isnan(q) || std::isnan(k) || std::isnan(df)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (!(k >= 2.0) || !(df > 0.0)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (!(q > 0.0)) {
		return 0.0;
	}
	// At or above the limit the outer integral is replaced by the limit it is
	// converging to. The measured jump at the switch is with the constant.
	if (df >= detail::kStudentizedNormalLimit) {
		return NormalRangeCDF(q, k);
	}

	double s_lo = 0.0;
	double s_hi = 0.0;
	detail::StudentizedRangeSpread(df, s_lo, s_hi);

	// The density of s, up to the constant factor the ratio below cancels:
	//   log g(s) = (df - 1)*log(s/s_mode) - (df/2)*(s*s - s_mode*s_mode)
	// measured from the mode, sqrt((df-1)/df), so the exponential never
	// overflows and the two terms that grow with df cancel most of each other
	// before they are exponentiated. For df <= 1 the density has no interior
	// mode (it is largest at s = 0), and s = 1 serves as the reference point.
	//
	// Leaving the normalising constant out is what makes the large-df end
	// work: written in full the constant is (df/2)*log(df/2) - lgamma(df/2),
	// a difference of two terms of size 1e11 at df = 1e10, which loses eleven
	// digits of the O(log df) answer. The ratio below never needs it.
	const double s_mode = df > 1.0 ? std::sqrt((df - 1.0) / df) : 1.0;
	const double log_s_mode = std::log(s_mode);
	const double sq_s_mode = s_mode * s_mode;
	const auto shape = [df, log_s_mode, sq_s_mode](double s) {
		return std::exp((df - 1.0) * (std::log(s) - log_s_mode) - 0.5 * df * (s * s - sq_s_mode));
	};

	// Beyond s_break the inner CDF is exactly 1, so the integrand there is the
	// density alone. Splitting at that point does three things: it gives the
	// panels below it the whole transition of W to resolve, which at small df
	// happens over a sliver of the range (at df = 1 and q = 298 the range in s
	// is [0, 7.1] and W rises from 0 to 1 within [0.005, 0.025]); it saves the
	// inner integral above it; and it makes that piece cancel exactly between
	// the numerator and the denominator.
	const double s_break =
	    std::min(std::max(detail::NormalRangeFlatPoint(k) / q, s_lo), s_hi);

	const auto weighted = [&shape, q, k](double s) { return shape(s) * NormalRangeCDF(q * s, k); };
	const double flat = detail::PanelledGaussLegendre<detail::kStudentizedNodes>(
	    s_break, s_hi, detail::kStudentizedPanels, shape);
	const double num = detail::PanelledGaussLegendre<detail::kStudentizedNodes>(
	                       s_lo, s_break, detail::kStudentizedPanels, weighted) +
	                   flat;
	const double den = detail::PanelledGaussLegendre<detail::kStudentizedNodes>(
	                       s_lo, s_break, detail::kStudentizedPanels, shape) +
	                   flat;

	// Dividing by the quadrature of the same density over the same nodes does
	// two things. It cancels the leading quadrature error of the density, and
	// it makes a q large enough that W is 1 at every node return exactly 1.0,
	// because the numerator and the denominator are then the same sum in the
	// same order.
	//
	// den is a positive density over a range of positive width, so it is
	// positive for every df this quadrature can represent. Below about
	// df = 1e-300 the shape underflows to zero at every node, and NaN is the
	// honest answer there rather than whatever 0/0 produces.
	if (!(den > 0.0)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	const double p = num / den;
	return p < 0.0 ? 0.0 : (p > 1.0 ? 1.0 : p);
}

//! The inverse of StudentizedRangeCDF in q. p = 0 gives 0, the floor of the
//! support; p = 1 gives +inf; p outside [0, 1], k < 2, df <= 0 or any NaN
//! argument gives NaN.
inline double StudentizedRangeQuantile(double p, double k, double df) {
	if (std::isnan(p) || std::isnan(k) || std::isnan(df)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (!(k >= 2.0) || !(df > 0.0)) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (p < 0.0 || p > 1.0) {
		return std::numeric_limits<double>::quiet_NaN();
	}
	if (p == 0.0) {
		return 0.0;
	}
	if (p == 1.0) {
		return std::numeric_limits<double>::infinity();
	}

	// Bracket from [0, 8], doubling the upper end until the CDF passes p. The
	// upper end has to move a long way at small df: the 0.99 quantile at k = 20
	// and df = 1 is 298. The cap stops a pathological input from looping
	// forever; 100 doublings reach 1e31 and beyond that the quantile is
	// reported as infinite.
	//
	// A NaN from the CDF is reported as NaN. Without that, every comparison
	// below would be false, the bracket would collapse to its lower end and the
	// function would return a number near 0 for an input it cannot compute.
	const int kMaxDoublings = 100;
	double lo = 0.0;
	double hi = 8.0;
	for (int i = 0;; i++) {
		const double at_hi = StudentizedRangeCDF(hi, k, df);
		if (std::isnan(at_hi)) {
			return std::numeric_limits<double>::quiet_NaN();
		}
		if (!(at_hi < p)) {
			break;
		}
		if (i >= kMaxDoublings) {
			return std::numeric_limits<double>::infinity();
		}
		hi *= 2.0;
	}
	// A fixed step count, so the result never depends on an adaptive stopping
	// rule that could differ between platforms. 100 steps shrink the bracket
	// by 2^-100, far below the accuracy of the CDF itself, so the CDF is the
	// limiting factor and not the bisection.
	for (int i = 0; i < 100; i++) {
		const double mid = 0.5 * (lo + hi);
		if (StudentizedRangeCDF(mid, k, df) < p) {
			lo = mid;
		} else {
			hi = mid;
		}
	}
	return 0.5 * (lo + hi);
}

} // namespace stats_duck
