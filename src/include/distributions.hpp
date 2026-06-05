#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>

// Centralized PDF / CDF / quantile implementations for the distributions used
// throughout stats_duck. Header-only so callers pay no link cost and the
// compiler can inline across the aggregate hot path.
//
// Numerical references:
//   - Numerical Recipes in C, 3rd ed., ch. 6 (gamma/beta functions and their
//     incomplete companions).
//   - P. J. Acklam, "An algorithm for computing the inverse normal
//     cumulative distribution function" (Acklam's rational approximation).

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

} // namespace stats_duck
