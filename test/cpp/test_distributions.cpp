// Direct C++ unit tests for the NaN rule of the distribution functions in
// distributions.hpp (#102): a NaN argument gives a NaN result, and no
// exception.
//
// Each function is called with NaN in each argument in turn, and with valid
// values in the others. The valid values alone must give a finite result, so
// a function that returned NaN for every input could not pass. The Weibull
// CDF takes x equal to the scale, because pow(1, NaN) is 1 under IEEE rules.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "distributions.hpp"

#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <string>
#include <vector>

static int g_checks = 0;
static int g_fail = 0;

using Args = std::vector<double>;

static std::string describe(double r) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "%.17g", r);
	return buf;
}

// Calls f with the valid arguments, then with NaN in each position in turn.
template <class F>
static void check_nan_args(const char *name, F f, const Args &valid) {
	++g_checks;
	try {
		double r = f(valid);
		if (!std::isfinite(r)) {
			++g_fail;
			std::printf("  FAIL: %s with valid arguments returned %s\n", name, describe(r).c_str());
		}
	} catch (const std::exception &e) {
		++g_fail;
		std::printf("  FAIL: %s with valid arguments threw: %s\n", name, e.what());
	}

	for (std::size_t pos = 0; pos < valid.size(); pos++) {
		Args args = valid;
		args[pos] = std::numeric_limits<double>::quiet_NaN();
		++g_checks;
		try {
			double r = f(args);
			if (!std::isnan(r)) {
				++g_fail;
				std::printf("  FAIL: %s with NaN in argument %zu returned %s\n", name, pos + 1, describe(r).c_str());
			}
		} catch (const std::exception &e) {
			++g_fail;
			std::printf("  FAIL: %s with NaN in argument %zu threw: %s\n", name, pos + 1, e.what());
		}
	}
}

static void test_continuous() {
	std::printf("continuous distributions\n");
	using namespace stats_duck;
	check_nan_args("NormalPDF", [](const Args &a) { return NormalPDF(a[0], a[1], a[2]); }, {0.3, 0.0, 1.0});
	check_nan_args("NormalCDF", [](const Args &a) { return NormalCDF(a[0], a[1], a[2]); }, {0.3, 0.0, 1.0});
	check_nan_args("NormalUpperTail", [](const Args &a) { return NormalUpperTail(a[0], a[1], a[2]); },
	               {0.3, 0.0, 1.0});
	check_nan_args("NormalQuantile", [](const Args &a) { return NormalQuantile(a[0], a[1], a[2]); },
	               {0.5, 0.0, 1.0});

	check_nan_args("StudentTPDF", [](const Args &a) { return StudentTPDF(a[0], a[1]); }, {1.0, 5.0});
	check_nan_args("StudentTCDF", [](const Args &a) { return StudentTCDF(a[0], a[1]); }, {1.0, 5.0});
	check_nan_args("StudentTUpperTail", [](const Args &a) { return StudentTUpperTail(a[0], a[1]); }, {1.0, 5.0});
	check_nan_args("StudentTQuantile", [](const Args &a) { return StudentTQuantile(a[0], a[1]); }, {0.5, 5.0});

	check_nan_args("ChiSquarePDF", [](const Args &a) { return ChiSquarePDF(a[0], a[1]); }, {2.0, 3.0});
	check_nan_args("ChiSquareCDF", [](const Args &a) { return ChiSquareCDF(a[0], a[1]); }, {2.0, 3.0});
	check_nan_args("ChiSquareUpperTail", [](const Args &a) { return ChiSquareUpperTail(a[0], a[1]); }, {2.0, 3.0});
	check_nan_args("ChiSquareQuantile", [](const Args &a) { return ChiSquareQuantile(a[0], a[1]); }, {0.5, 3.0});

	check_nan_args("FPDF", [](const Args &a) { return FPDF(a[0], a[1], a[2]); }, {1.5, 2.0, 3.0});
	check_nan_args("FCDF", [](const Args &a) { return FCDF(a[0], a[1], a[2]); }, {1.5, 2.0, 3.0});
	check_nan_args("FUpperTail", [](const Args &a) { return FUpperTail(a[0], a[1], a[2]); }, {1.5, 2.0, 3.0});
	check_nan_args("FQuantile", [](const Args &a) { return FQuantile(a[0], a[1], a[2]); }, {0.5, 2.0, 3.0});

	check_nan_args("GammaPDF", [](const Args &a) { return GammaPDF(a[0], a[1], a[2]); }, {1.0, 2.0, 1.5});
	check_nan_args("GammaCDF", [](const Args &a) { return GammaCDF(a[0], a[1], a[2]); }, {1.0, 2.0, 1.5});
	check_nan_args("GammaQuantile", [](const Args &a) { return GammaQuantile(a[0], a[1], a[2]); }, {0.5, 2.0, 1.5});

	check_nan_args("BetaPDF", [](const Args &a) { return BetaPDF(a[0], a[1], a[2]); }, {0.3, 2.0, 3.0});
	check_nan_args("BetaCDF", [](const Args &a) { return BetaCDF(a[0], a[1], a[2]); }, {0.3, 2.0, 3.0});
	check_nan_args("BetaQuantile", [](const Args &a) { return BetaQuantile(a[0], a[1], a[2]); }, {0.5, 2.0, 3.0});

	check_nan_args("ExponentialPDF", [](const Args &a) { return ExponentialPDF(a[0], a[1]); }, {1.0, 2.0});
	check_nan_args("ExponentialCDF", [](const Args &a) { return ExponentialCDF(a[0], a[1]); }, {1.0, 2.0});
	check_nan_args("ExponentialQuantile", [](const Args &a) { return ExponentialQuantile(a[0], a[1]); }, {0.5, 2.0});

	check_nan_args("WeibullPDF", [](const Args &a) { return WeibullPDF(a[0], a[1], a[2]); }, {1.5, 2.0, 1.5});
	check_nan_args("WeibullCDF", [](const Args &a) { return WeibullCDF(a[0], a[1], a[2]); }, {1.5, 2.0, 1.5});
	check_nan_args("WeibullQuantile", [](const Args &a) { return WeibullQuantile(a[0], a[1], a[2]); },
	               {0.5, 2.0, 1.5});

	check_nan_args("LogNormalPDF", [](const Args &a) { return LogNormalPDF(a[0], a[1], a[2]); }, {1.0, 0.0, 1.0});
	check_nan_args("LogNormalCDF", [](const Args &a) { return LogNormalCDF(a[0], a[1], a[2]); }, {1.0, 0.0, 1.0});
	check_nan_args("LogNormalQuantile", [](const Args &a) { return LogNormalQuantile(a[0], a[1], a[2]); },
	               {0.5, 0.0, 1.0});

	check_nan_args("StudentizedRangeCDF", [](const Args &a) { return StudentizedRangeCDF(a[0], a[1], a[2]); },
	               {3.0, 3.0, 10.0});
	check_nan_args("StudentizedRangeQuantile",
	               [](const Args &a) { return StudentizedRangeQuantile(a[0], a[1], a[2]); }, {0.5, 3.0, 10.0});
}

static void test_discrete() {
	std::printf("discrete distributions\n");
	using namespace stats_duck;
	check_nan_args("PoissonPMF", [](const Args &a) { return PoissonPMF(a[0], a[1]); }, {2.0, 3.0});
	check_nan_args("PoissonCDF", [](const Args &a) { return PoissonCDF(a[0], a[1]); }, {2.0, 3.0});
	check_nan_args("PoissonQuantile", [](const Args &a) { return PoissonQuantile(a[0], a[1]); }, {0.5, 3.0});

	check_nan_args("NegBinomPMF", [](const Args &a) { return NegBinomPMF(a[0], a[1], a[2]); }, {2.0, 3.0, 0.5});
	check_nan_args("NegBinomCDF", [](const Args &a) { return NegBinomCDF(a[0], a[1], a[2]); }, {2.0, 3.0, 0.5});
	check_nan_args("NegBinomQuantile", [](const Args &a) { return NegBinomQuantile(a[0], a[1], a[2]); },
	               {0.5, 3.0, 0.5});

	check_nan_args("HyperGeomPMF", [](const Args &a) { return HyperGeomPMF(a[0], a[1], a[2], a[3]); },
	               {1.0, 5.0, 5.0, 3.0});
	check_nan_args("HyperGeomCDF", [](const Args &a) { return HyperGeomCDF(a[0], a[1], a[2], a[3]); },
	               {1.0, 5.0, 5.0, 3.0});
	check_nan_args("HyperGeomQuantile", [](const Args &a) { return HyperGeomQuantile(a[0], a[1], a[2], a[3]); },
	               {0.5, 5.0, 5.0, 3.0});
}

int main() {
	std::printf("== test_distributions ==\n");
	test_continuous();
	test_discrete();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL DISTRIBUTION TESTS PASSED\n");
		return 0;
	}
	return 1;
}
