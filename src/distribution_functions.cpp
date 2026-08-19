#include "distribution_functions.hpp"
#include "register_documented.hpp"
#include "distributions.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/common/vector_operations/unary_executor.hpp"
#include "duckdb/common/vector_operations/binary_executor.hpp"
#include "duckdb/common/vector_operations/ternary_executor.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

// Scalar wrappers around stats_duck::NormalPDF / CDF / Quantile etc.
// Each distribution gets three SQL functions: d{name}, p{name}, q{name}
// following R's convention.
//
// Invalid parameters (e.g., negative sd, df <= 0, p outside [0,1]) are
// surfaced as NULL rather than raised errors so that a single bad input row
// doesn't abort an entire query.

namespace {

// Helper: run a stats_duck call safely. On exception, mark the output slot as
// NULL via the ValidityMask and return a zero sentinel. On NaN, also NULL.
template <typename CALL>
inline double SafeCall(CALL &&f, ValidityMask &mask, idx_t idx) {
	try {
		double r = f();
		if (std::isnan(r)) {
			mask.SetInvalid(idx);
			return 0.0;
		}
		return r;
	} catch (...) {
		mask.SetInvalid(idx);
		return 0.0;
	}
}

// ── Normal ──────────────────────────────────────────────────────────────────

static void DNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double x, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalPDF(x); }, mask, idx);
	    });
}

static void DNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double mean, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalPDF(x, mean); }, mask, idx);
	    });
}

static void DNorm3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double mean, double sd, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalPDF(x, mean, sd); }, mask, idx);
	    });
}

static void PNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double x, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalCDF(x); }, mask, idx);
	    });
}

static void PNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double mean, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalCDF(x, mean); }, mask, idx);
	    });
}

static void PNorm3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double mean, double sd, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalCDF(x, mean, sd); }, mask, idx);
	    });
}

static void QNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double p, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalQuantile(p); }, mask, idx);
	    });
}

static void QNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double mean, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalQuantile(p, mean); }, mask, idx);
	    });
}

static void QNorm3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double p, double mean, double sd, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NormalQuantile(p, mean, sd); }, mask, idx);
	    });
}

// ── Student's t ─────────────────────────────────────────────────────────────

static void DTExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double df, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::StudentTPDF(x, df); }, mask, idx);
	    });
}

static void PTExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double df, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::StudentTCDF(x, df); }, mask, idx);
	    });
}

static void QTExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double df, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::StudentTQuantile(p, df); }, mask, idx);
	    });
}

// ── Chi-square ──────────────────────────────────────────────────────────────

static void DChiSqExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double df, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ChiSquarePDF(x, df); }, mask, idx);
	    });
}

static void PChiSqExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double df, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ChiSquareCDF(x, df); }, mask, idx);
	    });
}

static void QChiSqExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double df, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ChiSquareQuantile(p, df); }, mask, idx);
	    });
}

// ── F ───────────────────────────────────────────────────────────────────────

static void DFExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double df1, double df2, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::FPDF(x, df1, df2); }, mask, idx);
	    });
}

static void PFExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double df1, double df2, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::FCDF(x, df1, df2); }, mask, idx);
	    });
}

static void QFExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double p, double df1, double df2, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::FQuantile(p, df1, df2); }, mask, idx);
	    });
}

// ── Gamma ───────────────────────────────────────────────────────────────────

static void DGamma2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double shape, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::GammaPDF(x, shape); }, mask, idx);
	    });
}

static void DGamma3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double shape, double rate, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::GammaPDF(x, shape, rate); }, mask, idx);
	    });
}

static void PGamma2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double shape, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::GammaCDF(x, shape); }, mask, idx);
	    });
}

static void PGamma3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double shape, double rate, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::GammaCDF(x, shape, rate); }, mask, idx);
	    });
}

static void QGamma2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double shape, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::GammaQuantile(p, shape); }, mask, idx);
	    });
}

static void QGamma3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double p, double shape, double rate, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::GammaQuantile(p, shape, rate); }, mask, idx);
	    });
}

// ── Beta ────────────────────────────────────────────────────────────────────

static void DBetaExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double alpha, double beta, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::BetaPDF(x, alpha, beta); }, mask, idx);
	    });
}

static void PBetaExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double alpha, double beta, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::BetaCDF(x, alpha, beta); }, mask, idx);
	    });
}

static void QBetaExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double p, double alpha, double beta, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::BetaQuantile(p, alpha, beta); }, mask, idx);
	    });
}

// ── Exponential ─────────────────────────────────────────────────────────────

static void DExpStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double x, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ExponentialPDF(x); }, mask, idx);
	    });
}

static void DExp2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double rate, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ExponentialPDF(x, rate); }, mask, idx);
	    });
}

static void PExpStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double x, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ExponentialCDF(x); }, mask, idx);
	    });
}

static void PExp2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double rate, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ExponentialCDF(x, rate); }, mask, idx);
	    });
}

static void QExpStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double p, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ExponentialQuantile(p); }, mask, idx);
	    });
}

static void QExp2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double rate, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::ExponentialQuantile(p, rate); }, mask, idx);
	    });
}

// ── Weibull ─────────────────────────────────────────────────────────────────
// dweibull(x, shape [, scale]); scale defaults to 1.

static void DWeibull2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double shape, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::WeibullPDF(x, shape); }, mask, idx);
	    });
}

static void DWeibull3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double shape, double scale, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::WeibullPDF(x, shape, scale); }, mask, idx);
	    });
}

static void PWeibull2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double shape, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::WeibullCDF(x, shape); }, mask, idx);
	    });
}

static void PWeibull3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double shape, double scale, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::WeibullCDF(x, shape, scale); }, mask, idx);
	    });
}

static void QWeibull2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double shape, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::WeibullQuantile(p, shape); }, mask, idx);
	    });
}

static void QWeibull3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double p, double shape, double scale, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::WeibullQuantile(p, shape, scale); }, mask, idx);
	    });
}

// ── Log-normal ──────────────────────────────────────────────────────────────
// dlnorm(x [, meanlog [, sdlog]]); both default to 0 and 1 respectively.

static void DLNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double x, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalPDF(x); }, mask, idx);
	    });
}

static void DLNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double meanlog, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalPDF(x, meanlog); }, mask, idx);
	    });
}

static void DLNorm3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double meanlog, double sdlog, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalPDF(x, meanlog, sdlog); }, mask, idx);
	    });
}

static void PLNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double x, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalCDF(x); }, mask, idx);
	    });
}

static void PLNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double x, double meanlog, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalCDF(x, meanlog); }, mask, idx);
	    });
}

static void PLNorm3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double x, double meanlog, double sdlog, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalCDF(x, meanlog, sdlog); }, mask, idx);
	    });
}

static void QLNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	UnaryExecutor::ExecuteWithNulls<double, double>(
	    args.data[0], result, args.size(),
	    [](double p, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalQuantile(p); }, mask, idx);
	    });
}

static void QLNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double meanlog, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalQuantile(p, meanlog); }, mask, idx);
	    });
}

static void QLNorm3Exec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double p, double meanlog, double sdlog, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::LogNormalQuantile(p, meanlog, sdlog); }, mask, idx);
	    });
}

// ── Poisson ─────────────────────────────────────────────────────────────────
// Discrete; lambda > 0. R: dpois(x, lambda) / ppois(q, lambda) / qpois(p, lambda).

static void DPoisExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double k, double lambda, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::PoissonPMF(k, lambda); }, mask, idx);
	    });
}

static void PPoisExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double q, double lambda, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::PoissonCDF(q, lambda); }, mask, idx);
	    });
}

static void QPoisExec(DataChunk &args, ExpressionState &, Vector &result) {
	BinaryExecutor::ExecuteWithNulls<double, double, double>(
	    args.data[0], args.data[1], result, args.size(),
	    [](double p, double lambda, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::PoissonQuantile(p, lambda); }, mask, idx);
	    });
}

// ── Negative Binomial ───────────────────────────────────────────────────────
// dnbinom(x, size, prob) / pnbinom(q, size, prob) / qnbinom(p, size, prob).

static void DNBinomExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double k, double size, double prob, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NegBinomPMF(k, size, prob); }, mask, idx);
	    });
}

static void PNBinomExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double k, double size, double prob, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NegBinomCDF(k, size, prob); }, mask, idx);
	    });
}

static void QNBinomExec(DataChunk &args, ExpressionState &, Vector &result) {
	TernaryExecutor::ExecuteWithNulls<double, double, double, double>(
	    args.data[0], args.data[1], args.data[2], result, args.size(),
	    [](double p, double size, double prob, ValidityMask &mask, idx_t idx) {
		    return SafeCall([&] { return stats_duck::NegBinomQuantile(p, size, prob); }, mask, idx);
	    });
}

// ── Hypergeometric ──────────────────────────────────────────────────────────
// dhyper(x, m, n, k) / phyper(q, m, n, k) / qhyper(p, m, n, k) — four
// arguments. DuckDB doesn't ship a 4-ary helper, so we drive the loop directly
// via UnifiedVectorFormat (which still handles constant / flat / dictionary
// input encodings correctly).

template <typename Fn>
static void RunQuaternaryWithNulls(DataChunk &args, Vector &result, Fn &&fn) {
	idx_t count = args.size();
	UnifiedVectorFormat a0, a1, a2, a3;
	args.data[0].ToUnifiedFormat(count, a0);
	args.data[1].ToUnifiedFormat(count, a1);
	args.data[2].ToUnifiedFormat(count, a2);
	args.data[3].ToUnifiedFormat(count, a3);
	auto v0 = UnifiedVectorFormat::GetData<double>(a0);
	auto v1 = UnifiedVectorFormat::GetData<double>(a1);
	auto v2 = UnifiedVectorFormat::GetData<double>(a2);
	auto v3 = UnifiedVectorFormat::GetData<double>(a3);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		auto i0 = a0.sel->get_index(i);
		auto i1 = a1.sel->get_index(i);
		auto i2 = a2.sel->get_index(i);
		auto i3 = a3.sel->get_index(i);
		if (!a0.validity.RowIsValid(i0) || !a1.validity.RowIsValid(i1) ||
		    !a2.validity.RowIsValid(i2) || !a3.validity.RowIsValid(i3)) {
			mask.SetInvalid(i);
			out[i] = 0.0;
			continue;
		}
		out[i] = SafeCall([&] { return fn(v0[i0], v1[i1], v2[i2], v3[i3]); }, mask, i);
	}
}

static void DHyperExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunQuaternaryWithNulls(args, result,
	                        [](double x, double m, double n, double k) {
		                        return stats_duck::HyperGeomPMF(x, m, n, k);
	                        });
}

static void PHyperExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunQuaternaryWithNulls(args, result,
	                        [](double q, double m, double n, double k) {
		                        return stats_duck::HyperGeomCDF(q, m, n, k);
	                        });
}

static void QHyperExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunQuaternaryWithNulls(args, result,
	                        [](double p, double m, double n, double k) {
		                        return stats_duck::HyperGeomQuantile(p, m, n, k);
	                        });
}

} // namespace

void RegisterDistributionFunctions(ExtensionLoader &loader) {
	const auto DBL = LogicalType::DOUBLE;

	// ── Normal ──────────────────────────────────────────────────────────────
	{
		ScalarFunctionSet dnorm("dnorm");
		dnorm.AddFunction(ScalarFunction({DBL}, DBL, DNormStdExec));
		dnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, DNorm2Exec));
		dnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, DNorm3Exec));
		statsduck::RegisterDocumented(loader, std::move(dnorm),
		                              "Normal distribution PDF; defaults mean = 0, sd = 1.",
		                              {"x", "mean", "sd"}, "SELECT dnorm(0.0)");
	}
	{
		ScalarFunctionSet pnorm("pnorm");
		pnorm.AddFunction(ScalarFunction({DBL}, DBL, PNormStdExec));
		pnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, PNorm2Exec));
		pnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PNorm3Exec));
		statsduck::RegisterDocumented(loader, std::move(pnorm),
		                              "Normal distribution CDF; defaults mean = 0, sd = 1.",
		                              {"q", "mean", "sd"}, "SELECT pnorm(1.96)");
	}
	{
		ScalarFunctionSet qnorm("qnorm");
		qnorm.AddFunction(ScalarFunction({DBL}, DBL, QNormStdExec));
		qnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, QNorm2Exec));
		qnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QNorm3Exec));
		statsduck::RegisterDocumented(loader, std::move(qnorm),
		                              "Normal distribution quantile; defaults mean = 0, sd = 1.",
		                              {"p", "mean", "sd"}, "SELECT qnorm(0.975)");
	}

	// ── Student's t ─────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, ScalarFunction("dt", {DBL, DBL}, DBL, DTExec),
	                              "Student's t distribution PDF.", {"x", "df"}, "SELECT dt(0.5, 10)");
	statsduck::RegisterDocumented(loader, ScalarFunction("pt", {DBL, DBL}, DBL, PTExec),
	                              "Student's t distribution CDF.", {"q", "df"}, "SELECT pt(2.04, 30)");
	statsduck::RegisterDocumented(loader, ScalarFunction("qt", {DBL, DBL}, DBL, QTExec),
	                              "Student's t distribution quantile.", {"p", "df"}, "SELECT qt(0.975, 30)");

	// ── Chi-square ──────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, ScalarFunction("dchisq", {DBL, DBL}, DBL, DChiSqExec),
	                              "Chi-square distribution PDF.", {"x", "df"}, "SELECT dchisq(3.0, 2)");
	statsduck::RegisterDocumented(loader, ScalarFunction("pchisq", {DBL, DBL}, DBL, PChiSqExec),
	                              "Chi-square distribution CDF.", {"q", "df"}, "SELECT pchisq(3.84, 1)");
	statsduck::RegisterDocumented(loader, ScalarFunction("qchisq", {DBL, DBL}, DBL, QChiSqExec),
	                              "Chi-square distribution quantile.", {"p", "df"}, "SELECT qchisq(0.95, 1)");

	// ── F ───────────────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, ScalarFunction("df", {DBL, DBL, DBL}, DBL, DFExec),
	                              "F distribution PDF.", {"x", "df1", "df2"}, "SELECT df(1.0, 3, 12)");
	statsduck::RegisterDocumented(loader, ScalarFunction("pf", {DBL, DBL, DBL}, DBL, PFExec),
	                              "F distribution CDF.", {"q", "df1", "df2"}, "SELECT pf(4.47, 3, 12)");
	statsduck::RegisterDocumented(loader, ScalarFunction("qf", {DBL, DBL, DBL}, DBL, QFExec),
	                              "F distribution quantile.", {"p", "df1", "df2"}, "SELECT qf(0.95, 3, 12)");

	// ── Gamma ───────────────────────────────────────────────────────────────
	// dgamma(x, shape) defaults rate = 1; dgamma(x, shape, rate) for the
	// explicit form. Matches R's `dgamma(x, shape, rate = 1)`.
	{
		ScalarFunctionSet dgamma("dgamma");
		dgamma.AddFunction(ScalarFunction({DBL, DBL}, DBL, DGamma2Exec));
		dgamma.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, DGamma3Exec));
		statsduck::RegisterDocumented(loader, std::move(dgamma),
		                              "Gamma distribution PDF; rate defaults to 1.",
		                              {"x", "shape", "rate"}, "SELECT dgamma(2.0, 3.0)");
	}
	{
		ScalarFunctionSet pgamma("pgamma");
		pgamma.AddFunction(ScalarFunction({DBL, DBL}, DBL, PGamma2Exec));
		pgamma.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PGamma3Exec));
		statsduck::RegisterDocumented(loader, std::move(pgamma),
		                              "Gamma distribution CDF; rate defaults to 1.",
		                              {"q", "shape", "rate"}, "SELECT pgamma(2.0, 3.0)");
	}
	{
		ScalarFunctionSet qgamma("qgamma");
		qgamma.AddFunction(ScalarFunction({DBL, DBL}, DBL, QGamma2Exec));
		qgamma.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QGamma3Exec));
		statsduck::RegisterDocumented(loader, std::move(qgamma),
		                              "Gamma distribution quantile; rate defaults to 1.",
		                              {"p", "shape", "rate"}, "SELECT qgamma(0.95, 3.0)");
	}

	// ── Beta ────────────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, ScalarFunction("dbeta", {DBL, DBL, DBL}, DBL, DBetaExec),
	                              "Beta distribution PDF on [0, 1].", {"x", "alpha", "beta"},
	                              "SELECT dbeta(0.5, 2.0, 3.0)");
	statsduck::RegisterDocumented(loader, ScalarFunction("pbeta", {DBL, DBL, DBL}, DBL, PBetaExec),
	                              "Beta distribution CDF.", {"q", "alpha", "beta"},
	                              "SELECT pbeta(0.5, 2.0, 3.0)");
	statsduck::RegisterDocumented(loader, ScalarFunction("qbeta", {DBL, DBL, DBL}, DBL, QBetaExec),
	                              "Beta distribution quantile.", {"p", "alpha", "beta"},
	                              "SELECT qbeta(0.95, 2.0, 3.0)");

	// ── Exponential ─────────────────────────────────────────────────────────
	// One-arg form defaults rate = 1 (matches R's `dexp(x, rate = 1)`).
	{
		ScalarFunctionSet dexp("dexp");
		dexp.AddFunction(ScalarFunction({DBL}, DBL, DExpStdExec));
		dexp.AddFunction(ScalarFunction({DBL, DBL}, DBL, DExp2Exec));
		statsduck::RegisterDocumented(loader, std::move(dexp),
		                              "Exponential distribution PDF; rate defaults to 1.",
		                              {"x", "rate"}, "SELECT dexp(1.0)");
	}
	{
		ScalarFunctionSet pexp("pexp");
		pexp.AddFunction(ScalarFunction({DBL}, DBL, PExpStdExec));
		pexp.AddFunction(ScalarFunction({DBL, DBL}, DBL, PExp2Exec));
		statsduck::RegisterDocumented(loader, std::move(pexp),
		                              "Exponential distribution CDF; rate defaults to 1.",
		                              {"q", "rate"}, "SELECT pexp(1.0)");
	}
	{
		ScalarFunctionSet qexp("qexp");
		qexp.AddFunction(ScalarFunction({DBL}, DBL, QExpStdExec));
		qexp.AddFunction(ScalarFunction({DBL, DBL}, DBL, QExp2Exec));
		statsduck::RegisterDocumented(loader, std::move(qexp),
		                              "Exponential distribution quantile (closed form); rate defaults to 1.",
		                              {"p", "rate"}, "SELECT qexp(0.95)");
	}

	// ── Weibull ─────────────────────────────────────────────────────────────
	// dweibull(x, shape) defaults scale = 1 (matches R's `dweibull(x, shape, scale = 1)`).
	{
		ScalarFunctionSet dweibull("dweibull");
		dweibull.AddFunction(ScalarFunction({DBL, DBL}, DBL, DWeibull2Exec));
		dweibull.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, DWeibull3Exec));
		statsduck::RegisterDocumented(loader, std::move(dweibull),
		                              "Weibull distribution PDF; scale defaults to 1.",
		                              {"x", "shape", "scale"}, "SELECT dweibull(1.0, 2.0)");
	}
	{
		ScalarFunctionSet pweibull("pweibull");
		pweibull.AddFunction(ScalarFunction({DBL, DBL}, DBL, PWeibull2Exec));
		pweibull.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PWeibull3Exec));
		statsduck::RegisterDocumented(loader, std::move(pweibull),
		                              "Weibull distribution CDF; scale defaults to 1.",
		                              {"q", "shape", "scale"}, "SELECT pweibull(1.0, 2.0)");
	}
	{
		ScalarFunctionSet qweibull("qweibull");
		qweibull.AddFunction(ScalarFunction({DBL, DBL}, DBL, QWeibull2Exec));
		qweibull.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QWeibull3Exec));
		statsduck::RegisterDocumented(loader, std::move(qweibull),
		                              "Weibull distribution quantile (closed form); scale defaults to 1.",
		                              {"p", "shape", "scale"}, "SELECT qweibull(0.95, 2.0)");
	}

	// ── Log-normal ──────────────────────────────────────────────────────────
	// dlnorm() / dlnorm(x, meanlog) / dlnorm(x, meanlog, sdlog). Defaults
	// meanlog = 0, sdlog = 1 — matches R's `dlnorm(x, meanlog = 0, sdlog = 1)`.
	{
		ScalarFunctionSet dlnorm("dlnorm");
		dlnorm.AddFunction(ScalarFunction({DBL}, DBL, DLNormStdExec));
		dlnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, DLNorm2Exec));
		dlnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, DLNorm3Exec));
		statsduck::RegisterDocumented(loader, std::move(dlnorm),
		                              "Log-normal distribution PDF; defaults meanlog = 0, sdlog = 1.",
		                              {"x", "meanlog", "sdlog"}, "SELECT dlnorm(1.0)");
	}
	{
		ScalarFunctionSet plnorm("plnorm");
		plnorm.AddFunction(ScalarFunction({DBL}, DBL, PLNormStdExec));
		plnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, PLNorm2Exec));
		plnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PLNorm3Exec));
		statsduck::RegisterDocumented(loader, std::move(plnorm),
		                              "Log-normal distribution CDF; defaults meanlog = 0, sdlog = 1.",
		                              {"q", "meanlog", "sdlog"}, "SELECT plnorm(1.0)");
	}
	{
		ScalarFunctionSet qlnorm("qlnorm");
		qlnorm.AddFunction(ScalarFunction({DBL}, DBL, QLNormStdExec));
		qlnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, QLNorm2Exec));
		qlnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QLNorm3Exec));
		statsduck::RegisterDocumented(loader, std::move(qlnorm),
		                              "Log-normal distribution quantile; defaults meanlog = 0, sdlog = 1.",
		                              {"p", "meanlog", "sdlog"}, "SELECT qlnorm(0.95)");
	}

	// ── Poisson ─────────────────────────────────────────────────────────────
	// Discrete: dpois(k, lambda) / ppois(q, lambda) / qpois(p, lambda).
	statsduck::RegisterDocumented(loader, ScalarFunction("dpois", {DBL, DBL}, DBL, DPoisExec),
	                              "Poisson PMF (discrete).", {"k", "lambda"}, "SELECT dpois(3, 2.5)");
	statsduck::RegisterDocumented(loader, ScalarFunction("ppois", {DBL, DBL}, DBL, PPoisExec),
	                              "Poisson CDF.", {"q", "lambda"}, "SELECT ppois(3, 2.5)");
	statsduck::RegisterDocumented(loader, ScalarFunction("qpois", {DBL, DBL}, DBL, QPoisExec),
	                              "Poisson quantile (integer search).", {"p", "lambda"},
	                              "SELECT qpois(0.95, 2.5)");

	// ── Negative Binomial ───────────────────────────────────────────────────
	// Discrete: dnbinom(k, size, prob) / pnbinom(q, size, prob) / qnbinom(p, size, prob).
	statsduck::RegisterDocumented(loader, ScalarFunction("dnbinom", {DBL, DBL, DBL}, DBL, DNBinomExec),
	                              "Negative-binomial PMF: failures before size successes.",
	                              {"k", "size", "prob"}, "SELECT dnbinom(2, 5, 0.5)");
	statsduck::RegisterDocumented(loader, ScalarFunction("pnbinom", {DBL, DBL, DBL}, DBL, PNBinomExec),
	                              "Negative-binomial CDF.", {"q", "size", "prob"},
	                              "SELECT pnbinom(2, 5, 0.5)");
	statsduck::RegisterDocumented(loader, ScalarFunction("qnbinom", {DBL, DBL, DBL}, DBL, QNBinomExec),
	                              "Negative-binomial quantile (integer search).", {"p", "size", "prob"},
	                              "SELECT qnbinom(0.95, 5, 0.5)");

	// ── Hypergeometric ──────────────────────────────────────────────────────
	// Discrete: dhyper(x, m, n, k) / phyper(q, m, n, k) / qhyper(p, m, n, k).
	statsduck::RegisterDocumented(loader, ScalarFunction("dhyper", {DBL, DBL, DBL, DBL}, DBL, DHyperExec),
	                              "Hypergeometric PMF (m successes, n failures, k draws).",
	                              {"x", "m", "n", "k"}, "SELECT dhyper(1, 5, 10, 4)");
	statsduck::RegisterDocumented(loader, ScalarFunction("phyper", {DBL, DBL, DBL, DBL}, DBL, PHyperExec),
	                              "Hypergeometric CDF.", {"q", "m", "n", "k"}, "SELECT phyper(1, 5, 10, 4)");
	statsduck::RegisterDocumented(loader, ScalarFunction("qhyper", {DBL, DBL, DBL, DBL}, DBL, QHyperExec),
	                              "Hypergeometric quantile.", {"p", "m", "n", "k"},
	                              "SELECT qhyper(0.95, 5, 10, 4)");
}

} // namespace duckdb
