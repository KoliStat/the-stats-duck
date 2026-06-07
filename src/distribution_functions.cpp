#include "distribution_functions.hpp"
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
		loader.RegisterFunction(dnorm);
	}
	{
		ScalarFunctionSet pnorm("pnorm");
		pnorm.AddFunction(ScalarFunction({DBL}, DBL, PNormStdExec));
		pnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, PNorm2Exec));
		pnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PNorm3Exec));
		loader.RegisterFunction(pnorm);
	}
	{
		ScalarFunctionSet qnorm("qnorm");
		qnorm.AddFunction(ScalarFunction({DBL}, DBL, QNormStdExec));
		qnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, QNorm2Exec));
		qnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QNorm3Exec));
		loader.RegisterFunction(qnorm);
	}

	// ── Student's t ─────────────────────────────────────────────────────────
	loader.RegisterFunction(ScalarFunction("dt", {DBL, DBL}, DBL, DTExec));
	loader.RegisterFunction(ScalarFunction("pt", {DBL, DBL}, DBL, PTExec));
	loader.RegisterFunction(ScalarFunction("qt", {DBL, DBL}, DBL, QTExec));

	// ── Chi-square ──────────────────────────────────────────────────────────
	loader.RegisterFunction(ScalarFunction("dchisq", {DBL, DBL}, DBL, DChiSqExec));
	loader.RegisterFunction(ScalarFunction("pchisq", {DBL, DBL}, DBL, PChiSqExec));
	loader.RegisterFunction(ScalarFunction("qchisq", {DBL, DBL}, DBL, QChiSqExec));

	// ── F ───────────────────────────────────────────────────────────────────
	loader.RegisterFunction(ScalarFunction("df", {DBL, DBL, DBL}, DBL, DFExec));
	loader.RegisterFunction(ScalarFunction("pf", {DBL, DBL, DBL}, DBL, PFExec));
	loader.RegisterFunction(ScalarFunction("qf", {DBL, DBL, DBL}, DBL, QFExec));

	// ── Gamma ───────────────────────────────────────────────────────────────
	// dgamma(x, shape) defaults rate = 1; dgamma(x, shape, rate) for the
	// explicit form. Matches R's `dgamma(x, shape, rate = 1)`.
	{
		ScalarFunctionSet dgamma("dgamma");
		dgamma.AddFunction(ScalarFunction({DBL, DBL}, DBL, DGamma2Exec));
		dgamma.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, DGamma3Exec));
		loader.RegisterFunction(dgamma);
	}
	{
		ScalarFunctionSet pgamma("pgamma");
		pgamma.AddFunction(ScalarFunction({DBL, DBL}, DBL, PGamma2Exec));
		pgamma.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PGamma3Exec));
		loader.RegisterFunction(pgamma);
	}
	{
		ScalarFunctionSet qgamma("qgamma");
		qgamma.AddFunction(ScalarFunction({DBL, DBL}, DBL, QGamma2Exec));
		qgamma.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QGamma3Exec));
		loader.RegisterFunction(qgamma);
	}

	// ── Beta ────────────────────────────────────────────────────────────────
	loader.RegisterFunction(ScalarFunction("dbeta", {DBL, DBL, DBL}, DBL, DBetaExec));
	loader.RegisterFunction(ScalarFunction("pbeta", {DBL, DBL, DBL}, DBL, PBetaExec));
	loader.RegisterFunction(ScalarFunction("qbeta", {DBL, DBL, DBL}, DBL, QBetaExec));

	// ── Exponential ─────────────────────────────────────────────────────────
	// One-arg form defaults rate = 1 (matches R's `dexp(x, rate = 1)`).
	{
		ScalarFunctionSet dexp("dexp");
		dexp.AddFunction(ScalarFunction({DBL}, DBL, DExpStdExec));
		dexp.AddFunction(ScalarFunction({DBL, DBL}, DBL, DExp2Exec));
		loader.RegisterFunction(dexp);
	}
	{
		ScalarFunctionSet pexp("pexp");
		pexp.AddFunction(ScalarFunction({DBL}, DBL, PExpStdExec));
		pexp.AddFunction(ScalarFunction({DBL, DBL}, DBL, PExp2Exec));
		loader.RegisterFunction(pexp);
	}
	{
		ScalarFunctionSet qexp("qexp");
		qexp.AddFunction(ScalarFunction({DBL}, DBL, QExpStdExec));
		qexp.AddFunction(ScalarFunction({DBL, DBL}, DBL, QExp2Exec));
		loader.RegisterFunction(qexp);
	}

	// ── Weibull ─────────────────────────────────────────────────────────────
	// dweibull(x, shape) defaults scale = 1 (matches R's `dweibull(x, shape, scale = 1)`).
	{
		ScalarFunctionSet dweibull("dweibull");
		dweibull.AddFunction(ScalarFunction({DBL, DBL}, DBL, DWeibull2Exec));
		dweibull.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, DWeibull3Exec));
		loader.RegisterFunction(dweibull);
	}
	{
		ScalarFunctionSet pweibull("pweibull");
		pweibull.AddFunction(ScalarFunction({DBL, DBL}, DBL, PWeibull2Exec));
		pweibull.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PWeibull3Exec));
		loader.RegisterFunction(pweibull);
	}
	{
		ScalarFunctionSet qweibull("qweibull");
		qweibull.AddFunction(ScalarFunction({DBL, DBL}, DBL, QWeibull2Exec));
		qweibull.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QWeibull3Exec));
		loader.RegisterFunction(qweibull);
	}

	// ── Log-normal ──────────────────────────────────────────────────────────
	// dlnorm() / dlnorm(x, meanlog) / dlnorm(x, meanlog, sdlog). Defaults
	// meanlog = 0, sdlog = 1 — matches R's `dlnorm(x, meanlog = 0, sdlog = 1)`.
	{
		ScalarFunctionSet dlnorm("dlnorm");
		dlnorm.AddFunction(ScalarFunction({DBL}, DBL, DLNormStdExec));
		dlnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, DLNorm2Exec));
		dlnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, DLNorm3Exec));
		loader.RegisterFunction(dlnorm);
	}
	{
		ScalarFunctionSet plnorm("plnorm");
		plnorm.AddFunction(ScalarFunction({DBL}, DBL, PLNormStdExec));
		plnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, PLNorm2Exec));
		plnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, PLNorm3Exec));
		loader.RegisterFunction(plnorm);
	}
	{
		ScalarFunctionSet qlnorm("qlnorm");
		qlnorm.AddFunction(ScalarFunction({DBL}, DBL, QLNormStdExec));
		qlnorm.AddFunction(ScalarFunction({DBL, DBL}, DBL, QLNorm2Exec));
		qlnorm.AddFunction(ScalarFunction({DBL, DBL, DBL}, DBL, QLNorm3Exec));
		loader.RegisterFunction(qlnorm);
	}

	// ── Poisson ─────────────────────────────────────────────────────────────
	// Discrete: dpois(k, lambda) / ppois(q, lambda) / qpois(p, lambda).
	loader.RegisterFunction(ScalarFunction("dpois", {DBL, DBL}, DBL, DPoisExec));
	loader.RegisterFunction(ScalarFunction("ppois", {DBL, DBL}, DBL, PPoisExec));
	loader.RegisterFunction(ScalarFunction("qpois", {DBL, DBL}, DBL, QPoisExec));

	// ── Negative Binomial ───────────────────────────────────────────────────
	// Discrete: dnbinom(k, size, prob) / pnbinom(q, size, prob) / qnbinom(p, size, prob).
	loader.RegisterFunction(ScalarFunction("dnbinom", {DBL, DBL, DBL}, DBL, DNBinomExec));
	loader.RegisterFunction(ScalarFunction("pnbinom", {DBL, DBL, DBL}, DBL, PNBinomExec));
	loader.RegisterFunction(ScalarFunction("qnbinom", {DBL, DBL, DBL}, DBL, QNBinomExec));

	// ── Hypergeometric ──────────────────────────────────────────────────────
	// Discrete: dhyper(x, m, n, k) / phyper(q, m, n, k) / qhyper(p, m, n, k).
	loader.RegisterFunction(ScalarFunction("dhyper", {DBL, DBL, DBL, DBL}, DBL, DHyperExec));
	loader.RegisterFunction(ScalarFunction("phyper", {DBL, DBL, DBL, DBL}, DBL, PHyperExec));
	loader.RegisterFunction(ScalarFunction("qhyper", {DBL, DBL, DBL, DBL}, DBL, QHyperExec));
}

} // namespace duckdb
