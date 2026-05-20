#include "distribution_functions.hpp"
#include "distributions.hpp"

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
}

} // namespace duckdb
