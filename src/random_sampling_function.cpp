#include "random_sampling_function.hpp"
#include "register_documented.hpp"

#include "distributions.hpp"

#include "duckdb/common/types/vector.hpp"
#include "duckdb/function/function_set.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <limits>
#include <random>

namespace duckdb {

namespace {

//===--------------------------------------------------------------------===//
// Thread-local RNG.
//
// Each worker thread holds its own std::mt19937_64 seeded from
// std::random_device on first use. Marking the registered functions VOLATILE
// keeps DuckDB from constant-folding them, so each row genuinely calls
// NextUniform() and the per-row independence assumption holds. A future
// iteration can add an explicit-seed override (e.g. a session-level
// stats_duck_seed() helper) — today the RNG is non-reproducible by design,
// matching the behavior of DuckDB's `random()`.
//===--------------------------------------------------------------------===//

static std::mt19937_64 &TLSRng() {
	thread_local std::mt19937_64 rng = [] {
		std::random_device rd;
		uint64_t seed = (static_cast<uint64_t>(rd()) << 32) | rd();
		return std::mt19937_64(seed);
	}();
	return rng;
}

//! Uniform sample on the open interval (0, 1). Take the top 53 bits of a single
//! mt19937_64 draw and scale by 2^-53, which gives every representable double
//! in [0, 1) equal probability mass. A u==0 nudge keeps the inverse-CDF
//! transforms below away from log(0) / qnorm(0). We deliberately do not use
//! std::uniform_real_distribution or std::generate_canonical — both have biased
//! tails on the MSVC STL.
static inline double NextUniform() {
	auto &rng = TLSRng();
	uint64_t x = rng();
	uint64_t bits = x >> 11; // 53-bit mantissa
	double u = static_cast<double>(bits) * (1.0 / static_cast<double>(1ULL << 53));
	if (u <= 0.0) {
		u = std::numeric_limits<double>::min();
	}
	return u;
}

//! Wrap a quantile-style function in the row-validity discipline used by the
//! rest of the distribution scalars (NaN / domain errors / domain-arg NULLs →
//! result NULL). Always consumes one uniform draw from the RNG, regardless of
//! whether validity is set — keeps the per-row RNG advance independent of
//! input mask state.
template <typename Fn>
static inline double SafeSampleOne(Fn &&fn, ValidityMask &mask, idx_t idx) {
	double u = NextUniform();
	try {
		double v = fn(u);
		if (std::isnan(v) || std::isinf(v)) {
			mask.SetInvalid(idx);
			return 0.0;
		}
		return v;
	} catch (...) {
		mask.SetInvalid(idx);
		return 0.0;
	}
}

//===--------------------------------------------------------------------===//
// Generic per-row executors.
//
// DuckDB's UnaryExecutor / BinaryExecutor have a fast path for constant-vector
// inputs that calls the user lambda *once per chunk* and replicates the output
// across the row range. That fast path is correct for pure functions but
// catastrophic for r* sampling — a query like `SELECT rexp(1.0) FROM range(N)`
// would produce ~N/STANDARD_VECTOR_SIZE unique values instead of N. We mark the
// functions VOLATILE so DuckDB doesn't constant-fold at the planner level, then
// also bypass UnaryExecutor / BinaryExecutor with these explicit per-row loops
// that genuinely draw one fresh uniform per output row.
//===--------------------------------------------------------------------===//

template <typename Quantile>
static inline void RunPerRow0(DataChunk &args, Vector &result, Quantile &&q) {
	idx_t count = args.size();
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		out[i] = SafeSampleOne([&](double u) { return q(u); }, mask, i);
	}
}

template <typename Quantile>
static inline void RunPerRow1(DataChunk &args, Vector &result, Quantile &&q) {
	idx_t count = args.size();
	UnifiedVectorFormat a0;
	args.data[0].ToUnifiedFormat(count, a0);
	auto v0 = UnifiedVectorFormat::GetData<double>(a0);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		auto i0 = a0.sel->get_index(i);
		if (!a0.validity.RowIsValid(i0)) {
			mask.SetInvalid(i);
			(void)NextUniform(); // keep RNG advance independent of input nulls
			out[i] = 0.0;
			continue;
		}
		double p0 = v0[i0];
		out[i] = SafeSampleOne([&](double u) { return q(u, p0); }, mask, i);
	}
}

template <typename Quantile>
static inline void RunPerRow2(DataChunk &args, Vector &result, Quantile &&q) {
	idx_t count = args.size();
	UnifiedVectorFormat a0, a1;
	args.data[0].ToUnifiedFormat(count, a0);
	args.data[1].ToUnifiedFormat(count, a1);
	auto v0 = UnifiedVectorFormat::GetData<double>(a0);
	auto v1 = UnifiedVectorFormat::GetData<double>(a1);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		auto i0 = a0.sel->get_index(i);
		auto i1 = a1.sel->get_index(i);
		if (!a0.validity.RowIsValid(i0) || !a1.validity.RowIsValid(i1)) {
			mask.SetInvalid(i);
			(void)NextUniform();
			out[i] = 0.0;
			continue;
		}
		double p0 = v0[i0];
		double p1 = v1[i1];
		out[i] = SafeSampleOne([&](double u) { return q(u, p0, p1); }, mask, i);
	}
}

template <typename Quantile>
static inline void RunPerRow3(DataChunk &args, Vector &result, Quantile &&q) {
	idx_t count = args.size();
	UnifiedVectorFormat a0, a1, a2;
	args.data[0].ToUnifiedFormat(count, a0);
	args.data[1].ToUnifiedFormat(count, a1);
	args.data[2].ToUnifiedFormat(count, a2);
	auto v0 = UnifiedVectorFormat::GetData<double>(a0);
	auto v1 = UnifiedVectorFormat::GetData<double>(a1);
	auto v2 = UnifiedVectorFormat::GetData<double>(a2);
	result.SetVectorType(VectorType::FLAT_VECTOR);
	auto out = FlatVector::GetData<double>(result);
	auto &mask = FlatVector::Validity(result);
	for (idx_t i = 0; i < count; i++) {
		auto i0 = a0.sel->get_index(i);
		auto i1 = a1.sel->get_index(i);
		auto i2 = a2.sel->get_index(i);
		if (!a0.validity.RowIsValid(i0) || !a1.validity.RowIsValid(i1) ||
		    !a2.validity.RowIsValid(i2)) {
			mask.SetInvalid(i);
			(void)NextUniform();
			out[i] = 0.0;
			continue;
		}
		double p0 = v0[i0];
		double p1 = v1[i1];
		double p2 = v2[i2];
		out[i] = SafeSampleOne([&](double u) { return q(u, p0, p1, p2); }, mask, i);
	}
}

//===--------------------------------------------------------------------===//
// rnorm: 0 / 1 / 2 args (defaults mean=0, sd=1).
//===--------------------------------------------------------------------===//

static void RNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow0(args, result, [](double u) { return stats_duck::NormalQuantile(u); });
}
static void RNorm1Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result,
	           [](double u, double mean) { return stats_duck::NormalQuantile(u, mean); });
}
static void RNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow2(args, result, [](double u, double mean, double sd) {
		return stats_duck::NormalQuantile(u, mean, sd);
	});
}

//===--------------------------------------------------------------------===//
// rt(df), rchisq(df).
//===--------------------------------------------------------------------===//

static void RTExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result,
	           [](double u, double df) { return stats_duck::StudentTQuantile(u, df); });
}
static void RChiSqExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result,
	           [](double u, double df) { return stats_duck::ChiSquareQuantile(u, df); });
}

//===--------------------------------------------------------------------===//
// rf(df1, df2).
//===--------------------------------------------------------------------===//

static void RFExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow2(args, result, [](double u, double df1, double df2) {
		return stats_duck::FQuantile(u, df1, df2);
	});
}

//===--------------------------------------------------------------------===//
// rgamma: 1 / 2 args (defaults rate=1).
//===--------------------------------------------------------------------===//

static void RGamma1Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result,
	           [](double u, double shape) { return stats_duck::GammaQuantile(u, shape); });
}
static void RGamma2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow2(args, result, [](double u, double shape, double rate) {
		return stats_duck::GammaQuantile(u, shape, rate);
	});
}

//===--------------------------------------------------------------------===//
// rbeta(alpha, beta).
//===--------------------------------------------------------------------===//

static void RBetaExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow2(args, result, [](double u, double alpha, double beta) {
		return stats_duck::BetaQuantile(u, alpha, beta);
	});
}

//===--------------------------------------------------------------------===//
// rexp: 0 / 1 args (defaults rate=1).
//===--------------------------------------------------------------------===//

static void RExpStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow0(args, result, [](double u) { return stats_duck::ExponentialQuantile(u); });
}
static void RExp1Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result,
	           [](double u, double rate) { return stats_duck::ExponentialQuantile(u, rate); });
}

//===--------------------------------------------------------------------===//
// rweibull(shape [, scale=1]).
//===--------------------------------------------------------------------===//

static void RWeibull1Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result,
	           [](double u, double shape) { return stats_duck::WeibullQuantile(u, shape); });
}
static void RWeibull2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow2(args, result, [](double u, double shape, double scale) {
		return stats_duck::WeibullQuantile(u, shape, scale);
	});
}

//===--------------------------------------------------------------------===//
// rlnorm: 0 / 1 / 2 args (defaults meanlog=0, sdlog=1).
//===--------------------------------------------------------------------===//

static void RLNormStdExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow0(args, result, [](double u) { return stats_duck::LogNormalQuantile(u); });
}
static void RLNorm1Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result, [](double u, double meanlog) {
		return stats_duck::LogNormalQuantile(u, meanlog);
	});
}
static void RLNorm2Exec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow2(args, result, [](double u, double meanlog, double sdlog) {
		return stats_duck::LogNormalQuantile(u, meanlog, sdlog);
	});
}

//===--------------------------------------------------------------------===//
// rpois(lambda). Discrete — inverse-CDF via the existing PoissonQuantile,
// which does monotone integer search seeded from the normal approximation.
//===--------------------------------------------------------------------===//

static void RPoisExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow1(args, result,
	           [](double u, double lambda) { return stats_duck::PoissonQuantile(u, lambda); });
}

//===--------------------------------------------------------------------===//
// rnbinom(size, prob) — inverse-CDF; NegBinomQuantile is monotone integer
// search seeded from the normal approximation, same shape as rpois.
//===--------------------------------------------------------------------===//

static void RNBinomExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow2(args, result, [](double u, double size, double prob) {
		return stats_duck::NegBinomQuantile(u, size, prob);
	});
}

//===--------------------------------------------------------------------===//
// rhyper(m, n, k) — inverse-CDF; HyperGeomQuantile sums the PMF up to where
// the cumulative crosses u, so the support is bounded by min(m, k).
//===--------------------------------------------------------------------===//

static void RHyperExec(DataChunk &args, ExpressionState &, Vector &result) {
	RunPerRow3(args, result, [](double u, double m, double n, double k) {
		return stats_duck::HyperGeomQuantile(u, m, n, k);
	});
}

//===--------------------------------------------------------------------===//
// Registration helpers.
//===--------------------------------------------------------------------===//

static ScalarFunction MakeVolatile(string name, vector<LogicalType> args, LogicalType ret,
                                   scalar_function_t fn) {
	ScalarFunction f(std::move(name), std::move(args), std::move(ret), fn);
	f.stability = FunctionStability::VOLATILE;
	return f;
}

} // namespace

void RegisterRandomSampling(ExtensionLoader &loader) {
	const auto DBL = LogicalType::DOUBLE;

	// ── rnorm ───────────────────────────────────────────────────────────────
	{
		ScalarFunctionSet set("rnorm");
		set.AddFunction(MakeVolatile("rnorm", {}, DBL, RNormStdExec));
		set.AddFunction(MakeVolatile("rnorm", {DBL}, DBL, RNorm1Exec));
		set.AddFunction(MakeVolatile("rnorm", {DBL, DBL}, DBL, RNorm2Exec));
		statsduck::RegisterDocumented(loader, std::move(set),
		                              "Random normal sample per row (volatile); defaults mean = 0, sd = 1.",
		                              {"mean", "sd"}, "SELECT rnorm() FROM range(5)");
	}

	// ── rt / rchisq ─────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, MakeVolatile("rt", {DBL}, DBL, RTExec),
	                              "Random Student's t sample per row (volatile).", {"df"},
	                              "SELECT rt(10.0) FROM range(5)");
	statsduck::RegisterDocumented(loader, MakeVolatile("rchisq", {DBL}, DBL, RChiSqExec),
	                              "Random chi-square sample per row (volatile).", {"df"},
	                              "SELECT rchisq(2.0) FROM range(5)");

	// ── rf ──────────────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, MakeVolatile("rf", {DBL, DBL}, DBL, RFExec),
	                              "Random F-distribution sample per row (volatile).", {"df1", "df2"},
	                              "SELECT rf(3.0, 12.0) FROM range(5)");

	// ── rgamma ──────────────────────────────────────────────────────────────
	{
		ScalarFunctionSet set("rgamma");
		set.AddFunction(MakeVolatile("rgamma", {DBL}, DBL, RGamma1Exec));
		set.AddFunction(MakeVolatile("rgamma", {DBL, DBL}, DBL, RGamma2Exec));
		statsduck::RegisterDocumented(loader, std::move(set),
		                              "Random gamma sample per row (volatile); rate defaults to 1.",
		                              {"shape", "rate"}, "SELECT rgamma(2.0) FROM range(5)");
	}

	// ── rbeta ───────────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, MakeVolatile("rbeta", {DBL, DBL}, DBL, RBetaExec),
	                              "Random beta sample per row (volatile).", {"alpha", "beta"},
	                              "SELECT rbeta(2.0, 3.0) FROM range(5)");

	// ── rexp ────────────────────────────────────────────────────────────────
	{
		ScalarFunctionSet set("rexp");
		set.AddFunction(MakeVolatile("rexp", {}, DBL, RExpStdExec));
		set.AddFunction(MakeVolatile("rexp", {DBL}, DBL, RExp1Exec));
		statsduck::RegisterDocumented(loader, std::move(set),
		                              "Random exponential sample per row (volatile); rate defaults to 1.",
		                              {"rate"}, "SELECT rexp() FROM range(5)");
	}

	// ── rweibull ────────────────────────────────────────────────────────────
	{
		ScalarFunctionSet set("rweibull");
		set.AddFunction(MakeVolatile("rweibull", {DBL}, DBL, RWeibull1Exec));
		set.AddFunction(MakeVolatile("rweibull", {DBL, DBL}, DBL, RWeibull2Exec));
		statsduck::RegisterDocumented(loader, std::move(set),
		                              "Random Weibull sample per row (volatile); scale defaults to 1.",
		                              {"shape", "scale"}, "SELECT rweibull(1.5) FROM range(5)");
	}

	// ── rlnorm ──────────────────────────────────────────────────────────────
	{
		ScalarFunctionSet set("rlnorm");
		set.AddFunction(MakeVolatile("rlnorm", {}, DBL, RLNormStdExec));
		set.AddFunction(MakeVolatile("rlnorm", {DBL}, DBL, RLNorm1Exec));
		set.AddFunction(MakeVolatile("rlnorm", {DBL, DBL}, DBL, RLNorm2Exec));
		statsduck::RegisterDocumented(loader, std::move(set),
		                              "Random log-normal sample per row (volatile); defaults meanlog = 0, "
		                              "sdlog = 1.",
		                              {"meanlog", "sdlog"}, "SELECT rlnorm() FROM range(5)");
	}

	// ── rpois ───────────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, MakeVolatile("rpois", {DBL}, DBL, RPoisExec),
	                              "Random Poisson sample per row (volatile).", {"lambda"},
	                              "SELECT rpois(2.5) FROM range(5)");

	// ── rnbinom ─────────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, MakeVolatile("rnbinom", {DBL, DBL}, DBL, RNBinomExec),
	                              "Random negative-binomial sample per row (volatile).", {"size", "prob"},
	                              "SELECT rnbinom(5.0, 0.5) FROM range(5)");

	// ── rhyper ──────────────────────────────────────────────────────────────
	statsduck::RegisterDocumented(loader, MakeVolatile("rhyper", {DBL, DBL, DBL}, DBL, RHyperExec),
	                              "Random hypergeometric sample per row (volatile).", {"m", "n", "k"},
	                              "SELECT rhyper(5.0, 10.0, 4.0) FROM range(5)");
}

} // namespace duckdb
