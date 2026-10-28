#pragma once

// stats_duck multiple-testing p-value adjustment — the methods behind the
// adjust_p SQL function, moved here so the emmeans kernel (#21) can share
// them without pulling in DuckDB.
//
// Methods follow R's p.adjust naming and formulas. For sorted ascending
// p_(1) <= ... <= p_(n) the canonical step-down / step-up adjustments are:
//
//   bonferroni:  q_(i) = min(1, n * p_(i))                     per-row, trivial
//   holm:        q_(i) = min(1, max_{j<=i} (n-j+1) * p_(j))    step-down cumulative max
//   hochberg:    q_(i) = min(1, min_{j>=i} (n-j+1) * p_(j))    step-up cumulative min
//   BH (fdr):    q_(i) = min(1, min_{j>=i} (n/j) * p_(j))      step-up cumulative min
//   BY:          q_(i) = min(1, min_{j>=i} (n*H_n/j) * p_(j))  BH with harmonic factor
//   none:        q_(i) = p_(i)
//
// An invalid entry (NULL at the SQL layer) is passed through to the output at
// the same position and excluded from n and from the rank computation.
//
// DuckDB-free and Eigen-free, like linalg.hpp: only the standard library, so
// a DuckDB-free kernel (emmeans) can include it directly.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace statsduck {

enum class AdjustMethod {
	BONFERRONI,
	HOLM,
	HOCHBERG,
	BENJAMINI_HOCHBERG,
	BENJAMINI_YEKUTIELI,
	NONE,
};

// Parse a method name. Case-sensitive, matching adjust_p's existing SQL
// surface. Accepts exactly "bonferroni", "holm", "hochberg", "BH", "fdr"
// (alias for BH), "BY" and "none". Returns false, leaving `out` untouched,
// on anything else — including "tukey", which needs the joint distribution
// of the estimates rather than a plain list of p-values, so it cannot be
// computed here.
inline bool parse_adjust(const std::string &name, AdjustMethod &out) {
	if (name == "bonferroni") {
		out = AdjustMethod::BONFERRONI;
	} else if (name == "holm") {
		out = AdjustMethod::HOLM;
	} else if (name == "hochberg") {
		out = AdjustMethod::HOCHBERG;
	} else if (name == "BH" || name == "fdr") {
		out = AdjustMethod::BENJAMINI_HOCHBERG;
	} else if (name == "BY") {
		out = AdjustMethod::BENJAMINI_YEKUTIELI;
	} else if (name == "none") {
		out = AdjustMethod::NONE;
	} else {
		return false;
	}
	return true;
}

// Canonical name for a method, e.g. echoed back by emmeans. "fdr" maps to
// the same canonical form as "BH", since they are the same method.
inline const char *adjust_name(AdjustMethod m) {
	switch (m) {
	case AdjustMethod::BONFERRONI:
		return "bonferroni";
	case AdjustMethod::HOLM:
		return "holm";
	case AdjustMethod::HOCHBERG:
		return "hochberg";
	case AdjustMethod::BENJAMINI_HOCHBERG:
		return "BH";
	case AdjustMethod::BENJAMINI_YEKUTIELI:
		return "BY";
	case AdjustMethod::NONE:
		return "none";
	}
	return "none";
}

// Adjust p_in (raw p-values) into p_out (resized to match). `valid` marks
// which entries are real; an invalid entry comes back NaN in p_out and does
// not count toward n or toward any entry's rank.
inline void adjust_p(const std::vector<double> &p_in, const std::vector<bool> &valid,
                      AdjustMethod method, std::vector<double> &p_out) {
	std::size_t total = p_in.size();
	double nan = std::numeric_limits<double>::quiet_NaN();
	p_out.assign(total, nan);

	// Collect valid indices.
	std::vector<std::size_t> idx_valid;
	idx_valid.reserve(total);
	for (std::size_t i = 0; i < total; i++) {
		if (valid[i]) {
			idx_valid.push_back(i);
		}
	}
	std::size_t n = idx_valid.size();
	if (n == 0) {
		return;
	}

	if (method == AdjustMethod::NONE) {
		for (std::size_t k : idx_valid) {
			p_out[k] = p_in[k];
		}
		return;
	}

	if (method == AdjustMethod::BONFERRONI) {
		double n_d = static_cast<double>(n);
		for (std::size_t k : idx_valid) {
			double q = n_d * p_in[k];
			p_out[k] = q > 1.0 ? 1.0 : q;
		}
		return;
	}

	// Step-down / step-up methods: sort the valid indices by raw p value.
	std::vector<std::size_t> order = idx_valid;
	std::sort(order.begin(), order.end(),
	          [&](std::size_t a, std::size_t b) { return p_in[a] < p_in[b]; });

	double n_d = static_cast<double>(n);

	switch (method) {
	case AdjustMethod::HOLM: {
		// Cumulative max from the smallest p upward; multiplier (n - i + 1) for
		// 1-based rank i. In 0-based form: multiplier (n - i) for i in [0, n).
		double cummax = 0.0;
		for (std::size_t i = 0; i < n; i++) {
			double q = (n_d - static_cast<double>(i)) * p_in[order[i]];
			if (q > cummax) {
				cummax = q;
			}
			p_out[order[i]] = cummax > 1.0 ? 1.0 : cummax;
		}
		break;
	}
	case AdjustMethod::HOCHBERG: {
		// Cumulative min from the largest p downward.
		double cummin = std::numeric_limits<double>::infinity();
		for (std::size_t i = n; i-- > 0;) {
			double q = (n_d - static_cast<double>(i)) * p_in[order[i]];
			if (q < cummin) {
				cummin = q;
			}
			p_out[order[i]] = cummin > 1.0 ? 1.0 : cummin;
		}
		break;
	}
	case AdjustMethod::BENJAMINI_HOCHBERG: {
		// Step-up FDR: q_(i) = min over j >= i of (n / rank_j) * p_(j).
		double cummin = std::numeric_limits<double>::infinity();
		for (std::size_t i = n; i-- > 0;) {
			double rank = static_cast<double>(i + 1); // 1-based
			double q = (n_d / rank) * p_in[order[i]];
			if (q < cummin) {
				cummin = q;
			}
			p_out[order[i]] = cummin > 1.0 ? 1.0 : cummin;
		}
		break;
	}
	case AdjustMethod::BENJAMINI_YEKUTIELI: {
		// Like BH but with the harmonic-number penalty for arbitrary dependence.
		double harmonic = 0.0;
		for (std::size_t k = 1; k <= n; k++) {
			harmonic += 1.0 / static_cast<double>(k);
		}
		double cummin = std::numeric_limits<double>::infinity();
		for (std::size_t i = n; i-- > 0;) {
			double rank = static_cast<double>(i + 1);
			double q = (n_d * harmonic / rank) * p_in[order[i]];
			if (q < cummin) {
				cummin = q;
			}
			p_out[order[i]] = cummin > 1.0 ? 1.0 : cummin;
		}
		break;
	}
	default:
		break; // unreachable: covered above
	}
}

} // namespace statsduck
