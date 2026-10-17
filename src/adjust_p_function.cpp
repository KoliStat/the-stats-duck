#include "adjust_p_function.hpp"
#include "register_documented.hpp"

#include "p_adjust.hpp" // statsduck::adjust_p, parse_adjust, adjust_name

#include "duckdb/function/scalar_function.hpp"
#include "duckdb/common/types/vector.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace duckdb {

// Multiple-testing corrections: adjusted p-values from a vector of raw
// p-values. The methods, their formulas, and the NULL-handling convention
// live in p_adjust.hpp now. This file is just the SQL binding. It unwraps
// the arguments, raises on an unknown method name, and writes the result
// list.

namespace {

static void AdjustPExec(DataChunk &args, ExpressionState &, Vector &result) {
	idx_t row_count = args.size();
	auto &list_input = args.data[0];
	auto &method_input = args.data[1];

	UnifiedVectorFormat list_fmt, method_fmt;
	list_input.ToUnifiedFormat(row_count, list_fmt);
	method_input.ToUnifiedFormat(row_count, method_fmt);
	auto list_entries = UnifiedVectorFormat::GetData<list_entry_t>(list_fmt);
	auto method_data = UnifiedVectorFormat::GetData<string_t>(method_fmt);

	// Child vector of the input list (the flat array of doubles).
	auto &input_child = ListVector::GetEntry(list_input);
	UnifiedVectorFormat child_fmt;
	input_child.ToUnifiedFormat(ListVector::GetListSize(list_input), child_fmt);
	auto child_data = UnifiedVectorFormat::GetData<double>(child_fmt);

	// Compute total output size = total input size (NULLs preserved as NULLs).
	idx_t total_output = 0;
	for (idx_t i = 0; i < row_count; i++) {
		auto idx = list_fmt.sel->get_index(i);
		if (list_fmt.validity.RowIsValid(idx)) {
			total_output += list_entries[idx].length;
		}
	}
	ListVector::Reserve(result, total_output);
	ListVector::SetListSize(result, total_output);

	auto result_list_entries = FlatVector::GetData<list_entry_t>(result);
	auto &result_child = ListVector::GetEntry(result);
	auto result_child_data = FlatVector::GetData<double>(result_child);
	auto &result_child_validity = FlatVector::Validity(result_child);

	std::vector<double> p_in_buf;
	std::vector<bool> valid_buf;
	std::vector<double> p_out_buf;
	idx_t out_offset = 0;

	for (idx_t i = 0; i < row_count; i++) {
		auto list_idx = list_fmt.sel->get_index(i);
		auto method_idx = method_fmt.sel->get_index(i);

		if (!list_fmt.validity.RowIsValid(list_idx) ||
		    !method_fmt.validity.RowIsValid(method_idx)) {
			FlatVector::SetNull(result, i, true);
			result_list_entries[i] = list_entry_t(out_offset, 0);
			continue;
		}

		auto entry = list_entries[list_idx];
		auto method_str = method_data[method_idx];
		std::string method_s(method_str.GetData(), method_str.GetSize());
		statsduck::AdjustMethod method;
		if (!statsduck::parse_adjust(method_s, method)) {
			throw InvalidInputException(
			    "adjust_p: unknown method '%s' — supported: 'bonferroni', 'holm', "
			    "'hochberg', 'BH' (or 'fdr'), 'BY', 'none'",
			    method_s);
		}

		// Extract input values + validity.
		p_in_buf.assign(entry.length, 0.0);
		valid_buf.assign(entry.length, false);
		for (idx_t j = 0; j < entry.length; j++) {
			auto child_idx = child_fmt.sel->get_index(entry.offset + j);
			if (child_fmt.validity.RowIsValid(child_idx)) {
				double v = child_data[child_idx];
				if (!std::isnan(v)) {
					p_in_buf[j] = v;
					valid_buf[j] = true;
				}
			}
		}

		statsduck::adjust_p(p_in_buf, valid_buf, method, p_out_buf);

		// Emit results.
		result_list_entries[i] = list_entry_t(out_offset, entry.length);
		for (idx_t j = 0; j < entry.length; j++) {
			if (valid_buf[j]) {
				result_child_data[out_offset + j] = p_out_buf[j];
			} else {
				result_child_validity.SetInvalid(out_offset + j);
			}
		}
		out_offset += entry.length;
	}
}

} // namespace

void RegisterAdjustP(ExtensionLoader &loader) {
	ScalarFunction fn("adjust_p",
	                  {LogicalType::LIST(LogicalType::DOUBLE), LogicalType::VARCHAR},
	                  LogicalType::LIST(LogicalType::DOUBLE), AdjustPExec);
	statsduck::RegisterDocumented(loader, std::move(fn),
	                              "Multiple-testing correction over a list of p-values "
	                              "(bonferroni, holm, hochberg, or BH/fdr).",
	                              {"pvals", "method"}, "SELECT adjust_p([0.01, 0.04, 0.20], 'holm')");
}

} // namespace duckdb
