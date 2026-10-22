#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace statsduck {

// Dense 0-based cluster ids from arbitrary string keys, in sorted-key order.
// Deliberately sort-based, NOT a hash map: std::sort uses operator<, so it
// avoids the libc++ std::__hash_memory symbol that a default-hashed unordered
// map keyed on std::string pulls into the wasm build (the bug that broke
// table_one in wasm; see check-wasm-string-hash.sh). The label values are
// irrelevant — only the partition they induce matters.
inline std::vector<int> densify_clusters(const std::vector<std::string> &keys) {
	const std::size_t n = keys.size();
	std::vector<int> order(n);
	for (std::size_t i = 0; i < n; i++) {
		order[i] = static_cast<int>(i);
	}
	std::sort(order.begin(), order.end(), [&](int a, int b) { return keys[a] < keys[b]; });
	std::vector<int> ids(n, 0);
	int cur = 0;
	for (std::size_t r = 0; r < n; r++) {
		if (r > 0 && keys[order[r]] != keys[order[r - 1]]) {
			cur++;
		}
		ids[order[r]] = cur;
	}
	return ids;
}

}  // namespace statsduck
