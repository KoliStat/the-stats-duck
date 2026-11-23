#pragma once

#include <algorithm>
#include <vector>

namespace stats_duck {

// The entries of a map in ascending key order, as pointers into the map.
//
// An unordered map iterates in an order set by its bucket layout and by the
// order of insertion. The insertion order follows the order the rows arrive
// in, which changes with DuckDB's parallel execution, and the bucket layout
// differs between standard libraries. A floating-point sum taken in that order
// can then differ in its last bits between runs and between platforms. A sum
// over these entries runs in one fixed order instead (#54).
//
// Sorting uses operator< on the keys, so it needs no hash. The keys of a map
// are distinct, so the order is unique.
template <class Map>
std::vector<const typename Map::value_type *> SortedEntries(const Map &map) {
	using Entry = typename Map::value_type;
	std::vector<const Entry *> entries;
	entries.reserve(map.size());
	for (const auto &entry : map) {
		entries.push_back(&entry);
	}
	std::sort(entries.begin(), entries.end(), [](const Entry *a, const Entry *b) { return a->first < b->first; });
	return entries;
}

} // namespace stats_duck
