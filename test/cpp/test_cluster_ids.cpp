// Direct C++ unit tests for statsduck::densify_clusters — converting
// VARCHAR cluster keys into dense zero-based ids by sorting rather than hashing.
//
// Builds standalone (no DuckDB) via scripts/run-cpp-tests.sh. Exit 0 = pass.

#include "cluster_ids.hpp"

#include <cstdio>
#include <limits>
#include <vector>

static int g_checks = 0;
static int g_fail = 0;

static void check(bool cond, const char *msg) {
	++g_checks;
	if (!cond) {
		++g_fail;
		std::printf("  FAIL: %s\n", msg);
	}
}

// Safe accessor for an int vector. A failing call can leave the vector
// empty, and indexing an empty vector directly is undefined behaviour: it
// can crash the binary before any later check gets to report, which turns a
// clear assertion failure into a confusing one. Out of range reads back as
// -1, which no real cluster id equals (ids are 0-based and positive).
static int at(const std::vector<int> &v, std::size_t i) {
	return i < v.size() ? v[i] : -1;
}

static void test_densify_clusters() {
	std::printf("densify_clusters\n");

	// Ids are assigned in ascending key order, not first-seen order, so the
	// partition is reproducible whatever order the rows arrived in.
	const std::vector<std::string> keys = {"c", "a", "b", "a", "c"};
	const std::vector<int> ids = statsduck::densify_clusters(keys);
	check(ids.size() == 5, "one id per key");
	check(at(ids, 1) == 0 && at(ids, 3) == 0, "both 'a' rows share id 0");
	check(at(ids, 2) == 1, "'b' is id 1");
	check(at(ids, 0) == 2 && at(ids, 4) == 2, "both 'c' rows share id 2");

	// The same partition from a different row order gives the same id per key.
	const std::vector<std::string> shuffled = {"a", "c", "b", "c", "a"};
	const std::vector<int> ids2 = statsduck::densify_clusters(shuffled);
	check(at(ids2, 0) == 0 && at(ids2, 2) == 1 && at(ids2, 1) == 2,
	      "ids follow the key, not the position");

	// Degenerate shapes.
	check(statsduck::densify_clusters({}).empty(), "no keys, no ids");
	const std::vector<int> one = statsduck::densify_clusters({"only"});
	check(one.size() == 1 && at(one, 0) == 0, "a single key is id 0");
	const std::vector<int> same = statsduck::densify_clusters({"x", "x", "x"});
	check(same.size() == 3 && at(same, 0) == 0 && at(same, 2) == 0,
	      "all-equal keys are one cluster");

	// The empty string is a key like any other, and sorts first.
	const std::vector<int> emptykey = statsduck::densify_clusters({"b", "", "b"});
	check(at(emptykey, 1) == 0 && at(emptykey, 0) == 1, "the empty string is a real key");

	// The highest id plus one is the cluster count, which is what fit_lm needs.
	int max_id = -1;
	for (const int v : ids) { if (v > max_id) max_id = v; }
	check(max_id + 1 == 3, "three distinct keys give ids 0..2");
}

int main() {
	std::printf("== test_cluster_ids ==\n");
	test_densify_clusters();

	std::printf("\n%d checks, %d failures\n", g_checks, g_fail);
	if (g_fail == 0) {
		std::printf("ALL CLUSTER_IDS TESTS PASSED\n");
		return 0;
	}
	return 1;
}
