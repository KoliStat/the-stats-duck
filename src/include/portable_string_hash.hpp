#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace stats_duck {

// FNV-1a hash for std::string keys.
//
// Why this exists: the default std::hash<std::string> lowers to libc++'s
// std::__hash_memory(const void*, size_t). When stats_duck is built as a WASM
// loadable extension, libc++ lives in the @duckdb/duckdb-wasm MAIN module and
// the side module imports the libc++ symbols it needs from it — but duckdb keys
// its own maps on duckdb::string_t (with its own hash), so it never instantiates
// or exports __hash_memory. The import is therefore left unresolved and the
// Emscripten loader stubs it with a throwing placeholder. Any
// std::unordered_map/set<std::string,...> that hashes a key then traps as
// "TypeError: t is not a function" at runtime in the browser (it crashed
// table_one via anova_oneway / chisq_independence). Hashing the bytes ourselves
// keeps the call inside the extension and removes the only unresolved import.
//
// See notes/engineering/2026-06-duckdb-version-must-match-duckdb-wasm.md.
struct PortableStringHash {
	std::size_t operator()(const std::string &s) const noexcept {
		std::uint64_t h = 1469598103934665603ull; // FNV-1a 64-bit offset basis
		for (unsigned char c : s) {
			h ^= c;
			h *= 1099511628211ull; // FNV-1a 64-bit prime
		}
		return static_cast<std::size_t>(h);
	}
};

} // namespace stats_duck
