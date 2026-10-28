duckdb_extension_load(stats_duck
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)

# DuckDB 2.0 links an extension into its targets (the unittest binary that
# runs test/sql, and the shell) only if the extension config asks for it;
# otherwise it is built loadable and left out, and every `require stats_duck`
# in the suite is skipped rather than run. 1.x linked it by default, and has
# no such function, hence the guard.
if(COMMAND duckdb_extension_statically_link)
    duckdb_extension_statically_link(stats_duck)
endif()
