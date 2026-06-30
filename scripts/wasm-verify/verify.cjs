// Load the built stats_duck wasm_eh extension into @duckdb/duckdb-wasm@1.32.0
// (async worker EH bundle — the path bedevere-wise uses) and run a probe matrix,
// table_one included. The extension is an Emscripten side module sharing the wasm
// instance with duckdb-eh.wasm, so the extension<->duckdb boundary is exercised
// exactly as in the browser.
//
//   Prereq:  build the extension (`make wasm_eh`), then `npm install` in this dir.
//   Run:     node verify.cjs   ->   per-call PASS/FAIL matrix + "PROBE_DONE".
//
// A "t/r is not a function" FAIL means the extension imports a libc++ symbol that
// duckdb-wasm's main module doesn't export (e.g. the std::__hash_memory regression
// that crashed anova/chisq — see src/include/portable_string_hash.hpp and
// notes/engineering/2026-06-duckdb-version-must-match-duckdb-wasm.md). The
// extension is served over a throwaway localhost http repo; patch-worker.mjs
// sanitizes a Windows-only colon in duckdb-wasm's extension cache path.

const duckdb = require('@duckdb/duckdb-wasm');
const WWorker = require('web-worker');
const path = require('path');
const fs = require('fs');
const http = require('http');
const { pathToFileURL } = require('url');

const HERE = __dirname;
const ROOT = path.resolve(HERE, '..', '..');
const DIST = path.join(HERE, 'node_modules', '@duckdb', 'duckdb-wasm', 'dist');
const REPO = path.join(ROOT, 'build', 'wasm_eh', 'repository');

const rows = (t) => { const o = []; for (const r of t) o.push(r.toJSON()); return o; };
const withTimeout = (p, ms, label) =>
  Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`TIMEOUT ${label} ${ms}ms`)), ms))]);
async function q(conn, sql, label, ms = 30000) {
  try { return await withTimeout(conn.query(sql), ms, label); }
  catch (e) { console.error(`\n[query-fail:${label}]\nSQL: ${sql}\nERROR: ${e && e.stack ? e.stack : e}`); throw e; }
}

(async () => {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    fs.readFile(path.join(REPO, url), (err, data) => {
      console.log(`[http] ${req.method} ${url} -> ${err ? '404 ' + err.code : data.length + 'B'}`);
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'application/wasm', 'Content-Length': data.length });
      res.end(req.method === 'HEAD' ? undefined : data);
    });
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;
  console.log(`[http] serving ${REPO} on http://127.0.0.1:${port}`);

  const worker = new WWorker(pathToFileURL(path.join(HERE, 'patch-worker.mjs')).href, { type: 'module' });
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(path.join(DIST, 'duckdb-eh.wasm'));
  await db.open({ path: ':memory:', allowUnsignedExtensions: true, query: { castBigIntToDouble: true } });
  const conn = await db.connect();

  const ver = rows(await q(conn, `SELECT library_version, source_id FROM pragma_version()`, 'ver'))[0];
  const plat = rows(await q(conn, `PRAGMA platform`, 'plat'))[0];
  console.log(`[duckdb-wasm] library_version=${ver.library_version} source_id=${ver.source_id} platform=${plat.platform}`);

  await q(conn, `SET custom_extension_repository='http://127.0.0.1:${port}'`, 'set-repo');
  await q(conn, `INSTALL stats_duck`, 'install');
  await q(conn, `LOAD stats_duck`, 'load');
  console.log('[duckdb] stats_duck INSTALL + LOAD ok');

  await q(conn, `CREATE TABLE penguins (
      body_mass_g DOUBLE, flipper_length_mm INTEGER, bill_length_mm DOUBLE,
      sex VARCHAR, species VARCHAR)`, 'create');
  await q(conn, `INSERT INTO penguins VALUES
      (3750,181,39.1,'male','Adelie'),   (3800,186,39.5,'female','Adelie'),
      (3250,195,40.3,'female','Adelie'), (3450,193,36.7,'female','Adelie'),
      (4500,210,46.1,'male','Gentoo'),   (5000,217,50.0,'male','Gentoo'),
      (4750,216,49.9,'female','Gentoo'), (4400,214,48.4,'female','Gentoo'),
      (3700,193,46.5,'male','Chinstrap'),(3550,191,50.5,'female','Chinstrap'),
      (3900,198,51.3,'male','Chinstrap'),(3650,190,49.2,'female','Chinstrap')`, 'insert');

  // Localize the ABI break: probe increasing complexity, each independently.
  const probes = [
    ['baseline-builtin', `SELECT count(*) AS n, sum(body_mass_g) AS s FROM penguins`],
    ['scalar-ext', `SELECT pnorm(1.96) AS p`],
    ['struct-agg-direct', `SELECT summary_stats(body_mass_g)::VARCHAR AS s FROM penguins`],
    ['anova-direct', `SELECT anova_oneway(body_mass_g, species)::VARCHAR AS a FROM penguins`],
    ['corr_matrix-tablefn', `SELECT count(*) AS n FROM corr_matrix('penguins', variables := ['body_mass_g','flipper_length_mm'])`],
    ['visualize', `VISUALIZE body_mass_g AS x, flipper_length_mm AS y FROM penguins DRAW point`],
    ['table_one-internalconn', `SELECT count(*) AS n FROM table_one('penguins', variables := ['body_mass_g','sex'], by := ['species'])`],
    // lm_fit: aggregate returning LIST<STRUCT> on the header-only lm_core kernel.
    // 'HC1' exercises the base path; the 'CR1' probes exercise the cluster path
    // (DensifyClusters + the cluster sandwich + the n_clusters field). The cluster
    // grouping is sort-based precisely so it imports no std::__hash_memory — these
    // probes are the runtime proof of that (species = 3 clusters of 4 rows).
    ['lm_fit-hc1', `SELECT (lm_fit(body_mass_g, [bill_length_mm], 'HC1')).k AS k FROM penguins`],
    ['lm_fit-cluster-cr1', `SELECT (lm_fit(body_mass_g, [bill_length_mm], 'CR1', species)).n_clusters AS g FROM penguins`],
    ['lm_fit-cluster-unnest', `SELECT count(*) AS n FROM (SELECT unnest((lm_fit(body_mass_g, [bill_length_mm], 'CR1', species)).coefficients) FROM penguins)`],
  ];
  const results = [];
  for (const [name, sql] of probes) {
    try {
      const r = rows(await withTimeout(conn.query(sql), 20000, name));
      console.log(`  [PASS] ${name} -> ${JSON.stringify(r[0]).slice(0, 80)}`);
      results.push([name, 'PASS']);
    } catch (e) {
      const msg = (e && e.message ? e.message : String(e)).split('\n')[0];
      console.log(`  [FAIL] ${name} -> ${msg}`);
      results.push([name, 'FAIL: ' + msg]);
    }
  }
  console.log('\n[summary] ' + results.map((r) => `${r[0]}=${r[1].startsWith('PASS') ? 'PASS' : 'FAIL'}`).join('  '));

  // Exact task repro query — show actual rows with p_value / effect_size.
  console.log('\n[exact repro] table_one(penguins, by species):');
  const repro = rows(await q(conn, `SELECT variable, level, stratum, display, p_value, effect_size
      FROM table_one('penguins',
        variables := ['body_mass_g','flipper_length_mm','sex'],
        by := ['species'])
      WHERE statistic IN ('mean (sd)','n (%)') OR p_value IS NOT NULL
      ORDER BY variable, stratum, level NULLS FIRST LIMIT 14`, 'repro'));
  for (const r of repro)
    console.log(`  ${r.variable} | ${r.level ?? '-'} | ${r.stratum} | ${r.display} | p=${r.p_value ?? '-'} | es=${r.effect_size ?? '-'}`);

  await conn.close();
  await db.terminate();
  server.close();
  console.log('\nPROBE_DONE');
  process.exit(0);
})().catch((e) => { console.error('\nVERIFY_FAIL:', e && e.stack ? e.stack : e); process.exit(1); });
