// Bedevere 0.15 release gate — verify.cjs's probe matrix plus the surfaces the
// 0.15 cycle newly relies on: the COPY stat-format writers (web `.export`
// offers xpt/sav/por/sas7bdat once capabilities.visualize is true), the
// type-override retarget (violin x:nominal must NOT retype the density axis —
// the-stats-duck 3753ea0), and the stacked-SCALE grammar. Same harness
// mechanics as verify.cjs: throwaway localhost repo + patched worker.
//
//   Place the candidate build at build/wasm_eh/repository/v1.4.3/wasm_eh/
//   then:  node verify-015.cjs   ->   matrix + "PROBE_DONE" (exit 0) or FAIL.

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
// Unique per-run paths — a pre-existing target makes copyFileToBuffer serve
// stale content in the node harness (real-fs mapping); the app never reuses
// paths either (__bedevere_export_<ts>).
const RUN = Date.now().toString(36);

const rows = (t) => { const o = []; for (const r of t) o.push(r.toJSON()); return o; };
const withTimeout = (p, ms, label) =>
  Promise.race([p, new Promise((_, rej) => setTimeout(() => rej(new Error(`TIMEOUT ${label} ${ms}ms`)), ms))]);

(async () => {
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    fs.readFile(path.join(REPO, url), (err, data) => {
      if (err) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'Content-Type': 'application/wasm', 'Content-Length': data.length });
      res.end(req.method === 'HEAD' ? undefined : data);
    });
  });
  await new Promise((r) => server.listen(0, '127.0.0.1', r));
  const port = server.address().port;

  const worker = new WWorker(pathToFileURL(path.join(HERE, 'patch-worker.mjs')).href, { type: 'module' });
  const db = new duckdb.AsyncDuckDB(new duckdb.VoidLogger(), worker);
  await db.instantiate(path.join(DIST, 'duckdb-eh.wasm'));
  await db.open({ path: ':memory:', allowUnsignedExtensions: true, query: { castBigIntToDouble: true } });
  const conn = await db.connect();

  const ver = rows(await conn.query(`SELECT library_version FROM pragma_version()`))[0];
  console.log(`[duckdb-wasm] library_version=${ver.library_version}`);

  await conn.query(`SET custom_extension_repository='http://127.0.0.1:${port}'`);
  await conn.query(`INSTALL stats_duck`);
  await conn.query(`LOAD stats_duck`);
  console.log('[duckdb] stats_duck INSTALL + LOAD ok');

  await conn.query(`CREATE TABLE penguins (
      body_mass_g DOUBLE, flipper_length_mm INTEGER, bill_length_mm DOUBLE,
      sex VARCHAR, species VARCHAR)`);
  await conn.query(`INSERT INTO penguins VALUES
      (3750,181,39.1,'male','Adelie'),   (3800,186,39.5,'female','Adelie'),
      (3250,195,40.3,'female','Adelie'), (3450,193,36.7,'female','Adelie'),
      (4500,210,46.1,'male','Gentoo'),   (5000,217,50.0,'male','Gentoo'),
      (4750,216,49.9,'female','Gentoo'), (4400,214,48.4,'female','Gentoo'),
      (3700,193,46.5,'male','Chinstrap'),(3550,191,50.5,'female','Chinstrap'),
      (3900,198,51.3,'male','Chinstrap'),(3650,190,49.2,'female','Chinstrap')`);

  // [name, sql, optional row-assertion(rows) -> throw to fail]
  const probes = [
    // verify.cjs base matrix (abbreviated labels, same coverage)
    ['baseline-builtin', `SELECT count(*) AS n FROM penguins`],
    ['scalar-ext', `SELECT pnorm(1.96) AS p`],
    ['struct-agg', `SELECT summary_stats(body_mass_g)::VARCHAR AS s FROM penguins`],
    ['anova', `SELECT anova_oneway(body_mass_g, species)::VARCHAR AS a FROM penguins`],
    ['table_one', `SELECT count(*) AS n FROM table_one('penguins', variables := ['body_mass_g','sex'], by := ['species'])`],
    ['lm_fit-cr1', `SELECT (lm_fit(body_mass_g, [bill_length_mm], 'CR1', species)).n_clusters AS g FROM penguins`],
    ['visualize-point', `VISUALIZE body_mass_g AS x, flipper_length_mm AS y FROM penguins DRAW point`],
    // 0.15: stacked SCALE options (the 0.14 desktop "Unexpected token 'ZERO'" class)
    ['visualize-stacked-scale', `VISUALIZE body_mass_g AS x, flipper_length_mm AS y FROM penguins DRAW point SCALE x LABEL 'mass' ZERO false`],
    // 0.15: violin x:nominal — spec must keep the density sweep quantitative
    // (pre-3753ea0 binaries retype it nominal -> page-wide ribbon)
    ['visualize-violin-override', `VISUALIZE species AS x:nominal, body_mass_g AS y FROM penguins DRAW violin`,
      (r) => {
        const spec = r[0].spec;
        if (!spec.includes('"x":{"field":"density","type":"quantitative"'))
          throw new Error('density axis retyped — pre-3753ea0 binary? spec: ' + spec.slice(0, 200));
        if (!spec.includes('"column":{"field":"x","type":"nominal"'))
          throw new Error('column facet missing/retyped: ' + spec.slice(0, 200));
      }],
    // 0.15: COPY stat-format writers + read-back through read_stat (the web
    // `.export` path: COPY to the wasm virtual FS, bytes read back after).
    // XPT (V5 transport) and POR cap variable names at 8 chars — stats_duck
    // binder-errors on longer ones by design — so export a projection with
    // short names.
    ['copy-prep', `CREATE TABLE p8 AS SELECT body_mass_g AS mass, flipper_length_mm AS flip, sex, species FROM penguins`],
    ['copy-xpt', `COPY p8 TO '/tmp/p_${RUN}.xpt' (FORMAT xpt)`],
    // The web `.export` path reads the COPY output back as bytes
    // (copyFileToBuffer), and the import path registers a buffer and
    // read_stats the registered name — model exactly those.
    ['export-bytes-xpt', async (conn, db) => {
      const bytes = await db.copyFileToBuffer(`/tmp/p_${RUN}.xpt`);
      if (!bytes || bytes.length < 80) throw new Error(`xpt export bytes: ${bytes && bytes.length}`);
      await db.registerFileBuffer(`reimport_${RUN}.xpt`, bytes);
    }],
    ['import-xpt', `SELECT count(*) AS n FROM read_stat('reimport_${RUN}.xpt')`,
      (r) => { if (Number(r[0].n) !== 12) throw new Error(`xpt export->import round-trip lost rows: ${r[0].n}`); }],
    ['copy-sav', `COPY p8 TO '/tmp/p_${RUN}.sav' (FORMAT sav)`],
    ['readback-sav', `SELECT count(*) AS n FROM read_stat('/tmp/p_${RUN}.sav')`,
      (r) => { if (Number(r[0].n) !== 12) throw new Error(`sav round-trip lost rows: ${r[0].n}`); }],
    ['copy-por', `COPY p8 TO '/tmp/p_${RUN}.por' (FORMAT por)`],
    ['export-bytes-por', async (conn, db) => {
      const bytes = await db.copyFileToBuffer(`/tmp/p_${RUN}.por`);
      if (!bytes || bytes.length < 80) throw new Error(`por export bytes: ${bytes && bytes.length}`);
    }],
    ['copy-sas7bdat', `COPY p8 TO '/tmp/p_${RUN}.sas7bdat' (FORMAT sas7bdat)`],
    ['readback-sas7bdat', `SELECT count(*) AS n FROM read_stat('/tmp/p_${RUN}.sas7bdat')`,
      (r) => { if (Number(r[0].n) !== 12) throw new Error(`sas7bdat round-trip lost rows: ${r[0].n}`); }],
  ];

  let failed = 0;
  for (const [name, sqlOrFn, assert] of probes) {
    try {
      if (typeof sqlOrFn === 'function') {
        await withTimeout(sqlOrFn(conn, db), 20000, name);
      } else {
        const r = rows(await withTimeout(conn.query(sqlOrFn), 20000, name));
        if (assert) assert(r);
      }
      console.log(`  [PASS] ${name}`);
    } catch (e) {
      const msg = (e && e.message ? e.message : String(e)).split('\n')[0];
      console.log(`  [FAIL] ${name} -> ${msg}`);
      failed++;
    }
  }

  await conn.close();
  await db.terminate();
  server.close();
  if (failed) { console.error(`\nVERIFY_FAIL: ${failed} probe(s) failed`); process.exit(1); }
  console.log('\nPROBE_DONE');
  process.exit(0);
})().catch((e) => { console.error('\nVERIFY_FAIL:', e && e.stack ? e.stack : e); process.exit(1); });
