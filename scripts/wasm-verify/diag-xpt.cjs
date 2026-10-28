// Diagnose the wasm 0-byte XPT: file state after COPY, path variants, v5 vs v8,
// and sav as control. Throwaway diagnostic — not part of the release gate.
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
  await db.open({ path: ':memory:', allowUnsignedExtensions: true });
  const conn = await db.connect();
  await conn.query(`SET custom_extension_repository='http://127.0.0.1:${port}'`);
  await conn.query(`INSTALL stats_duck`);
  await conn.query(`LOAD stats_duck`);
  await conn.query(`CREATE TABLE p8 AS SELECT 3750.0 AS mass, 181 AS flip, 'male' AS sex, 'Adelie' AS species FROM range(12)`);

  const tryQ = async (label, sql) => {
    try { const r = rows(await conn.query(sql)); console.log(`[ok] ${label}: ${JSON.stringify(r).slice(0, 160)}`); return r; }
    catch (e) { console.log(`[err] ${label}: ${(e.message || e).split('\n')[0]}`); return null; }
  };
  const bufLen = async (label, p) => {
    try { const b = await db.copyFileToBuffer(p); console.log(`[buf] ${label}: ${b ? b.length : 'null'} bytes`); }
    catch (e) { console.log(`[buf-err] ${label}: ${(e.message || e).split('\n')[0]}`); }
  };

  await tryQ('copy-xpt-tmp', `COPY p8 TO '/tmp/d1.xpt' (FORMAT xpt)`);
  await bufLen('d1.xpt', '/tmp/d1.xpt');
  await tryQ('copy-xpt-root', `COPY p8 TO 'd2.xpt' (FORMAT xpt)`);
  await bufLen('d2.xpt', 'd2.xpt');
  await tryQ('copy-xpt-v8', `COPY p8 TO '/tmp/d3.xpt' (FORMAT xpt, VERSION 8)`);
  await bufLen('d3.xpt', '/tmp/d3.xpt');
  await tryQ('copy-sav-control', `COPY p8 TO '/tmp/d4.sav' (FORMAT sav)`);
  await bufLen('d4.sav', '/tmp/d4.sav');
  await tryQ('glob-tmp', `SELECT file FROM glob('/tmp/*')`);
  await tryQ('blob-sizes', `SELECT filename, size FROM read_blob(['/tmp/d1.xpt','/tmp/d4.sav'])`);

  await conn.close(); await db.terminate(); server.close();
  console.log('DIAG_DONE'); process.exit(0);
})().catch((e) => { console.error('DIAG_FAIL:', e && e.stack ? e.stack : e); process.exit(1); });
