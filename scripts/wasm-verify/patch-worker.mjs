// Wrapper worker: duckdb-wasm's node runtime caches fetched extensions under
// ~/.duckdb/extensions/<host:port>/... and a ':' is an illegal Windows path
// component (NTFS reads it as an alternate-data-stream). Sanitize the host:port
// colon (digit ':' digit only — never the "C:" drive colon) consistently across
// every fs path op, so the cache writes and reads the same colon-free dir. Then
// load the real duckdb node worker.
import fs from 'fs';

const sani = (p) => (typeof p === 'string' ? p.replace(/(\d):(\d)/g, '$1_$2') : p);
const methods = [
  'mkdirSync', 'writeFileSync', 'readFileSync', 'existsSync', 'statSync', 'lstatSync',
  'openSync', 'accessSync', 'realpathSync', 'readdirSync', 'rmSync', 'unlinkSync',
  'appendFileSync', 'copyFileSync', 'renameSync', 'chmodSync', 'utimesSync',
];
for (const m of methods) {
  const orig = fs[m];
  if (typeof orig === 'function') fs[m] = (p, ...rest) => orig.call(fs, sani(p), ...rest);
}

await import(new URL('./node_modules/@duckdb/duckdb-wasm/dist/duckdb-node-eh.worker.cjs', import.meta.url).href);
