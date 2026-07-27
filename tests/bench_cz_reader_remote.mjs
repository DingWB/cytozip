/**
 * bench_cz_reader_remote.mjs — benchmark remote random-access query speed of
 * the JavaScript .cz reader (cz_reader.mjs) against real remote files served
 * over HTTP Range requests, the same access pattern WashU Epigenome Browser
 * uses for remote bgzip+tabix tracks.
 *
 * Run:
 *   /home/x-wding2/Software/conda/m3c/bin/node bench_cz_reader_remote.mjs
 */
import { CzReader } from '../cytozip/cz_reader.mjs';

const REF_URL    = 'https://neomorph.salk.edu/ftp/bican/hg38_with_chrL.allc.cz';
const SAMPLE_URL = 'https://neomorph.salk.edu/ftp/bican/UWA7648_CX1819_NAC_1_P10-1-K18-A10.cz';

// ── Instrument global fetch to count HTTP Range requests + bytes ─────────────
const origFetch = globalThis.fetch;
const httpStats = { requests: 0, bytes: 0, rangeReq: 0 };
globalThis.fetch = async (url, opts = {}) => {
  httpStats.requests++;
  const range = opts.headers && (opts.headers.Range || opts.headers.range);
  if (range) httpStats.rangeReq++;
  const resp = await origFetch(url, opts);
  // Only count payload bytes for Range (206) responses — HEAD/200 report the
  // full file length and would wildly inflate the transferred total.
  const cl = resp.headers.get('Content-Length');
  if (cl && range) httpStats.bytes += parseInt(cl, 10);
  return resp;
};
function snap() { return { ...httpStats }; }
function diff(a, b) {
  return {
    requests: b.requests - a.requests,
    rangeReq: b.rangeReq - a.rangeReq,
    bytes: b.bytes - a.bytes,
  };
}

const fmtBytes = (n) => n < 1024 ? `${n} B`
  : n < 1048576 ? `${(n / 1024).toFixed(1)} KB`
  : `${(n / 1048576).toFixed(2)} MB`;

function stats(times) {
  const s = [...times].sort((a, b) => a - b);
  const mean = s.reduce((x, y) => x + y, 0) / s.length;
  const p = (q) => s[Math.min(s.length - 1, Math.floor(q * s.length))];
  return { min: s[0], mean, median: p(0.5), p90: p(0.9), max: s[s.length - 1] };
}

// ── Network floor: cold single HTTP Range request latency ────────────────────
async function benchRawRange(url, label, n = 8) {
  // Fetch a small 64 KB slice at random offsets — no caching, fresh reader each
  // time — to measure the pure per-request round-trip floor (what a single
  // bgzip block fetch costs).
  const head = await origFetch(url, { method: 'HEAD' });
  const size = parseInt(head.headers.get('Content-Length'), 10);
  const times = [];
  for (let i = 0; i < n; i++) {
    const start = Math.floor(Math.random() * (size - 65536));
    const t0 = performance.now();
    const r = await origFetch(url, { headers: { Range: `bytes=${start}-${start + 65535}` } });
    await r.arrayBuffer();
    times.push(performance.now() - t0);
  }
  const st = stats(times);
  console.log(`\n[Network floor] ${label}  (64 KB cold Range x${n})`);
  console.log(`  file size: ${fmtBytes(size)}`);
  console.log(`  latency  : min ${st.min.toFixed(0)}  median ${st.median.toFixed(0)}  p90 ${st.p90.toFixed(0)}  max ${st.max.toFixed(0)} ms`);
  return st.median;
}

// ── Region query benchmark on the reference (queryable) file ─────────────────
async function benchRegionQuery() {
  console.log('\n' + '='.repeat(72));
  console.log('REGION QUERY BENCHMARK  —  hg38_with_chrL.allc.cz (reference)');
  console.log('='.repeat(72));

  const t0 = performance.now();
  const openBefore = snap();
  const r = await CzReader.fromUrl(REF_URL);
  const openMs = performance.now() - t0;
  const openIo = diff(openBefore, snap());
  console.log(`open (header + chunk index): ${openMs.toFixed(0)} ms  ` +
              `[${openIo.requests} HTTP req, ${fmtBytes(openIo.bytes)}]`);

  const chr1 = r.summaryChunks().find(c => c.dims[0] === 'chr1');
  console.log(`chr1: ${chr1.nrows.toLocaleString()} rows in ${chr1.nblocks} blocks`);

  // chr1 genomic span (hg38 ~248 Mb). Query random regions of varying width.
  const CHR1_MAX = 248_900_000;
  const widths = [1_000, 10_000, 100_000, 1_000_000];
  const REPEAT = 6;

  for (const w of widths) {
    const times = [];
    const ios = [];
    let sampleCount = 0;
    for (let i = 0; i < REPEAT; i++) {
      const start = Math.floor(Math.random() * (CHR1_MAX - w));
      const end = start + w;
      // Fresh reader each iteration → cold query (no tail/block cache reuse),
      // the realistic "user jumps to a new region" scenario.
      const rr = await CzReader.fromUrl(REF_URL);
      const before = snap();
      const t = performance.now();
      const recs = await rr.query('chr1', start, end, 0);
      const dt = performance.now() - t;
      times.push(dt);
      ios.push(diff(before, snap()));
      sampleCount = recs.length;
      rr.close();
    }
    const st = stats(times);
    const avgReq = ios.reduce((s, x) => s + x.requests, 0) / ios.length;
    const avgBytes = ios.reduce((s, x) => s + x.bytes, 0) / ios.length;
    console.log(
      `\nregion ${(w / 1000).toString().padStart(5)} kb  ` +
      `| query: median ${st.median.toFixed(0).padStart(4)} ms  p90 ${st.p90.toFixed(0).padStart(4)} ms  ` +
      `| ~${avgReq.toFixed(1)} HTTP req, ${fmtBytes(avgBytes)}  ` +
      `| ~${sampleCount} rows`
    );
  }

  // Warm query: reuse one reader, repeat queries (tail/index cached).
  console.log('\n-- warm (reader reused, chunk tail cached) --');
  const rw = await CzReader.fromUrl(REF_URL);
  await rw.query('chr1', 1_000_000, 1_010_000, 0); // prime chr1 tail
  for (const w of [1_000, 100_000]) {
    const times = [];
    for (let i = 0; i < 6; i++) {
      const start = Math.floor(Math.random() * (CHR1_MAX - w));
      const before = snap();
      const t = performance.now();
      await rw.query('chr1', start, start + w, 0);
      times.push(performance.now() - t);
    }
    const st = stats(times);
    console.log(`region ${(w / 1000).toString().padStart(5)} kb  | median ${st.median.toFixed(0)} ms  p90 ${st.p90.toFixed(0)} ms`);
  }
  rw.close();
}

// ── Main ─────────────────────────────────────────────────────────────────────
console.log('Node', process.version, '| DecompressionStream:', typeof DecompressionStream !== 'undefined');

await benchRawRange(REF_URL, 'reference allc.cz');
await benchRawRange(SAMPLE_URL, 'sample UWA7648 .cz');
await benchRegionQuery();

console.log('\nTotal HTTP requests:', httpStats.requests, '| total transferred:', fmtBytes(httpStats.bytes));
