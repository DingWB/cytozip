/**
 * Test cz_reader.js against a local HTTP server serving a known .cz file.
 * Run: node --experimental-vm-modules tests/test_cz_reader.mjs
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { CzReader } from '../cytozip/cz_reader.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const TEST_FILE = '/tmp/test_js.cz';

// ── Simple HTTP server with Range support ──────────────────────────────────
function startServer(filePath) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const stat = fs.statSync(filePath);
      const size = stat.size;

      if (req.method === 'HEAD') {
        res.writeHead(200, { 'Content-Length': size, 'Accept-Ranges': 'bytes' });
        res.end();
        return;
      }

      const range = req.headers.range;
      if (range) {
        const [startStr, endStr] = range.replace('bytes=', '').split('-');
        const start = parseInt(startStr, 10);
        const end = endStr ? parseInt(endStr, 10) : size - 1;
        res.writeHead(206, {
          'Content-Range': `bytes ${start}-${end}/${size}`,
          'Accept-Ranges': 'bytes',
          'Content-Length': end - start + 1,
          'Content-Type': 'application/octet-stream',
        });
        fs.createReadStream(filePath, { start, end }).pipe(res);
      } else {
        res.writeHead(200, { 'Content-Length': size, 'Content-Type': 'application/octet-stream' });
        fs.createReadStream(filePath).pipe(res);
      }
    });
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, url: `http://127.0.0.1:${port}/${path.basename(filePath)}` });
    });
  });
}

// ── Test runner ──────────────────────────────────────────────────────────────
let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (!condition) {
    console.error(`  FAIL: ${msg}`);
    failed++;
  } else {
    passed++;
  }
}

function assertEquals(actual, expected, msg) {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a !== e) {
    console.error(`  FAIL: ${msg}\n    expected: ${e}\n    actual:   ${a}`);
    failed++;
  } else {
    passed++;
  }
}

async function main() {
  if (!fs.existsSync(TEST_FILE)) {
    console.error(`Test file not found: ${TEST_FILE}`);
    console.error('Run the Python test data generator first.');
    process.exit(1);
  }

  const { server, url } = await startServer(TEST_FILE);
  console.log(`Server running at ${url}`);

  try {
    // ── Test 1: Open and read header ──
    console.log('Test 1: Header parsing');
    const reader = await CzReader.fromUrl(url);
    assert(reader.header !== null, 'header should not be null');
    assertEquals(reader.header.magic, 'CZIP', 'magic');
    assertEquals(reader.header.formats, ['Q', 'H', 'H'], 'formats');
    assertEquals(reader.header.columns, ['pos', 'mc', 'cov'], 'columns');
    assertEquals(reader.header.dimensions, ['chrom'], 'dimensions');
    assert(reader.header.totalSize > 0, 'totalSize > 0');

    // ── Test 2: Chunk index ──
    console.log('Test 2: Chunk index');
    const dims = reader.dims;
    assert(dims.length === 2, `expected 2 dims, got ${dims.length}`);
    assert(dims.includes('chr1'), 'should have chr1');
    assert(dims.includes('chr2'), 'should have chr2');

    // ── Test 3: Summary ──
    console.log('Test 3: Summary chunks');
    const summary = reader.summaryChunks();
    assertEquals(summary.length, 2, 'should have 2 chunks');
    const chr1Summary = summary.find(s => s.dims[0] === 'chr1');
    assertEquals(chr1Summary.nrows, 100, 'chr1 should have 100 rows');
    const chr2Summary = summary.find(s => s.dims[0] === 'chr2');
    assertEquals(chr2Summary.nrows, 50, 'chr2 should have 50 rows');

    // ── Test 4: Fetch all records ──
    console.log('Test 4: Fetch chr1');
    const chr1Recs = await reader.fetch('chr1');
    assertEquals(chr1Recs.length, 100, 'chr1: 100 records');
    // First record: pos=0, mc=0, cov=1
    assertEquals(chr1Recs[0], [0, 0, 1], 'chr1 first record');
    // Last record: pos=990, mc=99, cov=100
    assertEquals(chr1Recs[99], [990, 99, 100], 'chr1 last record');
    // Spot check: record 50 → pos=500, mc=50, cov=51
    assertEquals(chr1Recs[50], [500, 50, 51], 'chr1 record 50');

    console.log('Test 5: Fetch chr2');
    const chr2Recs = await reader.fetch('chr2');
    assertEquals(chr2Recs.length, 50, 'chr2: 50 records');
    assertEquals(chr2Recs[0], [0, 0, 1], 'chr2 first record');
    assertEquals(chr2Recs[49], [245, 49, 50], 'chr2 last record');

    // ── Test 6: Query ──
    console.log('Test 6: Query chr1 [500, 600]');
    const queryResults = await reader.query('chr1', 500, 600, 0);
    assertEquals(queryResults.length, 11, 'query should return 11 records');
    assertEquals(queryResults[0][0], 500, 'first query result pos=500');
    assertEquals(queryResults[10][0], 600, 'last query result pos=600');
    // Verify mc and cov values
    assertEquals(queryResults[0], [500, 50, 51], 'query first full record');
    assertEquals(queryResults[10], [600, 60, 61], 'query last full record');

    // ── Test 7: Query edge cases ──
    console.log('Test 7: Query edge cases');
    const q2 = await reader.query('chr1', 0, 0, 0);
    assertEquals(q2.length, 1, 'query [0,0] should return 1 record');
    assertEquals(q2[0], [0, 0, 1], 'query [0,0] record');

    const q3 = await reader.query('chr1', 990, 990, 0);
    assertEquals(q3.length, 1, 'query [990,990] should return 1 record');
    assertEquals(q3[0], [990, 99, 100], 'query [990,990] record');

    const q4 = await reader.query('chr1', 995, 1000, 0);
    assertEquals(q4.length, 0, 'query [995,1000] should return 0 records');

    // ── Test 8: fetchChunkBytes ──
    console.log('Test 8: fetchChunkBytes');
    const raw = await reader.fetchChunkBytes('chr1');
    // 100 records × 12 bytes each = 1200 bytes
    assertEquals(raw.byteLength, 1200, 'chr1 raw should be 1200 bytes');

    reader.close();

    // ── Test 9: String columns ──
    console.log('Test 9: String columns');
    const TEST_FILE_STR = '/tmp/test_js_str.cz';
    if (fs.existsSync(TEST_FILE_STR)) {
      const { server: srv2, url: url2 } = await startServer(TEST_FILE_STR);
      try {
        const r2 = await CzReader.fromUrl(url2);
        assertEquals(r2.header.formats, ['H', '10s', 'c'], 'str formats');
        assertEquals(r2.header.columns, ['val', 'name', 'strand'], 'str columns');
        const recs = await r2.fetch('g1');
        assertEquals(recs.length, 20, 'g1: 20 records');
        assertEquals(recs[0][0], 0, 'first val=0');
        assertEquals(recs[0][1], 'gene00000', 'first name');
        assertEquals(recs[0][2], '+', 'first strand=+');
        assertEquals(recs[1][2], '-', 'second strand=-');
        assertEquals(recs[19][0], 19, 'last val=19');
        assertEquals(recs[19][1], 'gene00019', 'last name');
        r2.close();
      } finally {
        srv2.close();
      }
    } else {
      console.log('  SKIP: /tmp/test_js_str.cz not found');
    }

    console.log(`\n${'='.repeat(40)}`);
    console.log(`Results: ${passed} passed, ${failed} failed`);
    if (failed > 0) process.exitCode = 1;
  } finally {
    server.close();
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
