/**
 * cz_reader.js — JavaScript reader for the cytozip (.cz) binary format.
 *
 * Reads .cz files from remote HTTP servers using Range requests.
 * Designed for browser-based visualization of DNA methylation data.
 *
 * Usage (browser):
 *   import { CzReader } from './cz_reader.js';
 *   const reader = await CzReader.fromUrl('https://example.com/data.cz');
 *   console.log(reader.header);
 *   console.log(reader.chunkIndex);
 *   const records = await reader.fetch('chr1');
 *   const queried = await reader.query('chr1', 1000, 2000);
 *   reader.close();
 *
 * Usage (Node.js):
 *   Same API — uses globalThis.fetch (Node 18+).
 */

// ─── Constants ───────────────────────────────────────────────────────────────
const CZ_MAGIC   = 'CZIP';
const BLOCK_MAGIC = 0x4243; // 'CB' little-endian → bytes [0x43, 0x42] → uint16LE = 0x4243
const CHUNK_MAGIC = 0x4343; // 'CC'
const INDEX_MAGIC = 'CZIX';
const BLOCK_MAX_LEN = 65535;

// ─── Struct format helpers ───────────────────────────────────────────────────
// Map Python struct format chars to {size, read(dataView, offset)}
const FORMAT_MAP = {
  'b': { size: 1, read: (dv, o) => dv.getInt8(o) },
  'B': { size: 1, read: (dv, o) => dv.getUint8(o) },
  'h': { size: 2, read: (dv, o) => dv.getInt16(o, true) },
  'H': { size: 2, read: (dv, o) => dv.getUint16(o, true) },
  'i': { size: 4, read: (dv, o) => dv.getInt32(o, true) },
  'I': { size: 4, read: (dv, o) => dv.getUint32(o, true) },
  'q': { size: 8, read: (dv, o) => dv.getBigInt64(o, true) },
  'Q': { size: 8, read: (dv, o) => dv.getBigUint64(o, true) },
  'f': { size: 4, read: (dv, o) => dv.getFloat32(o, true) },
  'd': { size: 8, read: (dv, o) => dv.getFloat64(o, true) },
  'e': { size: 2, read: (dv, o) => _readFloat16(dv, o) },
  // 'c' and 's' handled specially (variable-width strings)
};

/** IEEE 754 half-precision → JS number (for format 'e'). */
function _readFloat16(dv, offset) {
  const bits = dv.getUint16(offset, true);
  const sign = (bits >> 15) & 1;
  const exp  = (bits >> 10) & 0x1f;
  const frac = bits & 0x3ff;
  if (exp === 0) return (sign ? -1 : 1) * 2 ** -14 * (frac / 1024);
  if (exp === 31) return frac ? NaN : (sign ? -Infinity : Infinity);
  return (sign ? -1 : 1) * 2 ** (exp - 15) * (1 + frac / 1024);
}

/**
 * Parse a Python struct format string (without '<') into an array of field
 * descriptors: { fmt, size, read, strLen? }.
 * Handles: 'B', 'H', 'I', 'Q', 'h', 'i', 'q', 'f', 'd', 'e', 'Ns', 'c'
 */
function parseFormats(fmtStrings) {
  const fields = [];
  for (const fmt of fmtStrings) {
    const lastChar = fmt[fmt.length - 1];
    if (lastChar === 's') {
      const n = fmt.length > 1 ? parseInt(fmt.slice(0, -1), 10) : 1;
      fields.push({
        fmt, size: n, strLen: n,
        read: (dv, o) => {
          const bytes = new Uint8Array(dv.buffer, dv.byteOffset + o, n);
          // Trim trailing nulls
          let end = n;
          while (end > 0 && bytes[end - 1] === 0) end--;
          return new TextDecoder().decode(bytes.subarray(0, end));
        }
      });
    } else if (lastChar === 'c') {
      fields.push({
        fmt, size: 1, strLen: 1,
        read: (dv, o) => {
          const b = dv.getUint8(o);
          return b === 0 ? '' : String.fromCharCode(b);
        }
      });
    } else {
      const info = FORMAT_MAP[lastChar];
      if (!info) throw new Error(`Unsupported format: ${fmt}`);
      fields.push({ fmt, ...info });
    }
  }
  return fields;
}

/** Compute total byte size of one record from parsed fields. */
function unitSize(fields) {
  return fields.reduce((s, f) => s + f.size, 0);
}

/**
 * Unpack records from an ArrayBuffer/Uint8Array using parsed field descriptors.
 * Returns an array of arrays (each inner array = one record's column values).
 */
function unpackRecords(buffer, fields, unit) {
  const dv = new DataView(buffer.buffer || buffer, buffer.byteOffset || 0, buffer.byteLength || buffer.length);
  const n = Math.floor(dv.byteLength / unit);
  const records = new Array(n);
  for (let i = 0; i < n; i++) {
    let off = i * unit;
    const row = new Array(fields.length);
    for (let j = 0; j < fields.length; j++) {
      row[j] = fields[j].read(dv, off);
      off += fields[j].size;
    }
    records[i] = row;
  }
  return records;
}

// ─── Raw DEFLATE decompression ───────────────────────────────────────────────
/**
 * Decompress raw DEFLATE data (no gzip/zlib header, wbits = -15).
 * Uses the browser's DecompressionStream API.
 */
async function inflateRaw(compressedBytes) {
  // DecompressionStream('deflate-raw') is available in modern browsers
  // and Node 18+ (behind --experimental-global-webcrypto in some builds).
  if (typeof DecompressionStream !== 'undefined') {
    const ds = new DecompressionStream('deflate-raw');
    const writer = ds.writable.getWriter();
    const reader = ds.readable.getReader();
    writer.write(compressedBytes);
    writer.close();
    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }
    if (chunks.length === 1) return chunks[0];
    const total = chunks.reduce((s, c) => s + c.byteLength, 0);
    const out = new Uint8Array(total);
    let pos = 0;
    for (const c of chunks) {
      out.set(c, pos);
      pos += c.byteLength;
    }
    return out;
  }
  // Fallback: try pako if available (bundled by user)
  if (typeof globalThis.pako !== 'undefined') {
    return globalThis.pako.inflateRaw(compressedBytes);
  }
  throw new Error(
    'No raw DEFLATE decompressor available. Use a browser with ' +
    'DecompressionStream support, or load pako.js first.'
  );
}

// ─── RemoteFile ──────────────────────────────────────────────────────────────
/**
 * File-like wrapper around HTTP Range requests with read-ahead caching.
 * Mirrors Python's RemoteFile: seek()/read()/tell()/close().
 */
class RemoteFile {
  /**
   * @param {string} url
   * @param {object} [opts]
   * @param {number} [opts.cacheSize=2097152]  Read-ahead cache (bytes).
   * @param {object} [opts.fetchOptions={}]    Extra options passed to fetch()
   *        (e.g. { headers: {...}, credentials: 'include' }).
   */
  constructor(url, opts = {}) {
    this.url = url;
    this._pos = 0;
    this._size = -1;
    this._cacheSize = opts.cacheSize ?? 2 * 1024 * 1024;
    this._fetchOpts = opts.fetchOptions ?? {};
    this._cacheStart = -1;
    this._cacheEnd = -1;
    this._cacheData = null;
  }

  /** Probe the server for the file size (HEAD, fallback to Range probe). */
  async init() {
    // Try HEAD first
    const headResp = await fetch(this.url, {
      method: 'HEAD', redirect: 'follow', ...this._fetchOpts,
    });
    const cl = headResp.headers.get('Content-Length');
    if (cl && headResp.status === 200) {
      this._size = parseInt(cl, 10);
      return;
    }
    // Fallback: Range probe
    const headers = { ...(this._fetchOpts.headers || {}), Range: 'bytes=0-0' };
    const resp = await fetch(this.url, {
      method: 'GET', redirect: 'follow',
      ...this._fetchOpts, headers,
    });
    const cr = resp.headers.get('Content-Range');  // "bytes 0-0/12345"
    if (cr && cr.includes('/')) {
      this._size = parseInt(cr.split('/').pop(), 10);
    } else {
      throw new Error(`Cannot determine file size for ${this.url}`);
    }
  }

  get size() { return this._size; }

  seek(offset, whence = 0) {
    if (whence === 0) this._pos = offset;
    else if (whence === 1) this._pos += offset;
    else if (whence === 2) this._pos = this._size + offset;
  }

  tell() { return this._pos; }

  /**
   * Read `size` bytes from the current position. Returns a Uint8Array.
   * Fetches from cache or makes an HTTP Range request.
   */
  async read(size) {
    if (size <= 0) return new Uint8Array(0);
    const start = this._pos;
    const end = Math.min(start + size, this._size);
    const need = end - start;
    if (need <= 0) return new Uint8Array(0);

    // Cache hit?
    if (this._cacheData && start >= this._cacheStart && end <= this._cacheEnd) {
      const off = start - this._cacheStart;
      this._pos = end;
      return new Uint8Array(this._cacheData.buffer, this._cacheData.byteOffset + off, need);
    }

    // Fetch with read-ahead
    const fetchEnd = Math.min(start + Math.max(size, this._cacheSize), this._size);
    const headers = {
      ...(this._fetchOpts.headers || {}),
      Range: `bytes=${start}-${fetchEnd - 1}`,
    };
    const resp = await fetch(this.url, {
      method: 'GET', redirect: 'follow',
      ...this._fetchOpts, headers,
    });
    if (resp.status !== 206 && resp.status !== 200) {
      throw new Error(`HTTP ${resp.status} fetching ${this.url} [${start}-${fetchEnd - 1}]`);
    }
    const arrayBuf = await resp.arrayBuffer();
    this._cacheData = new Uint8Array(arrayBuf);
    this._cacheStart = start;
    this._cacheEnd = start + this._cacheData.byteLength;
    this._pos = end;
    return new Uint8Array(this._cacheData.buffer, this._cacheData.byteOffset, need);
  }

  close() {
    this._cacheData = null;
  }
}

// ─── CzReader ────────────────────────────────────────────────────────────────
/**
 * Read-only reader for remote .cz files.
 *
 * @example
 *   const reader = await CzReader.fromUrl('https://example.com/data.cz');
 *   console.log(reader.header);
 *   const records = await reader.fetch('chr1');
 *   const queried = await reader.query('chr1', 1000, 2000);
 */
class CzReader {
  constructor(remoteFile) {
    this._handle = remoteFile;
    this.header = null;
    this.chunkIndex = null;    // Map<string, {start,size,dataLen,nblocks}>
    this._fields = null;       // parsed format descriptors
    this._unitSize = 0;
    this._strColMask = null;
    this._chunkTailCache = new Map();
  }

  /**
   * Open a remote .cz file.
   * @param {string} url
   * @param {object} [opts]  Options forwarded to RemoteFile.
   * @returns {Promise<CzReader>}
   */
  static async fromUrl(url, opts = {}) {
    const rf = new RemoteFile(url, opts);
    await rf.init();
    const reader = new CzReader(rf);
    await reader._readHeader();
    await reader._readChunkIndex();
    return reader;
  }

  // ── Header parsing ───────────────────────────────────────────────────────
  async _readHeader() {
    this._handle.seek(0);
    // Read first 200 bytes (enough for most headers)
    let buf = await this._handle.read(200);
    if (buf.byteLength < 4) throw new Error('File too small for .cz header');
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    let off = 0;

    // magic (4B)
    const magic = new TextDecoder().decode(buf.subarray(0, 4));
    if (magic !== CZ_MAGIC) throw new Error(`Not a .cz file (magic=${magic})`);
    off = 4;

    // version (float32 LE)
    const version = dv.getFloat32(off, true); off += 4;

    // total_size (uint64 LE)
    const totalSize = Number(dv.getBigUint64(off, true)); off += 8;
    if (totalSize === 0) throw new Error('File not completed (total_size=0)');

    // message
    const msgLen = dv.getUint16(off, true); off += 2;
    // Ensure we have enough data
    if (off + msgLen > buf.byteLength) {
      this._handle.seek(0);
      buf = await this._handle.read(Math.max(400, off + msgLen + 200));
    }
    let dv2 = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    const message = new TextDecoder().decode(buf.subarray(off, off + msgLen));
    off += msgLen;

    // n_cols (1B)
    const nCols = buf[off]; off += 1;

    // formats[]
    const formats = [];
    for (let i = 0; i < nCols; i++) {
      if (off >= buf.byteLength) {
        // Need more data (very unlikely for typical headers)
        this._handle.seek(0);
        buf = await this._handle.read(off + 200);
        dv2 = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
      }
      const fLen = buf[off]; off += 1;
      formats.push(new TextDecoder().decode(buf.subarray(off, off + fLen)));
      off += fLen;
    }

    // columns[]
    const columns = [];
    for (let i = 0; i < nCols; i++) {
      const nLen = buf[off]; off += 1;
      columns.push(new TextDecoder().decode(buf.subarray(off, off + nLen)));
      off += nLen;
    }

    // sort_col (1B): index of the "position" column whose first-in-block
    // values are cached at the end of each chunk tail. 0xFF means disabled.
    const sortColRaw = buf[off]; off += 1;
    const sortCol = sortColRaw === 0xff ? null : sortColRaw;

    // dimensions
    const nDims = buf[off]; off += 1;
    const dimensions = [];
    for (let i = 0; i < nDims; i++) {
      const dLen = buf[off]; off += 1;
      dimensions.push(new TextDecoder().decode(buf.subarray(off, off + dLen)));
      off += dLen;
    }

    this.header = {
      magic, version, totalSize, message,
      formats, columns, sortCol, dimensions,
      headerSize: off,
    };

    this._fields = parseFormats(formats);
    this._unitSize = unitSize(this._fields);
    this._strColMask = formats.map(f => {
      const last = f[f.length - 1];
      return last === 's' || last === 'c';
    });

    // Prepare per-block first_coord reader (only when sort_col is enabled)
    if (sortCol !== null) {
      const sortFmt = formats[sortCol];
      const entry = FORMAT_MAP[sortFmt];
      if (!entry) {
        throw new Error(`sort_col format '${sortFmt}' is not a supported integer format`);
      }
      this._sortColSize = entry.size;
      this._sortColRead = entry.read;
    } else {
      this._sortColSize = 0;
      this._sortColRead = null;
    }
  }

  // ── Chunk Index ──────────────────────────────────────────────────────────
  /** Read the chunk index from the end of the file (2 HTTP requests). */
  async _readChunkIndex() {
    const f = this._handle;
    const fileSize = f.size;
    if (fileSize < 36) {
      this.chunkIndex = new Map();
      return;
    }

    // Read last 36 bytes: chunk_index_offset(8B) + EOF(28B)
    f.seek(fileSize - 36);
    const tail = await f.read(36);
    const tailDv = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
    const indexOffset = Number(tailDv.getBigUint64(0, true));

    if (indexOffset === 0 || indexOffset >= fileSize) {
      this.chunkIndex = new Map();
      return;
    }

    // Read the entire chunk index
    f.seek(indexOffset);
    const idxBuf = await f.read(fileSize - 28 - indexOffset);
    const idxDv = new DataView(idxBuf.buffer, idxBuf.byteOffset, idxBuf.byteLength);
    let off = 0;

    // magic (4B)
    const magic = new TextDecoder().decode(idxBuf.subarray(0, 4));
    if (magic !== INDEX_MAGIC) {
      this.chunkIndex = new Map();
      return;
    }
    off = 4;

    const nChunks = Number(idxDv.getBigUint64(off, true)); off += 8;
    const nDims = this.header.dimensions.length;
    const index = new Map();

    for (let c = 0; c < nChunks; c++) {
      const dims = [];
      for (let d = 0; d < nDims; d++) {
        const dLen = idxBuf[off]; off += 1;
        dims.push(new TextDecoder().decode(idxBuf.subarray(off, off + dLen)));
        off += dLen;
      }
      const start   = Number(idxDv.getBigUint64(off, true)); off += 8;
      const size    = Number(idxDv.getBigUint64(off, true)); off += 8;
      const dataLen = Number(idxDv.getBigUint64(off, true)); off += 8;
      const nblocks = Number(idxDv.getBigUint64(off, true)); off += 8;
      const key = dims.join('\t');
      index.set(key, { dims, start, size, dataLen, nblocks });
    }

    this.chunkIndex = index;
  }

  /** List all dimension keys. */
  get dims() {
    return this.chunkIndex ? [...this.chunkIndex.keys()] : [];
  }

  /**
   * Get summary info for all chunks.
   * @returns {Array<{dims, start, size, dataLen, nblocks, nrows}>}
   */
  summaryChunks() {
    const result = [];
    for (const [key, info] of this.chunkIndex) {
      result.push({
        dims: info.dims,
        start: info.start,
        size: info.size,
        dataLen: info.dataLen,
        nblocks: info.nblocks,
        nrows: Math.floor(info.dataLen / this._unitSize),
      });
    }
    return result;
  }

  // ── Chunk tail loading ───────────────────────────────────────────────────
  /**
   * Load chunk tail metadata (block virtual offsets, etc.) for a given dim key.
   * Caches results to avoid re-reading.
   */
  async _loadChunkTail(dimKey) {
    if (this._chunkTailCache.has(dimKey)) return this._chunkTailCache.get(dimKey);

    const info = this.chunkIndex.get(dimKey);
    if (!info) throw new Error(`Unknown dimension: ${dimKey}`);

    // Chunk tail sits right after the compressed blocks.
    // Read from chunk_start + chunk_size (tail offset).
    const tailOffset = info.start + info.size;
    this._handle.seek(tailOffset);

    // tail header: data_len(8B) + n_blocks(8B) + virtual_offsets(N*8B)
    //   + [first_coords(N * sort_col_size) if sort_col enabled]
    //   + dim_values
    const sortSize = this._sortColSize || 0;
    const tailSize = 16 + info.nblocks * (8 + sortSize) + 256; // +256 for dim strings
    const tailBuf = await this._handle.read(tailSize);
    const dv = new DataView(tailBuf.buffer, tailBuf.byteOffset, tailBuf.byteLength);
    let off = 0;

    const dataLen = Number(dv.getBigUint64(off, true)); off += 8;
    const nblocks = Number(dv.getBigUint64(off, true)); off += 8;

    const blockVOs = new Array(nblocks);
    for (let i = 0; i < nblocks; i++) {
      blockVOs[i] = Number(dv.getBigUint64(off, true)); off += 8;
    }

    // first_coords (one entry per block) when sort_col is enabled — lets
    // query() do true numeric bisect without decompressing probe blocks.
    let firstCoords = null;
    if (sortSize > 0) {
      firstCoords = new Array(nblocks);
      for (let i = 0; i < nblocks; i++) {
        firstCoords[i] = _toNumber(this._sortColRead(dv, off));
        off += sortSize;
      }
    }

    const result = {
      start: info.start,
      size: info.size,
      dataLen,
      nblocks,
      blockVOs,
      firstCoords,
    };
    this._chunkTailCache.set(dimKey, result);
    return result;
  }

  // ── Block decompression ──────────────────────────────────────────────────
  /**
   * Decompress all blocks of a chunk from a compressed buffer.
   * @param {Uint8Array} compressed - raw compressed bytes (all blocks concatenated)
   * @returns {Promise<Uint8Array>} concatenated decompressed data
   */
  async _decompressBlocks(compressed) {
    const parts = [];
    let off = 0;
    while (off + 4 <= compressed.byteLength) {
      // Block magic: 'CB' as uint16 LE
      const magic = compressed[off] | (compressed[off + 1] << 8);
      if (magic !== BLOCK_MAGIC) break;
      const bsize = compressed[off + 2] | (compressed[off + 3] << 8);
      if (off + bsize > compressed.byteLength) break;
      // Compressed payload is between header(4B) and raw_len trailer(2B)
      const payload = compressed.subarray(off + 4, off + bsize - 2);
      const decompressed = await inflateRaw(payload);
      parts.push(decompressed instanceof Uint8Array ? decompressed : new Uint8Array(decompressed));
      off += bsize;
    }
    // Concatenate
    const total = parts.reduce((s, p) => s + p.byteLength, 0);
    const out = new Uint8Array(total);
    let pos = 0;
    for (const p of parts) {
      out.set(p, pos);
      pos += p.byteLength;
    }
    return out;
  }

  // ── Fetch ────────────────────────────────────────────────────────────────
  /**
   * Fetch all records for a given dimension.
   * @param {string|string[]} dim - dimension value(s), e.g. 'chr1' or ['cell1','chr1']
   * @returns {Promise<Array<Array>>} array of records (each record = array of column values)
   */
  async fetch(dim) {
    const dimKey = Array.isArray(dim) ? dim.join('\t') : dim;
    const info = this.chunkIndex.get(dimKey);
    if (!info) throw new Error(`Unknown dimension: ${dimKey}`);

    // Read all compressed blocks in one request
    this._handle.seek(info.start + 10); // skip chunk header (CC 2B + size 8B)
    const compressedSize = info.size - 10;
    const compressed = await this._handle.read(compressedSize);

    // Decompress all blocks
    const decompressed = await this._decompressBlocks(compressed);

    // Unpack records and convert BigInt → Number
    const records = unpackRecords(decompressed, this._fields, this._unitSize);
    return records.map(rec => this._decodeRecord(rec));
  }

  /**
   * Fetch raw decompressed bytes for a chunk (for typed array processing).
   * @param {string|string[]} dim
   * @returns {Promise<Uint8Array>}
   */
  async fetchChunkBytes(dim) {
    const dimKey = Array.isArray(dim) ? dim.join('\t') : dim;
    const info = this.chunkIndex.get(dimKey);
    if (!info) throw new Error(`Unknown dimension: ${dimKey}`);

    this._handle.seek(info.start + 10);
    const compressed = await this._handle.read(info.size - 10);
    return this._decompressBlocks(compressed);
  }

  // ── Query (binary search) ─────────────────────────────────────────────────
  /**
   * Query records in a genomic region [start, end] within a dimension.
   * Uses binary search on blocks (O(log N) decompressions) to find the
   * target block, then scans forward.
   *
   * @param {string|string[]} dim - dimension, e.g. 'chr1'
   * @param {number} start - start position (inclusive)
   * @param {number} end   - end position (inclusive)
   * @param {number} [queryCol=0] - column index to query on (default: first column)
   * @returns {Promise<Array<Array>>} matching records
   */
  async query(dim, start, end, queryCol = 0) {
    const dimKey = Array.isArray(dim) ? dim.join('\t') : dim;
    const tail = await this._loadChunkTail(dimKey);
    const vos = tail.blockVOs;
    const nblocks = vos.length;
    if (nblocks === 0) return [];

    // Fast path: if this file has a sort_col index and the caller is
    // querying that column, bisect the in-memory first_coords array — no
    // extra block decompressions needed to locate the start block.
    let startBlockIdx;
    if (tail.firstCoords !== null && queryCol === this.header.sortCol) {
      startBlockIdx = _bisectRight(tail.firstCoords, start) - 1;
      if (startBlockIdx < 0) startBlockIdx = 0;
    } else {
      // Fallback: probe block first-values via decompression (O(log N) blocks).
      startBlockIdx = await this._bisectBlockIndex(vos, start, queryCol, 0, nblocks);
    }

    // Read and decompress from startBlockIdx onward to collect matching records
    const results = [];
    let blockIdx = startBlockIdx;

    while (blockIdx < nblocks) {
      const decompressed = await this._readOneBlock(vos[blockIdx]);
      if (!decompressed) break;

      const records = unpackRecords(decompressed, this._fields, this._unitSize);
      let foundAny = false;

      for (const rec of records) {
        const val = _toNumber(rec[queryCol]);
        if (val > end) return results; // past the end, done
        if (val >= start) {
          results.push(this._decodeRecord(rec));
          foundAny = true;
        }
      }

      // If the first record of this block is already past 'end', stop
      if (records.length > 0 && _toNumber(records[0][queryCol]) > end) break;

      blockIdx++;
    }

    return results;
  }

  /**
   * Binary search on blocks to find the last block whose first record <= target.
   * Only decompresses O(log N) blocks.
   */
  async _bisectBlockIndex(vos, target, col, lo, hi) {
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      const val = await this._readBlockFirstValue(vos[mid], col);
      if (val === null || val <= target) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return Math.max(lo - 1, 0);
  }

  /** Read and decompress one block, return decompressed Uint8Array. */
  async _readOneBlock(virtualOffset) {
    const blockStart = Math.floor(virtualOffset / 65536); // vo >> 16
    this._handle.seek(blockStart);
    // Read block header (4 bytes): magic(2B) + bsize(2B)
    const hdr = await this._handle.read(4);
    if (hdr.byteLength < 4) return null;
    const magic = hdr[0] | (hdr[1] << 8);
    if (magic !== BLOCK_MAGIC) return null;
    const bsize = hdr[2] | (hdr[3] << 8);
    // Read the rest of the block (compressed payload + raw_len trailer)
    const rest = await this._handle.read(bsize - 4);
    if (rest.byteLength < bsize - 4) return null;
    // Payload is bytes [0 .. bsize-6), raw_len trailer is last 2 bytes
    const payload = rest.subarray(0, rest.byteLength - 2);
    return inflateRaw(payload);
  }

  /** Read the first record's column value from a block at the given virtual offset. */
  async _readBlockFirstValue(virtualOffset, col) {
    const blockStart = Math.floor(virtualOffset / 65536);
    const within = virtualOffset % 65536;
    const block = await this._readOneBlock(virtualOffset);
    if (!block || within + this._unitSize > block.byteLength) return null;
    // Read one record starting at 'within'
    const dv = new DataView(block.buffer, block.byteOffset + within, this._unitSize);
    let off = 0;
    for (let j = 0; j <= col; j++) {
      if (j === col) return _toNumber(this._fields[j].read(dv, off));
      off += this._fields[j].size;
    }
    return null;
  }

  /** Decode a record: convert BigInt to Number for numeric fields. */
  _decodeRecord(rec) {
    return rec.map((v, i) => {
      if (this._strColMask[i]) return v; // already a string
      return _toNumber(v);
    });
  }

  close() {
    this._handle.close();
  }
}

/** Convert BigInt to Number if needed. */
function _toNumber(v) {
  return typeof v === 'bigint' ? Number(v) : v;
}

/** bisect_right on a sorted numeric array — returns first idx where arr[idx] > target. */
function _bisectRight(arr, target) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid] <= target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// ─── Exports ─────────────────────────────────────────────────────────────────
// ES module export (works in browsers with <script type="module"> and bundlers)
export { CzReader, RemoteFile, unpackRecords, parseFormats };
