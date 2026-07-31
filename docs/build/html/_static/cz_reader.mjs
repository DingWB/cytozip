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
// Block on-disk layout (must match Python cz._py_load_bcz_block):
//   [magic 2B][block_size uint32 4B][deflate (block_size-10)B][data_len uint32 4B]
// block_size is the TOTAL block size (header + deflate payload + trailer).
const BLOCK_HEADER_BYTES = 6;   // magic(2) + block_size(4)
const BLOCK_TRAILER_BYTES = 4;  // data_len(4)
// Virtual offset layout (must match Python cz._VO_OFFSET_BITS): high bits hold
// the physical file offset of the compressed block, low VO_OFFSET_BITS hold the
// byte offset within the decompressed block data.
const VO_OFFSET_BITS = 20;
const VO_BLOCK_DIVISOR = 2 ** VO_OFFSET_BITS; // 1048576
const VO_OFFSET_MASK = VO_BLOCK_DIVISOR - 1;  // 0xFFFFF

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
   * @param {number} [opts.cacheSize=2097152]  Read-ahead window (bytes) used
   *        for large sequential reads (e.g. whole-chunk fetch()).
   * @param {number} [opts.randomReadAhead=262144]  Small read-ahead window
   *        (bytes) used for random access (query / metadata reads). Keeping
   *        this near one block avoids pulling megabytes for a point query.
   * @param {object} [opts.fetchOptions={}]    Extra options passed to fetch()
   *        (e.g. { headers: {...}, credentials: 'include' }).
   */
  constructor(url, opts = {}) {
    this.url = url;
    this._pos = 0;
    this._size = -1;
    this._cacheSize = opts.cacheSize ?? 2 * 1024 * 1024;
    this._randomReadAhead = opts.randomReadAhead ?? 256 * 1024;
    this._fetchOpts = opts.fetchOptions ?? {};
    this._cacheStart = -1;
    this._cacheEnd = -1;
    this._cacheData = null;
  }

  /** Small read-ahead window for random-access (query) reads. */
  get randomReadAhead() { return this._randomReadAhead; }

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
   *
   * @param {number} size
   * @param {object} [opts]
   * @param {number} [opts.readAhead]  Override the read-ahead window for this
   *        read (bytes). Defaults to the large sequential window (cacheSize).
   *        Pass `randomReadAhead` for random-access reads.
   */
  async read(size, opts = {}) {
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

    // Fetch with read-ahead (window is per-read: large for sequential whole-chunk
    // reads, small for random-access queries).
    const window = opts.readAhead != null ? opts.readAhead : this._cacheSize;
    const fetchEnd = Math.min(start + Math.max(size, window), this._size);
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
    // One-time metadata reads: small read-ahead (don't pull megabytes to open).
    const ra = { readAhead: this._handle.randomReadAhead };
    this._handle.seek(0);
    // Read first 200 bytes (enough for most headers)
    let buf = await this._handle.read(200, ra);
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
      buf = await this._handle.read(Math.max(400, off + msgLen + 200), ra);
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
        buf = await this._handle.read(off + 200, ra);
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

    // Per-column storage-encoding table (must match Python cz):
    //   n_enc (1B) then n_enc pairs of (col_idx 1B, enc_code 1B).
    // enc_code 0x00 = RAW (default), 0x01 = DELTA (in-block cumulative diffs
    // on an integer column). Columns not listed default to RAW.
    const ENC_DELTA = 0x01;
    const nEnc = buf[off]; off += 1;
    const deltaCols = [];
    for (let i = 0; i < nEnc; i++) {
      const colIdx = buf[off]; off += 1;
      const encCode = buf[off]; off += 1;
      if (encCode === ENC_DELTA) deltaCols.push(colIdx);
    }

    // chunk_dims (the chunk_key names)
    const nChunkKeys = buf[off]; off += 1;
    const chunk_keys = [];
    for (let i = 0; i < nChunkKeys; i++) {
      const dLen = buf[off]; off += 1;
      chunk_keys.push(new TextDecoder().decode(buf.subarray(off, off + dLen)));
      off += dLen;
    }

    this.header = {
      magic, version, totalSize, message,
      formats, columns, sortCol, deltaCols, chunk_keys,
      headerSize: off,
    };
    this._deltaCols = deltaCols;

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
    const ra = { readAhead: this._handle.randomReadAhead };
    f.seek(fileSize - 36);
    const tail = await f.read(36, ra);
    const tailDv = new DataView(tail.buffer, tail.byteOffset, tail.byteLength);
    const indexOffset = Number(tailDv.getBigUint64(0, true));

    if (indexOffset === 0 || indexOffset >= fileSize) {
      this.chunkIndex = new Map();
      return;
    }

    // Read the entire chunk index
    f.seek(indexOffset);
    const idxBuf = await f.read(fileSize - 28 - indexOffset, ra);
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
    const nChunkKeys = this.header.chunk_keys.length;
    const index = new Map();

    for (let c = 0; c < nChunks; c++) {
      const dims = [];
      for (let d = 0; d < nChunkKeys; d++) {
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

  /** List all chunk_key keys. */
  get chunkKeys() {
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
  async _loadChunkTail(chunkKey) {
    if (this._chunkTailCache.has(chunkKey)) return this._chunkTailCache.get(chunkKey);

    const info = this.chunkIndex.get(chunkKey);
    if (!info) throw new Error(`Unknown chunk_key: ${chunkKey}`);

    // Chunk tail sits right after the compressed blocks.
    // Read from chunk_start + chunk_size (tail offset).
    const tailOffset = info.start + info.size;
    this._handle.seek(tailOffset);

    // tail header: data_len(8B) + n_blocks(8B) + virtual_offsets(N*8B)
    //   + [first_coords(N * sort_col_size) if sort_col enabled]
    //   + chunk_key_values
    const sortSize = this._sortColSize || 0;
    const tailSize = 16 + info.nblocks * (8 + sortSize) + 256; // +256 for dim strings
    const tailBuf = await this._handle.read(tailSize, { readAhead: this._handle.randomReadAhead });
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
    this._chunkTailCache.set(chunkKey, result);
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
    const dv = new DataView(compressed.buffer, compressed.byteOffset, compressed.byteLength);
    while (off + BLOCK_HEADER_BYTES <= compressed.byteLength) {
      // Block magic: 'CB' as uint16 LE
      const magic = compressed[off] | (compressed[off + 1] << 8);
      if (magic !== BLOCK_MAGIC) break;
      const bsize = dv.getUint32(off + 2, true);
      if (bsize < BLOCK_HEADER_BYTES + BLOCK_TRAILER_BYTES || off + bsize > compressed.byteLength) break;
      // Deflate payload sits between the 6B header and the 4B data_len trailer.
      const payload = compressed.subarray(off + BLOCK_HEADER_BYTES, off + bsize - BLOCK_TRAILER_BYTES);
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

  /**
   * Decompress all blocks of a chunk, returning one Uint8Array per block
   * (boundaries preserved so per-block DELTA decoding works).
   * @param {Uint8Array} compressed
   * @returns {Promise<Uint8Array[]>}
   */
  async _decompressBlocksList(compressed) {
    const parts = [];
    let off = 0;
    const dv = new DataView(compressed.buffer, compressed.byteOffset, compressed.byteLength);
    while (off + BLOCK_HEADER_BYTES <= compressed.byteLength) {
      const magic = compressed[off] | (compressed[off + 1] << 8);
      if (magic !== BLOCK_MAGIC) break;
      const bsize = dv.getUint32(off + 2, true);
      if (bsize < BLOCK_HEADER_BYTES + BLOCK_TRAILER_BYTES || off + bsize > compressed.byteLength) break;
      const payload = compressed.subarray(off + BLOCK_HEADER_BYTES, off + bsize - BLOCK_TRAILER_BYTES);
      const decompressed = await inflateRaw(payload);
      parts.push(decompressed instanceof Uint8Array ? decompressed : new Uint8Array(decompressed));
      off += bsize;
    }
    return parts;
  }

  // ── Fetch ────────────────────────────────────────────────────────────────
  /**
   * Fetch all records for a given chunk_key.
   * @param {string|string[]} dim - chunk_key value(s), e.g. 'chr1' or ['cell1','chr1']
   * @returns {Promise<Array<Array>>} array of records (each record = array of column values)
   */
  async fetch(dim) {
    const chunkKey = Array.isArray(dim) ? dim.join('\t') : dim;
    const info = this.chunkIndex.get(chunkKey);
    if (!info) throw new Error(`Unknown chunk_key: ${chunkKey}`);

    // Read all compressed blocks in one request
    this._handle.seek(info.start + 10); // skip chunk header (CC 2B + size 8B)
    const compressedSize = info.size - 10;
    const compressed = await this._handle.read(compressedSize);

    // Decompress block by block so per-block DELTA decoding stays correct.
    const blocks = await this._decompressBlocksList(compressed);
    const out = [];
    for (const blk of blocks) {
      const records = unpackRecords(blk, this._fields, this._unitSize);
      this._applyDelta(records);
      for (const rec of records) out.push(this._decodeRecord(rec));
    }
    return out;
  }

  /**
   * Fetch raw decompressed bytes for a chunk (for typed array processing).
   * @param {string|string[]} dim
   * @returns {Promise<Uint8Array>}
   */
  async fetchChunkBytes(dim) {
    const chunkKey = Array.isArray(dim) ? dim.join('\t') : dim;
    const info = this.chunkIndex.get(chunkKey);
    if (!info) throw new Error(`Unknown chunk_key: ${chunkKey}`);

    this._handle.seek(info.start + 10);
    const compressed = await this._handle.read(info.size - 10);
    return this._decompressBlocks(compressed);
  }

  // ── Query (binary search) ─────────────────────────────────────────────────
  /**
   * Query records in a genomic region [start, end] within a chunk_key.
   *
   * Read-ahead strategy:
   *   - Single-block (point / small region) queries use a small read-ahead
   *     window (~one block), so a random jump costs only ~one block of bytes.
   *   - Multi-block queries (large regions) bulk-read the whole contiguous
   *     block span in a single Range request (large sequential read).
   *
   * @param {string|string[]} dim - chunk_key, e.g. 'chr1'
   * @param {number} start - start position (inclusive)
   * @param {number} end   - end position (inclusive)
   * @param {number} [queryCol=0] - column index to query on (default: first column)
   * @returns {Promise<Array<Array>>} matching records
   */
  async query(dim, start, end, queryCol = 0) {
    const chunkKey = Array.isArray(dim) ? dim.join('\t') : dim;
    const tail = await this._loadChunkTail(chunkKey);
    const vos = tail.blockVOs;
    const nblocks = vos.length;
    if (nblocks === 0) return [];

    // Fast path: sort_col index present and querying that column — bisect the
    // in-memory first_coords array for BOTH ends, then read the contiguous
    // block span [startBlockIdx, endBlockIdx] in one request.
    if (tail.firstCoords !== null && queryCol === this.header.sortCol) {
      let startBlockIdx = _bisectRight(tail.firstCoords, start) - 1;
      if (startBlockIdx < 0) startBlockIdx = 0;
      let endBlockIdx = _bisectRight(tail.firstCoords, end) - 1;
      if (endBlockIdx < startBlockIdx) endBlockIdx = startBlockIdx;

      // Byte range covering blocks [startBlockIdx, endBlockIdx]. Blocks are
      // contiguous, so block j ends where block j+1 begins (or at the chunk
      // tail offset for the last block).
      const firstByte = Math.floor(vos[startBlockIdx] / VO_BLOCK_DIVISOR);
      const lastByte = (endBlockIdx + 1 < nblocks)
        ? Math.floor(vos[endBlockIdx + 1] / VO_BLOCK_DIVISOR)
        : (tail.start + tail.size);
      const spanBytes = lastByte - firstByte;
      const nSpan = endBlockIdx - startBlockIdx + 1;

      // One block → small read-ahead; many blocks → read the whole span at once.
      const readAhead = nSpan > 1 ? spanBytes : this._handle.randomReadAhead;
      this._handle.seek(firstByte);
      const buf = await this._handle.read(spanBytes, { readAhead });
      const blocks = await this._decompressBlocksList(buf);

      const results = [];
      for (const blk of blocks) {
        const records = unpackRecords(blk, this._fields, this._unitSize);
        this._applyDelta(records); // reconstruct absolute values for DELTA columns
        if (records.length > 0 && _toNumber(records[0][queryCol]) > end) break;
        for (const rec of records) {
          const val = _toNumber(rec[queryCol]);
          if (val > end) return results;
          if (val >= start) results.push(this._decodeRecord(rec));
        }
      }
      return results;
    }

    // Fallback: no first_coords index — probe block first-values via
    // decompression (O(log N) blocks), then scan forward block by block.
    const startBlockIdx = await this._bisectBlockIndex(vos, start, queryCol, 0, nblocks);
    const results = [];
    let blockIdx = startBlockIdx;

    while (blockIdx < nblocks) {
      const decompressed = await this._readOneBlock(vos[blockIdx]);
      if (!decompressed) break;

      const records = unpackRecords(decompressed, this._fields, this._unitSize);
      this._applyDelta(records); // reconstruct absolute values for DELTA columns

      for (const rec of records) {
        const val = _toNumber(rec[queryCol]);
        if (val > end) return results; // past the end, done
        if (val >= start) {
          results.push(this._decodeRecord(rec));
        }
      }

      // If the first record of this block is already past 'end', stop
      if (records.length > 0 && _toNumber(records[0][queryCol]) > end) break;

      blockIdx++;
    }

    return results;
  }

  /**
   * Uniform per-block decompressed byte size for a chunk (all blocks except
   * the last are packed to the same size). Learned by decompressing block 0
   * once and cached on the chunk tail. Used for row-index ↔ byte-offset math.
   * @param {string} chunkKey
   * @returns {Promise<number>}
   */
  async _fullBlockBytes(chunkKey) {
    const tail = await this._loadChunkTail(chunkKey);
    if (tail._fullBlockBytes != null) return tail._fullBlockBytes;
    if (tail.nblocks === 0) { tail._fullBlockBytes = 0; return 0; }
    const b0 = await this._readOneBlock(tail.blockVOs[0]);
    tail._fullBlockBytes = b0 ? b0.byteLength : 0;
    return tail._fullBlockBytes;
  }

  /**
   * Query a genomic region AND return the global row-index range within the
   * chunk (needed to join a coordinate-less value file against this
   * reference). Requires a sort_col index on the query column and a
   * record-aligned (DELTA) layout — i.e. a reference .cz file.
   *
   * @param {string|string[]} dim - chunk_key, e.g. 'chr1'
   * @param {number} start - inclusive start position
   * @param {number} end   - inclusive end position
   * @param {number} [queryCol] - defaults to the file's sort_col
   * @returns {Promise<{rowStart:number, rowEnd:number, positions:number[], records:Array<Array>}>}
   *          rowStart inclusive, rowEnd exclusive; positions[i] ↔ row rowStart+i.
   */
  async queryWithIndex(dim, start, end, queryCol = this.header.sortCol) {
    const chunkKey = Array.isArray(dim) ? dim.join('\t') : dim;
    const tail = await this._loadChunkTail(chunkKey);
    const vos = tail.blockVOs;
    const nblocks = vos.length;
    const empty = { rowStart: 0, rowEnd: 0, positions: [], records: [] };
    if (nblocks === 0) return empty;
    if (tail.firstCoords === null || queryCol !== this.header.sortCol) {
      throw new Error('queryWithIndex requires a sort_col index on the query column');
    }

    const fullBytes = await this._fullBlockBytes(chunkKey);
    const recPerBlock = Math.floor(fullBytes / this._unitSize);

    let startBlk = _bisectRight(tail.firstCoords, start) - 1;
    if (startBlk < 0) startBlk = 0;
    let endBlk = _bisectRight(tail.firstCoords, end) - 1;
    if (endBlk < startBlk) endBlk = startBlk;

    const firstByte = Math.floor(vos[startBlk] / VO_BLOCK_DIVISOR);
    const lastByte = (endBlk + 1 < nblocks)
      ? Math.floor(vos[endBlk + 1] / VO_BLOCK_DIVISOR)
      : (tail.start + tail.size);
    const spanBytes = lastByte - firstByte;
    this._handle.seek(firstByte);
    const readAhead = endBlk > startBlk ? spanBytes : this._handle.randomReadAhead;
    const buf = await this._handle.read(spanBytes, { readAhead });
    const blocks = await this._decompressBlocksList(buf);

    const positions = [];
    const records = [];
    let rowStart = -1;
    let rowEnd = -1;
    for (let jj = 0; jj < blocks.length; jj++) {
      const base = (startBlk + jj) * recPerBlock;
      const recs = unpackRecords(blocks[jj], this._fields, this._unitSize);
      this._applyDelta(recs);
      for (let li = 0; li < recs.length; li++) {
        const val = _toNumber(recs[li][queryCol]);
        if (val > end) {
          return {
            rowStart: rowStart < 0 ? 0 : rowStart,
            rowEnd: rowEnd < 0 ? 0 : rowEnd + 1,
            positions, records,
          };
        }
        if (val >= start) {
          const gr = base + li;
          if (rowStart < 0) rowStart = gr;
          rowEnd = gr;
          positions.push(val);
          records.push(this._decodeRecord(recs[li]));
        }
      }
    }
    return {
      rowStart: rowStart < 0 ? 0 : rowStart,
      rowEnd: rowEnd < 0 ? 0 : rowEnd + 1,
      positions, records,
    };
  }

  /**
   * Fetch records for a contiguous global row-index range [rowStart, rowEnd)
   * within a chunk. Designed for coordinate-less value files (e.g. mc/cov)
   * whose rows align 1:1 with a reference. Only the covering blocks are
   * fetched/decompressed. Assumes a non-DELTA (RAW) value layout.
   *
   * @param {string|string[]} dim - chunk_key, e.g. 'chr1'
   * @param {number} rowStart - inclusive
   * @param {number} rowEnd   - exclusive
   * @returns {Promise<Array<Array>>} records for rows [rowStart, rowEnd)
   */
  async getRowValues(dim, rowStart, rowEnd) {
    const chunkKey = Array.isArray(dim) ? dim.join('\t') : dim;
    if (rowEnd <= rowStart) return [];
    const tail = await this._loadChunkTail(chunkKey);
    const vos = tail.blockVOs;
    const nblocks = vos.length;
    if (nblocks === 0) return [];

    const fullBytes = await this._fullBlockBytes(chunkKey);
    const unit = this._unitSize;
    const byteStart = rowStart * unit;
    const byteEnd = rowEnd * unit;

    let blockA = Math.floor(byteStart / fullBytes);
    let blockB = Math.floor((byteEnd - 1) / fullBytes);
    if (blockA < 0) blockA = 0;
    if (blockB >= nblocks) blockB = nblocks - 1;
    if (blockB < blockA) blockB = blockA;

    const firstByte = Math.floor(vos[blockA] / VO_BLOCK_DIVISOR);
    const lastByte = (blockB + 1 < nblocks)
      ? Math.floor(vos[blockB + 1] / VO_BLOCK_DIVISOR)
      : (tail.start + tail.size);
    const spanBytes = lastByte - firstByte;
    this._handle.seek(firstByte);
    const readAhead = blockB > blockA ? spanBytes : this._handle.randomReadAhead;
    const buf = await this._handle.read(spanBytes, { readAhead });
    const blocks = await this._decompressBlocksList(buf);

    // Concatenate the covering blocks (decompressed bytes are contiguous).
    const total = blocks.reduce((s, b) => s + b.byteLength, 0);
    const concat = new Uint8Array(total);
    let p = 0;
    for (const b of blocks) { concat.set(b, p); p += b.byteLength; }

    const localStart = byteStart - blockA * fullBytes;
    const localEnd = Math.min(byteEnd - blockA * fullBytes, concat.byteLength);
    if (localStart >= localEnd) return [];
    const slice = concat.subarray(localStart, localEnd);
    const recs = unpackRecords(slice, this._fields, this._unitSize);
    return recs.map(rec => this._decodeRecord(rec));
  }

  /**
   * Column-oriented variant of getRowValues: returns each column as a contiguous
   * typed array (numeric columns as Float64Array, string columns as Array),
   * avoiding the per-row `[a, b, …]` allocations that stress the GC and hurt
   * cache locality in hot drawing loops. Assumes a non-DELTA (RAW) value layout.
   *
   * @param {string|string[]} dim - chunk_key, e.g. 'chr1'
   * @param {number} rowStart - inclusive
   * @param {number} rowEnd   - exclusive
   * @returns {Promise<{n:number, columns:Array<Float64Array|Array>}>}
   */
  async getRowValuesColumns(dim, rowStart, rowEnd) {
    const fields = this._fields, nf = fields.length;
    const emptyCols = () => fields.map((f, j) => this._strColMask[j] ? [] : new Float64Array(0));
    const chunkKey = Array.isArray(dim) ? dim.join('\t') : dim;
    if (rowEnd <= rowStart) return { n: 0, columns: emptyCols() };
    const tail = await this._loadChunkTail(chunkKey);
    const vos = tail.blockVOs;
    const nblocks = vos.length;
    if (nblocks === 0) return { n: 0, columns: emptyCols() };

    const fullBytes = await this._fullBlockBytes(chunkKey);
    const unit = this._unitSize;
    const byteStart = rowStart * unit;
    const byteEnd = rowEnd * unit;

    let blockA = Math.floor(byteStart / fullBytes);
    let blockB = Math.floor((byteEnd - 1) / fullBytes);
    if (blockA < 0) blockA = 0;
    if (blockB >= nblocks) blockB = nblocks - 1;
    if (blockB < blockA) blockB = blockA;

    const firstByte = Math.floor(vos[blockA] / VO_BLOCK_DIVISOR);
    const lastByte = (blockB + 1 < nblocks)
      ? Math.floor(vos[blockB + 1] / VO_BLOCK_DIVISOR)
      : (tail.start + tail.size);
    const spanBytes = lastByte - firstByte;
    this._handle.seek(firstByte);
    const readAhead = blockB > blockA ? spanBytes : this._handle.randomReadAhead;
    const buf = await this._handle.read(spanBytes, { readAhead });
    const blocks = await this._decompressBlocksList(buf);

    const total = blocks.reduce((s, b) => s + b.byteLength, 0);
    const concat = new Uint8Array(total);
    let p = 0;
    for (const b of blocks) { concat.set(b, p); p += b.byteLength; }

    const localStart = byteStart - blockA * fullBytes;
    const localEnd = Math.min(byteEnd - blockA * fullBytes, concat.byteLength);
    if (localStart >= localEnd) return { n: 0, columns: emptyCols() };

    const n = Math.floor((localEnd - localStart) / unit);
    const dv = new DataView(concat.buffer, concat.byteOffset + localStart, n * unit);
    // Per-field byte offset within a record (computed once).
    const offs = new Array(nf);
    for (let j = 0, acc = 0; j < nf; j++) { offs[j] = acc; acc += fields[j].size; }
    const columns = new Array(nf);
    for (let j = 0; j < nf; j++) columns[j] = this._strColMask[j] ? new Array(n) : new Float64Array(n);
    for (let j = 0; j < nf; j++) {
      const read = fields[j].read, off = offs[j], colArr = columns[j], isStr = this._strColMask[j];
      for (let i = 0; i < n; i++) {
        const v = read(dv, i * unit + off);
        colArr[i] = isStr ? v : (typeof v === 'bigint' ? Number(v) : v);
      }
    }
    return { n, columns };
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
    const blockStart = Math.floor(virtualOffset / VO_BLOCK_DIVISOR); // vo >> 20
    this._handle.seek(blockStart);
    // Read block header (6 bytes): magic(2B) + block_size(uint32 4B).
    // Random access → small read-ahead (prefetch ~one block, not megabytes).
    const ra = { readAhead: this._handle.randomReadAhead };
    const hdr = await this._handle.read(BLOCK_HEADER_BYTES, ra);
    if (hdr.byteLength < BLOCK_HEADER_BYTES) return null;
    const magic = hdr[0] | (hdr[1] << 8);
    if (magic !== BLOCK_MAGIC) return null;
    const bsize = (hdr[2] | (hdr[3] << 8) | (hdr[4] << 16) | (hdr[5] << 24)) >>> 0;
    // Read the rest of the block (deflate payload + data_len trailer)
    const rest = await this._handle.read(bsize - BLOCK_HEADER_BYTES, ra);
    if (rest.byteLength < bsize - BLOCK_HEADER_BYTES) return null;
    // Payload excludes the trailing 4-byte data_len
    const payload = rest.subarray(0, rest.byteLength - BLOCK_TRAILER_BYTES);
    return inflateRaw(payload);
  }

  /** Read the first record's column value from a block at the given virtual offset. */
  async _readBlockFirstValue(virtualOffset, col) {
    const blockStart = Math.floor(virtualOffset / VO_BLOCK_DIVISOR);
    const within = virtualOffset % VO_BLOCK_DIVISOR;
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

  /**
   * Undo per-column DELTA encoding in place on a block's records.
   * For each delta column the stored values are within-block differences
   * (first record absolute, rest are diffs); a running cumulative sum
   * reconstructs the absolute values. Must be applied per block.
   * @param {Array<Array>} records - records of a single block (mutated in place)
   */
  _applyDelta(records) {
    if (!this._deltaCols || this._deltaCols.length === 0 || records.length < 2) return records;
    for (const col of this._deltaCols) {
      const isBig = typeof records[0][col] === 'bigint';
      let acc = records[0][col];
      for (let i = 1; i < records.length; i++) {
        acc = isBig ? acc + records[i][col] : acc + records[i][col];
        records[i][col] = acc;
      }
    }
    return records;
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
