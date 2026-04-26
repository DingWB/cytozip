# cz Format v2 — Design Specification

**Status**: Design draft — ready for implementation.
**Backwards compatibility**: NONE. v2 readers will not read v1 files; a
one-shot migration tool (`czip migrate-v1-to-v2`) will be provided.

---

## 1. Goals

| Goal | Mechanism |
|---|---|
| Reduce file size 5-15% on real workloads | Larger blocks (≤16 MB) → better DEFLATE/zstd window |
| Reduce decompress call count 15× | Same: ~33 k blocks → ~2 k |
| Codec choice per file | New `codec` byte in header (deflate / zstd / none) |
| Point query 100× faster | Block-level mini-index (record-id → within-block offset) |
| Cleaner mmap path | All blocks must be record-aligned |
| Faster integrity check | xxhash64 instead of CRC32 (DEFLATE-derived) |
| Cell-sparse data first-class | Sparse-layout schema option in header |
| Cloud-ready | Already covered by tail-CZIX + range reads (no v2 change needed) |

## 2. File magic & version

```
Bytes  0-3 : b"CZ\x02\x00"          # bumped magic; old was b"CZ\x01\x00"
Bytes  4-11: total_size  uint64     # whole-file size including chunk index
Bytes 12-15: header_crc32           # of bytes 16..header_end
```

Reading the first 16 bytes is enough to refuse v1 files cleanly.

## 3. Header (after first 16 bytes)

```
codec        : uint8     # 0=deflate, 1=zstd, 2=none, 3..255=reserved
codec_level  : uint8     # algorithm-specific (deflate L1-L12, zstd L1-L22)
flags        : uint16    # bit 0: sparse layout
                         # bit 1: blocks always record-aligned (set in v2)
                         # bit 2: block-mini-index present
                         # bit 3..15: reserved
xxhash_seed  : uint64    # for whole-file integrity, non-secret
n_columns    : uint8
columns      : repeat n_columns of:
                  name_len  uint8
                  name      bytes[name_len]
                  fmt_len   uint8
                  fmt       bytes[fmt_len]    # struct fmt e.g. "B", "Q", "3s"
sort_col_idx : uint8     # 0xFF = none
delta_col_mask: uint32   # bit i set ⇒ column i is delta-encoded
chunk_dims   : repeat n_dims of (name_len uint8, name bytes)
message_len  : uint32
message      : bytes[message_len]
```

## 4. Block structure (v2)

Each block:
```
magic         : 2 bytes "CB"
block_size    : uint32   # whole block size including header (was uint16)
n_records     : uint32   # ALWAYS record-aligned in v2; this is the # of
                         # logical records in the decompressed payload
mini_index_len: uint16   # 0 if no mini-index, else size of the
                         # offset list following decompressed payload
xxhash_low32  : uint32   # cheap per-block check; full 64 not needed
codec_payload : bytes[block_size - 16]
```

`block_size` upper bound: 16 MiB (`0x01000000`). 16 MB chosen because:
- Compresses 5-10% better than 64 KB on biological repeats
- Still fits in L3 cache on modern x86 (decompression locality)
- 16 MB × 2 k blocks = 32 GB max chunk; far above realistic single-cell sizes

`n_records` lets vectorised readers `np.frombuffer(view, count=n_records)`
without any byte arithmetic.

`mini_index_len > 0` ⇒ at the tail of the **decompressed** payload, after
`n_records * unit_size` bytes, there are `mini_index_len / 4` `uint32`
checkpoints. Checkpoint `i` is the byte offset (within decompressed
payload) of record `i * STRIDE`. Default STRIDE = 256.

## 5. Virtual offset (v2)

v1 packed: 48-bit file pos + 16-bit within-block offset (max 64 KB block).

v2 packs:  56-bit file pos + 24-bit within-block record index (NOT byte
offset — record index, since blocks are always record-aligned). 24-bit
record index fits 16 M records per block, matching block_size cap.

Encoded as one `uint80` ⇒ 10 bytes per voffset (was 8). The block-level
voffset table grows by 25%, but it is small relative to the data.

## 6. Sparse layout (optional, header flags bit 0)

When set, **all** records carry an implicit leading `int32 site_idx`
field that is delta-encoded (so contiguous filled regions are tiny).
The struct `formats` excludes `site_idx`; it is only in the layout. A
"missing" site has no record. This is what enables 10× wins on ATAC /
super-sparse modalities while still being losslessly invertible into
the dense layout when needed.

## 7. Chunk index (CZIX) — unchanged structurally

Same tail-CZIX magic ``b"CZIX"``. Index entries are 4×uint64 each
(start_offset, size, n_blocks, first_voffset). Reading the index is
still a single small range request after a tail seek.

## 8. Implementation plan

| # | File | What to do | Est LOC | Risk |
|---|------|------------|---------|------|
| 1 | `cytozip/cz_format_v2.py` (NEW) | Parsers/serialisers for v2 header + block + mini-index | 400 | low |
| 2 | `cytozip/cz_accel.pyx` | Add `_c_inflate_v2` (uint32 block_size, mini-index aware), `_c_get_records_by_ids_v2` (use mini-index for log-N point query) | 300 | high (must be correct + fast) |
| 3 | `cytozip/cz.py` | Branch Reader/Writer on magic byte 3; new codec dispatch (zstd via `import zstandard`); v2 voffset packing | 350 | medium |
| 4 | `cytozip/cz_reader.mjs` | Mirror v2 changes in JS reader | 200 | medium |
| 5 | `cytozip/migrate.py` (NEW) | One-shot v1 → v2 migrator (`czip migrate-v1-to-v2 a.cz a_v2.cz [--codec zstd]`) | 150 | low |
| 6 | `tests/test_v2.py` (NEW) | Round-trip tests, mini-index correctness, large-block stress, codec coverage | 300 | — |
| 7 | `docs/source/format_v2.md` | Polished public spec (this file is the design draft) | — | — |

**Total**: ~1500 LOC + tests. Estimated 3-5 working days.

## 9. Order of implementation

```
1. format_v2.py            (foundation, no Cython needed)
2. cz.py Reader v2 path    (read first; tests use hand-crafted files)
3. cz.py Writer v2 path
4. cz_accel.pyx v2 path    (only after pure-Python works)
5. JS reader
6. Migration tool + CLI
7. Comprehensive tests
8. Docs polish + release notes
```

## 10. Deferred / out of scope

- **Encryption**: no, do this with filesystem-level encryption.
- **Split files**: no, multi-file cz violates the "single artifact" model.
- **Append/edit in place**: no, cz remains write-once-read-many.
- **In-block parallel decompression**: empirically slower than current
  Cython libdeflate (verified Apr 2026 on 1.1 B-row mc/cov chunk).
  Cross-chunk parallelism is what users need; already provided via
  :meth:`Reader.chunks2numpy(n_workers=...)`.
