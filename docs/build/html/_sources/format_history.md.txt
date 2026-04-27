# cz Format — Change History

Historical record of on-disk format changes. The current format is
documented in [dev.md](../../dev.md) and
[developer_guide.md](developer_guide.md). The package version is still
`0.3` — the format itself is not separately versioned, so changes
listed below are tracked against package releases / commits rather than
a numbered format spec.

> Older `.cz` files written before any of these changes are no longer
> readable. Re-generate from source data when needed.

---

## Larger blocks (256 KiB − 1)

Block size cap was raised from 65 535 raw bytes to `(1 << 18) − 1`
(256 KiB − 1).

- `block_size` field grew from `<H` (2 B) to `<I` (4 B); `raw_len`
  grew the same way → block header overhead is now 10 B instead of 6 B.
- DEFLATE's 32 KiB sliding window saturates compression around 256 KiB,
  so file size shrinks 5–10 % on real biological data with no codec
  change.
- ~10× fewer blocks → ~10× fewer `inflate` calls per chunk read.

## Wider virtual offsets (44 + 20)

`uint64` virtual offsets were re-split from 48 + 16 to **44 + 20**:

- 44 bits file offset (16 TiB addressable — still ample).
- 20 bits within-block byte offset (matches the new 256 KiB block cap).

## Delta-encoded integer columns

Strictly-monotonic integer columns (typically genomic positions) can be
stored as first-order differences within each block (`_ENC_DELTA`).
DEFLATE compresses the delta stream to a small fraction of the raw
column. Decoding is one numpy `cumsum` per delta column per block.

The delta-column mask is recorded in the chunk header so old files
without the flag are not silently mis-read — but, as noted above,
pre-change files are not readable.

## Tail chunk index (CZIX)

A tail-appended chunk index lets HTTP / remote readers locate any chunk
in O(1) without scanning the whole file. Index entries store
`(start_offset, size, n_blocks, first_voffset)` per chunk; the index
itself is found via a single small range request near EOF.

---

## Ideas explored but **not** adopted

These were considered during earlier design rounds and intentionally
left out — recording here so they are not re-proposed without new
evidence:

- **Larger blocks (e.g. 16 MiB)** — beyond ~256 KiB the DEFLATE window
  no longer helps and per-block point-query latency degrades. 256 KiB
  was the empirical sweet spot.
- **Per-block mini record-id index** — point queries are already fast
  enough via the per-block first-coord array + a single inflate; the
  extra complexity wasn't justified.
- **Switching the integrity hash from CRC32 to xxhash64** — CRC32 is
  free as a side-effect of DEFLATE on most paths; the upgrade had no
  measurable benefit.
- **Sparse-layout schema flag** — single-cell sparsity is handled well
  enough by chunk-level partitioning + delta-encoded positions.
- **Alternative codecs (zstd / none) selected per file** — DEFLATE via
  libdeflate already saturates the I/O budget; carrying a codec byte
  added compatibility surface without a clear win.
- **Encryption / multi-file artefacts** — out of scope; use
  filesystem-level encryption and keep the "single artefact" model.
