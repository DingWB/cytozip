"""Measure sparsity and estimate RLE savings on existing per-cell cz files.

Schema: (mc:B, cov:B), chunk_dims=['chrom']. Each row corresponds to one
reference site; cov==0 means the cell has no coverage at that site.

For each cell, walk the existing cz blocks (cz already chunks at <=65535B
per block, so this matches the natural compression unit). For each block:
  - count zeros in mc / cov
  - measure raw bytes (= block size)
  - run-length encode each column independently and measure RLE bytes
  - DEFLATE both and compare

Usage:  python measure_sparsity.py [N_CELLS]
"""
import os
import sys
import glob
import zlib
import time
import numpy as np
import cytozip as cz


REC_DTYPE = np.dtype([('mc', np.uint8), ('cov', np.uint8)])


def rle_encode(arr):
    """Return (values, lengths_uint32) for the run-length encoding."""
    if arr.size == 0:
        return arr[:0], np.empty(0, dtype=np.uint32)
    diff = np.diff(arr)
    starts = np.concatenate(([0], np.nonzero(diff)[0] + 1))
    ends = np.concatenate((starts[1:], [arr.size]))
    return arr[starts], (ends - starts).astype(np.uint32)


def measure(cz_path, log=print):
    t0 = time.time()
    r = cz.Reader(cz_path)
    assert r.header['formats'] == ['B', 'B'], r.header['formats']
    n_total = 0
    n_zero_cov = 0
    n_zero_mc = 0
    raw_bytes_total = 0
    rle_bytes_total = 0
    raw_deflate_total = 0
    rle_deflate_total = 0

    keys = list(r.chunk_key2offset)
    n_blocks = 0
    for ki, key in enumerate(keys):
        r._load_chunk(r.chunk_key2offset[key], jump=False)
        r._load_block(start_offset=r._chunk_start_offset + 10)
        chunk_n = 0
        chunk_blocks = 0
        # Block boundaries are NOT necessarily record-aligned; carry the
        # leftover bytes of the last partial record across blocks.
        carry = b''
        while r._block_raw_length > 0:
            buf = carry + r._buffer
            n_full = len(buf) // REC_DTYPE.itemsize
            usable = n_full * REC_DTYPE.itemsize
            carry = buf[usable:]
            if n_full > 0:
                arr = np.frombuffer(buf[:usable], dtype=REC_DTYPE,
                                    count=n_full)
                mc = arr['mc']
                cov = arr['cov']
                n = arr.size
                n_total += n
                chunk_n += n
                n_zero_cov += int((cov == 0).sum())
                n_zero_mc += int((mc == 0).sum())

                # raw payload (mc bytes + cov bytes, column-tiled)
                mc_b = mc.tobytes()
                cov_b = cov.tobytes()
                raw_block = mc_b + cov_b
                raw_bytes_total += len(raw_block)
                raw_deflate_total += len(zlib.compress(raw_block, 6))

                # RLE per column
                mc_v, mc_l = rle_encode(mc)
                cov_v, cov_l = rle_encode(cov)
                rle_block = (mc_v.tobytes() + mc_l.tobytes()
                             + cov_v.tobytes() + cov_l.tobytes())
                rle_bytes_total += len(rle_block)
                rle_deflate_total += len(zlib.compress(rle_block, 6))
                chunk_blocks += 1
                n_blocks += 1
            r._load_block()
        log(f'  [{ki+1}/{len(keys)}] {key} n={chunk_n:,} blocks={chunk_blocks} '
            f't={time.time()-t0:.1f}s', flush=True)
    r.close()
    file_size = os.path.getsize(cz_path)
    log(f'  done in {time.time()-t0:.1f}s ({n_blocks:,} blocks)', flush=True)
    return {
        'path': cz_path,
        'n_total': n_total,
        'n_blocks': n_blocks,
        'zero_cov_ratio': n_zero_cov / n_total,
        'zero_mc_ratio': n_zero_mc / n_total,
        'raw_payload_MB': raw_bytes_total / 1e6,
        'rle_payload_MB': rle_bytes_total / 1e6,
        'raw_deflate_MB': raw_deflate_total / 1e6,
        'rle_deflate_MB': rle_deflate_total / 1e6,
        'file_size_MB': file_size / 1e6,
    }


def main():
    paths = sorted(glob.glob('cytozip_example_data/output/cz/*.cz'))
    if not paths:
        print('no files', file=sys.stderr)
        sys.exit(1)
    n_lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(paths)
    paths = paths[:n_lim]
    aggs = []
    for p in paths:
        print(f'== {os.path.basename(p)} ==', flush=True)
        m = measure(p)
        aggs.append(m)
        print(f'  n_total={m["n_total"]:,}  blocks={m["n_blocks"]:,}  '
              f'zero_cov={m["zero_cov_ratio"]*100:.2f}%  '
              f'zero_mc={m["zero_mc_ratio"]*100:.2f}%', flush=True)
        print(f'  raw_payload  ={m["raw_payload_MB"]:7.2f}MB  '
              f'raw+DEFLATE ={m["raw_deflate_MB"]:7.2f}MB', flush=True)
        print(f'  RLE_payload  ={m["rle_payload_MB"]:7.2f}MB  '
              f'RLE+DEFLATE ={m["rle_deflate_MB"]:7.2f}MB  '
              f'(saving {m["raw_deflate_MB"]/max(m["rle_deflate_MB"],1e-9):.2f}x)',
              flush=True)
        print(f'  on-disk cz   ={m["file_size_MB"]:7.2f}MB', flush=True)
    print(f'\n=== mean across {len(aggs)} cells ===', flush=True)
    keys = ['zero_cov_ratio', 'zero_mc_ratio',
            'raw_payload_MB', 'rle_payload_MB',
            'raw_deflate_MB', 'rle_deflate_MB', 'file_size_MB']
    means = {k: float(np.mean([m[k] for m in aggs])) for k in keys}
    print(f'  zero_cov: {means["zero_cov_ratio"]*100:.2f}%   '
          f'zero_mc: {means["zero_mc_ratio"]*100:.2f}%')
    print(f'  raw payload {means["raw_payload_MB"]:.2f}MB '
          f'-> raw+DEFLATE {means["raw_deflate_MB"]:.2f}MB '
          f'-> RLE+DEFLATE {means["rle_deflate_MB"]:.2f}MB')
    saving = means['raw_deflate_MB'] / max(means['rle_deflate_MB'], 1e-9)
    print(f'  RLE saves {saving:.2f}x over plain DEFLATE')
    print(f'  on-disk current cz: {means["file_size_MB"]:.2f}MB')


if __name__ == '__main__':
    main()
