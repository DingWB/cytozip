"""Shared memory-budget and process-pool helpers.

Every parallel entry point in cytozip (``cz_to_anndata*``, ``merge_cz``,
``mc_summary``, ``call_dmr*``, ``call_peaks*``, ``index_context``,
``pivot_*``, ``allc2cz``) sizes its worker pool with :func:`cap_workers`:
``jobs`` is clamped so ``jobs x (estimated per-worker peak)`` fits the
memory budget -- ``max_mem`` when given, otherwise 80% of
:func:`available_memory_bytes` (free RAM, further limited by the tightest
memory cgroup, e.g. a SLURM job limit). Pools are created with the ``fork``
context (:func:`make_pool`) so large read-only state is inherited
copy-on-write, and tasks are fed through a bounded window
(:func:`bounded_map` / :func:`bounded_unordered`) so finished-but-unconsumed
results can never pile up with the number of tasks.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from loguru import logger

_MEM_UNITS = {"": 1, "b": 1, "k": 1 << 10, "kb": 1 << 10, "m": 1 << 20,
              "mb": 1 << 20, "g": 1 << 30, "gb": 1 << 30, "t": 1 << 40,
              "tb": 1 << 40}

# Rough fixed per-process cost of a worker (interpreter + numpy + cytozip).
WORKER_BASE_BYTES = 150 << 20


def parse_mem(value) -> Optional[int]:
    """``'200G'`` / ``'512MB'`` / ``int`` bytes -> bytes (``None`` passthrough)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) or type(value).__module__ == "numpy":
        return int(value)
    m = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]*)\s*", str(value))
    if not m or m.group(2).lower() not in _MEM_UNITS:
        raise ValueError(f"cannot parse memory size {value!r} (e.g. '200G')")
    return int(float(m.group(1)) * _MEM_UNITS[m.group(2).lower()])


def _read_int_file(path) -> Optional[int]:
    try:
        with open(path) as fh:
            txt = fh.read().strip()
        return None if txt in ("max", "") else int(txt)
    except (OSError, ValueError):
        return None


def _cgroup_paths():
    """``(mount_dir, rel_path, limit_file, usage_file)`` of every memory
    cgroup hierarchy this process belongs to (v2 unified and/or v1)."""
    try:
        with open("/proc/self/cgroup") as fh:
            lines = [ln.strip().split(":", 2) for ln in fh if ln.strip()]
    except OSError:
        return []
    paths = []
    for parts in lines:
        if len(parts) != 3:
            continue
        hid, ctrls, rel = parts
        if hid == "0" and ctrls == "":           # cgroup v2 unified
            paths.append(("/sys/fs/cgroup", rel, "memory.max",
                          "memory.current"))
        elif "memory" in ctrls.split(","):       # cgroup v1
            paths.append(("/sys/fs/cgroup/memory", rel,
                          "memory.limit_in_bytes", "memory.usage_in_bytes"))
    return paths


def cgroup_memory_headroom() -> Optional[int]:
    """``limit - usage`` of the tightest memory cgroup this process sits in
    (v2 or v1, walking up the hierarchy -- e.g. SLURM sets the job limit on
    a parent of the task cgroup). ``None`` when unlimited / unknown."""
    best = None
    for mount, rel, lim_f, use_f in _cgroup_paths():
        rel = rel.strip("/")
        comps = rel.split("/") if rel else []
        # Walk from the leaf up to the mount root.
        for depth in range(len(comps), -1, -1):
            d = os.path.join(mount, *comps[:depth]) if depth else mount
            lim = _read_int_file(os.path.join(d, lim_f))
            if lim is None or lim >= (1 << 60):
                continue
            use = _read_int_file(os.path.join(d, use_f)) or 0
            head = max(0, lim - use)
            best = head if best is None else min(best, head)
    return best


def cgroup_usage_file() -> Optional[str]:
    """Usage file of the tightest limited memory cgroup (see
    :func:`cgroup_memory_headroom`), or ``None``."""
    best = None
    for mount, rel, lim_f, use_f in _cgroup_paths():
        rel = rel.strip("/")
        comps = rel.split("/") if rel else []
        for depth in range(len(comps), -1, -1):
            d = os.path.join(mount, *comps[:depth]) if depth else mount
            lim = _read_int_file(os.path.join(d, lim_f))
            if lim is None or lim >= (1 << 60):
                continue
            cand = (lim, os.path.join(d, use_f))
            if best is None or cand[0] < best[0]:
                best = cand
    return None if best is None else best[1]


def available_memory_bytes() -> Optional[int]:
    """Best-effort free-memory estimate: min(free physical RAM, cgroup
    headroom). Returns ``None`` when nothing can be determined."""
    cands = []
    try:
        cands.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    head = cgroup_memory_headroom()
    if head is not None:
        cands.append(head)
    return min(cands) if cands else None


def memory_budget(max_mem=None, fraction=0.8) -> Optional[int]:
    """``max_mem`` parsed to bytes, else ``fraction`` of the detected free
    memory (``None`` when neither is known)."""
    budget = parse_mem(max_mem)
    if budget is not None:
        return budget
    avail = available_memory_bytes()
    return None if avail is None else int(avail * fraction)


def fmt_bytes(n) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def cap_workers(jobs, est_worker_bytes, max_mem=None, parent_bytes=0,
                what="worker") -> int:
    """Clamp ``jobs`` so ``jobs * est_worker_bytes + parent_bytes`` fits in
    the memory budget (``max_mem`` or 80% of detected free memory)."""
    n = int(jobs) if jobs and int(jobs) > 1 else 1
    if n <= 1 or not est_worker_bytes:
        return n
    budget = memory_budget(max_mem, 0.8)
    if budget is None:
        return n
    fit = max(1, int((budget - parent_bytes) // est_worker_bytes))
    if fit < n:
        logger.warning(
            f"[cytozip] jobs={n} would need ~{fmt_bytes(n * est_worker_bytes)} "
            f"(~{fmt_bytes(est_worker_bytes)} per {what}) but the memory "
            f"budget is {fmt_bytes(budget)}; using jobs={fit}. Pass "
            f"max_mem= to override the budget.")
        return fit
    return n


def make_pool(n_workers, initializer=None, initargs=()):
    """``ProcessPoolExecutor`` preferring ``fork`` so the (possibly large)
    read-only ``initargs`` are inherited copy-on-write instead of pickled
    once per worker."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    ctx = None
    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
    return ProcessPoolExecutor(max_workers=n_workers, initializer=initializer,
                               initargs=initargs, mp_context=ctx)


def bounded_map(ex, fn, items, window):
    """Ordered ``ex.map`` with at most ``window`` tasks in flight, so
    finished-but-unconsumed results cannot pile up with ``len(items)``."""
    from collections import deque
    it = iter(items)
    pending = deque()
    for _ in range(max(1, int(window))):
        try:
            pending.append(ex.submit(fn, next(it)))
        except StopIteration:
            break
    while pending:
        res = pending.popleft().result()
        try:
            pending.append(ex.submit(fn, next(it)))
        except StopIteration:
            pass
        yield res


def bounded_unordered(ex, fn, items, window):
    """Unordered counterpart of :func:`bounded_map`: yields results as they
    finish while keeping at most ``window`` tasks in flight."""
    from concurrent.futures import FIRST_COMPLETED, wait
    it = iter(items)
    pending = set()
    for _ in range(max(1, int(window))):
        try:
            pending.add(ex.submit(fn, next(it)))
        except StopIteration:
            break
    while pending:
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for fut in done:
            res = fut.result()
            try:
                pending.add(ex.submit(fn, next(it)))
            except StopIteration:
                pass
            yield res


class PerProcessReader:
    """One cached :class:`~cytozip.cz.Reader` of a ``catcz``'d file per
    worker process, opened with ``Reader(chunk_index=...)`` so the (large)
    CZIX index is never parsed in a worker.

    The parent passes ``{chrom: {dims: chunk_start_offset}}`` to :meth:`init`
    (as a pool initializer argument, inherited copy-on-write under ``fork``);
    tasks then carry only the chrom name and call :meth:`get` with that
    chrom's offsets, which re-points the cached reader at exactly the chunks
    of the current task. Used by ``merge_cz`` and ``pivot_*``.
    """

    def __init__(self):
        self.chrom_offsets = {}
        self._reader = None
        self._path = None

    def init(self, chrom_offsets):
        """Pool initializer: install the fork-shared offsets."""
        self.close()
        self.chrom_offsets = chrom_offsets

    def offsets(self, chrom):
        return self.chrom_offsets.get(chrom)

    def get(self, path, dim_offsets):
        """Cached reader of ``path`` seeing exactly ``dim_offsets``."""
        from .cz import Reader
        if self._reader is None or self._path != path:
            self.close()
            self._reader = Reader(path, mmap_advise="sequential",
                                  chunk_index=dim_offsets)
            self._path = path
        else:
            # Re-point at this task's chrom only (no union growing back to
            # the full index as a worker visits every chrom).
            self._reader.chunk_key2offset = dict(dim_offsets)
        return self._reader

    def close(self):
        if self._reader is not None:
            self._reader.close()
        self._reader = None
        self._path = None
