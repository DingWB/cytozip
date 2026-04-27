# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""In-process BS-seq pileup using htslib's bam_mplp_* C API directly.

Replaces the ``samtools mpileup`` subprocess + stdout text parsing path
in :func:`cytozip.bam.bam_to_cz`. Produces byte-equivalent (modulo u8
truncation) mC / cov counts as ``samtools mpileup -Q q -q m -B -f`` does,
because we use the **same** htslib pileup engine plus the public
``bam_mplp_init_overlaps()`` mate-overlap detector.

Filter order (mirrors ``samtools mpileup``):
  1. Read-level (in ``mplp_func`` callback):
     - default flag filter: BAM_FUNMAP|BAM_FSECONDARY|BAM_FQCFAIL|BAM_FDUP
     - MAPQ < min_mapq
  2. ``bam_mplp_init_overlaps`` zeroes the lower-quality mate's base qual.
  3. Per-base (post-pileup):
     - ``is_del`` / ``is_refskip``: skipped
     - base qual < min_base_quality: skipped

API: :class:`PileupCounter` exposes :meth:`iter_chrom`, which yields
``(positions, mc, cov)`` numpy arrays for each chrom containing only
reference C/G sites with cov > 0.
"""
from libc.stdint cimport (int8_t, int32_t, int64_t,
                          uint8_t, uint16_t, uint32_t, uint64_t)
from libc.stdlib cimport malloc, free
from libc.string cimport memset, strcmp
import numpy as np
cimport numpy as cnp

cnp.import_array()

# ---------------------------------------------------------------------------
# htslib C API declarations. Keep minimal — only what we need.
# ---------------------------------------------------------------------------
cdef extern from "htslib/hts.h":
    ctypedef int64_t hts_pos_t
    ctypedef struct htsFile:
        pass
    htsFile *hts_open(const char *fn, const char *mode) nogil
    int hts_close(htsFile *fp) nogil


cdef extern from "htslib/sam.h":
    # Flag constants.
    int BAM_FUNMAP
    int BAM_FSECONDARY
    int BAM_FQCFAIL
    int BAM_FDUP
    int BAM_FPAIRED
    int BAM_FPROPER_PAIR

    # Core record (subset).
    ctypedef struct bam1_core_t:
        hts_pos_t pos
        int32_t tid
        uint16_t bin
        uint8_t qual
        uint8_t l_extranul
        uint16_t flag
        uint16_t l_qname
        uint32_t n_cigar
        int32_t l_qseq
        int32_t mtid
        hts_pos_t mpos
        hts_pos_t isize

    ctypedef struct bam1_t:
        bam1_core_t core
        uint64_t id
        uint8_t *data
        int l_data
        int m_data

    ctypedef struct bam_pileup1_t:
        bam1_t *b
        int32_t qpos
        int indel
        int level
        # Bitfield: is_del:1, is_head:1, is_tail:1, is_refskip:1, ...
        # Cython can't represent bitfields directly; we expose helpers below.
        uint32_t flag

    ctypedef struct sam_hdr_t:
        int32_t n_targets
        char **target_name

    sam_hdr_t *sam_hdr_read(htsFile *fp) nogil
    void sam_hdr_destroy(sam_hdr_t *h) nogil
    int sam_hdr_name2tid(sam_hdr_t *h, const char *ref) nogil

    ctypedef struct hts_idx_t:
        pass
    hts_idx_t *sam_index_load(htsFile *fp, const char *fn) nogil
    void hts_idx_destroy(hts_idx_t *idx) nogil

    ctypedef struct hts_itr_t:
        pass
    hts_itr_t *sam_itr_queryi(hts_idx_t *idx, int tid,
                              hts_pos_t beg, hts_pos_t end) nogil
    int sam_itr_next(htsFile *fp, hts_itr_t *iter, bam1_t *b) nogil
    void hts_itr_destroy(hts_itr_t *iter) nogil

    bam1_t *bam_init1() nogil
    void bam_destroy1(bam1_t *b) nogil

    # Pileup C API (multi-input variant; we use n=1 + overlap detection).
    ctypedef int (*bam_plp_auto_f)(void *data, bam1_t *b) nogil

    ctypedef struct __bam_mplp_t:
        pass
    ctypedef __bam_mplp_t *bam_mplp_t

    bam_mplp_t bam_mplp_init(int n, bam_plp_auto_f func, void **data) nogil
    int bam_mplp_init_overlaps(bam_mplp_t iter) nogil
    void bam_mplp_set_maxcnt(bam_mplp_t iter, int maxcnt) nogil
    void bam_mplp_destroy(bam_mplp_t iter) nogil
    int bam_mplp64_auto(bam_mplp_t iter, int *_tid, hts_pos_t *_pos,
                        int *n_plp, const bam_pileup1_t **plp) nogil


cdef extern from "htslib/faidx.h":
    ctypedef struct faidx_t:
        pass
    faidx_t *fai_load(const char *fn) nogil
    void fai_destroy(faidx_t *fai) nogil
    char *fai_fetch(const faidx_t *fai, const char *reg, int *seq_len) nogil
    int faidx_seq_len(const faidx_t *fai, const char *seq) nogil


# ---------------------------------------------------------------------------
# C helpers. Replicate the htslib macros that Cython can't see.
# ---------------------------------------------------------------------------
cdef extern from *:
    """
    /* htslib macros, replicated as inline functions for Cython use. */
    #include "htslib/sam.h"
    static inline uint8_t *cz_bam_get_seq(bam1_t *b) {
        return b->data + (b->core.n_cigar << 2) + b->core.l_qname;
    }
    static inline uint8_t *cz_bam_get_qual(bam1_t *b) {
        return b->data + (b->core.n_cigar << 2) + b->core.l_qname
               + ((b->core.l_qseq + 1) >> 1);
    }
    static inline int cz_bam_seqi(uint8_t *s, int i) {
        return (s[i >> 1] >> ((~i & 1) << 2)) & 0xf;
    }
    static inline int cz_plp_is_del(const bam_pileup1_t *p) { return p->is_del; }
    static inline int cz_plp_is_refskip(const bam_pileup1_t *p) { return p->is_refskip; }
    """
    uint8_t *cz_bam_get_seq(bam1_t *b) nogil
    uint8_t *cz_bam_get_qual(bam1_t *b) nogil
    int cz_bam_seqi(uint8_t *s, int i) nogil
    int cz_plp_is_del(const bam_pileup1_t *p) nogil
    int cz_plp_is_refskip(const bam_pileup1_t *p) nogil


# htslib base encoding: 0=., 1=A, 2=C, 4=G, 8=T, 15=N.
cdef int8_t _BASE_A = 1
cdef int8_t _BASE_C = 2
cdef int8_t _BASE_G = 4
cdef int8_t _BASE_T = 8


# ---------------------------------------------------------------------------
# State passed to mplp_func via void*.
# ---------------------------------------------------------------------------
cdef struct PlpData:
    htsFile *fp
    hts_itr_t *iter
    int min_mapq
    int default_flag_filter
    int skip_orphans  # 1 = drop paired-but-not-proper-pair reads (mpileup default)


cdef int _mplp_func(void *data, bam1_t *b) noexcept nogil:
    """Read callback for bam_mplp_init. Returns next read or <0 on EOF.

    Mirrors samtools' mplp_func: discard reads with default skip flags,
    below MAPQ threshold, and (by default) "orphan" paired reads —
    i.e. paired reads without the BAM_FPROPER_PAIR flag set. This last
    filter is critical for m3C/hisat-3n BAMs: chimeric and split reads
    keep BAM_FPAIRED but lose BAM_FPROPER_PAIR, and samtools mpileup
    drops them by default (only kept with ``-A``). ALLCools'
    ``bam_to_allc`` does not pass ``-A``, so we must drop them too to
    stay byte-equivalent.
    """
    cdef PlpData *d = <PlpData*>data
    cdef int ret
    while True:
        ret = sam_itr_next(d.fp, d.iter, b)
        if ret < 0:
            return ret
        if (b.core.flag & d.default_flag_filter) != 0:
            continue
        if b.core.qual < d.min_mapq:
            continue
        if d.skip_orphans:
            if (b.core.flag & BAM_FPAIRED) != 0 and \
                    (b.core.flag & BAM_FPROPER_PAIR) == 0:
                continue
        return ret


cdef class PileupCounter:
    """In-process BAM → per-site (mc, cov) counter for BS-seq.

    Lifecycle: construct with paths and filters, then call
    :meth:`iter_chrom` for each chromosome (in any order), or
    :meth:`iter_all` to walk all chroms in BAM order.
    """
    cdef htsFile *fp
    cdef sam_hdr_t *hdr
    cdef hts_idx_t *idx
    cdef faidx_t *fai
    cdef int min_mapq
    cdef int min_base_quality
    cdef int max_depth
    cdef bytes _bam_path
    cdef bytes _fasta_path

    def __cinit__(self, bam_path, fasta_path,
                  int min_mapq=10, int min_base_quality=20,
                  int max_depth=8000):
        self.fp = NULL
        self.hdr = NULL
        self.idx = NULL
        self.fai = NULL
        self.min_mapq = min_mapq
        self.min_base_quality = min_base_quality
        self.max_depth = max_depth

        self._bam_path = bam_path.encode("utf-8") \
            if isinstance(bam_path, str) else bam_path
        self._fasta_path = fasta_path.encode("utf-8") \
            if isinstance(fasta_path, str) else fasta_path

        self.fp = hts_open(self._bam_path, b"rb")
        if self.fp == NULL:
            raise IOError(f"Failed to open BAM: {bam_path}")
        self.hdr = sam_hdr_read(self.fp)
        if self.hdr == NULL:
            raise IOError(f"Failed to read BAM header: {bam_path}")
        self.idx = sam_index_load(self.fp, self._bam_path)
        if self.idx == NULL:
            raise IOError(
                f"Failed to load BAM index for {bam_path}; "
                "run `samtools index <bam>` first."
            )
        self.fai = fai_load(self._fasta_path)
        if self.fai == NULL:
            raise IOError(
                f"Failed to load FASTA index for {fasta_path}; "
                "run `samtools faidx <fa>` first."
            )

    def __dealloc__(self):
        if self.fai != NULL:
            fai_destroy(self.fai)
            self.fai = NULL
        if self.idx != NULL:
            hts_idx_destroy(self.idx)
            self.idx = NULL
        if self.hdr != NULL:
            sam_hdr_destroy(self.hdr)
            self.hdr = NULL
        if self.fp != NULL:
            hts_close(self.fp)
            self.fp = NULL

    @property
    def references(self):
        """Tuple of chromosome names from the BAM header."""
        cdef int i
        cdef list names = []
        for i in range(self.hdr.n_targets):
            names.append(self.hdr.target_name[i].decode("ascii"))
        return tuple(names)

    def iter_chrom(self, chrom):
        """Pileup one chromosome.

        Returns
        -------
        positions : np.ndarray[uint32]
            1-based reference positions (matches allc.tsv.gz).
        ref_bases : np.ndarray[bytes]  (single-byte b'C' / b'G')
            Reference base at each position (always C or G).
        mc : np.ndarray[uint32]
            Methylated read count.
        cov : np.ndarray[uint32]
            Total covered read count.

        Only positions with cov > 0 AND ref base in {C, G} are returned.
        """
        if isinstance(chrom, str):
            chrom_b = chrom.encode("ascii")
        else:
            chrom_b = chrom

        cdef int tid = sam_hdr_name2tid(self.hdr, chrom_b)
        if tid < 0:
            raise KeyError(f"chrom {chrom!r} not in BAM header")

        cdef int seq_len = faidx_seq_len(self.fai, chrom_b)
        if seq_len < 0:
            raise KeyError(f"chrom {chrom!r} not in FASTA index")

        # Fetch the entire chrom sequence as uppercase ASCII.
        cdef int fetched_len = 0
        cdef char *seq_c = fai_fetch(self.fai, chrom_b, &fetched_len)
        if seq_c == NULL:
            raise IOError(f"fai_fetch failed for {chrom!r}")
        # Uppercase in-place.
        cdef int i
        for i in range(fetched_len):
            if seq_c[i] >= b'a' and seq_c[i] <= b'z':
                seq_c[i] = seq_c[i] - 32

        # Allocate output buffers (over-allocate to chrom_len; trim later).
        # Most chroms have ~10% C+G, so this is ~5x larger than needed.
        cdef int est_cap = fetched_len // 5 + 16
        if est_cap < 1024:
            est_cap = 1024
        cdef cnp.ndarray[cnp.uint32_t, ndim=1] pos_arr = \
            np.empty(est_cap, dtype=np.uint32)
        cdef cnp.ndarray[cnp.uint8_t, ndim=1] ref_arr = \
            np.empty(est_cap, dtype=np.uint8)
        cdef cnp.ndarray[cnp.uint32_t, ndim=1] mc_arr = \
            np.empty(est_cap, dtype=np.uint32)
        cdef cnp.ndarray[cnp.uint32_t, ndim=1] cov_arr = \
            np.empty(est_cap, dtype=np.uint32)
        cdef int n_out = 0
        cdef int cap = est_cap

        # Set up pileup iterator.
        cdef PlpData pdata
        pdata.fp = self.fp
        pdata.iter = sam_itr_queryi(self.idx, tid, 0, seq_len)
        if pdata.iter == NULL:
            free(seq_c)
            raise IOError(f"sam_itr_queryi failed for {chrom!r}")
        pdata.min_mapq = self.min_mapq
        pdata.default_flag_filter = (BAM_FUNMAP | BAM_FSECONDARY
                                     | BAM_FQCFAIL | BAM_FDUP)
        pdata.skip_orphans = 1  # mpileup default; required for ALLCools parity

        cdef void *data_ptr = <void*>(&pdata)
        cdef bam_mplp_t mplp = bam_mplp_init(1, _mplp_func, &data_ptr)
        if mplp == NULL:
            hts_itr_destroy(pdata.iter)
            free(seq_c)
            raise MemoryError("bam_mplp_init failed")
        bam_mplp_init_overlaps(mplp)
        bam_mplp_set_maxcnt(mplp, self.max_depth)

        cdef int plp_tid = -1
        cdef hts_pos_t plp_pos = 0
        cdef int n_plp = 0
        cdef const bam_pileup1_t *plp_p = NULL
        cdef const bam_pileup1_t *p = NULL
        cdef int j
        cdef int mc_count, cov_count
        cdef int base_idx
        cdef uint8_t *seq_data
        cdef uint8_t *qual_data
        cdef int min_baseq = self.min_base_quality
        cdef char ref_base
        cdef bint is_C, is_G
        cdef int b_enc
        cdef bint base_unconverted, base_converted

        with nogil:
            while bam_mplp64_auto(mplp, &plp_tid, &plp_pos,
                                  &n_plp, &plp_p) > 0:
                if plp_tid != tid:
                    # Defensive: should not happen since we restricted
                    # the iterator to this tid.
                    continue
                if plp_pos < 0 or plp_pos >= fetched_len:
                    continue
                ref_base = seq_c[plp_pos]
                is_C = (ref_base == b'C')
                is_G = (ref_base == b'G')
                if not (is_C or is_G):
                    continue

                mc_count = 0
                cov_count = 0
                for j in range(n_plp):
                    p = &plp_p[j]
                    if cz_plp_is_del(p) or cz_plp_is_refskip(p):
                        continue
                    qual_data = cz_bam_get_qual(p.b)
                    if qual_data[p.qpos] < min_baseq:
                        continue
                    seq_data = cz_bam_get_seq(p.b)
                    b_enc = cz_bam_seqi(seq_data, p.qpos)

                    if is_C:
                        # Forward strand. Reads aligning to '+' strand
                        # of a C ref base contribute on uppercase channel.
                        # samtools mpileup outputs '.' for ref-match, 'T'
                        # for mismatch when on forward strand.
                        # Forward strand reads have flag 16 (BAM_FREVERSE)
                        # cleared. mpileup considers BAM_FREVERSE for case.
                        if (p.b.core.flag & 16) == 0:
                            # Forward read on a C ref site:
                            #   read base == C → mC (unconverted, methylated)
                            #   read base == T → converted (unmethylated)
                            if b_enc == _BASE_C:
                                mc_count += 1
                                cov_count += 1
                            elif b_enc == _BASE_T:
                                cov_count += 1
                            # else: A/G/N: not counted (mismatch / non-BS)
                        # Reverse read on C ref site: in mpileup these
                        # show as lowercase 'c'/'t'/etc. ALLCools' code
                        # only counts '.' and 'T' on a C ref site, so
                        # reverse reads are effectively ignored. Match
                        # that here (skip).
                    else:  # is_G
                        # Reverse strand of a CpG. mpileup outputs ','
                        # for ref-match (lowercase, reverse read). ALLCools
                        # counts ',' (unconverted, methylated G) and 'a'
                        # (converted G→A on the bottom strand).
                        if (p.b.core.flag & 16) != 0:
                            # Reverse read on G ref site:
                            #   read base == G → mC (unconverted)
                            #   read base == A → converted
                            if b_enc == _BASE_G:
                                mc_count += 1
                                cov_count += 1
                            elif b_enc == _BASE_A:
                                cov_count += 1
                        # Forward read on G ref site: ignored (matches ALLCools).

                if cov_count == 0:
                    continue

                # Append to output (grow buffers if needed).
                if n_out >= cap:
                    with gil:
                        cap = cap * 2
                        pos_arr = np.resize(pos_arr, cap)
                        ref_arr = np.resize(ref_arr, cap)
                        mc_arr = np.resize(mc_arr, cap)
                        cov_arr = np.resize(cov_arr, cap)
                pos_arr[n_out] = <uint32_t>(plp_pos + 1)  # 1-based
                ref_arr[n_out] = <uint8_t>ref_base
                mc_arr[n_out] = <uint32_t>mc_count
                cov_arr[n_out] = <uint32_t>cov_count
                n_out += 1

        bam_mplp_destroy(mplp)
        hts_itr_destroy(pdata.iter)
        free(seq_c)

        # Trim to actual size.
        return (pos_arr[:n_out].copy(),
                ref_arr[:n_out].copy(),
                mc_arr[:n_out].copy(),
                cov_arr[:n_out].copy())
