import unittest
import tempfile
import os
import struct

from cytozip.cz import Writer, Reader


class TestDeltaEncoding(unittest.TestCase):
    """Test delta encoding (v2.0) write and read round-trip."""

    def _make_test_data(self):
        """Create sorted position data with mc/cov columns."""
        # Simulate an allc-like file: pos(Q), mc(H), cov(H)
        records = [
            (100, 3, 10),
            (250, 5, 12),
            (300, 0, 8),
            (1000, 2, 15),
            (1500, 7, 20),
            (2000, 1, 5),
        ]
        fmt = "QHH"
        data = b"".join(struct.pack(f"<{fmt}", *r) for r in records)
        return records, fmt, data

    def test_delta_roundtrip_fetch(self):
        """Write with delta_cols=[0], read back via fetch, verify values match."""
        records, fmt, data = self._make_test_data()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "delta.cz")
            w = Writer(
                Output=path,
                Formats=["Q", "H", "H"],
                Columns=["pos", "mc", "cov"],
                Dimensions=["chrom"],
                delta_cols=[0],
            )
            w.write_chunk(data, ["chr1"])
            w.close()

            r = Reader(path)
            self.assertAlmostEqual(r.header["version"], 0.1, places=1)
            self.assertEqual(r._delta_cols, [0])
            got = [row for row in r.__fetch__(("chr1",))]
            r.close()

            self.assertEqual(len(got), len(records))
            for i, (expected, actual) in enumerate(zip(records, got)):
                self.assertEqual(expected, actual, f"Record {i} mismatch")

    def test_delta_roundtrip_fetch_chunk_bytes(self):
        """Write with delta, read back via fetch_chunk_bytes, verify."""
        records, fmt, data = self._make_test_data()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "delta2.cz")
            w = Writer(
                Output=path,
                Formats=["Q", "H", "H"],
                Columns=["pos", "mc", "cov"],
                Dimensions=["chrom"],
                delta_cols=[0],
            )
            w.write_chunk(data, ["chr1"])
            w.close()

            r = Reader(path)
            raw = r.fetch_chunk_bytes(("chr1",))
            r.close()

            # Unpack and compare
            st = struct.Struct(f"<{fmt}")
            got = [st.unpack_from(raw, i * st.size) for i in range(len(raw) // st.size)]
            self.assertEqual(len(got), len(records))
            for i, (expected, actual) in enumerate(zip(records, got)):
                self.assertEqual(expected, actual, f"Record {i} mismatch")

    def test_no_delta_still_works(self):
        """Verify files without delta still work correctly."""
        records, fmt, data = self._make_test_data()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "nodelta.cz")
            w = Writer(
                Output=path,
                Formats=["Q", "H", "H"],
                Columns=["pos", "mc", "cov"],
                Dimensions=["chrom"],
            )
            w.write_chunk(data, ["chr1"])
            w.close()

            r = Reader(path)
            self.assertAlmostEqual(r.header["version"], 0.1, places=1)
            self.assertEqual(r._delta_cols, [])
            got = [row for row in r.__fetch__(("chr1",))]
            r.close()
            self.assertEqual(len(got), len(records))
            for i, (expected, actual) in enumerate(zip(records, got)):
                self.assertEqual(expected, actual)

    def test_delta_multi_chunk(self):
        """Delta encoding with multiple chunks (chroms)."""
        fmt = "QHH"
        chr1_records = [(100, 1, 5), (200, 2, 10), (500, 3, 15)]
        chr2_records = [(50, 4, 20), (300, 5, 25)]
        chr1_data = b"".join(struct.pack(f"<{fmt}", *r) for r in chr1_records)
        chr2_data = b"".join(struct.pack(f"<{fmt}", *r) for r in chr2_records)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "multi.cz")
            w = Writer(
                Output=path,
                Formats=["Q", "H", "H"],
                Columns=["pos", "mc", "cov"],
                Dimensions=["chrom"],
                delta_cols=[0],
            )
            w.write_chunk(chr1_data, ["chr1"])
            w.write_chunk(chr2_data, ["chr2"])
            w.close()

            r = Reader(path)
            got1 = [row for row in r.__fetch__(("chr1",))]
            got2 = [row for row in r.__fetch__(("chr2",))]
            r.close()

            self.assertEqual(len(got1), 3)
            self.assertEqual(len(got2), 2)
            for exp, act in zip(chr1_records, got1):
                self.assertEqual(exp, act)
            for exp, act in zip(chr2_records, got2):
                self.assertEqual(exp, act)

    def test_delta_large_data_multi_block(self):
        """Delta encoding across multiple blocks."""
        fmt = "QHH"
        unit_size = struct.calcsize(fmt)  # 12 bytes
        aligned_block = (65535 // unit_size) * unit_size  # 65532
        records_per_block = aligned_block // unit_size  # 5461

        # Generate enough records to span 3+ blocks
        n_records = records_per_block * 3 + 100
        all_records = []
        pos = 1000
        for i in range(n_records):
            pos += (i % 50) + 1  # varying deltas
            mc = i % 256
            cov = (i * 3) % 65536
            all_records.append((pos, mc, cov))
        data = b"".join(struct.pack(f"<{fmt}", *r) for r in all_records)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "large.cz")
            w = Writer(
                Output=path,
                Formats=["Q", "H", "H"],
                Columns=["pos", "mc", "cov"],
                Dimensions=["chrom"],
                delta_cols=[0],
            )
            w.write_chunk(data, ["chr1"])
            w.close()

            r = Reader(path)
            got = [row for row in r.__fetch__(("chr1",))]
            r.close()

            self.assertEqual(len(got), n_records)
            for i in range(n_records):
                self.assertEqual(all_records[i], got[i], f"Record {i} mismatch")


class TestChunkIndex(unittest.TestCase):
    """Test chunk index at end of file."""

    def test_chunk_index_present(self):
        """Chunk index is written and readable."""
        fmt = "QHH"
        data = struct.pack(f"<{fmt}", 100, 5, 10)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "idx.cz")
            w = Writer(
                Output=path,
                Formats=["Q", "H", "H"],
                Columns=["pos", "mc", "cov"],
                Dimensions=["chrom"],
            )
            w.write_chunk(data, ["chr1"])
            w.close()

            r = Reader(path)
            idx = r.read_chunk_index()
            r.close()

            self.assertIsNotNone(idx)
            self.assertIn(("chr1",), idx)
            entry = idx[("chr1",)]
            self.assertEqual(entry["nblocks"], 1)
            self.assertEqual(entry["data_len"], struct.calcsize(fmt))

    def test_chunk_index_multi_chunk(self):
        """Chunk index with multiple chunks."""
        fmt = "HH"
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "idx2.cz")
            w = Writer(
                Output=path,
                Formats=["H", "H"],
                Columns=["mc", "cov"],
                Dimensions=["chrom"],
            )
            w.write_chunk(struct.pack(f"<{fmt}", 1, 2), ["chr1"])
            w.write_chunk(struct.pack(f"<{fmt}", 3, 4), ["chr2"])
            w.write_chunk(struct.pack(f"<{fmt}", 5, 6), ["chr3"])
            w.close()

            r = Reader(path)
            idx = r.read_chunk_index()
            r.close()

            self.assertEqual(len(idx), 3)
            self.assertIn(("chr1",), idx)
            self.assertIn(("chr2",), idx)
            self.assertIn(("chr3",), idx)

    def test_chunk_index_matches_summary(self):
        """Chunk index data matches summary_chunks data."""
        fmt = "QHH"
        unit = struct.calcsize(fmt)
        data1 = b"".join(struct.pack(f"<{fmt}", i * 100, i, i * 2) for i in range(10))
        data2 = b"".join(struct.pack(f"<{fmt}", i * 50, i, i * 3) for i in range(20))

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "match.cz")
            w = Writer(
                Output=path,
                Formats=["Q", "H", "H"],
                Columns=["pos", "mc", "cov"],
                Dimensions=["chrom"],
            )
            w.write_chunk(data1, ["chr1"])
            w.write_chunk(data2, ["chr2"])
            w.close()

            r = Reader(path)
            idx = r.read_chunk_index()
            chunk_info = r.chunk_info

            for dim_tuple in idx:
                entry = idx[dim_tuple]
                loc = chunk_info.index.get_loc(dim_tuple)
                row = chunk_info.iloc[loc]
                self.assertEqual(entry["start"], row["chunk_start_offset"])
                self.assertEqual(entry["data_len"], row["chunk_nrows"] * unit)
                self.assertEqual(entry["nblocks"], row["chunk_nblocks"])
            r.close()


if __name__ == "__main__":
    unittest.main()
