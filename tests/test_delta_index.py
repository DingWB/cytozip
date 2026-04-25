import unittest
import tempfile
import os
import struct

from cytozip.cz import Writer, Reader


class TestChunkIndex(unittest.TestCase):
    """Test chunk index at end of file."""

    def test_chunk_index_present(self):
        """Chunk index is written and readable."""
        fmt = "QHH"
        data = struct.pack(f"<{fmt}", 100, 5, 10)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "idx.cz")
            w = Writer(
                output=path,
                formats=["Q", "H", "H"],
                columns=["pos", "mc", "cov"],
                chunk_dims=["chrom"],
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
                output=path,
                formats=["H", "H"],
                columns=["mc", "cov"],
                chunk_dims=["chrom"],
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
                output=path,
                formats=["Q", "H", "H"],
                columns=["pos", "mc", "cov"],
                chunk_dims=["chrom"],
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
