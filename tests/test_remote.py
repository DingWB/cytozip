import unittest
import tempfile
import os
import struct
import threading
import http.server
import functools

from cytozip.cz import Writer, Reader, RemoteFile


def _start_http_server(directory, port=0):
    """Start a simple HTTP server serving files from `directory`.
    Returns (server, url_base) where url_base is 'http://127.0.0.1:<port>'.
    """
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=directory)
    server = http.server.HTTPServer(('127.0.0.1', port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


class TestRemoteFile(unittest.TestCase):
    """Test RemoteFile with a local HTTP server."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tmpdir = self.td.name
        # Create a test file
        self.test_data = b"Hello, World! " * 1000  # 14KB
        with open(os.path.join(self.tmpdir, "test.bin"), "wb") as f:
            f.write(self.test_data)
        self.server, self.base_url = _start_http_server(self.tmpdir)

    def tearDown(self):
        self.server.shutdown()
        self.td.cleanup()

    def test_read_all(self):
        rf = RemoteFile(f"{self.base_url}/test.bin", cache_size=1024)
        data = rf.read(len(self.test_data))
        rf.close()
        self.assertEqual(data, self.test_data)

    def test_seek_and_read(self):
        rf = RemoteFile(f"{self.base_url}/test.bin", cache_size=1024)
        rf.seek(10)
        data = rf.read(5)
        self.assertEqual(data, self.test_data[10:15])
        rf.close()

    def test_seek_from_end(self):
        rf = RemoteFile(f"{self.base_url}/test.bin", cache_size=1024)
        rf.seek(-10, 2)
        data = rf.read(10)
        self.assertEqual(data, self.test_data[-10:])
        rf.close()

    def test_cache_hit(self):
        rf = RemoteFile(f"{self.base_url}/test.bin", cache_size=4096)
        rf.read(10)  # triggers fetch of 4096 bytes
        # Next read should be a cache hit
        data = rf.read(10)
        self.assertEqual(data, self.test_data[10:20])
        rf.close()


class TestRemoteReader(unittest.TestCase):
    """Test Reader with remote .cz files via local HTTP server."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.tmpdir = self.td.name
        self.server, self.base_url = _start_http_server(self.tmpdir)

    def tearDown(self):
        self.server.shutdown()
        self.td.cleanup()

    def _write_test_cz(self, filename, records, fmt="QHH",
                       formats=["Q", "H", "H"], columns=["pos", "mc", "cov"]):
        path = os.path.join(self.tmpdir, filename)
        data = b"".join(struct.pack(f"<{fmt}", *r) for r in records)
        w = Writer(output=path, formats=formats, columns=columns,
                   dimensions=["chrom"])
        w.write_chunk(data, ["chr1"])
        w.close()
        return path

    def test_remote_from_url(self):
        """Reader.from_url opens a remote .cz file."""
        records = [(100, 3, 10), (200, 5, 12), (300, 0, 8)]
        self._write_test_cz("test.cz", records)
        url = f"{self.base_url}/test.cz"
        r = Reader.from_url(url)
        self.assertTrue(r._is_remote)
        self.assertIn(("chr1",), r.dim2chunk_start)
        r.close()

    def test_remote_url_in_init(self):
        """Reader(url) auto-detects HTTP URLs."""
        records = [(100, 3, 10), (200, 5, 12)]
        self._write_test_cz("test2.cz", records)
        url = f"{self.base_url}/test2.cz"
        r = Reader(url)
        self.assertTrue(r._is_remote)
        r.close()

    def test_remote_fetch(self):
        """fetch() works on remote .cz files."""
        records = [(100, 3, 10), (200, 5, 12), (300, 0, 8)]
        self._write_test_cz("fetch.cz", records)
        url = f"{self.base_url}/fetch.cz"
        r = Reader.from_url(url)
        got = list(r.fetch(("chr1",)))
        r.close()
        self.assertEqual(len(got), 3)
        for exp, act in zip(records, got):
            self.assertEqual(exp, tuple(act))

    def test_remote_fetch_chunk_bytes(self):
        """fetch_chunk_bytes() works on remote .cz files."""
        records = [(100, 3, 10), (200, 5, 12)]
        self._write_test_cz("bytes.cz", records)
        url = f"{self.base_url}/bytes.cz"
        r = Reader.from_url(url)
        raw = r.fetch_chunk_bytes(("chr1",))
        r.close()
        st = struct.Struct("<QHH")
        got = [st.unpack_from(raw, i * st.size) for i in range(len(raw) // st.size)]
        self.assertEqual(len(got), 2)
        for exp, act in zip(records, got):
            self.assertEqual(exp, act)

    def test_remote_chunk_index(self):
        """read_chunk_index() works on remote .cz files."""
        records = [(100, 3, 10)]
        self._write_test_cz("idx.cz", records)
        url = f"{self.base_url}/idx.cz"
        r = Reader.from_url(url)
        idx = r.read_chunk_index()
        r.close()
        self.assertIsNotNone(idx)
        self.assertIn(("chr1",), idx)

    def test_remote_multi_chunk(self):
        """Remote reading with multiple chunks."""
        path = os.path.join(self.tmpdir, "multi.cz")
        fmt = "QHH"
        w = Writer(output=path, formats=["Q", "H", "H"],
                   columns=["pos", "mc", "cov"], dimensions=["chrom"])
        chr1 = [(100, 1, 5), (200, 2, 10)]
        chr2 = [(50, 3, 15), (300, 4, 20), (500, 5, 25)]
        w.write_chunk(b"".join(struct.pack(f"<{fmt}", *r) for r in chr1), ["chr1"])
        w.write_chunk(b"".join(struct.pack(f"<{fmt}", *r) for r in chr2), ["chr2"])
        w.close()

        url = f"{self.base_url}/multi.cz"
        r = Reader.from_url(url)
        got1 = list(r.fetch(("chr1",)))
        got2 = list(r.fetch(("chr2",)))
        r.close()
        self.assertEqual(len(got1), 2)
        self.assertEqual(len(got2), 3)
        for exp, act in zip(chr1, got1):
            self.assertEqual(exp, tuple(act))
        for exp, act in zip(chr2, got2):
            self.assertEqual(exp, tuple(act))

    def test_remote_delta(self):
        """Remote reading with .cz files (delta removed, test kept for coverage)."""
        records = [(100, 3, 10), (250, 5, 12), (300, 0, 8), (1000, 2, 15)]
        self._write_test_cz("delta.cz", records)
        url = f"{self.base_url}/delta.cz"
        r = Reader.from_url(url)
        got = list(r.fetch(("chr1",)))
        r.close()
        self.assertEqual(len(got), 4)
        for exp, act in zip(records, got):
            self.assertEqual(exp, tuple(act))

    def test_remote_query(self):
        """query() works on remote .cz files."""
        records = [(100, 3, 10), (200, 5, 12), (300, 0, 8),
                   (400, 2, 15), (500, 7, 20)]
        self._write_test_cz("query.cz", records)
        url = f"{self.base_url}/query.cz"
        r = Reader.from_url(url)
        got = list(r.query(dimension="chr1", start=200, end=400,
                           query_col=[0], printout=False))
        # query returns records where record[s] >= start and record[e] <= end
        # Records with pos 200, 300 should match (pos >= 200 and pos <= 400)
        self.assertTrue(len(got) >= 2)
        r.close()

    def test_remote_print_header(self):
        """print_header works on remote files."""
        records = [(100, 3, 10)]
        self._write_test_cz("header.cz", records)
        url = f"{self.base_url}/header.cz"
        r = Reader.from_url(url)
        r.print_header()  # should not raise
        r.close()

    def test_local_unchanged(self):
        """Local files still work exactly as before."""
        records = [(100, 3, 10), (200, 5, 12)]
        path = self._write_test_cz("local.cz", records)
        r = Reader(path)
        self.assertFalse(r._is_remote)
        got = list(r.fetch(("chr1",)))
        r.close()
        self.assertEqual(len(got), 2)
        for exp, act in zip(records, got):
            self.assertEqual(exp, tuple(act))


def _figshare_session():
    """Create a requests.Session configured for Figshare downloads."""
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://figshare.com/",
        "Accept": "*/*",
    })
    session.get("https://figshare.com")  # acquire cookies
    return session


@unittest.skipUnless(
    os.environ.get("CYTOZIP_TEST_FIGSHARE", "0") == "1",
    "Set CYTOZIP_TEST_FIGSHARE=1 to run Figshare remote tests",
)
class TestFigshareRemote(unittest.TestCase):
    """Integration tests against real Figshare-hosted .cz files."""

    # UCI2424_CX1718BS0102_MGM1_1_P10-1-I17-A16.with_coordinate.cz
    URL_WITH_COORD = "https://figshare.com/ndownloader/files/63531984"
    # FC_E17a_3C_1-1-I3-F13.cz  (no coordinates)
    URL_NO_COORD = "https://figshare.com/ndownloader/files/63531981"

    @classmethod
    def setUpClass(cls):
        cls.session = _figshare_session()

    def test_figshare_with_coordinate_header(self):
        """Read header from Figshare .cz file with coordinates."""
        reader = Reader.from_url(self.URL_WITH_COORD, session=self.session)
        self.assertTrue(reader._is_remote)
        self.assertIn("formats", reader.header)
        self.assertIn("columns", reader.header)
        self.assertGreater(len(reader.header["formats"]), 0)
        reader.close()

    def test_figshare_with_coordinate_summary(self):
        """summary_chunks returns a DataFrame for Figshare .cz file."""
        import pandas as pd
        reader = Reader.from_url(self.URL_WITH_COORD, session=self.session)
        df = reader.summary_chunks(printout=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        reader.close()

    def test_figshare_with_coordinate_query(self):
        """query() on Figshare .cz file with coordinates returns records."""
        reader = Reader.from_url(self.URL_WITH_COORD, session=self.session)
        records = list(reader.query(
            dimension="chr9", start=60610139, end=60610151, printout=False,
        ))
        self.assertGreater(len(records), 0)
        reader.close()

    def test_figshare_no_coordinate_header(self):
        """Read header from Figshare .cz file without coordinates."""
        reader = Reader.from_url(self.URL_NO_COORD, session=self.session)
        self.assertTrue(reader._is_remote)
        self.assertIn("formats", reader.header)
        reader.close()

    def test_figshare_no_coordinate_summary(self):
        """summary_chunks returns a DataFrame for Figshare .cz (no coords)."""
        import pandas as pd
        reader = Reader.from_url(self.URL_NO_COORD, session=self.session)
        df = reader.summary_chunks(printout=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        reader.close()

    def test_figshare_no_coordinate_fetch(self):
        """fetch() on Figshare .cz file returns records."""
        reader = Reader.from_url(self.URL_NO_COORD, session=self.session)
        dims = list(reader.dim2chunk_start.keys())
        self.assertGreater(len(dims), 0)
        # Fetch first chunk, read a few records
        records = []
        for i, rec in enumerate(reader.fetch(dims[0])):
            records.append(rec)
            if i >= 9:
                break
        self.assertEqual(len(records), 10)
        reader.close()


if __name__ == "__main__":
    unittest.main()
