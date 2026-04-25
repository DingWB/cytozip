import unittest
import tempfile
import os
import struct

from cytozip.cz import Writer, Reader

class TestUInt(unittest.TestCase):
    def test_uint32_roundtrip(self):
        vals = [0, 1, 123456789, 2**32 - 1]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'u.cz')
            w = Writer(output=path, formats=['I'], columns=['val'], chunk_dims=['chr'])
            data = b''.join(struct.pack("<I", v) for v in vals)
            w.write_chunk(data, ['chr1'])
            w.close()

            r = Reader(path)
            got = [row[0] for row in r.__fetch__(('chr1',), s=0, e=1)]
            r.close()
            self.assertEqual(got, vals)

if __name__ == '__main__':
    unittest.main()
