import os
import tempfile
import unittest

from cytozip.cz import _parse_chrom_whitelist


class TestParseChromWhitelist(unittest.TestCase):
    def test_chrom_sizes_file(self):
        # cz.py defines a module-level open() returning a Reader; reading a plain
        # text chrom.sizes file must still go through the builtin open
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'chrom.sizes')
            with open(path, 'w') as fh:
                fh.write("chr1\t248956422\nchr2\t242193529\n\nchrM\t16569\n")
            self.assertEqual(_parse_chrom_whitelist(path), {'chr1', 'chr2', 'chrM'})

    def test_other_inputs(self):
        self.assertIsNone(_parse_chrom_whitelist(None))
        self.assertEqual(_parse_chrom_whitelist('chr1,chr2'), {'chr1', 'chr2'})
        self.assertEqual(_parse_chrom_whitelist('chrX'), {'chrX'})
        self.assertEqual(_parse_chrom_whitelist(['chr1', 'chr2']), {'chr1', 'chr2'})


if __name__ == '__main__':
    unittest.main()
