## bam to allc bug:
```
error log: [  2/72] FAIL UWA7648_CX1819_NAC_1_P1-1-I3-A20         allc=  42.7s/   810MB  cz=  52.8s/  1164MB
     allc_err: /x-wding2/Software/conda/m3c/lib/python3.10/codecs.py", line 322, in decode
    (result, consumed) = self._buffer_decode(data, self.errors, final)
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb1 in position 4224: invalid start byte
Command exited with non-zero status 1
```