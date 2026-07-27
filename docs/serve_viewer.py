"""Local serve helper for cz_viewer.html: static file server + CORS/Range proxy.

Serves the repository over one port AND, for paths under
``/proxy?url=<encoded target>``, forwards GET/HEAD (preserving the Range header)
to the target with permissive CORS headers. This lets cz_viewer.html read remote
.cz files whose server does NOT send ``Access-Control-Allow-Origin`` (e.g. a plain
Apache FTP mirror), which the browser would otherwise block.

Usage:
    python docs/serve_viewer.py            # serves repo root on :8877
    python docs/serve_viewer.py 9000       # custom port

Then open  http://127.0.0.1:8877/docs/source/_static/cz_viewer.html  and, in the
page, set the "CORS proxy prefix" field to  /proxy?url=  so .cz fetches are routed
through the proxy. (The cz_reader.mjs module URL can stay on raw.githubusercontent
once the updated reader is pushed, or point it at /cytozip/cz_reader.mjs to use
this repo.)
"""
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PASS = ("Content-Length", "Content-Range", "Accept-Ranges", "Content-Type")
# Default document root = repo root (parent of this docs/ folder).
_DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else _DEFAULT_ROOT


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=ROOT, **k)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers",
                         "Content-Range, Accept-Ranges, Content-Length, Content-Type")

    def _is_proxy(self):
        return urllib.parse.urlparse(self.path).path == "/proxy"

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def _forward(self, method):
        qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        target = qs.get("url", [None])[0]
        if not target:
            self.send_response(400); self._cors(); self.end_headers(); return
        headers = {}
        if self.headers.get("Range"):
            headers["Range"] = self.headers.get("Range")
        try:
            resp = urllib.request.urlopen(
                urllib.request.Request(target, method=method, headers=headers), timeout=60)
        except urllib.error.HTTPError as e:
            resp = e
        except Exception as e:
            self.send_response(502); self._cors(); self.end_headers()
            if method == "GET":
                self.wfile.write(str(e).encode())
            return
        self.send_response(resp.status)
        for h in PASS:
            v = resp.headers.get(h)
            if v is not None:
                self.send_header(h, v)
        self._cors(); self.end_headers()
        if method == "GET":
            try:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            except Exception:
                pass

    def do_GET(self):
        if self._is_proxy():
            self._forward("GET")
        else:
            super().do_GET()

    def do_HEAD(self):
        if self._is_proxy():
            self._forward("HEAD")
        else:
            super().do_HEAD()

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8877
    print(f"Serving {ROOT} on http://127.0.0.1:{port}")
    print(f"Open    http://127.0.0.1:{port}/docs/source/_static/cz_viewer.html")
    print("In the page, set CORS proxy prefix to  /proxy?url=")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
