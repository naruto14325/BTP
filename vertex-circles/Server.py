"""
Small backend for the "Vertex circles in motion" site.

Serves the static frontend from ./public and exposes the model
parameters (eyebrow poses, gait settings) as JSON, so the numbers
live in one place instead of being hard-coded in the browser code.

Run:  python3 server.py          (then open http://localhost:8000)
      python3 server.py 9000     (pick another port)
"""

import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC_DIR = ROOT / "public"
PRESETS_FILE = ROOT / "data" / "presets.json"


def load_presets():
    with open(PRESETS_FILE, encoding="utf-8") as f:
        return json.load(f)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def do_GET(self):
        path = self.path.split("?", 1)[0]

        if path in ("/api/presets", "/data/presets.json"):
            # Re-read on every request so edits to presets.json show up on refresh.
            self.send_json(load_presets())
        elif path.startswith("/api/presets/"):
            key = path.rsplit("/", 1)[-1]
            presets = load_presets()
            if key in presets:
                self.send_json(presets[key])
            else:
                self.send_json({"error": f"unknown preset '{key}'"}, status=404)
        else:
            super().do_GET()

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving on http://localhost:{port}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
