#!/usr/bin/env python3
"""Static server for the WJM WebDemoMarkers glyph-detection demo.

    python3 serve.py                      # http://localhost:8000  (camera OK on localhost)
    python3 serve.py --host 0.0.0.0       # reachable on the LAN  (needs HTTPS for the camera)
    python3 serve.py --https --cert cert.pem --key key.pem

`getUserMedia` only runs in a secure context: `localhost`/`127.0.0.1` over plain
HTTP is fine, but a phone hitting this box over the LAN needs HTTPS. Make a throwaway
cert with:

    openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem \\
            -days 365 -subj "/CN=$(hostname)" \\
            -addext "subjectAltName=IP:<this-machine-ip>"
"""

from __future__ import annotations

import argparse
import http.server
import socket
import ssl
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".wasm": "application/wasm",
        ".json": "application/json",
    }

    def end_headers(self) -> None:
        # never cache during a demo/dev session
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:  # quieter
        print(f"  {self.address_string()} - {fmt % args}")


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--https", action="store_true", help="serve over TLS (needs --cert/--key)")
    ap.add_argument("--cert", default="cert.pem")
    ap.add_argument("--key", default="key.pem")
    args = ap.parse_args()

    handler = partial(Handler, directory=str(ROOT))
    httpd = http.server.ThreadingHTTPServer((args.host, args.port), handler)

    scheme = "http"
    if args.https:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile=args.cert, keyfile=args.key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        scheme = "https"

    ip = lan_ip()
    print(f"WJM WebDemoMarkers — serving {ROOT}")
    print(f"  local : {scheme}://localhost:{args.port}/")
    if args.host in ("0.0.0.0", "::"):
        print(f"  phone : {scheme}://{ip}:{args.port}/   "
              f"({'open this on the device' if args.https else 'HTTP over LAN blocks the camera — use --https'})")
    print("  Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
