#!/usr/bin/env python3
"""Static file server for the generated report, with cache headers that suit it.

`python3 -m http.server` revalidates every file on every visit, which over an
ssh tunnel means a round trip per thumbnail each time a page is reopened. The
report's images are content-addressed (see cached_png() in generate_report.py),
so their URL changes whenever their bytes do and they never need revalidating.
"""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

IMAGE_PREFIX = "/images/"
# A year, the longest max-age worth expressing (RFC 9111 suggests capping here).
IMMUTABLE = "public, max-age=31536000, immutable"
# Pages are rewritten in place by './ri report', so they cannot be immutable.
# A minute of freshness is enough to make clicking between a run, its images
# and its table free, while a reload still revalidates immediately.
PAGE_CACHE = "max-age=60"


class ReportHandler(SimpleHTTPRequestHandler):
    # The default HTTP/1.0 closes the connection after every file, which over an
    # ssh tunnel means a fresh round trip per thumbnail. 1.1 keeps it open.
    protocol_version = "HTTP/1.1"

    def end_headers(self):
        self.send_header(
            "Cache-Control",
            IMMUTABLE if self.path.startswith(IMAGE_PREFIX) else PAGE_CACHE,
        )
        super().end_headers()


def serve(bind, port, directory):
    handler = functools.partial(ReportHandler, directory=directory)
    with ThreadingHTTPServer((bind, port), handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def self_check():
    import os
    import tempfile
    import threading
    import urllib.request

    with tempfile.TemporaryDirectory() as tmp_dir:
        os.makedirs(os.path.join(tmp_dir, "images"))
        with open(os.path.join(tmp_dir, "index.html"), "w") as f:
            f.write("<p>report</p>")
        with open(os.path.join(tmp_dir, "images", "abc123.png"), "wb") as f:
            f.write(b"not really a png")

        handler = functools.partial(ReportHandler, directory=tmp_dir)
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as httpd:
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{httpd.server_port}"
            try:
                page = urllib.request.urlopen(base + "/index.html")
                image = urllib.request.urlopen(base + "/images/abc123.png")
            finally:
                httpd.shutdown()

    assert page.headers["Cache-Control"] == PAGE_CACHE, page.headers.items()
    assert image.headers["Cache-Control"] == IMMUTABLE, image.headers.items()
    assert page.version == 11, page.version  # keep-alive, not a connection per file
    print("ok report_server serves images immutable and pages briefly fresh")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("port", type=int, nargs="?")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--directory")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        self_check()
        return
    if args.port is None or not args.directory:
        parser.error("port and --directory are required when serving")
    serve(args.bind, args.port, args.directory)


if __name__ == "__main__":
    main()
