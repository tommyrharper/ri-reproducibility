#!/usr/bin/env python3
"""Static file server for the generated report, with cache headers that suit it.

`python3 -m http.server` revalidates every file on every visit, which over an
ssh tunnel means a round trip per thumbnail each time a page is reopened. The
report's images are content-addressed (see cached_png() in generate_report.py),
so their URL changes whenever their bytes do and they never need revalidating.
"""

import argparse
import functools
import hashlib
import os
import re
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


# Same construction as generate_report.REPORT_VERSION, recomputed rather than
# imported: that module pulls in common.py, which needs a newer Python than the
# host may have, and serving must not depend on it.
GENERATOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_report.py")
PAGE_VERSION_RE = re.compile(r'<meta name="report-version" content="([0-9a-f]+)">')


def stale_report_warning(directory, generator=GENERATOR):
    """A line to print when the pages on disk predate the current generator."""
    index = os.path.join(directory, "index.html")
    if not (os.path.exists(index) and os.path.exists(generator)):
        return None
    with open(index, encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    match = PAGE_VERSION_RE.search(head)
    current = hashlib.sha256(open(generator, "rb").read()).hexdigest()[:12]
    if match and match.group(1) == current:
        return None
    return "note: these pages predate the current report code - run './ri report' to rebuild"


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

    with tempfile.TemporaryDirectory() as tmp_dir:
        generator = os.path.join(tmp_dir, "generate_report.py")
        with open(generator, "w") as f:
            f.write("# pretend generator\n")
        version = hashlib.sha256(open(generator, "rb").read()).hexdigest()[:12]
        index = os.path.join(tmp_dir, "index.html")

        def write_index(stamp):
            with open(index, "w") as f:
                f.write(f'<meta name="report-version" content="{stamp}">')

        write_index(version)
        assert stale_report_warning(tmp_dir, generator) is None
        write_index("deadbeef0000")
        assert stale_report_warning(tmp_dir, generator) is not None
        os.remove(index)
        assert stale_report_warning(tmp_dir, generator) is None

    print("ok report_server serves images immutable, pages briefly fresh, warns when stale")


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
    warning = stale_report_warning(args.directory)
    if warning:
        print(warning, flush=True)
    serve(args.bind, args.port, args.directory)


if __name__ == "__main__":
    main()
