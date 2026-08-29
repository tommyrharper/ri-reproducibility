#!/usr/bin/env python3
"""Check the `wsclean-zygote` fork server's protocol, inside the WSClean image.

The zygote is what every WSClean evaluation now runs through
(`zygote_run()` in scripts/lib/nested_sampling/common.py), so a break in its
request/reply framing stops a search rather than slowing one down. This is the
host side of that contract exercised against the real binary: it needs the
image, so `./ri self-check zygote` runs it and CI cannot.

What it does not check is that a forked image equals a non-forked one - that
needs a Measurement Set, and it is what the FITS comparison and the end-to-end
A/B in docs/nested-sampling-wsclean-zygote.md are for.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ZYGOTE = "wsclean-zygote"


def request(workdir: Path, out: Path, err: Path, *argv: str) -> str:
    return "\t".join([str(workdir), str(out), str(err), *argv]) + "\n"


def main() -> int:
    if shutil.which(ZYGOTE) is None:
        print(f"FAIL: {ZYGOTE} is not on PATH; run this inside the WSClean image")
        return 1

    server = subprocess.Popen(
        [ZYGOTE], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    )

    def ask(line: str) -> tuple[int, float, int]:
        server.stdin.write(line)
        server.stdin.flush()
        reply = server.stdout.readline()
        assert reply, "zygote closed its stdout"
        code, seconds, peak = reply.split("\t")
        return int(code), float(seconds), int(peak)

    with tempfile.TemporaryDirectory() as tmp:
        # A directory whose name has a space in it, because the request is tab
        # separated and nothing quotes it.
        root = Path(tmp) / "eval 0001"
        root.mkdir()
        out, err = root / "stdout.log", root / "stderr.log"

        code, seconds, peak = ask(request(root, out, err, "wsclean", "--version"))
        assert code == 0, code
        assert "WSClean version" in out.read_text(), out.read_text()
        assert err.read_text() == "", err.read_text()
        # Both come from wait4()'s rusage, and replace a `/usr/bin/time -v` this
        # no longer forks. A zero here means the reply is being made up.
        assert seconds > 0.0, seconds
        assert peak > 1024 * 1024, peak

        # The server keeps serving: a failing request must not end it, and the
        # next one must still be answered.
        code, _, _ = ask(request(root, out, err, "wsclean", "--no-such-flag"))
        assert code != 0, code
        assert "no-such-flag" in out.read_text() + err.read_text()

        # An unparseable request is answered rather than desynchronising the
        # stream - one lost reply would wedge the rank that sent it.
        assert ask("garbage\n")[0] == 126

        code, _, _ = ask(request(root, out, err, "wsclean", "--version"))
        assert code == 0, code

        # cwd is honoured: a relative output path lands in the evaluation
        # directory rather than wherever the rank happened to be.
        assert Path.cwd() != root, "the check below would be vacuous"
        code, _, _ = ask(request(root, "relative.log", "relative.err", "wsclean", "--version"))
        assert code == 0, code
        assert "WSClean version" in (root / "relative.log").read_text()

    server.stdin.close()
    assert server.wait(timeout=30) == 0, server.returncode
    print("wsclean zygote self-check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
