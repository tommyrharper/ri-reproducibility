#!/usr/bin/env python3
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
        root = Path(tmp) / "eval 0001"
        root.mkdir()
        out, err = root / "stdout.log", root / "stderr.log"

        code, seconds, peak = ask(request(root, out, err, "wsclean", "--version"))
        assert code == 0, code
        assert "WSClean version" in out.read_text(), out.read_text()
        assert err.read_text() == "", err.read_text()
        assert seconds > 0.0, seconds
        assert peak > 1024 * 1024, peak

        code, _, _ = ask(request(root, out, err, "wsclean", "--no-such-flag"))
        assert code != 0, code
        assert "no-such-flag" in out.read_text() + err.read_text()

        assert ask("garbage\n")[0] == 126

        code, _, _ = ask(request(root, out, err, "wsclean", "--version"))
        assert code == 0, code

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
