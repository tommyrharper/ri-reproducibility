#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

import common  # noqa: E402

SIMULATE_SOURCE = REPO_ROOT / "scripts" / "lib" / "nested_sampling" / "simulate_point_source_ms.py"

# Recorded slowest stages; update bounds with measurements.
SLOWEST_SIMULATE_SECONDS = 0.60
SLOWEST_CONVERT_SECONDS = 1.42
SLOWEST_PREDICT_SECONDS = 0.34


def predict_wait_seconds() -> float:
    match = re.search(r"^PREDICT_WAIT_SECONDS = ([\d.]+)$", SIMULATE_SOURCE.read_text(), re.MULTILINE)
    assert match, "PREDICT_WAIT_SECONDS is no longer a plain literal in simulate_point_source_ms.py"
    return float(match.group(1))


def check_timeout_ladder() -> None:
    predict = predict_wait_seconds()
    assert predict < common.SIMULATE_REPLY_TIMEOUT, (
        f"the worker's own bound ({predict}s) must expire before the rank's "
        f"({common.SIMULATE_REPLY_TIMEOUT}s), or the rank kills the worker "
        "before it can fix itself"
    )
    assert common.SIMULATE_REPLY_TIMEOUT > 2 * predict, (
        f"the rank's bound ({common.SIMULATE_REPLY_TIMEOUT}s) leaves no room for "
        f"the worker's two attempts at {predict}s"
    )
    assert common.SIMULATE_REPLY_TIMEOUT < common.SHELL_REPLY_TIMEOUT <= common.IMAGING_REPLY_TIMEOUT

    assert predict > 3 * SLOWEST_PREDICT_SECONDS, (
        f"PREDICT_WAIT_SECONDS={predict}s is too close to the slowest predict "
        f"on record ({SLOWEST_PREDICT_SECONDS}s)"
    )
    slowest_covered = max(SLOWEST_SIMULATE_SECONDS, SLOWEST_CONVERT_SECONDS)
    assert common.SIMULATE_REPLY_TIMEOUT > 3 * slowest_covered, (
        f"SIMULATE_REPLY_TIMEOUT={common.SIMULATE_REPLY_TIMEOUT}s is too close to "
        f"the slowest stage it covers ({slowest_covered}s, the R2D2 convert)"
    )


def check_fifo_kill_pattern() -> None:
    base = Path("/repo/results/nested-sampling/run/.simulate-workers/1")
    pattern = common.fifo_worker_pgrep_pattern(base)

    argv = f"python3 /opt/ri-nested-sampling/simulate_point_source_ms.py --serve --fifo {base}"
    assert re.search(pattern, argv), f"{pattern!r} no longer matches the worker argv {argv!r}"

    sibling = argv.replace(str(base), str(base.parent / "12"))
    assert not re.search(pattern, sibling), f"{pattern!r} also matches rank 12"

    carrier = f"sh -c p=$(pgrep -f '{pattern}') || exit 0; kill -9 $(pgrep -P $p) $p"
    assert not re.search(pattern, carrier), f"{pattern!r} matches the shell running it"
    assert not re.search(pattern, f"pgrep -f {pattern}"), f"{pattern!r} matches its own pgrep"


def check_worker_died_is_not_a_score() -> None:
    assert common.WORKER_DIED < 0, "WORKER_DIED must not collide with a real exit status"
    assert common.FAILURE_OBJECTIVE > 0
    assert common.WORKER_RETRY_DELAYS and all(d >= 0 for d in common.WORKER_RETRY_DELAYS)


def main() -> None:
    for check in (
        common.self_check_worker_timeout,
        check_timeout_ladder,
        check_fifo_kill_pattern,
        check_worker_died_is_not_a_score,
    ):
        check()
        print(f"ok   {check.__name__}")
    print("all watchdog checks passed")


if __name__ == "__main__":
    main()
