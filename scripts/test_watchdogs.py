#!/usr/bin/env python3
"""Regression checks for the three layers that survive a MeqTrees deadlock.

A predict that never answers used to stop a whole run: the rank waiting on it
blocked forever and the other ranks burned a core each in PolyChord's
collective behind it. Three layers now stand between that and a stopped run,
and each has a way of failing silently, which is what these pin down:

  1. the worker bounds its own predict and replaces its meqserver
     (PREDICT_WAIT_SECONDS, scripts/lib/nested_sampling/simulate_point_source_ms.py)
  2. the rank bounds the worker's reply and kills the worker
     (SIMULATE_REPLY_TIMEOUT, scripts/lib/nested_sampling/common.py)
  3. a worker that will not come back stops the run rather than being scored
     (WORKER_DIED, same file)

Only the parts that need no Docker live here, so CI runs them on every change.
The rest - the real Timba conversation, and a wedged worker dying instead of
answering - need a meqserver, and are self-checks inside the images:
`./ri self-check` runs those.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "lib" / "nested_sampling"))

import common  # noqa: E402

SIMULATE_SOURCE = REPO_ROOT / "scripts" / "lib" / "nested_sampling" / "simulate_point_source_ms.py"

# The slowest each bounded stage has ever run, over the 17,644 evaluations on
# record when these bounds were set. Update alongside the bounds themselves, not
# to make a failure go away: a bound that no longer clears its stage fires on
# work that was only slow, and kills a worker that was doing its job.
SLOWEST_SIMULATE_SECONDS = 0.60
SLOWEST_CONVERT_SECONDS = 1.42
SLOWEST_PREDICT_SECONDS = 0.34


def predict_wait_seconds() -> float:
    """PREDICT_WAIT_SECONDS, read out of the source rather than imported.

    simulate_point_source_ms.py imports casacore and numpy at module scope and
    runs only inside the meqtrees image, which is also the reason it cannot
    import common.py: the image bind-mounts that file at build time to bake the
    skeleton cache and never installs it. The two constants therefore never
    share an interpreter, and the one relationship between them that matters
    can only be checked by reading it.
    """
    match = re.search(r"^PREDICT_WAIT_SECONDS = ([\d.]+)$", SIMULATE_SOURCE.read_text(), re.MULTILINE)
    assert match, "PREDICT_WAIT_SECONDS is no longer a plain literal in simulate_point_source_ms.py"
    return float(match.group(1))


def check_worker_reply_timeout() -> None:
    """common.py's own check that a silent worker is killed, retried, dropped.

    It lives there because it needs that module's internals; it is called from
    here because until this file existed nothing ran it outside a container.
    """
    common.self_check_worker_timeout()


def check_timeout_ladder() -> None:
    """Each bound has to fire before the one above it, with room over real work.

    The ordering is the whole design. If the worker's own bound ever exceeds the
    rank's, the rank kills the worker before it can replace its meqserver, and
    every deadlock silently costs a killed worker and a lost pool slot again -
    the exact behaviour the in-process layer was added to stop. Nothing else
    would report that; the run would just get slower.
    """
    predict = predict_wait_seconds()
    assert predict < common.SIMULATE_REPLY_TIMEOUT, (
        f"the worker's own bound ({predict}s) must expire before the rank's "
        f"({common.SIMULATE_REPLY_TIMEOUT}s), or the rank kills the worker "
        "before it can fix itself"
    )
    # Two full attempts plus a meqserver restart have to fit, or the rank's
    # bound cuts the recovery short on the retry rather than on the failure.
    assert common.SIMULATE_REPLY_TIMEOUT > 2 * predict, (
        f"the rank's bound ({common.SIMULATE_REPLY_TIMEOUT}s) leaves no room for "
        f"the worker's two attempts at {predict}s"
    )
    assert common.SIMULATE_REPLY_TIMEOUT < common.SHELL_REPLY_TIMEOUT <= common.IMAGING_REPLY_TIMEOUT

    # Every bound has to clear the slowest honest run of the stage it covers.
    assert predict > 3 * SLOWEST_PREDICT_SECONDS, (
        f"PREDICT_WAIT_SECONDS={predict}s is too close to the slowest predict "
        f"on record ({SLOWEST_PREDICT_SECONDS}s)"
    )
    # SIMULATE_REPLY_TIMEOUT covers the R2D2-only MS-to-.mat convert as well as
    # the simulate, and the convert is the slower of the two.
    slowest_covered = max(SLOWEST_SIMULATE_SECONDS, SLOWEST_CONVERT_SECONDS)
    assert common.SIMULATE_REPLY_TIMEOUT > 3 * slowest_covered, (
        f"SIMULATE_REPLY_TIMEOUT={common.SIMULATE_REPLY_TIMEOUT}s is too close to "
        f"the slowest stage it covers ({slowest_covered}s, the R2D2 convert)"
    )


def check_fifo_kill_pattern() -> None:
    """The pattern that kills a wedged pooled worker must still match one.

    A pattern that stops matching fails silently and expensively: the kill is a
    no-op, the worker keeps its half of the FIFO pair open, so the next attempt
    reconnects to the same wedged process and every retry finds the same corpse.
    Nothing raises. The three ways it can go wrong are all covered below.
    """
    base = Path("/repo/results/nested-sampling/run/.simulate-workers/1")
    pattern = common.fifo_worker_pgrep_pattern(base)

    # 1. It matches the argv scripts/run-nested-sampling.sh actually launches.
    argv = f"python3 /opt/ri-nested-sampling/simulate_point_source_ms.py --serve --fifo {base}"
    assert re.search(pattern, argv), f"{pattern!r} no longer matches the worker argv {argv!r}"

    # 2. It does not match a rank this one's number is a prefix of. Without the
    #    anchor, killing rank 1 would take out ranks 10 through 19 with it.
    sibling = argv.replace(str(base), str(base.parent / "12"))
    assert not re.search(pattern, sibling), f"{pattern!r} also matches rank 12"

    # 3. It does not match a command line that merely quotes it. The `sh -c`
    #    that runs the kill carries this pattern verbatim, and a pattern that
    #    matched it would have the kill take out the shell doing the killing.
    carrier = f"sh -c p=$(pgrep -f '{pattern}') || exit 0; kill -9 $(pgrep -P $p) $p"
    assert not re.search(pattern, carrier), f"{pattern!r} matches the shell running it"
    assert not re.search(pattern, f"pgrep -f {pattern}"), f"{pattern!r} matches its own pgrep"


def check_worker_died_is_not_a_score() -> None:
    """WORKER_DIED must stay outside the range a real objective can take.

    A failed evaluation scores FAILURE_OBJECTIVE and PolyChord maximizes it, so
    a host fault that reached the sampler by the same door would make the search
    chase dead workers instead of the algorithm. The two are only kept apart by
    WORKER_DIED being a returncode no simulate or imager can return.
    """
    assert common.WORKER_DIED < 0, "WORKER_DIED must not collide with a real exit status"
    assert common.FAILURE_OBJECTIVE > 0
    assert common.WORKER_RETRY_DELAYS and all(d >= 0 for d in common.WORKER_RETRY_DELAYS)


def main() -> None:
    for check in (
        check_worker_reply_timeout,
        check_timeout_ladder,
        check_fifo_kill_pattern,
        check_worker_died_is_not_a_score,
    ):
        check()
        print(f"ok   {check.__name__}")
    print("all watchdog checks passed")


if __name__ == "__main__":
    main()
