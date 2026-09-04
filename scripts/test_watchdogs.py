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

# Idle single-rank predict against makems' NTimes, measured on the meqtrees
# image (nchan makes almost no difference: 1200x4 and 1200x8 are 1.82s and
# 1.88s). The ladder has to hold against the top of this, not against the
# bottom, which is the whole reason both bounds scale.
MEASURED_PREDICT_SECONDS = {9: 0.04, 120: 0.20, 300: 0.49, 449: 0.72, 600: 0.97, 897: 1.42, 1200: 1.88}
# The largest NTimes the parameter space can ask for: 20 minutes at a 1s dump.
MOST_TIME_SAMPLES = 1200


def worker_default_predict_wait() -> float:
    """The bound a worker run straight from the CLI, with no rank sizing it, uses."""
    match = re.search(r"^PREDICT_WAIT_SECONDS = ([\d.]+)$", SIMULATE_SOURCE.read_text(), re.MULTILINE)
    assert match, "PREDICT_WAIT_SECONDS is no longer a plain literal in simulate_point_source_ms.py"
    return float(match.group(1))


def check_timeout_ladder() -> None:
    """The ladder has to hold at every shape, not just at the smallest one."""
    for n_times in (0, 1, *MEASURED_PREDICT_SECONDS, MOST_TIME_SAMPLES):
        predict = common.predict_wait_seconds(n_times)
        reply = common.simulate_reply_timeout(n_times)
        assert predict < reply, (
            f"at NTimes={n_times} the worker's own bound ({predict}s) must expire "
            f"before the rank's ({reply}s), or the rank kills the worker before "
            "it can fix itself"
        )
        assert reply > 2 * predict, (
            f"at NTimes={n_times} the rank's bound ({reply}s) leaves no room for "
            f"the worker's two attempts at {predict}s"
        )
        assert reply < common.SHELL_REPLY_TIMEOUT <= common.IMAGING_REPLY_TIMEOUT, (
            f"at NTimes={n_times} the rank's bound ({reply}s) is no longer inside "
            f"the imaging bounds around it"
        )

    # The bound never tightens as the job grows, which is what made one slow
    # point fatal: 15 minutes at a 1s dump measured 1.42s of predict against a
    # flat 3s bound, and lost the race on a loaded host.
    for n_times, measured in MEASURED_PREDICT_SECONDS.items():
        predict = common.predict_wait_seconds(n_times)
        assert predict > 3 * measured, (
            f"at NTimes={n_times} the bound ({predict}s) is too close to the "
            f"measured predict ({measured}s)"
        )

    # A worker nobody sized - a direct CLI run - still gets today's bound.
    assert worker_default_predict_wait() <= common.predict_wait_seconds(0)

    assert common.predict_wait_seconds(0) > 3 * SLOWEST_PREDICT_SECONDS, (
        f"the smallest bound ({common.predict_wait_seconds(0)}s) is too close to "
        f"the slowest predict on record ({SLOWEST_PREDICT_SECONDS}s)"
    )
    slowest_covered = max(SLOWEST_SIMULATE_SECONDS, SLOWEST_CONVERT_SECONDS)
    assert common.simulate_reply_timeout() > 3 * slowest_covered, (
        f"the smallest rank bound ({common.simulate_reply_timeout()}s) is too close "
        f"to the slowest stage it covers ({slowest_covered}s, the R2D2 convert)"
    )


def check_bounds_scale_with_the_measurement_set() -> None:
    """The evaluation's shape, not a constant, is what sizes both bounds."""
    small = {"observation_minutes": 0.3, "integration_seconds": 2}
    large = {"observation_minutes": 20.0, "integration_seconds": 1}
    assert common.evaluation_time_samples(small) == 9, common.evaluation_time_samples(small)
    assert common.evaluation_time_samples(large) == MOST_TIME_SAMPLES, common.evaluation_time_samples(large)
    # A degenerate observation still asks for one time sample, not zero.
    assert common.evaluation_time_samples({"observation_minutes": 0.0, "integration_seconds": 10}) == 1

    assert common.predict_wait_seconds(MOST_TIME_SAMPLES) > 3 * common.predict_wait_seconds(9), (
        f"the worker bound barely moves across the parameter space "
        f"({common.predict_wait_seconds(9)}s to "
        f"{common.predict_wait_seconds(MOST_TIME_SAMPLES)}s); it is scalar in all but name"
    )
    # The rank's bound carries a deliberately large fixed base, so it grows less
    # steeply than the worker's. What has to hold is that its headroom never
    # shrinks: every extra second the worker may spend on its two predicts is a
    # second the rank has already agreed to wait.
    for n_times in (*MEASURED_PREDICT_SECONDS, MOST_TIME_SAMPLES):
        grew = common.simulate_reply_timeout(n_times) - common.simulate_reply_timeout(9)
        allowed = common.WORKER_PREDICT_ATTEMPTS * (
            common.predict_wait_seconds(n_times) - common.predict_wait_seconds(9)
        )
        assert grew >= allowed, (
            f"from NTimes=9 to {n_times} the worker gained {allowed}s of predict "
            f"but the rank only gained {grew}s of patience"
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
        check_bounds_scale_with_the_measurement_set,
        check_fifo_kill_pattern,
        check_worker_died_is_not_a_score,
    ):
        check()
        print(f"ok   {check.__name__}")
    print("all watchdog checks passed")


if __name__ == "__main__":
    main()
