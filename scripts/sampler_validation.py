#!/usr/bin/env python3
"""Many-seed validation of NS_ADOPT_CONTOUR and NS_EARLY_STOP.

docs/csd3-sampler-validation.md holds the design and pass rule; this is its
tooling. `manifest` lists the array tasks, `extract` turns a finished run into
one JSON record, `volumes` adds each kept chain's contours on PolyChord's own
volumes, `analyse` turns those records into the report's tables.
Stdlib only, because it runs on compute nodes and in CI.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import re
import statistics
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from statistics import NormalDist
from typing import Any

PARTS: dict[str, dict[str, Any]] = {
    "A": {"seeds": list(range(9101, 9125)), "arms": ["stock", "stock-repeat", "adopt", "stop", "adopt+stop"],
          "shadow_seeds": list(range(9101, 9107)), "shadow_arms": ["shadow", "adopt+shadow"]},
    "B": {"seeds": list(range(9201, 9207)), "arms": ["stock", "stock-repeat", "adopt+stop"],
          "shadow_seeds": [], "shadow_arms": []},
}
PART_CPUS = {"A": 38, "B": 304}
ARM_ENV = {
    "stock": (0, 0), "stock-repeat": (0, 0), "adopt": (1, 0), "stop": (0, 1), "adopt+stop": (1, 1),
    "shadow": (0, 2), "adopt+shadow": (1, 2),
}
EARLY_STOP_RATIOS = {3: 0.82, 4: 0.86, 5: 0.84, 6: 0.82, 8: 0.84, 10: 0.87, 12: 0.90, 15: 0.93, 18: 0.95}
QUANTILES = (("q05", 0.05), ("q50", 0.50), ("q95", 0.95))
STATISTICS = ("mean", "sd", "q05", "q50", "q95")
LOG_X_POINTS = (-1, -2, -3, -4, -5, -6, -7, -8)
SHUFFLE_SEED = 20261030
FLIPS = 4000


def manifest(part: str) -> list[tuple[int, str]]:
    """Task order: seed by seed, arms shuffled within each seed, so no arm is
    always first onto the nodes. Shadow tasks come last."""
    spec = PARTS[part]
    rng = random.Random(SHUFFLE_SEED)
    tasks: list[tuple[int, str]] = []
    for seed in spec["seeds"]:
        arms = list(spec["arms"])
        rng.shuffle(arms)
        tasks += [(seed, arm) for arm in arms]
    for seed in spec["shadow_seeds"]:
        arms = list(spec["shadow_arms"])
        rng.shuffle(arms)
        tasks += [(seed, arm) for arm in arms]
    return tasks


def weighted_stats(values: list[float], weights: list[float]) -> dict[str, float]:
    total = sum(weights)
    mean = sum(w * v for v, w in zip(values, weights)) / total
    variance = sum(w * (v - mean) ** 2 for v, w in zip(values, weights)) / total
    ordered = sorted(zip(values, weights))
    stats = {"mean": mean, "sd": math.sqrt(variance)}
    for name, q in QUANTILES:
        running = 0.0
        for value, weight in ordered:
            running += weight
            if running >= q * total:
                stats[name] = value
                break
    return stats


def read_table(path: Path) -> list[list[float]]:
    # Fortran drops the E in three-digit exponents: 0.1-100 means 0.1E-100.
    fix = re.compile(r"(?<=\d)([+-]\d{3})$")
    return [[float(fix.sub(r"E\1", cell)) for cell in line.split()] for line in path.open() if line.strip()]


def parameter_names(chains: Path, stem: str, columns: int) -> list[str]:
    path = chains / f"{stem}.paramnames"
    names = [line.split()[0].rstrip("*") for line in path.open()] if path.exists() else []
    return names if len(names) == columns else [f"column{i + 1}" for i in range(columns)]


def wrong_stops(evaluations: list[dict[str, Any]]) -> dict[str, int] | None:
    """Replay the shipped rule over a shadow run's checks."""
    seen = above = would = wrong = 0
    for record in evaluations:
        checks = record.get("early_checks")
        contour = record.get("contour")
        if not checks or contour is None or contour <= -1e20 or record["objective"] >= 1e20:
            continue
        seen += 1
        is_above = record["objective"] > contour
        above += is_above
        stops = any(k in EARLY_STOP_RATIOS and rms < EARLY_STOP_RATIOS[k] * contour for k, rms in checks)
        would += stops
        wrong += stops and is_above
    return {"checked": seen, "above_contour": above, "would_stop": would, "wrong_stops": wrong} if seen else None


def extract(run_dir: Path, stem: str = "r2d2_vlaa") -> dict[str, Any]:
    chains = run_dir / "chains"
    text = (chains / f"{stem}.stats").read_text()
    number = r"(-?\d*\.\d+E?[+-]?\d+)"
    log_z, err = re.search(rf"^log\(Z\)\s*=\s*{number} \+/-\s*{number}", text, re.M).groups()
    record: dict[str, Any] = {
        "log_z": float(log_z), "log_z_err": float(err),
        "ndead": int(re.search(r"ndead:\s*(\d+)", text).group(1)),
        "nlike": int(re.search(r"nlike:\s*(\d+)", text).group(1)),
        "clusters": int(re.search(r"ncluster:\s*\d+ /\s*(\d+)", text).group(1)),
    }
    rows = read_table(chains / f"{stem}.txt")
    weights = [row[0] for row in rows]
    columns = len(rows[0]) - 2
    names = parameter_names(chains, stem, columns)
    record["posterior"] = {
        name: weighted_stats([row[i + 2] for row in rows], weights) for i, name in enumerate(names)
    }
    # Failed evaluations sit in the dead file at PolyChord's logzero.
    dead = sorted(row[0] for row in read_table(chains / f"{stem}_dead.txt") if row[0] > -1e20)
    summary = json.loads((run_dir / "summary.json").read_text())
    nlive = summary["polychord"]["nlive"]
    record["max_log_l"] = dead[-1]
    record["contour_at_log_x"] = {str(x): dead[-x * nlive] for x in LOG_X_POINTS if -x * nlive < len(dead)}
    record["wrong_stop"] = wrong_stops(summary.get("evaluations") or [])
    return record


def contours_at_volume(rows: list[list[float]]) -> dict[str, float]:
    """Objective at each ln X, from a .txt chain's weights, which carry PolyChord's own volumes.

    The dead file also holds chains that returned below the contour and were thrown away, so
    contour_at_log_x, indexed by dead-point count, sits on the wrong axis wherever chains go stale.
    """
    points = sorted(((-row[1] / 2, row[0]) for row in rows if row[0] > 0), reverse=True)
    total = sum(w * math.exp(-log_l) for log_l, w in points)
    contours: dict[str, float] = {}
    pending = sorted(LOG_X_POINTS)
    volume = 0.0
    for log_l, w in points:
        volume += w * math.exp(-log_l) / total
        while pending and volume >= math.exp(pending[0]):
            contours[str(pending.pop(0))] = log_l
    return contours


def add_volumes(path: Path, keep: Path) -> None:
    """Add contour_at_volume to every record, from the chains kept beside the run records."""
    rows = load([path])
    for r in rows:
        name = f"val-{r['part']}-{r['seed']}-{r['arm']}".replace("+", "-")
        with gzip.open(keep / name / "r2d2_vlaa.txt.gz", "rt") as chain:
            r["contour_at_volume"] = contours_at_volume([[float(v) for v in line.split()] for line in chain])
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def record(args: argparse.Namespace) -> None:
    """One line for a run, whatever happened to it, written atomically."""
    out = {"part": args.part, "seed": args.seed, "arm": args.arm, "exit": args.exit_code, "start": args.start,
           "end": args.end, "job": args.job, "nodes": args.nodes}
    try:
        out.update(extract(args.run_dir))
        # A shadow run exists only for this count, and a clean exit deletes its directory.
        if "shadow" in args.arm and not out["wrong_stop"]:
            raise ValueError("shadow run has no early checks")
        out.update(wall_s=json.loads(args.bench.read_text())["wall_s"], status="ok" if args.exit_code == 0 else "failed")
        if args.exit_code != 0:
            out["why"] = f"exit {args.exit_code}"
    except Exception as error:  # a broken run is a row, not a lost run
        out.update(status="failed", why=f"exit {args.exit_code}, extract: {error!r}")
    temp = args.out.with_suffix(".tmp")
    temp.write_text(json.dumps(out) + "\n")
    temp.rename(args.out)
    if out["status"] != "ok":
        raise SystemExit(out["why"])


def t_critical(df: int, p: float = 0.975) -> float:
    """Student's t quantile, Cornish-Fisher; 3 digits at df >= 5, which is all we use."""
    z = NormalDist().inv_cdf(p)
    return (z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
            + (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / (384 * df ** 3))


def mean_ci(values: list[float]) -> tuple[float, float, float, float]:
    n = len(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n)
    half = t_critical(n - 1) * se
    return mean, se, mean - half, mean + half


def sign_test(values: list[float]) -> float:
    plus = sum(v > 0 for v in values)
    minus = sum(v < 0 for v in values)
    n = plus + minus
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(plus, minus) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def sign_flips(n: int, rng: random.Random) -> list[tuple[int, ...]]:
    if n <= 12:
        return list(product((1, -1), repeat=n))
    return [(1,) * n] + [tuple(rng.choice((1, -1)) for _ in range(n)) for _ in range(FLIPS)]


def flip_p(values: list[float], flips: list[tuple[int, ...]]) -> float:
    observed = abs(sum(values))
    hits = sum(abs(sum(s * v for s, v in zip(flip, values))) >= observed * (1 - 1e-12) for flip in flips)
    return hits / len(flips)


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    running = 0.0
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    for i, (name, p) in enumerate(ordered):
        running = max(running, min(1.0, p * (len(ordered) - i)))
        adjusted[name] = running
    return adjusted


def scatter_p(arm: list[float], stock: list[float], rng: random.Random) -> float:
    """One-sided: how often swapping labels within seeds gives an sd ratio as high."""
    observed = statistics.stdev(arm) / statistics.stdev(stock)
    hits = 0
    for _ in range(FLIPS):
        a, b = [], []
        for x, y in zip(arm, stock):
            x, y = (x, y) if rng.random() < 0.5 else (y, x)
            a.append(x)
            b.append(y)
        hits += statistics.stdev(a) / statistics.stdev(b) >= observed
    return (hits + 1) / (FLIPS + 1)


def compare(runs: dict[str, dict[int, dict[str, Any]]], arm: str, rng: random.Random) -> dict[str, Any]:
    """Arm against stock, paired by seed, on seeds where both finished."""
    seeds = sorted(set(runs[arm]) & set(runs["stock"]))
    a = [runs[arm][s] for s in seeds]
    b = [runs["stock"][s] for s in seeds]
    z = [(x["log_z"] - y["log_z"]) / math.hypot(x["log_z_err"], y["log_z_err"]) for x, y in zip(a, b)]
    mean, se, low, high = mean_ci(z)
    out: dict[str, Any] = {
        "seeds": seeds, "z": z, "mean": mean, "se": se, "low": low, "high": high, "sd": statistics.stdev(z),
        "sign_p": sign_test(z), "plus": sum(v > 0 for v in z),
        "scatter_arm": statistics.stdev(x["log_z"] for x in a),
        "scatter_stock": statistics.stdev(y["log_z"] for y in b),
        "scatter_p": scatter_p([x["log_z"] for x in a], [y["log_z"] for y in b], rng),
    }
    flips = sign_flips(len(seeds), rng)
    params = list(a[0]["posterior"])
    out["holm"] = {}
    for stat in STATISTICS:
        raw = {
            p: flip_p([x["posterior"][p][stat] - y["posterior"][p][stat] for x, y in zip(a, b)], flips)
            for p in params
        }
        adjusted = holm(raw)
        out["holm"][stat] = {"raw": raw, "adjusted": adjusted}
    out["significant"] = sorted(
        f"{p} {stat}" for stat, table in out["holm"].items() for p, q in table["adjusted"].items() if q < 0.05
    )
    for label, key in (("wall", "wall_s"), ("evals", "nlike")):
        ratios = [100 * (x[key] / y[key] - 1) for x, y in zip(a, b)]
        out[label] = mean_ci(ratios)
    return out


def verdict(result: dict[str, Any], control_significant: int) -> str:
    """The design's pass rule, unchanged. Inconclusive is whatever is neither."""
    held = (abs(result["mean"]) <= 0.25 and result["low"] > -0.5 and result["high"] < 0.5
            and len(result["significant"]) <= control_significant and result["scatter_p"] >= 0.05)
    if held:
        return "PASS"
    shifted = abs(result["mean"]) > 0.25 and (result["low"] > 0 or result["high"] < 0)
    if shifted or len(result["significant"]) > control_significant or result["scatter_p"] < 0.05:
        return "FAIL"
    return "INCONCLUSIVE"


def tail_differences(runs: dict[str, dict[int, dict[str, Any]]], arms: list[str],
                     key: str = "contour_at_log_x") -> list[str]:
    """Secondary: paired difference from stock at each tail point. * = interval excludes 0."""
    points: list[tuple[str, Any]] = [("max objective", lambda r: r.get("max_log_l"))]
    points += [(f"ln X {x}", lambda r, x=x: r.get(key, {}).get(str(x))) for x in LOG_X_POINTS]
    lines = ["| point | " + " | ".join(arms) + " |", "| --- |" + " --- |" * len(arms)]
    for label, read in points:
        cells = []
        for arm in arms:
            pairs = [(read(runs[arm][s]), read(runs["stock"][s])) for s in sorted(set(runs[arm]) & set(runs["stock"]))]
            diffs = [a - b for a, b in pairs if a is not None and b is not None]
            if len(diffs) < 3:
                cells.append("-")
                continue
            mean, _, low, high = mean_ci(diffs)
            cells.append(f"{mean:+.2e} ({low:+.2e} to {high:+.2e}){'*' if low > 0 or high < 0 else ''}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


def spend(rows: list[dict[str, Any]], part: str) -> str:
    """Core-hours of every recorded run, failed ones too, and how far each arm spread over nodes."""
    hours = sum((datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])).total_seconds() / 3600
                * PART_CPUS[part] for r in rows if r.get("start"))
    nodes: dict[str, set[str]] = {}
    for r in rows:
        nodes.setdefault(r["arm"], set()).add(r.get("nodes", "?"))
    spread = ", ".join(f"{arm} {len(seen)}" for arm, seen in nodes.items())
    return f"Spent {hours:,.0f} core-hours. Distinct node lists per arm: {spread}."


def load(paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        records += [json.loads(line) for line in path.open() if line.strip()]
    return records


def analyse(records: list[dict[str, Any]], part: str) -> str:
    rng = random.Random(SHUFFLE_SEED + 1)
    rows = [r for r in records if r["part"] == part]
    ok = [r for r in rows if r.get("status") == "ok"]
    runs: dict[str, dict[int, dict[str, Any]]] = {}
    for r in ok:
        runs.setdefault(r["arm"], {})[r["seed"]] = r
    recorded = {(r["seed"], r["arm"]) for r in rows}
    missing = [task for task in manifest(part) if task not in recorded]
    lines = [f"Part {part}: {len(ok)} of {len(rows)} recorded runs finished, {len(missing)} tasks have no record."]
    lines += [f"- no record: seed {seed} {arm}" for seed, arm in missing]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"- not used: seed {r['seed']} {r['arm']}: {r.get('status')} ({r.get('why', 'no reason recorded')})")
    if "stock" not in runs or len(runs["stock"]) < 3:
        return "\n".join(lines + ["Too few stock runs to compare."])
    arms = [a for a in PARTS[part]["arms"] if a != "stock" and len(set(runs.get(a, {})) & set(runs["stock"])) >= 3]
    results = {a: compare(runs, a, rng) for a in arms}
    control = len(results["stock-repeat"]["significant"]) if "stock-repeat" in results else 0
    lines += ["", "| arm | seeds | mean z | SE | 95% CI | sd z | sign (+/n, p) | scatter / stock's (p) | Holm hits | verdict |",
              "| --- | ---: | ---: | ---: | --- | ---: | --- | --- | ---: | --- |"]
    for arm, r in results.items():
        n = len(r["seeds"])
        lines.append(
            f"| {arm} | {n} | {r['mean']:+.3f} | {r['se']:.3f} | {r['low']:+.3f} to {r['high']:+.3f} | {r['sd']:.3f} "
            f"| {r['plus']}/{n}, {r['sign_p']:.3g} | {r['scatter_arm'] / r['scatter_stock']:.3f} ({r['scatter_p']:.3g}) "
            f"| {len(r['significant'])} | {verdict(r, control)} |")
    lines += ["", "| arm | wall vs stock (%, 95% CI) | evaluations vs stock (%, 95% CI) | Holm hits named |",
              "| --- | --- | --- | --- |"]
    for arm, r in results.items():
        wall, evals = r["wall"], r["evals"]
        lines.append(f"| {arm} | {wall[0]:+.1f} ({wall[2]:+.1f} to {wall[3]:+.1f}) | {evals[0]:+.1f} "
                     f"({evals[2]:+.1f} to {evals[3]:+.1f}) | {', '.join(r['significant']) or 'none'} |")
    lines += ["", "Calibration (scatter of log(Z) across seeds against the quoted error):", "",
              "| arm | runs | mean log(Z) | scatter | rms quoted error | scatter / quoted (+/- approx) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in PARTS[part]["arms"]:
        values = list(runs.get(arm, {}).values())
        if len(values) < 3:
            continue
        log_z = [v["log_z"] for v in values]
        quoted = math.sqrt(statistics.fmean(v["log_z_err"] ** 2 for v in values))
        ratio = statistics.stdev(log_z) / quoted
        lines.append(f"| {arm} | {len(values)} | {statistics.fmean(log_z):.6f} | {statistics.stdev(log_z):.3e} | "
                     f"{quoted:.3e} | {ratio:.3f} +/- {ratio / math.sqrt(2 * (len(values) - 1)):.3f} |")
    lines += ["", "Per arm, mean over seeds:", "", "| arm | runs | log(Z) error | ndead | nlike | wall (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in PARTS[part]["arms"] + PARTS[part]["shadow_arms"]:
        values = list(runs.get(arm, {}).values())
        if values:
            cells = [statistics.fmean(v[k] for v in values) for k in ("log_z_err", "ndead", "nlike", "wall_s")]
            lines.append(f"| {arm} | {len(values)} | {cells[0]:.3e} | {cells[1]:.0f} | {cells[2]:.0f} | {cells[3]:.0f} |")
    lines += ["", "Smallest Holm-adjusted p across parameters, per statistic:", "",
              "| arm | " + " | ".join(STATISTICS) + " |", "| --- |" + " ---: |" * len(STATISTICS)]
    for arm, r in results.items():
        lines.append(f"| {arm} | " + " | ".join(f"{min(r['holm'][s]['adjusted'].values()):.3g}" for s in STATISTICS) + " |")
    lines += ["", "Secondary, outside the pass rule: the tail, which is what the search is for.", "",
              "| arm | runs | mean max objective | mean contour at ln X -4 | at -6 | at -8 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm in PARTS[part]["arms"]:
        values = list(runs.get(arm, {}).values())
        if not values:
            continue
        cells = [statistics.fmean(r["max_log_l"] for r in values)]
        for x in ("-4", "-6", "-8"):
            have = [r["contour_at_log_x"][x] for r in values if x in r.get("contour_at_log_x", {})]
            cells.append(statistics.fmean(have) if have else float("nan"))
        lines.append(f"| {arm} | {len(values)} | " + " | ".join(f"{c:.5f}" for c in cells) + " |")
    lines += ["", "Paired difference from stock (mean, 95% CI, * = excludes 0):", ""] + tail_differences(runs, arms)
    if any("contour_at_volume" in r for r in ok):
        lines += ["", "Post hoc, outside the design: the same on PolyChord's volumes (contour_at_volume), stock mean first:", ""]
        lines += ["| point | stock mean |", "| --- | ---: |"]
        for x in LOG_X_POINTS:
            have = [r["contour_at_volume"][str(x)] for r in runs["stock"].values()]
            lines.append(f"| ln X {x} | {statistics.fmean(have):.5f} |")
        lines += [""] + tail_differences(runs, arms, "contour_at_volume")
    lines += ["", spend(rows, part)]
    for arm in PARTS[part]["shadow_arms"]:
        counts = [r["wrong_stop"] for r in ok if r["arm"] == arm and r.get("wrong_stop")]
        if counts:
            wrong, above = sum(c["wrong_stops"] for c in counts), sum(c["above_contour"] for c in counts)
            lines.append(f"\nWrong stops, {arm}, {len(counts)} runs: {wrong} of {above} above-contour "
                         f"evaluations ({100 * wrong / above:.3f}%), per run "
                         + ", ".join(str(c["wrong_stops"]) for c in counts) + ".")
    return "\n".join(lines)


def self_check() -> None:
    tasks = manifest("A")
    assert len(tasks) == 132 and len(set(tasks)) == 132
    assert len(manifest("B")) == 18
    assert tasks[:5] != [(9101, arm) for arm in PARTS["A"]["arms"]]

    assert abs(t_critical(23) - 2.069) < 0.002 and abs(t_critical(5) - 2.571) < 0.01
    stats = weighted_stats([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
    assert stats["mean"] == 2.5 and stats["q50"] == 2.0 and stats["q95"] == 4.0 and abs(stats["sd"] - 1.118) < 0.001
    assert sign_test([1.0] * 6) == 2 / 64 and sign_test([1.0, -1.0]) == 1.0
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.5})
    assert abs(adjusted["a"] - 0.03) < 1e-12 and adjusted["b"] == 0.08 and adjusted["c"] == 0.5

    assert read_table(_scratch("0.1-100 0.2E+001\n")) == [[0.1e-100, 2.0]]

    volumes = contours_at_volume([[0.9, 0.0], [0.09 * math.e, -2.0], [0.01 * math.e ** 2, -4.0], [0.0, 1e30]])
    assert volumes["-8"] == volumes["-5"] == 2.0 and volumes["-4"] == volumes["-3"] == 1.0
    assert volumes["-2"] == volumes["-1"] == 0.0

    shadow = [
        {"objective": 2.0, "contour": 1.0, "early_checks": [[3, 0.5]]},  # wrong
        {"objective": 0.5, "contour": 1.0, "early_checks": [[3, 0.5]]},  # right
        {"objective": 2.0, "contour": 1.0, "early_checks": [[3, 0.9], [4, 1.5]]},  # no stop
        {"objective": 2.0, "contour": None, "early_checks": [[3, 0.5]]},  # no contour
    ]
    assert wrong_stops(shadow) == {"checked": 3, "above_contour": 2, "would_stop": 2, "wrong_stops": 1}

    rng = random.Random(5)

    def fake(seed: int, arm: str, shift: float, jitter: float) -> dict[str, Any]:
        err = 0.0004
        base = rng.gauss(0.017, 0.0001)
        return {
            "part": "A", "seed": seed, "arm": arm, "status": "ok", "log_z": base + shift * err * math.sqrt(2)
            + rng.gauss(0, err * jitter), "log_z_err": err, "wall_s": 1000.0, "ndead": 1400, "nlike": 70000, "max_log_l": 0.03,
            "contour_at_log_x": {"-4": 0.02},
            "posterior": {p: {s: rng.gauss(1.0 + (shift if p == "bad" else 0), 0.02) for s in STATISTICS}
                          for p in ("good", "bad")},
        }

    records = [fake(s, arm, 0.0, 0.3) for s in PARTS["A"]["seeds"] for arm in ("stock", "stock-repeat", "adopt")]
    records += [fake(s, "stop", 1.0, 0.3) for s in PARTS["A"]["seeds"]]
    records += [fake(s, "adopt+stop", 0.0, 3.0) for s in PARTS["A"]["seeds"]]
    records.append({"part": "A", "seed": 9101, "arm": "adopt", "status": "failed", "why": "exit 1"})
    table = analyse(records, "A")
    verdicts = dict(re.findall(r"^\| ([a-z+-]+) \| \d+ .*\| (PASS|FAIL|INCONCLUSIVE) \|$", table, re.M))
    assert verdicts["adopt"] == "PASS" and verdicts["stock-repeat"] == "PASS", table
    assert verdicts["stop"] == "FAIL" and verdicts["adopt+stop"] == "FAIL", table
    assert "not used: seed 9101 adopt: failed (exit 1)" in table
    assert re.search(r"^\| ln X -4 \| [+-]\d", table, re.M) and "| ln X -1 | - |" in table, table
    assert "Distinct node lists per arm: stock 1" in table and "Spent 0 core-hours" in table, table
    assert spend([{"arm": "x", "nodes": "n", "start": "2026-10-30T10:00:00+00:00",
                   "end": "2026-10-30T10:30:00+00:00"}], "A").startswith("Spent 19 core-hours")
    print("sampler_validation self-check OK")


def _scratch(text: str) -> Path:
    import tempfile
    handle = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    handle.write(text)
    handle.close()
    return Path(handle.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    commands = parser.add_subparsers(dest="command")
    p = commands.add_parser("manifest")
    p.add_argument("part", choices=PARTS)
    p = commands.add_parser("extract")
    p.add_argument("run_dir", type=Path)
    p = commands.add_parser("record")
    for name, kind in (("run_dir", Path), ("out", Path), ("part", str), ("seed", int), ("arm", str), ("exit_code", int),
                       ("start", str), ("end", str), ("job", str), ("nodes", str), ("bench", Path)):
        p.add_argument(name, type=kind)
    p = commands.add_parser("volumes")
    p.add_argument("records", type=Path)
    p.add_argument("keep", type=Path)
    p = commands.add_parser("analyse")
    p.add_argument("part", choices=PARTS)
    p.add_argument("records", type=Path, nargs="+")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.command == "manifest":
        for i, (seed, arm) in enumerate(manifest(args.part)):
            adopt, stop = ARM_ENV[arm]
            print(i, seed, arm, adopt, stop)
    elif args.command == "extract":
        print(json.dumps(extract(args.run_dir)))
    elif args.command == "record":
        record(args)
    elif args.command == "volumes":
        add_volumes(args.records, args.keep)
    elif args.command == "analyse":
        print(analyse(load(args.records), args.part))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
