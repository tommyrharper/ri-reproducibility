#!/usr/bin/env python3
"""List searched and pinned nested-sampling dimensions."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib" / "nested_sampling"))

from common import load_all_parameter_specs, load_parameter_space  # noqa: E402


def main() -> None:
    enabled_names = {spec["name"] for spec in load_parameter_space()}
    rows = []
    for spec in load_all_parameter_specs():
        name = str(spec["name"])
        status = "on" if name in enabled_names else "off"
        box = ("[[receiver_band]]" if spec.get("kind") == "band_start"
               else f"{spec.get('min', '?')} to {spec.get('max', '?')}")
        pinned = "" if status == "on" else f" (pinned at {spec['default']})"
        rows.append((name, status, box, spec.get("kind", ""), pinned))

    name_width = max(len(row[0]) for row in rows)
    status_width = max(len(row[1]) for row in rows)
    for name, status, box, kind, pinned in rows:
        kind_note = f" ({kind})" if kind else ""
        print(f"{name:<{name_width}}  {status:<{status_width}}  {box}{kind_note}{pinned}")

    searched = len(enabled_names)
    total = len(rows)
    print(f"\n{searched}/{total} dimensions searched")


if __name__ == "__main__":
    main()
