#!/usr/bin/env python3
"""List the nested-sampling parameter space and which dimensions are searched.

`enabled = false` in defaults.toml pins a dimension out of the search instead
of deleting it (see the comment above `[[parameter_space]]` there); this is
the read-only view of that state, further overridden the same way a search
would be by NS_ENABLE_PARAMS / NS_DISABLE_PARAMS (what `./ri search
--enable-param` / `--disable-param` set).

Usage:

  uv run scripts/list-parameter-space.py
  NS_DISABLE_PARAMS=channel_count uv run scripts/list-parameter-space.py
"""

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
        if spec.get("kind") == "band_start":
            box = "[[receiver_band]]"
        else:
            box = f"{spec.get('min', '?')} to {spec.get('max', '?')}"
        pinned = "" if status == "on" else f" (pinned at {spec.get('default', spec.get('min', 0.0))})"
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
