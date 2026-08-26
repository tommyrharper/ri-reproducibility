# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Nested-sampling PoC infrastructure, run commands, and the per-stage profiler (`make nested-sampling-profile RUN=...`) are documented in `docs/nested-sampling.md`; use those Makefile/script entrypoints instead of ad hoc container wiring.
- The `polychord` and `meqtrees` images both bake in files from `scripts/lib/nested_sampling` at build time (no live mount); after editing those files, rebuild every image that copies them (`scripts/build.sh polychord`, `scripts/build.sh meqtrees`) before running the PoC, or it silently runs the stale baked-in code. `polychord` copies the whole directory, `meqtrees` only the three simulate-side scripts (`simulate_point_source_ms.py`, `point_source_forest.py`, `ms_to_r2d2_mat.py`) - except that the `meqtrees` build also bind-mounts `poc_common.py` to bake the MS skeleton cache, so a `PARAMETER_SPACE` change needs both images rebuilt.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
