# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

- Nested-sampling PoC infrastructure, run commands, and the per-stage profiler (`make nested-sampling-profile RUN=...`) are documented in `docs/nested-sampling.md`; use those Makefile/script entrypoints instead of ad hoc container wiring.
- The `polychord` image bakes in `scripts/lib/nested_sampling` at build time (no live mount); after editing those files, rebuild with `docker build --platform linux/arm64 -f docker/polychord/Dockerfile -t ri-reproducibility/polychord:lite .` (or `make build-polychord`) before running the PoC, or it silently runs the stale baked-in code.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
