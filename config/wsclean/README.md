# config/wsclean/

WSClean has no canonical YAML/parset config format, so this directory stores
versioned command invocations for diffable imaging runs.

- `smoke-test.args` - exact CLI arguments read by
  `scripts/smoke-test-wsclean.sh`.

Nested sampling builds per-evaluation flags in
`scripts/lib/nested_sampling/polychord_wsclean.py`; fixed flags are recorded
in `summary.json` under `wsclean_fixed_hyperparameters`.
