# config/wsclean/

WSClean is driven entirely by command-line flags; it has no canonical
YAML/parset config file format (unlike R2D2-RI). This directory holds
documented, versioned command invocations instead, so a specific imaging
run is diffable the same way a config file would be.

- `smoke-test.args` - the exact WSClean CLI arguments used by
  `scripts/smoke-test-wsclean.sh`, kept here as the single source of
  truth the script reads from (rather than being duplicated inline).

The nested-sampling search does not read from here: its per-evaluation
WSClean flags are built in
`scripts/lib/nested_sampling/polychord_wsclean.py`, and the fixed
ones are recorded in each run's `summary.json` under
`wsclean_fixed_hyperparameters`.
