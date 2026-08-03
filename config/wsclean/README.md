# config/wsclean/

WSClean is driven entirely by command-line flags; it has no canonical
YAML/parset config file format (unlike R2D2-RI). This directory holds
documented, versioned command invocations instead, so that a specific
imaging run is reproducible and diffable the same way a config file
would be.

- `smoke-test.args` - the exact WSClean CLI arguments used by
  `scripts/smoke-test-wsclean.sh`, kept here as the single source of
  truth the script reads from (rather than being duplicated inline).

Add one `*.args` file per experiment as this repo's benchmarking grows;
reference its path from the corresponding entry in
`benchmarks/REPRODUCTION_PLAN.md` and from the manifest written by
`scripts/record-environment.sh --config config/wsclean/<file>.args`.
