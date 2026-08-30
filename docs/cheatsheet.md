# Cheatsheet

`./ri` is the only entrypoint; run it from the repository root. `--help` shows
the complete interface.

## Setup and checks

```bash
./ri build                         # build all images
./ri smoke                         # smoke-test both imagers
./ri self-check                    # host and image checks
./ri --dry-run search wsclean      # preview commands without running them
```

## Searches

```bash
./ri search wsclean
./ri search r2d2 --metric sigma_res
./ri search wsclean --nlive 20 --num-repeats 5 --max-ndead 20
./ri resume results/nested-sampling/<run>
```

## Results

```bash
./ri runs                         # list runs and completion state
./ri health results/nested-sampling/<run>
./ri profile results/nested-sampling/<run>
./ri report                       # generate HTML report
./ri serve                        # serve report for remote viewing
./ri merge <run-a> <run-b>
./ri clean                        # remove generated outputs and repo images
```

Details: [nested sampling](nested-sampling.md), [run health](run-health.md),
[robustness](robustness.md), [performance](nested-sampling-speed.md).
