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
./ri search wsclean --then r2d2       # the second starts only if the first finished
./ri resume results/nested-sampling/<run>
```

## Results

```bash
./ri runs                         # list runs and completion state
./ri health results/nested-sampling/<run>
./ri profile results/nested-sampling/<run>
./ri bench                        # throughput per commit, to catch a regression
./ri bench run wsclean --repeat 3 # add rows for the commit checked out now
./ri report                       # generate HTML report
./ri serve                        # serve report for remote viewing
./ri merge <run-a> <run-b>
./ri clean                        # remove generated outputs and repo images
```

## Figures and samples

```bash
./ri plot gui                     # interactive corner plots of the newest run
./ri plot gui results/nested-sampling/<run>
./ri plot fits                    # render FITS images to PNG in results/
./ri plot likelihood              # R2D2 vs WSClean failure-score figures
./ri plot likelihood --last       # ...for the last two directly comparable runs
```

Every pair is kept in `reports/likelihood-comparisons/`; `./ri report` collects
them onto a page linked from the top of the index.

## Watching the run that is going

```bash
./ri health results/nested-sampling/<run>   # is it healthy, and where is it
./ri report --live                # an HTML page per live run, marked live
./ri plot gui --live              # its samples so far, as a snapshot
```

Both read what the run has written so far, and neither disturbs it. Rerun
either one for a later picture; nothing updates on its own.

Details: [nested sampling](nested-sampling.md), [run health](run-health.md),
[robustness](robustness.md), [performance](nested-sampling-speed.md).
