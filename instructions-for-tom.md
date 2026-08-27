# Instructions for me

- `NS_NLIVE` number of live points (default `8`)
  - 8 for testing
  - 100-500 for serious exploration
- `NS_NUM_REPEATS` how much PolyChord explores inside the likelihood constraint before generating a replacement live point (default `2`)
  - 5D where D is dimensionality, so as we have 5 dimensions, 25 makes sense
- `NS_MAX_NDEAD` maximum number of dead points (default `12`)
  - 12-100 for testing
  - much higher for the real deal, or remove the cap entirely and use polychords convergence criteria
- `NS_MPI_PROCS` message passing interfaces. i.e. number of parallel threads (default `min(NS_NLIVE, host CPUs)`)
  - 8 should be the most i can handle
  - Don't have to configure, should be derived from `NS_NLIVE`
- `NS_METRIC` which metric we are maximising (default `total_rms_jy` - RMS of (cleaned image − truth) over the whole image)
  - `off_source_rms_jy` off-source RMS in Jy/beam
  - `sigma_res` paper data-fidelity `||residual_dirty||_2 / ||dirty||_2`
  - PolyChord maximises, which looks for the worst images 
- `OUTPUT_DIR` where the results are saved (default `results/nested-sampling/wsclean-vlaa-<UTC>` or `r2d2-vlaa-<UTC>`)
  - leave empty

## key commands

```zsh
./ri plot gui
./ri merge
./ri search wsclean
./ri search r2d2
./ri search wsclean --nlive 20 --num-repeats 5 --max-ndead 20
./ri search r2d2 --nlive 20 --num-repeats 5 --max-ndead 20
./ri report
./ri report --last 1
./ri report --run wsclean-vlaa-merged-20260818T125606Z
./ri report --run results/nested-sampling/r2d2-vlaa-20260817T120607Z
./ri report --upgrade
./ri report --force
./ri plot gui results/nested-sampling/r2d2-vlaa-merged-20260818T125604Z
./ri plot likelihood # the R2D2 vs WSClean overlay image

```

## current search approach:

Currently we have 5 dimesions, so `d=5`
- `nlive` should be `25d` for a serious run
- `--num-repeats` can start around `2d`, up to `5d`
- `--max-ndead` set to `-1` to make unlimited so we have a precision stopping criteria (you have to use `--max-ndead=-1` with the `=` sign to catch the negative symbol.
- `--seed` choose something random
- `--metric` currently going for `total_rms_jy`

Current command:
```zsh
./ri search r2d2 --nlive=50 --num-repeats=10 --max-ndead=-1 --seed=123 --metric=total_rms_jy
```
