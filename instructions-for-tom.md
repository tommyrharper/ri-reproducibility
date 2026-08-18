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
- `NS_METRIC` which metric we are maximising (default `off_source_rms_jy`)
  - `total_rms_jy` RMS of (cleaned image − truth) over the whole image
  - `sigma_res` paper data-fidelity `||residual_dirty||_2 / ||dirty||_2`
  - PolyChord maximises, which looks for the worst images 
- `OUTPUT_DIR` where the results are saved (default `results/nested-sampling-poc/wsclean-vlaa-<UTC>` or `r2d2-vlaa-<UTC>`)
  - leave empty

## key commands

```zsh
make anesthetic-gui
make merge-nested-sampling
make nested-sampling-poc
make nested-sampling-r2d2-poc
NS_NLIVE=20 NS_NUM_REPEATS=5 NS_MAX_NDEAD=20 make nested-sampling-poc
NS_NLIVE=20 NS_NUM_REPEATS=5 NS_MAX_NDEAD=20 make nested-sampling-r2d2-poc
make benchmark-report
make nested-sampling-report
make anesthetic-gui RUN=results/nested-sampling-poc/r2d2-vlaa-20260817T155938Z
```
