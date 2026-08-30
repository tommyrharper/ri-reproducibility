# data/

Bind-mounted at `/data`; only this README is tracked. Nested sampling does
not use it: evaluations create their own Measurement Sets under
`results/nested-sampling/<run>/evaluations/`.

Add experiment-specific inputs here when needed:

- WSClean Measurement Sets (`*.ms/`)
- R2D2 `.mat` files
- Ground-truth FITS images

Document each large file's source, checksum and license alongside its use.
