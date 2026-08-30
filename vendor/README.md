# vendor/

Intentionally empty. Docker builds clone WSClean and R2D2-RI at the pinned
commits in `versions.env`; source is never committed here.

For local inspection:

```bash
git clone --recurse-submodules https://gitlab.com/aroffringa/wsclean.git vendor/wsclean
git -C vendor/wsclean checkout $(grep WSCLEAN_GIT_COMMIT ../versions.env | cut -d= -f2)
```

Clones remain untracked and local.
