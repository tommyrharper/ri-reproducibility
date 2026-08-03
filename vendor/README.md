# vendor/

Intentionally empty. WSClean and R2D2-RI source is never vendored into
this Git repository; both are cloned at pinned commits during the Docker
image builds (see `versions.env` and `docker/*/Dockerfile`).

This directory exists as a documented, gitignored landing spot if you
ever need to inspect upstream source locally outside a container, e.g.:

```bash
git clone --recurse-submodules https://gitlab.com/aroffringa/wsclean.git vendor/wsclean
git -C vendor/wsclean checkout $(grep WSCLEAN_GIT_COMMIT ../versions.env | cut -d= -f2)
```

Anything cloned here stays untracked and local to your machine.
