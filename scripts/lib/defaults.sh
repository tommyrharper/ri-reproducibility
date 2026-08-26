# Single source of truth for the runtime defaults shared by scripts/*.sh.
#
# Every value here stays overridable from the environment: `:=` only assigns
# when the variable is unset or empty, so `NS_SEED=7 make nested-sampling-poc`
# still runs with seed 7.
#
# Source this after REPO_ROOT is set (CHECKPOINTS_DIR and RESULTS_DIR are
# relative to it). Values that genuinely differ per script - OUTPUT_DIR, for
# instance - stay in that script.
#
# Upstream software revisions are pinned in versions.env, not here.

# The one case where the internal name differs from the environment variable:
# DOCKER_DEFAULT_PLATFORM is what Docker itself reads and what .env.example
# documents, PLATFORM is what the scripts pass to `--platform`.
PLATFORM="${DOCKER_DEFAULT_PLATFORM:-linux/arm64}"

: "${R2D2_IMAGE:=ri-reproducibility/r2d2:cpu}"
: "${WSCLEAN_IMAGE:=ri-reproducibility/wsclean:v3.7}"
: "${MEQTREES_IMAGE:=ri-reproducibility/meqtrees:kern-10}"
: "${POLYCHORD_IMAGE:=ri-reproducibility/polychord:lite}"

: "${CHECKPOINTS_DIR:=${REPO_ROOT}/checkpoints}"
: "${RESULTS_DIR:=${REPO_ROOT}/results}"

: "${RUN_ID:=$(date -u +%Y%m%dT%H%M%SZ)}"

: "${NS_NLIVE:=8}"
: "${NS_NUM_REPEATS:=2}"
: "${NS_MAX_NDEAD:=12}"
: "${NS_SEED:=41}"
: "${NS_METRIC:=off_source_rms_jy}"
