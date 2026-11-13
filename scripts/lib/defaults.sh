# shellcheck shell=bash  # sourced, so no shebang
# Load scalar defaults from defaults.toml; environment values win.

if ! command -v uv >/dev/null 2>&1; then
  echo "defaults.sh: uv is required to read defaults.toml - see README.md" >&2
  exit 1
fi

# Assign before eval: empty output would hide missing or malformed defaults.
if ! _ri_defaults="$(
  REPO_ROOT="${REPO_ROOT}" uv run --no-project --python ">=3.11" python - <<'PYEOF'
import os
import shlex
import sys
import tomllib

repo_root = os.environ["REPO_ROOT"]
path = os.path.join(repo_root, "defaults.toml")

with open(path, "rb") as handle:
    defaults = tomllib.load(handle)

for key, value in defaults.items():
    # Python reads arrays and tables directly; scalars become environment variables.
    if isinstance(value, (list, dict)):
        continue
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        sys.exit(f"{path}: {key} must be a string or number, got {type(value).__name__}")
    # Environment wins.
    if os.environ.get(key):
        continue
    print(f"export {key}={shlex.quote(str(value).replace('{REPO_ROOT}', repo_root))}")
PYEOF
)"; then
  echo "defaults.sh: could not load ${REPO_ROOT}/defaults.toml" >&2
  exit 1
fi

eval "${_ri_defaults}"
unset _ri_defaults

# Generate a positive per-run seed unless overridden for exact replay.
: "${NS_SEED:=$(((RANDOM << 15 | RANDOM) + 1))}"
export NS_SEED

# Host-derived platform, overridable for cross-builds; Docker and scripts use it.
if [[ -z "${DOCKER_DEFAULT_PLATFORM:-}" ]]; then
  case "$(uname -m)" in
    x86_64 | amd64) DOCKER_DEFAULT_PLATFORM="linux/amd64" ;;
    aarch64 | arm64) DOCKER_DEFAULT_PLATFORM="linux/arm64" ;;
    *)
      echo "defaults.sh: no image is built for host architecture '$(uname -m)';" \
        "set DOCKER_DEFAULT_PLATFORM explicitly to override" >&2
      exit 1
      ;;
  esac
fi
export DOCKER_DEFAULT_PLATFORM
# shellcheck disable=SC2034  # read by the sourcing script
PLATFORM="${DOCKER_DEFAULT_PLATFORM}"

# The runtime: Apptainer (CSD3 ships it with no module to load; older nodes
# spell it singularity). The images travel as SIFs under images/, built from
# the Docker images by `./ri images export|import` - see docs/cluster.md.
if [ -z "${APPTAINER:-}" ]; then
  APPTAINER="$(command -v apptainer || command -v singularity || true)"
fi
export APPTAINER
SIF_DIR="${SIF_DIR:-${REPO_ROOT}/images}"
# shellcheck disable=SC2034  # read by the sourcing scripts
WSCLEAN_SIF="${SIF_DIR}/wsclean.sif"
# shellcheck disable=SC2034
R2D2_SIF="${SIF_DIR}/r2d2.sif"
# shellcheck disable=SC2034
R2D2_CUDA_SIF="${SIF_DIR}/r2d2-cuda.sif"
# shellcheck disable=SC2034
MEQTREES_SIF="${SIF_DIR}/meqtrees.sif"
# shellcheck disable=SC2034
POLYCHORD_SIF="${SIF_DIR}/polychord.sif"

# The Docker branch rebuilt images in front of every run; here the SIFs are
# whatever `./ri images import` last produced, so a run only checks they exist.
#   ns_require_sifs <sif>...
ns_require_sifs() {
  local sif
  [ -n "${APPTAINER}" ] || { echo "FATAL: neither apptainer nor singularity is on PATH (on CSD3: module load singularity/current)" >&2; exit 1; }
  for sif in "$@"; do
    [ -f "${sif}" ] || { echo "FATAL: ${sif} is missing - ./ri images import (docs/cluster.md)" >&2; exit 1; }
  done
}

# Hash Dockerfiles and copied or bind-mounted files. Include names so renames
# invalidate builds; exclude __pycache__ to match .dockerignore. Paths are
# relative to REPO_ROOT, which is part of what is hashed.
# Usage: inputs_hash <flavour> <path...>
inputs_hash() {
  local flavour="$1"
  shift
  (
    cd "${REPO_ROOT}" || exit
    printf '%s\n' "${flavour}"
    find "$@" -name '__pycache__' -prune -o -type f -print | LC_ALL=C sort
    find "$@" -name '__pycache__' -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 cat
  ) | sha256sum | cut -d' ' -f1
}

# The ri.build-inputs scripts/build.sh gives a WSClean image built from this tree.
ns_wsclean_build_inputs() {
  inputs_hash "${PLATFORM} TARGET_CPU=${WSCLEAN_TARGET_CPU-x86-64-v3}" \
    docker/wsclean/Dockerfile docker/wsclean/patches docker/wsclean/src
}

# The ri.wsclean-patches scripts/patch-wsclean-sif.sh gives a SIF it layers.
ns_wsclean_patches_hash() {
  cat "${REPO_ROOT}"/docker/wsclean/patches/*.patch "${REPO_ROOT}"/docker/wsclean/src/*.cpp \
    | sha256sum | cut -c1-16
}

# A SIF carries only the WSClean patches of the tree it was built from, and
# each later patch is a speed-up a run silently goes without: the 21 Sep
# wsclean.sif stops at 0006 (docs/csd3-speed.md, rounds 43 and 70).
#   ns_wsclean_sif_current <sif>   true unless the SIF lags this tree
ns_wsclean_sif_current() {
  local labels patches
  labels="$("${APPTAINER}" inspect --labels "$1" 2>/dev/null)" || return 0
  patches="$(awk -F': ' '$1 == "ri.wsclean-patches" { print $2 }' <<<"${labels}")"
  if [ -n "${patches}" ]; then
    [ "${patches}" = "$(ns_wsclean_patches_hash)" ]
  else
    [ "$(awk -F': ' '$1 == "ri.build-inputs" { print $2 }' <<<"${labels}")" = "$(ns_wsclean_build_inputs)" ]
  fi
}

# Warns, never refuses, since the images are still correct. Inside a job,
# ns_use_patched_wsclean_sif usually makes the warning moot.
#   ns_warn_stale_wsclean_sif <sif>
ns_warn_stale_wsclean_sif() {
  ns_wsclean_sif_current "$1" && return 0
  echo "WARNING: $1 was not built from this tree's docker/wsclean, so this run may lack" \
    "some WSClean speed-up patches. A search's Slurm job builds and caches a patched" \
    "copy (NS_PATCH_WSCLEAN=0 to skip); by hand: ./ri images patch-wsclean (docs/cluster.md)" >&2
}

# Inside a Slurm job, points WSCLEAN_SIF at a copy of it patched to this tree
# when it lags: built once by patch-wsclean-sif.sh (~2 minutes with the job's
# CPUs) into the work dir's cache, then reused by every run. Round 89 measured
# the shipped 0001-0006 SIF at 184 evals/s steady against 258 patched; the
# patches are result-preserving. Any failure keeps the stock SIF.
# NS_PATCH_WSCLEAN=0 keeps it too (an A/B that means to run a stale SIF).
#   ns_use_patched_wsclean_sif [build jobs]
ns_use_patched_wsclean_sif() {
  local base="${WSCLEAN_SIF}" cache out tmp
  [ "${NS_PATCH_WSCLEAN:-1}" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ] && [ -f "${base}" ] || return 0
  ns_wsclean_sif_current "${base}" && return 0
  cache="${RI_WORK_DIR:-$(dirname "$(cd "${REPO_ROOT}" && pwd -P)")}/.cache/ri"
  # Keyed by the base too: the same patches over another import are another SIF.
  out="${cache}/wsclean-$(ns_image_id "${base}" | cut -c1-12)-p$(ns_wsclean_patches_hash | cut -c1-12).sif"
  if [ ! -f "${out}" ]; then
    echo "${base} lags this tree's WSClean patches: building ${out} (~2 minutes, once)"
    mkdir -p "${cache}" || return 0
    # No lock: two jobs racing here build the same bytes, and the rename is atomic.
    tmp="${out}.$$.partial"
    if ! WSCLEAN_BASE_SIF="${base}" BUILD_JOBS="${1:-4}" \
      "${REPO_ROOT}/scripts/patch-wsclean-sif.sh" "${tmp}" >"${tmp}.log" 2>&1 \
      || ! mv -f "${tmp}" "${out}"; then
      echo "WARNING: building a patched wsclean.sif failed (${tmp}.log); this run uses ${base}" >&2
      rm -f "${tmp}"
      return 0
    fi
    rm -f "${tmp}.log"
  fi
  echo "Using ${out}: ${base} patched to this tree (NS_PATCH_WSCLEAN=0 to skip)."
  WSCLEAN_SIF="${out}"
}

# The build-input hash scripts/build.sh labelled the image with, which the SIF
# keeps; what run.env and the manifests record as the image id. A SIF from
# patch-wsclean-sif.sh inherits its base's build-input label, so its patches
# hash is appended: otherwise ./ri bench groups a patched and an unpatched
# wsclean.sif as one image (docs/csd3-speed.md, round 71).
#   ns_image_id <sif>
ns_image_id() {
  local id=""
  [ -f "${1:-}" ] && [ -n "${APPTAINER}" ] \
    && id="$("${APPTAINER}" inspect --labels "$1" 2>/dev/null | awk -F': ' '
      $1 == "ri.build-inputs" { id = $2 }
      $1 == "ri.wsclean-patches" { patches = $2 }
      END { if (id != "" && patches != "") id = id "+p" substr(patches, 1, 12); print id }')"
  printf '%s\n' "${id:-unknown}"
}
