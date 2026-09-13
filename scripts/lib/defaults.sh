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

# The build-input hash scripts/build.sh labelled the image with, which the SIF
# keeps; what run.env and the manifests record as the image id.
#   ns_image_id <sif>
ns_image_id() {
  local id=""
  [ -f "${1:-}" ] && [ -n "${APPTAINER}" ] \
    && id="$("${APPTAINER}" inspect --labels "$1" 2>/dev/null | awk -F': ' '$1 == "ri.build-inputs" { print $2 }')"
  printf '%s\n' "${id:-unknown}"
}
