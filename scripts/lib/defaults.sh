# shellcheck shell=bash  # sourced, so no shebang
# Loads the shared runtime defaults from defaults.toml at the repository root.
#
# defaults.toml is the file to edit; this one only reads it. Every scalar key
# there names an environment variable and is applied only when that variable
# is unset or empty, so overrides from the environment always win. The arrays
# of tables in it (parameter_space) are for the Python side and skipped here.
#
# Source this after REPO_ROOT is set - it is needed both to find the file and
# to expand the `{REPO_ROOT}` placeholder in it.
#
# uv is used rather than the system python3 because reading TOML needs
# tomllib, which is stdlib from 3.11 onwards and the system python3 is older
# on some hosts. uv is already required to run the host-side scripts (see
# README.md), and `--no-project` keeps this off the project venv, so it costs
# ~30ms and installs nothing.

if ! command -v uv >/dev/null 2>&1; then
  echo "defaults.sh: uv is required to read defaults.toml - see README.md" >&2
  exit 1
fi

# Assigned before being eval'd rather than `eval "$(...)"` directly: eval
# succeeds on empty input, which would swallow a missing or malformed
# defaults.toml and leave every default unset.
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
    # Arrays and tables (parameter_space) are read directly by the Python side
    # from defaults.toml; only scalars name an environment variable.
    if isinstance(value, (list, dict)):
        continue
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        sys.exit(f"{path}: {key} must be a string or number, got {type(value).__name__}")
    # Environment wins: only fill in variables that are unset or empty.
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

# The PolyChord seed is deliberately not in defaults.toml either, and for the
# same reason: it is a property of the run, not of the project. A committed
# value makes every search - here, on the next host, and on the tenth rerun -
# walk the same points in the same order, so the second search costs a day and
# discovers what the first one already did. Randomised per run instead.
#
# Still overridable, which is what repeating one run exactly needs, and only
# ever generated when NS_SEED is unset or empty - so `./ri search --seed 41`
# and `./ri resume` (which exports run.env's NS_SEED before this is sourced)
# both keep the seed they were given. The value a run actually used is in its
# run.env and summary.json.
#
# Bash's own RANDOM rather than /dev/urandom: two 15-bit draws are a big
# enough space for this, and it costs no subprocess. Positive and below 2^30,
# because PolyChord treats a seed <= 0 as "seed from the clock" and the
# per-evaluation noise seeds derived from it (stable_seed, common.py) are
# taken modulo 2^31-1.
: "${NS_SEED:=$(((RANDOM << 15 | RANDOM) + 1))}"
export NS_SEED

# The Docker platform is deliberately not in defaults.toml: it is a property
# of the host, not of the project, so any committed value is wrong on half
# the machines this repo runs on. Derived from the host architecture here and
# still overridable from the environment, which is what a host cross-building
# for the other architecture needs.
#
# DOCKER_DEFAULT_PLATFORM is what Docker itself reads (so `docker build`,
# `docker run`, and compose.yaml all pick it up once it is exported);
# PLATFORM is what the scripts pass to an explicit `--platform`.
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

# Host path of the Docker socket, for the containers that drive other
# containers. Rootless Docker listens on $XDG_RUNTIME_DIR/docker.sock, not
# /var/run/docker.sock, and a bind mount needs the real host path; DOCKER_HOST
# is what points the CLI at it. Also not in defaults.toml - a property of the
# host, not the project.
if [ -z "${DOCKER_SOCKET:-}" ]; then
  case "${DOCKER_HOST:-}" in
    unix://*) DOCKER_SOCKET="${DOCKER_HOST#unix://}" ;;
    *) DOCKER_SOCKET="/var/run/docker.sock" ;;
  esac
fi
