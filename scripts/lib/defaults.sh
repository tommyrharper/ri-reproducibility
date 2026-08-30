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

# Resolve socket path for containers that drive Docker, including rootless hosts.
if [ -z "${DOCKER_SOCKET:-}" ]; then
  case "${DOCKER_HOST:-}" in
    unix://*) DOCKER_SOCKET="${DOCKER_HOST#unix://}" ;;
    *) DOCKER_SOCKET="/var/run/docker.sock" ;;
  esac
fi
