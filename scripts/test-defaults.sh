#!/usr/bin/env bash
# Self-check for scripts/lib/defaults.sh, the loader every other script
# sources. Covers the rules that matter: the Docker platform follows the host,
# defaults.toml supplies the project values, and the environment always wins
# over both.
#
# Run it directly - `scripts/test-defaults.sh` - or via CI. Needs uv (as
# defaults.sh itself does) but not Docker.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

failures=0

# Load the defaults in a clean child shell and print one value from it.
#
#   load PLATFORM                       -> the shell variable PLATFORM
#   load env:DOCKER_DEFAULT_PLATFORM    -> the same name from the environment,
#                                          which only passes if it is exported
#
# DOCKER_DEFAULT_PLATFORM is cleared first so the host-derived path runs; any
# trailing KEY=VALUE arguments are set for that load only.
load() {
  local var="$1"
  shift
  # Single-quoted on purpose: the child shell expands these itself, from the
  # positional arguments passed after the script.
  # shellcheck disable=SC2016
  env -u DOCKER_DEFAULT_PLATFORM "$@" bash -c '
    REPO_ROOT="$1"
    # shellcheck source=scripts/lib/defaults.sh
    source "${REPO_ROOT}/scripts/lib/defaults.sh"
    case "$2" in
      env:*) printenv "${2#env:}" ;;
      *) printf "%s\n" "${!2}" ;;
    esac
  ' _ "${REPO_ROOT}" "${var}"
}

check() {
  local what="$1" expected="$2" actual="$3"
  if [[ "${actual}" == "${expected}" ]]; then
    echo "ok   ${what}"
  else
    echo "FAIL ${what}: expected '${expected}', got '${actual}'" >&2
    failures=$((failures + 1))
  fi
}

case "$(uname -m)" in
  x86_64 | amd64) host_platform="linux/amd64" ;;
  aarch64 | arm64) host_platform="linux/arm64" ;;
  *)
    echo "skip: no image is built for host architecture $(uname -m)"
    exit 0
    ;;
esac

check "the platform is derived from the host" \
  "${host_platform}" "$(load PLATFORM)"

check "the platform is exported, so Docker itself reads it" \
  "${host_platform}" "$(load env:DOCKER_DEFAULT_PLATFORM)"

check "the environment overrides the derived platform" \
  "linux/riscv64" "$(load PLATFORM DOCKER_DEFAULT_PLATFORM=linux/riscv64)"

check "defaults.toml supplies the project values" \
  "ri-reproducibility/wsclean:v3.7" "$(load WSCLEAN_IMAGE)"

check "the environment overrides defaults.toml" \
  "7" "$(load NS_SEED NS_SEED=7)"

check "{REPO_ROOT} is expanded" \
  "${REPO_ROOT}/results" "$(load RESULTS_DIR)"

if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} check(s) failed" >&2
  exit 1
fi
echo "all checks passed"
