#!/usr/bin/env bash
# Self-check defaults, platform detection, and environment precedence.
# Direct or CI use; needs uv, not Docker.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

failures=0

# Load defaults in a clean child shell; trailing KEY=VALUE arguments override it.
load() {
  local var="$1"
  shift
  # Child shell expands positional arguments itself.
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
  "7" "$(load NS_NLIVE NS_NLIVE=7)"

check "the environment overrides the generated seed" \
  "41" "$(load NS_SEED NS_SEED=41)"

# Reruns must not inherit the generated seed; compare two loads, not a value.
check "the seed is different on every load" \
  "different" "$([ "$(load NS_SEED)" = "$(load NS_SEED)" ] && echo same || echo different)"

check "the generated seed is exported, so the run scripts see it" \
  "ok" "$([ -n "$(load env:NS_SEED)" ] && echo ok)"

check "{REPO_ROOT} is expanded" \
  "${REPO_ROOT}/results" "$(load RESULTS_DIR)"

# A stand-in apptainer whose `inspect --labels` prints LABELS.
fake_bin="$(mktemp -d)"
trap 'rm -rf "${fake_bin}"' EXIT
# shellcheck disable=SC2016  # expanded by the stand-in, not here
printf '#!/bin/sh\nprintf "%%b" "${LABELS}"\n' >"${fake_bin}/apptainer"
chmod +x "${fake_bin}/apptainer"
# Prints "warned" or "quiet" for a SIF labelled with LABELS.
stale() {
  # shellcheck disable=SC2016
  env -u DOCKER_DEFAULT_PLATFORM LABELS="$1" APPTAINER="${fake_bin}/apptainer" bash -c '
    REPO_ROOT="$1"
    # shellcheck source=scripts/lib/defaults.sh
    source "${REPO_ROOT}/scripts/lib/defaults.sh"
    ns_warn_stale_wsclean_sif some.sif 2>&1 | grep -q WARNING && echo warned || echo quiet
  ' _ "${REPO_ROOT}"
}
# Runs one defaults.sh function in a clean child shell.
call() {
  # shellcheck disable=SC2016
  env -u DOCKER_DEFAULT_PLATFORM bash -c '
    REPO_ROOT="$1"
    # shellcheck source=scripts/lib/defaults.sh
    source "${REPO_ROOT}/scripts/lib/defaults.sh"
    "$2"
  ' _ "${REPO_ROOT}" "$1"
}
patches="$(call ns_wsclean_patches_hash)"
inputs="$(call ns_wsclean_build_inputs)"

check "a SIF built from this tree's docker/wsclean is not stale" \
  "quiet" "$(stale "ri.build-inputs: ${inputs}\n")"
check "a SIF built from another tree's docker/wsclean is stale" \
  "warned" "$(stale "ri.build-inputs: 0123\n")"
check "a SIF layered with this tree's patches is not stale" \
  "quiet" "$(stale "ri.build-inputs: 0123\nri.wsclean-patches: ${patches}\n")"
check "a SIF layered with other patches is stale" \
  "warned" "$(stale "ri.build-inputs: ${inputs}\nri.wsclean-patches: 0123\n")"

# Prints ns_image_id for a SIF labelled with LABELS.
image_id() {
  touch "${fake_bin}/some.sif"
  # shellcheck disable=SC2016
  env -u DOCKER_DEFAULT_PLATFORM LABELS="$1" APPTAINER="${fake_bin}/apptainer" bash -c '
    REPO_ROOT="$1"
    # shellcheck source=scripts/lib/defaults.sh
    source "${REPO_ROOT}/scripts/lib/defaults.sh"
    ns_image_id "$2"
  ' _ "${REPO_ROOT}" "${fake_bin}/some.sif"
}
check "an image id is its build-input label" \
  "abcd" "$(image_id "ri.build-inputs: abcd\n")"
check "a patched wsclean.sif's id carries its patches" \
  "abcd+p0123456789ab" "$(image_id "ri.build-inputs: abcd\nri.wsclean-patches: 0123456789abcdef\n")"
check "an unlabelled SIF's id is unknown" \
  "unknown" "$(image_id "")"

if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} check(s) failed" >&2
  exit 1
fi
echo "all checks passed"
