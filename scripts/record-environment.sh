#!/usr/bin/env bash
# Write a machine-readable JSON manifest capturing everything needed to
# reproduce a single experiment run, per README.md "Reproducibility
# metadata". Intended to wrap an actual run, e.g.:
#
#   scripts/record-environment.sh \
#     --tool wsclean \
#     --image ri-reproducibility/wsclean:v3.7 \
#     --config config/wsclean/example.cfg \
#     -- docker run --rm ri-reproducibility/wsclean:v3.7 -name /results/x ...
#
# Everything after `--` is the exact command that was (or will be) run;
# it is recorded verbatim, not re-interpreted. This script does not
# execute it - call it, then run the command yourself, or wrap it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST_DIR="${REPO_ROOT}/benchmarks/manifests"
mkdir -p "${MANIFEST_DIR}"

TOOL=""
IMAGE=""
CONFIG_FILE=""
COMMAND=()

while [ $# -gt 0 ]; do
  case "$1" in
    --tool) TOOL="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --config) CONFIG_FILE="$2"; shift 2 ;;
    --) shift; COMMAND=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [ -z "${TOOL}" ] || [ -z "${IMAGE}" ]; then
  echo "usage: $0 --tool <wsclean|r2d2|meqtrees|polychord> --image <image[:tag]> [--config <path>] [-- <command...>]" >&2
  exit 1
fi

MANIFEST_PATH="${MANIFEST_DIR}/${TOOL}-$(date -u +%Y%m%dT%H%M%SZ).json"

# NB: `cmd 2>/dev/null || fallback` is not safe here - some of these
# commands (e.g. `git rev-parse HEAD` in a repo with no commits yet)
# print partial/wrong output to stdout AND fail, so `||` leaks that
# partial output into the command substitution alongside the fallback.
# Check exit status explicitly instead.
REPO_GIT_REV="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null)" \
  || REPO_GIT_REV="unknown (no commits yet?)"
REPO_GIT_DIRTY="$(git -C "${REPO_ROOT}" diff --quiet 2>/dev/null && echo false || echo true)"

IMAGE_ID="$(docker image inspect "${IMAGE}" --format '{{.Id}}' 2>/dev/null)" \
  || IMAGE_ID="unknown"
IMAGE_DIGEST="$(docker image inspect "${IMAGE}" --format '{{index .RepoDigests 0}}' 2>/dev/null)" \
  || IMAGE_DIGEST="none-local-build"
IMAGE_CREATED="$(docker image inspect "${IMAGE}" --format '{{.Created}}' 2>/dev/null)" \
  || IMAGE_CREATED="unknown"
CONTAINER_ARCH="$(docker image inspect "${IMAGE}" --format '{{.Architecture}}' 2>/dev/null)" \
  || CONTAINER_ARCH="unknown"

CONFIG_CHECKSUM=""
if [ -n "${CONFIG_FILE}" ] && [ -f "${REPO_ROOT}/${CONFIG_FILE}" ]; then
  CONFIG_CHECKSUM="$(shasum -a 256 "${REPO_ROOT}/${CONFIG_FILE}" | awk '{print $1}')"
fi

TIMESTAMP_UTC="${TIMESTAMP_UTC:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}" \
TOOL="$TOOL" IMAGE="$IMAGE" \
REPO_GIT_REV="$REPO_GIT_REV" REPO_GIT_DIRTY="$REPO_GIT_DIRTY" \
IMAGE_ID="$IMAGE_ID" IMAGE_DIGEST="$IMAGE_DIGEST" IMAGE_CREATED="$IMAGE_CREATED" CONTAINER_ARCH="$CONTAINER_ARCH" \
HOST_OS="$(uname -s)" HOST_ARCH="$(uname -m)" HOST_KERNEL="$(uname -r)" \
CPU_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^ //' || echo unknown)" \
CPU_COUNT="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo unknown)" \
DOCKER_CPUS="$(docker info --format '{{.NCPU}}' 2>/dev/null || echo unknown)" \
DOCKER_MEM="$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo unknown)" \
CONFIG_FILE="$CONFIG_FILE" CONFIG_CHECKSUM="$CONFIG_CHECKSUM" \
MANIFEST_PATH="$MANIFEST_PATH" \
python3 - "${COMMAND[@]}" <<'PYEOF'
import json, sys, os

manifest = {
    "timestamp_utc": os.environ["TIMESTAMP_UTC"],
    "tool": os.environ["TOOL"],
    "repository": {
        "git_revision": os.environ["REPO_GIT_REV"],
        "dirty_working_tree": os.environ["REPO_GIT_DIRTY"] == "true",
    },
    "image": {
        "reference": os.environ["IMAGE"],
        "id": os.environ["IMAGE_ID"],
        "digest": os.environ["IMAGE_DIGEST"],
        "created": os.environ["IMAGE_CREATED"],
        "container_architecture": os.environ["CONTAINER_ARCH"],
    },
    "host": {
        "os": os.environ["HOST_OS"],
        "arch": os.environ["HOST_ARCH"],
        "kernel": os.environ["HOST_KERNEL"],
        "cpu_model": os.environ["CPU_MODEL"],
        "cpu_count": os.environ["CPU_COUNT"],
        "docker_allocated_cpus": os.environ["DOCKER_CPUS"],
        "docker_allocated_memory_bytes": os.environ["DOCKER_MEM"],
    },
    "config_file": os.environ["CONFIG_FILE"] or None,
    "config_file_sha256": os.environ["CONFIG_CHECKSUM"] or None,
    "environment_variables": {
        k: v for k, v in os.environ.items()
        if k in ("OMP_NUM_THREADS", "WSCLEAN_THREADS", "HOST_UID", "HOST_GID")
    },
    "command": sys.argv[1:] or None,
    "notes": (
        "Input/output/checkpoint file checksums and random seeds are "
        "experiment-specific and are not captured here - add them to "
        "this manifest once the run in question defines them."
    ),
}

path = os.environ["MANIFEST_PATH"]
with open(path, "w") as f:
    json.dump(manifest, f, indent=2)
    f.write("\n")
print(f"wrote {path}")
PYEOF
