#!/usr/bin/env bash
# Carry the four images from a Docker host to the cluster as SIF files.
#
#   scripts/images.sh export [DIR]   # Docker host: docker save -> DIR/<name>.tar
#   scripts/images.sh import [DIR]   # cluster:     DIR/<name>.tar -> images/<name>.sif
#
# The cluster has no Docker, so the Dockerfiles are built elsewhere and their
# images travel as plain `docker save` archives (rsync them into DIR). Apptainer
# builds a SIF from such an archive as an ordinary user; the ENTRYPOINT, ENV
# and labels come across, so `apptainer run wsclean.sif --version` behaves like
# the container did. See docs/cluster.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ACTION="${1:-}"
DIR="${2:-${REPO_ROOT}/images/archives}"
SIF_DIR="${REPO_ROOT}/images"

# Docker tags as scripts/build.sh names them.
NAMES=(wsclean r2d2 meqtrees polychord)
tag_of() {
  case "$1" in
    wsclean) echo ri-reproducibility/wsclean:v3.7 ;;
    r2d2) echo ri-reproducibility/r2d2:cpu ;;
    meqtrees) echo ri-reproducibility/meqtrees:kern-10 ;;
    polychord) echo ri-reproducibility/polychord:lite ;;
  esac
}

case "${ACTION}" in
  export)
    mkdir -p "${DIR}"
    for name in "${NAMES[@]}"; do
      echo "==> ${DIR}/${name}.tar <- $(tag_of "${name}")"
      docker save "$(tag_of "${name}")" -o "${DIR}/${name}.tar"
    done
    echo "OK: rsync -avz ${DIR}/ <cluster>:$(realpath --relative-to="${REPO_ROOT}" "${DIR}" 2>/dev/null || echo "${DIR}")/ then ./ri images import"
    ;;
  import)
    APPTAINER="$(command -v apptainer || command -v singularity)" \
      || { echo "FATAL: neither apptainer nor singularity is on PATH" >&2; exit 1; }
    # Unpacking the r2d2 archive needs ~6GB of scratch; keep it beside the
    # output rather than trusting /tmp on a login node.
    export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${SIF_DIR}/.tmp}"
    export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${SIF_DIR}/.cache}"
    mkdir -p "${SIF_DIR}" "${APPTAINER_TMPDIR}" "${APPTAINER_CACHEDIR}"
    for name in "${NAMES[@]}"; do
      archive="${DIR}/${name}.tar"
      [ -f "${archive}" ] || { echo "FATAL: ${archive} not found - run 'images export' on the Docker host and rsync ${DIR}/" >&2; exit 1; }
      if [ -f "${SIF_DIR}/${name}.sif" ] && [ ! "${archive}" -nt "${SIF_DIR}/${name}.sif" ]; then
        echo "==> ${SIF_DIR}/${name}.sif is newer than its archive, skipping"
        continue
      fi
      echo "==> ${SIF_DIR}/${name}.sif <- ${archive}"
      "${APPTAINER}" build --force "${SIF_DIR}/${name}.sif" "docker-archive://${archive}"
    done
    rm -rf "${APPTAINER_TMPDIR}"
    echo "OK: images in ${SIF_DIR}"
    ;;
  *)
    echo "usage: $0 export|import [DIR]" >&2
    exit 1
    ;;
esac
