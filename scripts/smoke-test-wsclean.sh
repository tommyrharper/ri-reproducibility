#!/usr/bin/env bash
# Smallest meaningful WSClean smoke test: verify the binary reports its
# version, then run a real (tiny) imaging pass on an ASTRON-hosted,
# checksum-verified test Measurement Set so the whole gridding -> FFT ->
# deconvolution pipeline is exercised, not just `--version`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${WSCLEAN_IMAGE:-ri-reproducibility/wsclean:v3.7}"

MS_URL="https://support.astron.nl/software/ci_data/wsclean/JVLA-MultiBand-S1_C5-minimal.ms.tar.bz2"
MS_SHA256="7c8d41b5ff59c8736b1223e6b855a96e410f27fb4be05d179c8292fdb78cdc7e"
MS_ARCHIVE_NAME="JVLA-MultiBand-S1_C5-minimal.ms.tar.bz2"
MS_DIR_NAME="JVLA-MultiBand-S1_C5-minimal.ms"

FIXTURE_DIR="${REPO_ROOT}/results/.smoke-test-fixtures/wsclean"
OUTPUT_DIR="${REPO_ROOT}/results/smoke-test-wsclean"

mkdir -p "${FIXTURE_DIR}" "${OUTPUT_DIR}"

echo "==> [1/3] wsclean --version"
docker run --rm --platform linux/arm64 "${IMAGE}" --version

echo "==> [2/3] fetching + verifying test Measurement Set fixture"
if [ ! -d "${FIXTURE_DIR}/${MS_DIR_NAME}" ]; then
  ARCHIVE_PATH="${FIXTURE_DIR}/${MS_ARCHIVE_NAME}"
  if [ ! -f "${ARCHIVE_PATH}" ]; then
    curl -fsSL "${MS_URL}" -o "${ARCHIVE_PATH}"
  fi
  echo "${MS_SHA256}  ${ARCHIVE_PATH}" | shasum -a 256 -c - \
    || { echo "FATAL: checksum mismatch for ${ARCHIVE_PATH}" >&2; exit 1; }
  tar -xjf "${ARCHIVE_PATH}" -C "${FIXTURE_DIR}"
else
  echo "    fixture already present at ${FIXTURE_DIR}/${MS_DIR_NAME}, skipping download"
fi

echo "==> [3/3] running a minimal WSClean imaging pass"
ARGS_FILE="${REPO_ROOT}/config/wsclean/smoke-test.args"
WSCLEAN_ARGS=()
while IFS= read -r line; do
  WSCLEAN_ARGS+=("${line}")
done < <(grep -v '^[[:space:]]*#' "${ARGS_FILE}" | grep -v '^[[:space:]]*$')
docker run --rm --platform linux/arm64 \
  -v "${FIXTURE_DIR}:/data:ro" \
  -v "${OUTPUT_DIR}:/results" \
  --entrypoint wsclean \
  "${IMAGE}" \
  -name /results/smoke \
  -temp-dir /results \
  "${WSCLEAN_ARGS[@]}" \
  "/data/${MS_DIR_NAME}"

echo "OK: WSClean smoke test complete. Output FITS files in ${OUTPUT_DIR}"
