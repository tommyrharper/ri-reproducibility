#!/usr/bin/env bash
# Fetch (or explain how to fetch) the pretrained R2D2 DNN checkpoints.
#
# This CANNOT be fully automated: researchportal.hw.ac.uk serves these
# files behind a Cloudflare bot challenge that rejects non-browser
# requests (verified 2026-08-03 - `curl -I` gets HTTP 403 with a
# Cloudflare challenge page, not the file). This script tries anyway (in
# case that changes), detects the failure precisely, and otherwise prints
# exact manual-download instructions instead of a cryptic download error.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/lib/defaults.sh
source "${REPO_ROOT}/scripts/lib/defaults.sh"

REALISATION="${1:-R2D2_A1_T2_Realisation1.zip}"
URL_BASE="https://researchportal.hw.ac.uk/files"

declare -A REALISATION_URLS=(
  [R2D2_A1_T2_Realisation1.zip]="146289536"
  [R2D2_A1_T2_Realisation2.zip]="146314171"
  [R2D2_A1_T2_Realisation3.zip]="146314172"
  [R2D2_A1_T2_Realisation4.zip]="146314173"
  [R2D2_A1_T2_Realisation5.zip]="146314174"
  [R2D2_A2_T2_Realisation1.zip]="146314175"
  [R2D2_A2_T2_Realisation2.zip]="146289537"
  [R2D2_A2_T2_Realisation3.zip]="146314176"
  [R2D2_A2_T2_Realisation4.zip]="146314177"
  [R2D2_A2_T2_Realisation5.zip]="146314178"
)

if [ -z "${REALISATION_URLS[${REALISATION}]:-}" ]; then
  echo "FATAL: unknown realisation '${REALISATION}'." >&2
  echo "Known values: ${!REALISATION_URLS[*]}" >&2
  exit 1
fi

FILE_ID="${REALISATION_URLS[${REALISATION}]}"
URL="${URL_BASE}/${FILE_ID}/${REALISATION}"
DEST="${CHECKPOINTS_DIR}/${REALISATION}"

mkdir -p "${CHECKPOINTS_DIR}"

if [ -f "${DEST}" ]; then
  echo "==> ${DEST} already present (presumably placed manually) - checksumming, not re-downloading"
  shasum -a 256 "${DEST}" | tee -a "${CHECKPOINTS_DIR}/CHECKSUMS.sha256"
  echo "OK: ${DEST}"
  echo "NOTE: no upstream checksum is published for this file (checked"
  echo "the Heriot-Watt Research Portal landing page); the sha256 above"
  echo "is self-recorded so future downloads can be compared against it,"
  echo "not verified against an authoritative upstream value."
  exit 0
fi

echo "==> attempting automated download of ${REALISATION} (~3.5-5.3 GB)"
HTTP_STATUS="$(curl -sL -o "${DEST}.partial" -w '%{http_code}' "${URL}" || echo "000")"

if [ "${HTTP_STATUS}" = "200" ] && file "${DEST}.partial" | grep -qi zip; then
  mv "${DEST}.partial" "${DEST}"
  shasum -a 256 "${DEST}" | tee -a "${CHECKPOINTS_DIR}/CHECKSUMS.sha256"
  echo "OK: downloaded ${DEST}"
  echo "NOTE: no upstream checksum is published for this file (checked"
  echo "the Heriot-Watt Research Portal landing page); the sha256 above"
  echo "is self-recorded so future downloads can be compared against it,"
  echo "not verified against an authoritative upstream value."
  exit 0
fi

rm -f "${DEST}.partial"
cat >&2 <<EOF

FATAL: automated download failed (HTTP status: ${HTTP_STATUS}).

This is expected: researchportal.hw.ac.uk serves checkpoint files behind
a Cloudflare bot challenge that rejects non-browser HTTP clients.

Manual download required:
  1. Open in a browser: https://${URL#https://}
  2. Save the file to: ${DEST}
  3. Re-run this script (it will checksum instead of re-downloading):
       $(basename "$0") ${REALISATION}

All ten realisation archives (5 U-Net "A1_T2", 5 U-WDSR "A2_T2") are
listed at the dataset landing page:
  https://researchportal.hw.ac.uk/en/datasets/robust-r2d2-dnn-series-for-monochromatic-intensity-imaging-with-v
DOI: 10.17861/e3060b95-4fe6-4b61-9f72-d77653c305bb (CC BY)

For a smoke test / baseline, one realisation of one architecture is
enough - the default (R2D2_A1_T2_Realisation1.zip) matches
config/r2d2/R2D2_U-Net.yaml's ckpt_realisations: 1.
EOF
exit 1
