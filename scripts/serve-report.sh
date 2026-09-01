#!/usr/bin/env bash
# Serve report over HTTP; default bind is loopback and report is unauthenticated.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${REPORT_PORT:-8000}"
BIND="${REPORT_BIND:-127.0.0.1}"
REPORT_DIR="${REPO_ROOT}/reports/nested-sampling-report"

if [[ ! -f "${REPORT_DIR}/index.html" ]]; then
  echo "refuse: no report at ${REPORT_DIR} - run './ri report' first" >&2
  exit 1
fi

# IPv6 literals need brackets in URLs and SSH local-forward specs.
bracketed() {
  case "$1" in
    *:*) echo "[$1]" ;;
    *) echo "$1" ;;
  esac
}

echo "Serving ${REPORT_DIR} on ${BIND}:${PORT}"
if [[ "${BIND}" == "127.0.0.1" || "${BIND}" == "localhost" || "${BIND}" == "::1" ]]; then
  # Prefer SSH's server address, then the first local address, then hostname.
  ADDRESS="$(awk '{print $3}' <<<"${SSH_CONNECTION:-}")"
  if [[ -z "${ADDRESS}" ]]; then
    ADDRESS="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  fi
  # SSH host stays unbracketed; bracket only the -L destination.
  SSH_HOST="${REPORT_SSH_HOST:-$(whoami)@${ADDRESS:-$(hostname)}}"
  echo
  echo "On your local machine:"
  echo "    ssh -N -L ${PORT}:$(bracketed "${BIND}"):${PORT} ${SSH_HOST}"
  echo "then open http://localhost:${PORT}/"
else
  echo
  echo "Open http://$(bracketed "${BIND}"):${PORT}/ - the report is unauthenticated,"
  echo "so anything that can reach this host on this port can read it."
fi
echo
echo "Ctrl-C to stop."

exec python3 "${REPO_ROOT}/scripts/lib/report_server.py" "${PORT}" \
  --bind "${BIND}" --directory "${REPORT_DIR}"
