#!/usr/bin/env bash
# Serve the generated HTML report over HTTP, so it can be read from a browser
# on another machine - this repo normally runs on a headless remote host.
#
#   scripts/serve-report.sh
#   REPORT_PORT=9000 scripts/serve-report.sh
#   REPORT_BIND=0.0.0.0 scripts/serve-report.sh
#
# Nothing on this host is exposed and no port is opened: the server binds to
# loopback, and the `ssh -L` line printed on startup is what makes it
# reachable - that tunnel runs on your own machine. REPORT_BIND=0.0.0.0 opts
# out of that and serves the report, unauthenticated, to anything that can
# reach this host.
#
# A single `scp` of one page would not work: the pages link to sibling PNGs
# under images/, which is nearly all of the report's bytes. Serving the
# directory keeps those links intact.
#
# Runs in the foreground; Ctrl-C stops it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PORT="${REPORT_PORT:-8000}"
BIND="${REPORT_BIND:-127.0.0.1}"
REPORT_DIR="${REPO_ROOT}/reports/nested-sampling-report"

if [[ ! -f "${REPORT_DIR}/index.html" ]]; then
  echo "refuse: no report at ${REPORT_DIR} - run './ri report' first" >&2
  exit 1
fi

echo "Serving ${REPORT_DIR} on ${BIND}:${PORT}"
if [[ "${BIND}" == "127.0.0.1" || "${BIND}" == "localhost" || "${BIND}" == "::1" ]]; then
  # Whoever runs the tunnel needs a host to point it at. `hostname -I` is this
  # host's own view of its addresses: right on a cloud box, wrong behind NAT
  # and unavailable on macOS - set REPORT_SSH_HOST when it guesses badly.
  # `|| true` because macOS exits non-zero here, which `set -e` would otherwise
  # turn into the server never starting at all.
  ADDRESS="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  SSH_HOST="${REPORT_SSH_HOST:-$(whoami)@${ADDRESS:-$(hostname)}}"
  echo
  echo "On your local machine:"
  echo "    ssh -N -L ${PORT}:127.0.0.1:${PORT} ${SSH_HOST}"
  echo "then open http://localhost:${PORT}/"
else
  echo
  echo "Open http://${BIND}:${PORT}/ - the report is unauthenticated, so anything"
  echo "that can reach this host on this port can read it."
fi
echo
echo "Ctrl-C to stop."

exec python3 -m http.server "${PORT}" --bind "${BIND}" --directory "${REPORT_DIR}"
