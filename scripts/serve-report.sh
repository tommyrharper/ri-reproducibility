#!/usr/bin/env bash
# Serve the generated HTML report over HTTP, so it can be read from a browser
# on another machine - this repo normally runs on a headless remote host.
#
#   scripts/serve-report.sh
#   REPORT_PORT=9000 scripts/serve-report.sh
#   REPORT_BIND=0.0.0.0 scripts/serve-report.sh
#
# Nothing is served to the network: the default bind is loopback, and the
# `ssh -L` line printed on startup is what makes it reachable - that tunnel
# runs on your own machine. The report is unauthenticated, so a loopback bind
# still leaves it readable by anyone with an account on this host, and by any
# container sharing its network namespace. REPORT_BIND=0.0.0.0 drops even that
# and serves it to anything that can reach the host.
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

# An IPv6 literal needs brackets everywhere it is followed by :port - in a URL,
# and in ssh's -L forward spec.
bracketed() {
  case "$1" in
    *:*) echo "[$1]" ;;
    *) echo "$1" ;;
  esac
}

echo "Serving ${REPORT_DIR} on ${BIND}:${PORT}"
if [[ "${BIND}" == "127.0.0.1" || "${BIND}" == "localhost" || "${BIND}" == "::1" ]]; then
  # Whoever runs the tunnel needs a host to point it at, and this script cannot
  # know how you reached the box - so it works down from the best evidence it
  # has to the weakest guess:
  #
  # 1. sshd sets SSH_CONNECTION to "<client ip> <client port> <server ip>
  #    <server port>" in every session it starts. Its third field is the
  #    address this session actually arrived on, which is the one thing here
  #    that is not a guess - it stays right behind NAT and on a multi-homed
  #    host, where the alternatives below are wrong.
  # 2. `hostname -I` is the host's own view of its addresses: right on a cloud
  #    box with a public IP, wrong behind NAT. `|| true` because it does not
  #    exist on macOS and exits non-zero there, which `set -e` would otherwise
  #    turn into the server never starting at all.
  # 3. Failing both, the hostname, which at least names the machine.
  #
  # The username is `whoami` throughout - the account running this, which is
  # not necessarily the one you SSH in as. REPORT_SSH_HOST overrides the lot.
  ADDRESS="$(awk '{print $3}' <<<"${SSH_CONNECTION:-}")"
  if [[ -z "${ADDRESS}" ]]; then
    ADDRESS="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  fi
  # Not bracketed: ssh wants a bare address after the @ ("ssh user@::1" works,
  # "ssh user@[::1]" fails to resolve). Brackets are only for the -L spec below.
  SSH_HOST="${REPORT_SSH_HOST:-$(whoami)@${ADDRESS:-$(hostname)}}"
  # Forward to whatever was actually bound, not to a hardcoded 127.0.0.1: a
  # REPORT_BIND=::1 server is not listening on the IPv4 loopback at all, and
  # `localhost` resolves to whichever of the two the host's getaddrinfo prefers.
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

exec python3 -m http.server "${PORT}" --bind "${BIND}" --directory "${REPORT_DIR}"
