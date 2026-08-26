# syntax=docker/dockerfile:1
#
# MeqTrees/Cattery runtime for VLA Measurement Set simulation.
#
# Build:
#   docker build --platform linux/arm64 -f docker/meqtrees/Dockerfile \
#     -t ri-reproducibility/meqtrees:kern-10 .

FROM ubuntu:24.04

ARG KERNSUITE_SIGNING_FINGERPRINT=813B0DC8B7D042E7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/usr/lib/python3/dist-packages/Cattery:/usr/lib/python3/dist-packages/Cattery/Calico:/usr/lib/python3/dist-packages/Cattery/Calico/OMS

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x${KERNSUITE_SIGNING_FINGERPRINT}" \
        | gpg --dearmor > /etc/apt/keyrings/kernsuite.gpg \
    && printf '%s\n' \
        'deb [signed-by=/etc/apt/keyrings/kernsuite.gpg] https://ppa.launchpadcontent.net/kernsuite/kern-10/ubuntu noble main' \
        > /etc/apt/sources.list.d/kernsuite.list \
    && apt-get update \
    && printf 'path-include=/usr/share/doc/makems/*\n' > /etc/dpkg/dpkg.cfg.d/zzz-include-makems-doc \
    && apt-get install -y --no-install-recommends \
        casacore-data=20250826-160001-5 \
        makems=1.5.6-1kern1 \
        meqtrees=1.8.0-1kern1 \
        meqtrees-timba=1.11-1kern1 \
        python3-meqtrees-cattery=1.8.0-1kern1 \
        python3-casacore=3.7.1-1kern2 \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Timba's stop_default_mqs() reaps the meqserver child with waitpid(WNOHANG)
# and then sleeps a fixed 10s before re-checking, so every predict pays ~10s of
# pure shutdown wait for a run whose real work takes <0.5s. Poll 100x finer,
# same ~200s ceiling before the SIGKILL fallback.
RUN python3 - <<'EOF'
import pathlib

path = pathlib.Path("/usr/lib/python3/dist-packages/Timba/Apps/meqserver.py")
source = path.read_text()
for old, new in (("      for i in range(20):", "      for i in range(2000):"),
                 ("        time.sleep(10);", "        time.sleep(0.1);")):
    assert source.count(old) == 1, f"meqserver.py shutdown poll changed upstream: {old!r}"
    source = source.replace(old, new)
path.write_text(source)
EOF

COPY scripts/lib/nested_sampling/simulate_point_source_ms.py /opt/ri-nested-sampling/simulate_point_source_ms.py
COPY scripts/lib/nested_sampling/point_source_forest.py /opt/ri-nested-sampling/point_source_forest.py
COPY scripts/lib/nested_sampling/ms_to_r2d2_mat.py /opt/ri-nested-sampling/ms_to_r2d2_mat.py

# Every (NTimes, NFrequencies) shape the parameter space can ask for, built
# here so that no run ever calls makems: a fresh makems is ~0.11s against
# ~0.004s for a cache hit, and a default 8-rank run used to take ~7 of those
# misses inside the sampler plus one per worker in warm_forest(). ~1.2s of build
# time and ~18MB of image for -0.17s end to end. common.py is bind-mounted
# rather than copied so the shapes come from the one authoritative
# PARAMETER_SPACE without the runtime image growing a fourth script.
RUN --mount=type=bind,source=scripts/lib/nested_sampling/common.py,target=/opt/ri-nested-sampling/common.py \
    python3 /opt/ri-nested-sampling/simulate_point_source_ms.py --prebuild-skeletons

# Same reason as the polychord image: anything without a valid .pyc is
# recompiled by the first process in each fresh container. Ubuntu ships .pyc for
# its own python3 packages, so here this is only these three files and the
# Timba module the patch above rewrote.
RUN python3 -m compileall -q /opt/ri-nested-sampling /usr/lib/python3/dist-packages/Timba/Apps || true

WORKDIR /workspace
VOLUME ["/data", "/results"]

ENTRYPOINT ["python3", "/opt/ri-nested-sampling/simulate_point_source_ms.py"]
CMD ["--help"]
