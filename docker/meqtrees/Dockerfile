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

COPY scripts/lib/nested_sampling/simulate_point_source_ms.py /opt/ri-nested-sampling/simulate_point_source_ms.py
COPY scripts/lib/nested_sampling/point_source_forest.py /opt/ri-nested-sampling/point_source_forest.py
COPY scripts/lib/nested_sampling/ms_to_r2d2_mat.py /opt/ri-nested-sampling/ms_to_r2d2_mat.py

WORKDIR /workspace
VOLUME ["/data", "/results"]

ENTRYPOINT ["python3", "/opt/ri-nested-sampling/simulate_point_source_ms.py"]
CMD ["--help"]
