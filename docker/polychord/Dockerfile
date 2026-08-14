# syntax=docker/dockerfile:1
#
# PolyChordLite plus the repo's nested-sampling PoC driver.
#
# Build:
#   docker build --platform linux/arm64 -f docker/polychord/Dockerfile \
#     -t ri-reproducibility/polychord:lite .

ARG POLYCHORD_GIT_URL=https://github.com/PolyChord/PolyChordLite.git
ARG POLYCHORD_GIT_COMMIT=370f6af5d59a4d3af2ed3333acfba152fdb7a4cf

FROM python:3.11-slim-bookworm AS build

ARG POLYCHORD_GIT_URL
ARG POLYCHORD_GIT_COMMIT

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    gfortran \
    git \
    libopenmpi-dev \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:${PATH}

RUN pip install --upgrade pip \
    && pip install \
        "git+${POLYCHORD_GIT_URL}@${POLYCHORD_GIT_COMMIT}" \
        astropy

FROM python:3.11-slim-bookworm AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    docker.io \
    libgfortran5 \
    libopenmpi3 \
    openmpi-bin \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/venv /opt/venv
COPY scripts/lib/nested_sampling /opt/ri-nested-sampling

WORKDIR /workspace
VOLUME ["/data", "/results"]

ENTRYPOINT ["python3", "/opt/ri-nested-sampling/polychord_wsclean_poc.py"]
CMD ["--help"]
