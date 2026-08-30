# syntax=docker/dockerfile:1
# PolyChordLite plus repo nested-sampling driver. Build with:
# docker build --platform linux/arm64 -f docker/polychord/Dockerfile -t ri-reproducibility/polychord:lite .

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
        astropy \
        mpi4py

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

# python:3.11-slim ships no .pyc for most of the stdlib, and pip left the venv
# with .pyc whose recorded mtimes no longer match after the COPY above. So the
# first Python process in a fresh container recompiled ~60 modules and wrote
# them into the container layer: an 8-rank `mpirun python3 -c "import numpy,
# pypolychord, common"` cost 0.99s cold against 0.22s once one process had
# paid it. Compiling here makes every container start warm, which is what the
# run script would otherwise have to buy with a throwaway warm-up exec on the
# critical path. `|| true`: compileall fails the build on any module that is not
# valid source for this interpreter, and the venv carries a few.
RUN python3 -m compileall -q /usr/local/lib/python3.11 /opt/venv/lib || true

COPY scripts/lib/nested_sampling /opt/ri-nested-sampling
RUN python3 -m compileall -q /opt/ri-nested-sampling

WORKDIR /workspace

ENTRYPOINT ["python3", "/opt/ri-nested-sampling/polychord_wsclean.py"]
CMD ["--help"]
