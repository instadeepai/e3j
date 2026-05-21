# Copyright (c) 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Base image with cuda since we need nvcc and cuda toolkit
# Note: could not find cuda:13.1.0-cudnn image on dockerhub,
#       so cudnn9-cuda-13-1 is installed with apt.
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED="yes"
WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

ENV UV_GROUPS="--group cuda13_local --group exp"
#ENV UV_GROUPS="--group cuda13_local --group exp --group openeq --group cuet"

# Install system dependencies
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends git g++ make cmake vim\
    && apt-get clean autoclean \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Make sure cmake finds g++ and nvcc
ENV CXX="/usr/bin/g++"
ENV CUDACXX="/usr/local/cuda/bin/nvcc"

# Make the uv env python the default system wide
ENV PATH="/app/.venv/bin:$PATH"

# Depending on how Docker run is executed, the current ENV variables may be overwritten.
# Make sure they are reloaded when logging in to the container
RUN echo "export PATH=/app/.venv/bin:\$PATH\n" >> /root/.profile

COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY uv.lock uv.lock

COPY lib/e3j_ops/pyproject.toml lib/e3j_ops/pyproject.toml
COPY lib/e3j_ops/README.md lib/e3j_ops/README.md

# Check that we have g++ and nvcc
RUN echo $(which g++) || echo "g++ not found"
RUN echo $(which nvcc) || echo "nvcc not found"

RUN uv sync $UV_GROUPS --no-install-project --no-dev --frozen

# Print important versions/refs
RUN echo "Built image with versions:" &&\
    echo "=========================" \
    && uv pip freeze | grep jaxlib

COPY lib lib
COPY src src

# Install e3j into environment
RUN uv sync --extra ops $UV_GROUPS --inexact --frozen

# Copy recipes scripts and tests
RUN mkdir bin/
COPY Makefile Makefile
COPY scripts scripts
COPY tests tests
COPY pytest.ini pytest.ini
