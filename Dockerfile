# /root/.local/lib/python3.10/site-packages/
# /opt/conda/lib/python3.10/site-packages
# pip /opt/conda/bin/pip
# python /opt/conda/bin/python
# can set command to
#
# - sleep
# - 1d
# to ssh into box and test things
ARG PYTORCH_CUDA_VERSION=2.0.0-cuda11.7-cudnn8
FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-devel as main

ARG USERID=1001
ARG GROUPID=1001
ARG USERNAME=dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -q \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    # essential for running. GCC is for Torch triton
    git git-lfs build-essential \
    # essential for testing
    libgl-dev libglib2.0-0 zip make \
    # devbox niceties
    curl vim tmux less sudo \
    # CircleCI
    ssh rsync git vim graphviz xdg-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tini: reaps zombie processes and forwards signals
ENTRYPOINT ["/usr/bin/tini", "--"]

# Simulate virtualenv activation
ENV VIRTUAL_ENV="/opt/venv"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python3 -m venv "${VIRTUAL_ENV}" --system-site-packages \
    && addgroup --gid ${GROUPID} ${USERNAME} \
    && adduser --uid ${USERID} --gid ${GROUPID} --disabled-password --gecos '' ${USERNAME} \
    && usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && mkdir -p "/workspace" \
    && chown -R ${USERNAME}:${USERNAME} "${VIRTUAL_ENV}" "/workspace"
USER ${USERNAME}
WORKDIR "/workspace"


# Copy whole repo and install
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install packaging && rm -rf "${HOME}/.cache"
RUN pip install jupyter causal-conv1d mamba-ssm wandb transformer-lens mamba-lens git+https://github.com/Phylliida/ACDC.git --upgrade && rm -rf "${HOME}/.cache"
RUN pip install graphviz

# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
