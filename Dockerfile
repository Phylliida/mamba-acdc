ARG PYTORCH_CUDA_VERSION=2.0.0-cuda11.7-cudnn8

FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-devel as envpool-environment
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y golang-1.18 git \
    # Linters
      clang-format clang-tidy \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Bazel will use /bin/python3 as the Python interpreter for running tests, even if it uses /opt/conda/bin/python3 to
# build everything else. So we make sure /bin/python3 points to the right place
RUN rm /bin/python3 && ln -s /opt/conda/bin/python3 /bin/python3

ENV PATH=/usr/lib/go-1.18/bin:/root/go/bin:$PATH
RUN go install github.com/bazelbuild/bazelisk@v1.19.0 && ln -sf $HOME/go/bin/bazelisk $HOME/go/bin/bazel
RUN go install github.com/bazelbuild/buildtools/buildifier@v0.0.0-20231115204819-d4c9dccdfbb1
# Install Go linting tools
RUN go install github.com/google/addlicense@v1.1.1

ENV USE_BAZEL_VERSION=6.4.0
RUN bazel version

WORKDIR /app

# Copy the whole repository
COPY third_party/envpool .

# Deal with the fact that envpool is a submodule and has no .git directory
RUN rm .git
# Copy the .git repository for this submodule
COPY .git/modules/third_party/envpool ./.git
# Remove config line stating that the worktree for this repo is elsewhere
RUN sed -e 's/^.*worktree =.*$//' .git/config > .git/config.new && mv .git/config.new .git/config

# Abort if repo is dirty
RUN echo "$(git status --porcelain --ignored=traditional)" \
    && if ! { [ -z "$(git status --porcelain --ignored=traditional)" ] \
    ; }; then exit 1; fi

FROM envpool-environment as envpool-ci
# Warm up the build for CI
RUN make bazel-build

FROM envpool-environment as envpool
RUN make bazel-release

FROM pytorch/pytorch:${PYTORCH_CUDA_VERSION}-devel as main-pre-pip

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

FROM main-pre-pip as main-pip-tools
RUN pip install "pip-tools ~=7.4.1"

FROM main-pre-pip as main
# Install all dependencies, which should be explicit in `requirements.txt`

# Copy whole repo and install
COPY --chown=${USERNAME}:${USERNAME} . .
RUN pip install packaging && rm -rf "${HOME}/.cache"
RUN pip install causal-conv1d mamba-ssm wandb transformer-lens mamba-lens && rm -rf "${HOME}/.cache"


# Default command to run -- may be changed at runtime
CMD ["/bin/bash"]
