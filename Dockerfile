FROM pytorch/pytorch as base

ARG UID=10000
ARG GID=101
ARG USERNAME=dev

# Update the package list, install sudo, create a non-root user, and grant password-less sudo permissions
RUN apt update && \
    apt install -y sudo && \
    addgroup --gid $GID ${USERNAME} && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install some useful packages
RUN apt update && \
    apt install -y rsync git vim graphviz xdg-utils

# Set the non-root user as the default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}
