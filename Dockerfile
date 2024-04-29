FROM ghcr.io/alignmentresearch/flamingo-devbox:latest


ARG UID=10000
ARG GID=101
ARG USERNAME=dev
# Install some useful packages
RUN sudo apt update
RUN sudo DEBIAN_FRONTEND=noninteractive apt install -y rsync git vim graphviz xdg-utils

# Set the non-root user as the default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}
