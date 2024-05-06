FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


ARG UID=10000
ARG GID=101
ARG USERNAME=dev

RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 9A2FD067A2E3EF7B
# Update the package list, install sudo, create a non-root user, and grant password-less sudo permissions
RUN  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 605C66F00D6C9793 \
    0E98404D386FA1D9 648ACFD622F3D138 && apt update && \
    apt install -y sudo && \
    addgroup --gid $GID ${USERNAME} && \
    adduser --uid $UID --gid $GID --disabled-password --gecos "" ${USERNAME} && \
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install some useful packages
RUN sudo apt update
RUN sudo DEBIAN_FRONTEND=noninteractive apt install -y rsync git vim graphviz xdg-utils

RUN pip install packaging 

RUN pip install causal-conv1d mamba-ssm 

# Set the non-root user as the default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}
