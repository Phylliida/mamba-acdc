FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel


ARG UID=10000
ARG GID=101
ARG USERNAME=dev

RUN apt-get install sudo

# make python point to the right place
#RUN rm /bin/python3 && ln -s /opt/conda/bin/python3 /bin/python3
# Install some useful packages

#RUN echo $(whereis python) && echo "bees" && wow
RUN sudo DEBIAN_FRONTEND=noninteractive apt install -y rsync git vim graphviz xdg-utils
RUN pip install pip --upgrade

RUN pip3 install --user packaging 

RUN pip3 install --user causal-conv1d mamba-ssm wandb transformer-lens mamba-lens

#RUN python -v -c "import wandb" | wow

#RUN cp -r /root/.local/lib/ /opt/conda/lib/python3.10/

# Set the non-root user as the default user
USER ${USERNAME}
WORKDIR /home/${USERNAME}
