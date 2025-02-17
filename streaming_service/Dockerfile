FROM ubuntu:22.04

RUN apt update
RUN apt install -y python3 python3.11 python3.11-venv sudo libgl1 iproute2 libglib2.0-0
RUN apt-get install -y python3-opencv

ARG USERNAME=noroot
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

WORKDIR /workdir

# Copy the current directory contents into the container at /workdir
COPY . /workdir
# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

#RUN ./env/bin/python3 main.py --dataset /dataset
CMD ["./env/bin/python","app2.py"]