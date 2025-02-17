FROM nvcr.io/nvidia/pytorch:25.01-py3-igpu

RUN apt update
RUN apt install -y libgl1 iproute2

WORKDIR /workdir

# Copy the current directory contents into the container at /workdir
COPY . /workdir
# [Optional] Set the default user. Omit if you want to keep the default as root.
USER ubuntu

# RUN python3.11 -m venv env2
# RUN sudo ./env2/bin/pip install -r requirements.txt
# USER $USERNAME
CMD ["./env/bin/python src/main.py"]
