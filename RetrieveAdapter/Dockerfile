# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

FROM nvcr.io/nvidia/pytorch:21.08-py3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y apt-utils 2>&1 | grep -v "debconf: delaying package configuration, since apt-utils is not installed"
RUN apt-get install -y --no-install-recommends tmux
RUN apt-get install -y --no-install-recommends openssh-server openssh-client
RUN apt-get install -y --no-install-recommends libsm6 libxext6 libxrender-dev
RUN ln -fs /usr/share/zoneinfo/America/Los_Angelos /etc/localtime

RUN pip install \
    imageio \
    imageio-ffmpeg==0.4.4 \
    pyspng==0.1.0 \
    tensorboardX \
    cython \
    mujoco-py==1.50.1.56 \
    mpi4py==3.0.0 \
    ipdb \
    glfw \
    gym[mujoco] \
    gym[robotics] \
    pylint \
    tqdm \
    autopep8 \
    pyyaml \
    fuzzywuzzy \
    python-Levenshtein \
    stanford-corenlp \
    pytorch-pretrained-BERT \
    slackclient \
    virtualenv \
    ipykernel \
    jupyter \
    matplotlib \
    scipy \
    sklearn \
    Pillow \
    h5py \
    keras \
    lmdb \
    opencv-python \
    moviepy \
    requests \
    cryptography \
    six \
    scikit-image \
    boto3 \
    tensorflow-gpu \
    notebook \
    ipywidgets \
    pydevd-pycharm~=212.5457.59

WORKDIR /home/Claude
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
COPY ../chromedriver /usr/bin/chromedriver
RUN DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install ../google-chrome-stable_current_amd64.deb
RUN chmod a+x run.sh
# Comment this line below if you do not want to auto start the server
ENTRYPOINT sh run.sh
