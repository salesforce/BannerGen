# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
# Copyright (c) 2023 Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache License 2.0
# For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
# By Chia-Chih Chen, chiachih.chen@salesforce.com

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
    scikit-learn \
    Pillow==9.5.0  \
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
    notebook \
    ipywidgets \
    pydevd-pycharm~=212.5457.59

# Exposing applicaiton on 5000 so that it can be accessible on default host(127.0.0.1) and port(5000).
EXPOSE 5000
RUN mkdir -p /home/Claude
WORKDIR /home/Claude
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . /home/Claude
COPY ../chromedriver /usr/bin/chromedriver
COPY activations.py /opt/conda/lib/python3.8/site-packages/transformers/activations.py
RUN DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install ../google-chrome-stable_current_amd64.deb
RUN chmod a+x run.sh
ENV CONTAINER_NAME "LayoutDETR"
# Comment this line below if you do not want to auto start the server
# ENTRYPOINT sh run.sh
ENTRYPOINT ["/bin/sh","-c","sleep infinity"]
