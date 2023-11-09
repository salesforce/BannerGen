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
RUN apt-get install -y --no-install-recommends git
RUN apt-get install -y --no-install-recommends openssh-server openssh-client
RUN apt-get install -y --no-install-recommends libsm6 libxext6 libxrender-dev
RUN ln -fs /usr/share/zoneinfo/America/Los_Angelos /etc/localtime

EXPOSE 4000
WORKDIR /home/
RUN pip install --upgrade pip
RUN git clone https://github.com/huggingface/transformers.git
RUN pip install -q ./transformers
RUN mkdir -p /home/Claude
WORKDIR /home/Claude
COPY . /home/Claude
RUN pip install -r requirements.txt
COPY chromedriver /usr/bin/chromedriver
COPY quantize.py /opt/conda/lib/python3.8/site-packages/taming/modules/vqvae/quantize.py
RUN DEBIAN_FRONTEND=noninteractive apt-get --assume-yes install ./google-chrome-stable_current_amd64.deb
RUN chmod a+x run.sh
ENV CONTAINER_NAME "InstructPix2Pix"
# Comment this line below if you do not want to auto start the server
# ENTRYPOINT sh run.sh
ENTRYPOINT ["/bin/sh","-c","sleep infinity"]