# Copyright (c) 2023 Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache License 2.0
# For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/
# By Chia-Chih Chen, chiachih.chen@salesforce.com


# Using Base image alpine with installed python3
FROM frolvlad/alpine-python3

#Making default entry as python will launch api1.py
#ENTRYPOINT ["/bin/sh","-c","sleep infinity"]

# Exposing applicaiton on 5000 so that it can be accessible on default host(127.0.0.1) and port(5000).
EXPOSE 6000
RUN mkdir -p /usr/src/app
COPY . /usr/src/app
WORKDIR /usr/src/app
RUN pip install -r requirements.txt
RUN chmod a+x run.sh
# Comment this line below if you do not want to auto start the server
# ENTRYPOINT sh run.sh
ENTRYPOINT ["/bin/sh","-c","sleep infinity"]