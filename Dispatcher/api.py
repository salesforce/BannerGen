#!/usr/bin/env python3
"""
Copyright (c) 2023 Salesforce, Inc.

All rights reserved.

SPDX-License-Identifier: Apache License 2.0

For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/

By Chia-Chih Chen, chiachih.chen@salesforce.com
"""
from flask import Flask, request, jsonify, make_response
import aiohttp
import asyncio
import async_timeout
import pdb


loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
app = Flask(__name__)
app.root_path = '/usr/src/app'
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

app = Flask(__name__)
container_dict = {'layoutdetr': 'http://localhost:5000/', 'instructpix2pix': 'http://localhost:4000/', 'retrieveadapter': 'http://localhost:3000/'}
TIMEOUT_ASYNC = 16

# Given the model pod url and request type, post the request and wait for the response asynchronously
async def call_container(container_url, request_type, payload):
    if request_type == 'prediction':
        async with aiohttp.ClientSession() as session, async_timeout.timeout(TIMEOUT_ASYNC):
            async with session.post(container_url + request_type, json=payload) as response:
                return await response.text()
    elif request_type == 'upload':
        with open(payload, 'rb') as f:
            async with aiohttp.ClientSession() as session, async_timeout.timeout(TIMEOUT_ASYNC):
                async with session.post(container_url + request_type, data={'image': f}) as response:
                    return await response.text()


async def perform_calls(request_type, payload):
    tasks = []
    for container_url in container_dict.values():
        tasks.append(call_container(container_url, request_type, payload))
    responses = await asyncio.gather(*tasks)

    return responses


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    # todo: relay serialized image
    file = request.files['file']
    file.save(file.filename)
    responses = loop.run_until_complete(perform_calls('upload', file.filename))
    return jsonify(responses[0])


@app.route('/prediction', methods=['POST'])
def prediction():
    request_dict = request.get_json()
    request_dict['numResultsPerModel'] = {'LayoutDETR': 1, 'InstructPix2Pix': 3, 'Retrieve-Adaptor': 4}
    responses = loop.run_until_complete(perform_calls('prediction', request_dict))
    return jsonify(responses[0])
