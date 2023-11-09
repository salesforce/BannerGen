import requests
import json
from PIL import Image
import io
import base64
import uuid

url = 'http://34.72.100.219:8080/'  # backend external IP

# upload background image
task = 'upload'
files = {'file': open('./test/data/dark_flooring.jpg','rb')}
res = requests.post(url + task, files=files)

# call generation
task = 'prediction'
with open('./test/data/banner_content.json') as f:
    banner_dict = json.load(f)
response_byte = requests.post(url + task, json=banner_dict)
response_dict = json.loads(json.loads(response_byte._content.decode("utf-8")))

# retrieve screenshots and image embedded htmls from response
for i in range(len(response_dict['generatedResults'])):
    fname = str(uuid.uuid4())
    screenshot = Image.open(io.BytesIO(base64.b64decode(response_dict['generatedResults'][i]['screenshot'])))
    screenshot.save(fname + '.png')
    with open(fname + '.html', 'w') as f:
        f.write(response_dict['generatedResults'][i]['html'])
