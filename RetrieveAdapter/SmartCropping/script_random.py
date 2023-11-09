import os
from PIL import Image
import numpy as np
import math
import smartcrop
import json

src_dir = '../../datasets/clean_testing_background'
tgt_dir = '../../datasets/clean_testing_background_smart_crop_random'
os.makedirs(tgt_dir, exist_ok=True)

sc = smartcrop.SmartCrop()

for file in sorted(os.listdir(src_dir)):
	if '.png' in file:
		tgt_path = '%s/%s_smaller.png' % (tgt_dir, file[:-4])
		print(tgt_path)
		if os.path.exists(tgt_path):
			continue

		src_path = '%s/%s' % (src_dir, file)
		im = np.array(Image.open(src_path))
		if len(im.shape) < 3:
			im = np.dstack((im, im, im))
		h, w, c = im.shape
		if c > 3:
			im = im[:,:,:3]
		im = Image.fromarray(im)
		h_scale = math.exp(np.random.uniform(low=math.log(0.5), high=math.log(1.0)))
		w_scale = math.exp(np.random.uniform(low=math.log(0.5), high=math.log(1.0)))

		
		h_new = int(float(h) * h_scale)
		w_new = int(float(w) * w_scale)
		results = sc.crop(im, w_new, h_new)
		box = (results['top_crop']['x'],
           	   results['top_crop']['y'],
               results['top_crop']['width'] + results['top_crop']['x'],
               results['top_crop']['height'] + results['top_crop']['y'])
		im_new = im.crop(box)
		im_new.save(tgt_path)