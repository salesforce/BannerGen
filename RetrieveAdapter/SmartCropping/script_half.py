import os
from PIL import Image
import numpy as np
import math
import smartcrop
import json

src_dir = '../../datasets/clean_testing_background'
tgt_dir = '../../datasets/clean_testing_background_smart_crop_half'
os.makedirs(tgt_dir, exist_ok=True)

sc = smartcrop.SmartCrop()

for file in sorted(os.listdir(src_dir)):
	if '.png' in file:
		src_path = '%s/%s' % (src_dir, file)
		im = np.array(Image.open(src_path))
		if len(im.shape) < 3:
			im = np.dstack((im, im, im))
		h, w, c = im.shape
		if c > 3:
			im = im[:,:,:3]
		im = Image.fromarray(im)

		w_small = 700
		mul = float(w_small) / float(w)
		h_small = int(float(h) * mul)
		im_small = im.resize((w_small, h_small))

		for w_new, h_new, max_scale in [(w_small, h_small//2, 1.0), (w_small, h_small//4, 1.0), (w_small//2, h_small//2, 0.5)]:
			tgt_path = '%s/%s_width_%.2f_height_%.2f.png' % (tgt_dir, file[:-4], float(w_new)/float(w_small), float(h_new)/float(h_small))
			print(tgt_path)
			results = sc.crop(im_small, w_new, h_new, max_scale=max_scale)
			box = (int(float(results['top_crop']['x']) / mul),
           	   	   int(float(results['top_crop']['y']) / mul),
                   int(float(results['top_crop']['width'] + results['top_crop']['x']) / mul),
                   int(float(results['top_crop']['height'] + results['top_crop']['y']) / mul))
			im_new = im.crop(box)
			im_new.save(tgt_path)