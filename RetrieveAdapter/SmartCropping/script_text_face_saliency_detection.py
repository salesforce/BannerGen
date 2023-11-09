import os
from PIL import Image
import numpy as np

src_dir = './din'
tgt_dir = './dout'
os.makedirs(tgt_dir, exist_ok=True)

for file in sorted(os.listdir(src_dir)):
	if '.png' in file:
		src_path = '%s/%s' % (src_dir, file)
		im = np.array(Image.open(src_path))
		h = im.shape[0]
		w = im.shape[1]
		r = float(w) / float(h)
		for (w_new, h_new) in [(900, 900)]:
		#for (w_new, h_new) in [(1400, 1050), (1600, 900), (600, 800), (1080, 1920), (1080, 1080)]:
			tgt_path = '%s/%s_width_%d_height_%d.png' % (tgt_dir, file[:-4], w_new, h_new)
			print(tgt_path)
			os.system("CUDA_VISIBLE_DEVICES=0 python3 smartcrop_PaddleOCR_mtcnn_u2net.py -W %d -H %d -i '%s' -o '%s' -draw" % (w_new, h_new, src_path, tgt_path))
			'''
			if w_temp <= w and h_temp <= h:
				w_new = w_temp
				h_new = h_temp
				tgt_path = '%s/%s_width_%d_height_%d_downsize.png' % (tgt_dir, file[:-4], w_temp, h_temp)
				print(tgt_path)
				os.system("python3 smartcrop_PaddleOCR_mtcnn_u2net.py -W %d -H %d -i '%s' -o '%s' -draw" % (w_new, h_new, src_path, tgt_path))
				#tgt_path = '%s/%s_width_%d_height_%d_no_downsize.png' % (tgt_dir, file[:-4], w_temp, h_temp)
				#print(tgt_path)
				#os.system("python3 smartcrop_PaddleOCR_mtcnn_u2net.py -W %d -H %d -i '%s' -o '%s' -n -t -f -draw" % (w_new, h_new, src_path, tgt_path))
			else:
				r_new = float(w_temp) / float(h_temp)
				if r_new > r:
					w_new = w
					h_new = int(w_new / r_new)
				else:
					h_new = h
					w_new = int(h_new * r_new)
				tgt_path = '%s/%s_width_%d_height_%d.png' % (tgt_dir, file[:-4], w_temp, h_temp)
				print(tgt_path)
				os.system("python3 smartcrop_PaddleOCR_mtcnn_u2net.py -W %d -H %d -i '%s' -o '%s' -n -draw" % (w_new, h_new, src_path, tgt_path))
			'''
