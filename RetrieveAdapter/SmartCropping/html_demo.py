import os
from pathlib import Path
import base64

def main():

	new_side_length = 256

	base_path = '../../datasets'
	folder_list = []
	folder_list.append('smart_cropping_testing_set_provided_by_Abigail')
	folder_list.append('smart_cropping_testing_set_provided_by_Abigail_resize_PaddleOCR_mtcnn_u2net_text_prioritized_outpaint')
	folder_list.append('smart_cropping_testing_set_provided_by_Abigail_resize_PaddleOCR_mtcnn_u2net_face_prioritized_outpaint')
	folder_list.append('smart_cropping_testing_set_provided_by_Abigail_resize_PaddleOCR_mtcnn_u2net_saliency_prioritized_outpaint')
	folder_list.append('smart_cropping_testing_set_provided_by_Abigail_resize_seam_carving_downsize')
	priority = ['None', 'text prioritized', 'face prioritized', 'other saliency prioritized', 'seam carving']
	data_list = sorted(Path('%s/%s' % (base_path, folder_list[0])).glob('*.png'))

	html = open('demo.html', 'w+')
	message = """<html><body>
			<table border="1">"""
	html.write(message)

	for data in data_list:
		data = str(data)
		for (w_temp, h_temp) in [(1400, 1050), (1600, 900), (600, 800), (1080, 1920), (1080, 1080)]:
			for count, folder in enumerate(folder_list):
				if count == 0:
					path = data
					message = '<tr>'
					caption = data[data.rfind('/')+1:-4]
				else:
					path = '%s_width_%d_height_%d.png' % (data.replace(folder_list[0], folder_list[count])[:-4], w_temp, h_temp)
					if not os.path.isfile(path):
						path = '%s width_%d_height_%d.png' % (data.replace(folder_list[0], folder_list[count])[:-4], w_temp, h_temp)
					message = ''
					caption = '%dx%d %s' % (w_temp, h_temp, priority[count])
				print(path)

				#message += """
				#	<td width="%d%%"><div align="center">
				#	<figure>
				#	<img width="%d" src="%s">
				#	<figcaption>%s</figcaption>
				#	<figure>
				#	</div></td>""" % (100/len(folder_list), new_side_length, path, caption)

				try:
					im = base64.b64encode(open(path, 'rb').read()).decode('utf-8')
					tag = '<img width="256" src="data:image/png;base64,%s">' % (im)
				except:
					tag = '<img width="%d" src="">' % new_side_length
				message += """
					<td width="%d%%"><div align="center">
					<figure>
					%s
					<figcaption>%s</figcaption>
					<figure>
					</div></td>""" % (100/len(folder_list), tag, caption)
				
				html.write(message)

	message = """</table>
		</body></html>"""
	html.write(message)		
	html.close()

if __name__ == '__main__':
    main()