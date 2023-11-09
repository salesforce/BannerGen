from pathlib import Path
import os

bg_dir = '/export/home/projects/datasets/clean_testing_background'

#strings = ["10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW|Excludes limited time deals and clearance",
#		   "10% OFF Cork Flooring 10% OFF Cork Flooring 10% OFF Cork Flooring|Code: EXTRA10 Code: EXTRA10 Code: EXTRA10|SHOP NOW SHOP NOW SHOP NOW|excludes limited time deals and clearance excludes limited time deals and clearance excludes limited time deals and clearance",
#		   "10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW|Do NOT miss it|Big Deal and Hot Sale!|excludes limited time deals and clearance"]
#string_labels = ["header|body text|button|disclaimer / footnote",
#				 "header|body text|button|disclaimer / footnote",
#				 "header|body text|button|header|header|disclaimer / footnote"]

strings = ["This is a header text string|This is a long long long long long long body text string|BUTTON|This is a disclaimer or footnote text string"]
string_labels = ["header|body text|button|disclaimer"]

network = '/export/home/projects/webpage_generation/stylegan3_detr_genRec_uncondDis_gIoU_fixedTextEncoder_shallowTextDecoder_unifiedNoise_textNoImageCond_backgroundCond_paddingImageInput_CNN_overlapping_alignment_losses_D_LM_D_visualDecoder/training-runs/layoutganpp/ads_banner_collection_manual_3x_mask_50cls_2len_5z/00001-layoutganpp-ads_banner_collection_manual_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-007800.pkl'

for count, fname in enumerate(sorted(Path(bg_dir).glob('*.png'))):
	#if '139410' not in str(fname):
	#	continue
	file_name = str(fname).split('/')[-1][:-4]
	for count2, string in enumerate(strings):
		#if count2 == 0:
		#	string_type = 'regular'
		#elif count2 == 1:
		#	string_type = 'long'
		#elif count2 == 2:
		#	string_type = 'many'
		output = 'simplified_API_generated_single_samples/clean_testing_background/val_bg_256_3x_mask/%s.png' % file_name
		#if os.path.isfile(output):
		#	continue
		cmd = "CUDA_VISIBLE_DEVICES=1 python simplified_gen_single_sample_API.py --seeds=0-4 \
				--network='%s' \
				--bg='%s' \
				--bg-preprocessing=256 \
				--strings='%s' \
				--string-labels='%s' \
				--outfile='%s' \
				--out-jittering-strength=0 \
				--out-postprocessing=horizontal_center_aligned" % (network, str(fname), string, string_labels[count2], output)
		print(cmd)
		os.system(cmd)