from pathlib import Path
import os

bg_dir = '/export/home/projects/datasets/clean_testing_background'

#strings = ["10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW|Excludes limited time deals and clearance",
#		   "10% OFF Cork Flooring 10% OFF Cork Flooring 10% OFF Cork Flooring|Code: EXTRA10 Code: EXTRA10 Code: EXTRA10|SHOP NOW SHOP NOW SHOP NOW|excludes limited time deals and clearance excludes limited time deals and clearance excludes limited time deals and clearance",
#		   "10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW|Do NOT miss it|Big Deal and Hot Sale!|excludes limited time deals and clearance"]
#string_labels = ["header|body text|button|disclaimer / footnote",
#				 "header|body text|button|disclaimer / footnote",
#				 "header|body text|button|header|header|disclaimer / footnote"]

strings_list = []
strings_list.append("Placeholder for header|This is a placeholder for body text string|BUTTON|This is a placeholder for disclaimer or footnote text string")
strings_list.append("Input header here|Input your body text here|Design your button here|Input your disclaimer or footnote here (optional)")
strings_list.append("Header is highlighted|Body text is more narrative than header text|Button is button|A disclaimer is generally any statement intended to specify or delimit the scope of rights and obligations that may be exercised and enforced by parties in a legally recognized relationship.")
strings_list.append("Shout Out|A wall of verbal and lengthy and expatiatory words|Press here|Disclaimer is a type of text that is less important than body text")
strings_list.append("I am a HEADER!!!|I am a body text with concrete information about this banner|Click Me|I appear as disclaimer information here")
string_labels = "header|body text|button|disclaimer / footnote"

network = '/export/home/projects/webpage_generation/stylegan3_detr_genRec_uncondDis_gIoU_fixedTextEncoder_shallowTextDecoder_unifiedNoise_textNoImageCond_backgroundCond_paddingImageInput_CNN_overlapping_alignment_losses_D_LM_D_visualDecoder/training-runs/layoutganpp/ads_banner_collection_manual_3x_mask_50cls_2len_5z/00001-layoutganpp-ads_banner_collection_manual_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-007800.pkl'

for count, fname in enumerate(sorted(Path(bg_dir).glob('*.png'))):
	file_name = 'ours_' + str(fname).split('/')[-1][:-4].replace(' ', '_')
	output = 'API_generated_single_samples/clean_testing_background/val_bg_3x_mask/%s' % file_name
	#if os.path.isfile(output):
	#	continue
	cmd = "CUDA_VISIBLE_DEVICES=1 python gen_single_sample_API.py --seeds=0 \
			--network='%s' \
			--bg='%s' \
			--bg-preprocessing=256 \
			--strings='%s' \
			--string-labels='%s' \
			--outfile='%s' \
			--out-jittering-strength=0 \
			--out-postprocessing=horizontal_center_aligned" % (network, str(fname), strings_list[count%len(strings_list)], string_labels, output)
	print(cmd)
	os.system(cmd)