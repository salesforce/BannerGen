***********************************************
* Data preparation
***********************************************
python dataset_tool.py \
--source=data/dataset/rico/raw/semantic_annotations \
--dest=data/dataset/rico/zip

python dataset_tool.py \
--source=data/dataset/enrico/raw/semantic_annotations \
--dest=data/dataset/enrico/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection/raw/json_png_pseudo_label_blue_bbox_filtered \
--dest=data/dataset/ads_banner_collection/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_improved/raw/improved_json_png_pseudo_label_blue_bbox_filtered \
--dest=data/dataset/ads_banner_collection_improved/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_improved_blur/raw/improved_json_png_pseudo_label_blue_bbox_filtered \
--dest=data/dataset/ads_banner_collection_improved_blur/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_improved_jpeg/raw/improved_json_png_pseudo_label_blue_bbox_filtered \
--dest=data/dataset/ads_banner_collection_improved_jpeg/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_improved_rec/raw/improved_json_png_pseudo_label_blue_bbox_filtered \
--dest=data/dataset/ads_banner_collection_improved_rec/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_improved_3x_mask/raw/improved_json_png_pseudo_label_blue_bbox_filtered \
--dest=data/dataset/ads_banner_collection_improved_3x_mask/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_improved_edge/raw/improved_json_png_pseudo_label_blue_bbox_filtered \
--dest=data/dataset/ads_banner_collection_improved_edge/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_manual/raw/manual_json_png_gt_label \
--dest=data/dataset/ads_banner_collection_manual/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_manual_3x_mask/raw/manual_json_png_gt_label \
--dest=data/dataset/ads_banner_collection_manual_3x_mask/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_manual_3x_mask/raw/manual_json_png_gt_label \
--dest=data/dataset/ads_banner_collection_manual_3x_mask_header_consolidated/zip

python dataset_tool.py \
--source=data/dataset/ads_banner_collection_manual_3x_mask/raw/manual_json_png_gt_label \
--dest=data/dataset/ads_banner_collection_manual_3x_mask_header_consolidated_3labels/zip

***********************************************
* Layoutganpp single GPU
***********************************************
CUDA_VISIBLE_DEVICES=0 python train.py --gpus=1 --batch=64 --workers=8 --tick=1 --snap=10 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--metrics=none \
--data=data/dataset/rico/zip/train.zip \
--outdir=training-runs/layoutganpp/rico

***********************************************
* Layoutganpp multiple GPU
***********************************************
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=16 --workers=16 --tick=1 --snap=10 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,maximum_iou50k_overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/enrico/zip/train.zip \
--outdir=training-runs/layoutganpp/enrico

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=100 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/clay/zip/train.zip \
--outdir=training-runs/layoutganpp/clay_noTextLen_textRec0.01 \
--resume=training-runs/layoutganpp/clay_noTextLen_textRec0.01/00000-layoutganpp-clay-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-001500.pkl \
--resume-kimg=1500

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=16 --workers=16 --tick=1 --snap=100 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=35.0 --alignment-weight=850.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/clay/zip/train.zip \
--outdir=training-runs/layoutganpp/clay_overlapping35_alignment850

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=400.0 --text-rec-weight=0.06 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=1024 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_noTextLen_bboxRec400_textRec0.06 \
--resume=training-runs/layoutganpp/ads_banner_collection_noTextLen_bboxRec400_textRec0.06/00000-layoutganpp-ads_banner_collection-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-000200.pkl \
--resume-kimg=200

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=1024 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=16 --workers=16 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.1 --text-len-rec-weight=10.0 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 --z-rec-weight=10.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=256 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_improved/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_improved_background_size_256 \
--resume=training-runs/layoutganpp/ads_banner_collection_improved_background_size_256/00000-layoutganpp-ads_banner_collection_improved-gpus8-batch16-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-003427.pkl \
--resume-kimg=3427

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=16 --workers=16 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=128 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_improved/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_improved_background_size_128 \
--resume=training-runs/layoutganpp/ads_banner_collection_improved_background_size_128/00000-layoutganpp-ads_banner_collection_improved-gpus8-batch16-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-011692.pkl \
--resume-kimg=11692

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=1024 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_improved_blur/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_improved_blur

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=1024 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_improved_jpeg/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_improved_jpeg \
--resume=training-runs/layoutganpp/ads_banner_collection_improved_jpeg/00000-layoutganpp-ads_banner_collection_improved_jpeg-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-004200.pkl \
--resume-kimg=4200

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=1024 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_improved_rec/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_improved_rec \
--resume=training-runs/layoutganpp/ads_banner_collection_improved_rec/00000-layoutganpp-ads_banner_collection_improved_rec-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-002800.pkl \
--resume-kimg=2800

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=1024 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_improved_3x_mask/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_improved_3x_mask \
--resume=training-runs/layoutganpp/ads_banner_collection_improved_3x_mask/00000-layoutganpp-ads_banner_collection_improved_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-005000.pkl \
--resume-kimg=5000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=1.0 --bbox-rec-weight=10.0 --text-rec-weight=0.01 --im-rec-weight=1.0 \
--overlapping-weight=7.0 --alignment-weight=17.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-layers=8 \
--background-size=1024 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_improved_edge/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_improved_edge \
--resume=training-runs/layoutganpp/ads_banner_collection_improved_edge/00000-layoutganpp-ads_banner_collection_improved_edge-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-000200.pkl \
--resume-kimg=200

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --gpus=8 --batch=8 --workers=8 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=50.0 --bbox-rec-weight=500.0 --text-rec-weight=0.1 --text-len-rec-weight=2.0 --im-rec-weight=0.5 \
--bbox-giou-weight=4.0 --overlapping-weight=7.0 --alignment-weight=17.0 --z-rec-weight=5.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-encoder-layers=12 --bert-num-decoder-layers=2 \
--background-size=256 --im-f-dim=512 \
--metrics=layout_fid50k_train,layout_fid50k_val,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_train,overlap50k_alignment50k_layoutwise_iou50k_layoutwise_docsim50k_val,fid50k_train,fid50k_val \
--data=data/dataset/ads_banner_collection_manual_3x_mask/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_manual_3x_mask_50cls_2len_5z \
--resume=training-runs/layoutganpp/ads_banner_collection_manual_3x_mask_50cls_2len_5z/00000-layoutganpp-ads_banner_collection_manual_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-004000.pkl \
--resume-kimg=4000

***********************************************
* Rendering
***********************************************

CUDA_VISIBLE_DEVICES=0 python train.py --gpus=1 --batch=1 --workers=1 --tick=1 --snap=20 \
--cfg=layoutganpp --aug=noaug \
--gamma=0.0 --pl-weight=0.0 \
--bbox-cls-weight=50.0 --bbox-rec-weight=500.0 --text-rec-weight=0.1 --text-len-rec-weight=2.0 --im-rec-weight=0.5 \
--bbox-giou-weight=4.0 --overlapping-weight=7.0 --alignment-weight=17.0 --z-rec-weight=5.0 \
--z-dim=4 --g-f-dim=256 --g-num-heads=4 --g-num-layers=8 --d-f-dim=256 --d-num-heads=4 --d-num-layers=8 \
--bert-f-dim=768 --bert-num-heads=4 --bert-num-encoder-layers=12 --bert-num-decoder-layers=2 \
--background-size=256 --im-f-dim=512 \
--metrics=rendering_val \
--data=data/dataset/ads_banner_collection_manual/zip/train.zip \
--outdir=training-runs/layoutganpp/ads_banner_collection_manual_3x_mask_50cls_2len_5z \
--resume=training-runs/layoutganpp/ads_banner_collection_manual_3x_mask_50cls_2len_5z/00001-layoutganpp-ads_banner_collection_manual_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-007800.pkl \
--resume-kimg=25000

***********************************************
* Interpolated video generation
***********************************************

CUDA_VISIBLE_DEVICES=3 python gen_video.py --seeds=0-23 --grid=6x4 \
--network=training-runs/layoutganpp/clay/00000-layoutganpp-clay-gpus8-batch16-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-002016.pkl \
--data=data/dataset/clay/zip/val.zip \
--output=interpolation_video/clay/val.mp4

CUDA_VISIBLE_DEVICES=2 python gen_video.py --seeds=0-23 --grid=6x4 \
--network=training-runs/layoutganpp/ads_banner_collection/00000-layoutganpp-ads_banner_collection-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-006000.pkl \
--data=data/dataset/ads_banner_collection/zip/train.zip \
--output=interpolation_video/ads_banner_collection/train.mp4

CUDA_VISIBLE_DEVICES=2 python gen_video.py --seeds=0-23 --grid=6x4 \
--network=training-runs/layoutganpp/ads_banner_collection_manual_3x_mask/00000-layoutganpp-ads_banner_collection_manual_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-001200.pkl \
--data=data/dataset/floors_product_samples/zip/val.zip \
--output=interpolation_video/floors_product_samples/val.mp4

CUDA_VISIBLE_DEVICES=2 python gen_video.py --seeds=0-23 --grid=6x4 \
--network=training-runs/layoutganpp/ads_banner_collection_manual_3x_mask/00000-layoutganpp-ads_banner_collection_manual_3x_mask-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-010800.pkl \
--background-size=256 \
--data=data/dataset/ads_banner_collection_manual_3x_mask/zip/val.zip \
--output=interpolation_video/ads_banner_collection_manual_3x_mask/val.mp4

***********************************************
* Ranked image generation
***********************************************
CUDA_VISIBLE_DEVICES=0 python gen_images.py --num-samples=20 --seeds=0-19 \
--network=training-runs/layoutganpp/clay/00001-layoutganpp-clay-gpus4-batch4-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-005000.pkl \
--data=data/dataset/clay/zip/train.zip \
--outdir=generated_samples/clay/train

CUDA_VISIBLE_DEVICES=0 python gen_images.py --num-samples=20 --seeds=0-19 \
--network=training-runs/layoutganpp/ads_banner_collection/00000-layoutganpp-ads_banner_collection-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-006000.pkl \
--data=data/dataset/ads_banner_collection/zip/train.zip \
--outdir=generated_samples/ads_banner_collection/train

CUDA_VISIBLE_DEVICES=0 python gen_images.py --num-samples=20 --seeds=0-19 \
--network=training-runs/layoutganpp/ads_banner_collection_improved/00002-layoutganpp-ads_banner_collection_improved-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-019200.pkl \
--data=data/dataset/ads_banner_collection_improved/zip/val.zip \
--outdir=generated_samples/ads_banner_collection_improved/val

CUDA_VISIBLE_DEVICES=0 python gen_images.py --num-samples=35 --seeds=0-19 \
--network=training-runs/layoutganpp/ads_banner_collection_improved/00002-layoutganpp-ads_banner_collection_improved-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-019200.pkl \
--data=data/dataset/floors_product_samples/zip/val.zip \
--outdir=generated_samples/floors_product_samples/val

***********************************************
* Ranked single sample generation (API)
***********************************************
CUDA_VISIBLE_DEVICES=0 python gen_single_sample_API.py --seeds=0-4 \
--network=training-runs/layoutganpp/ads_banner_collection_improved/00002-layoutganpp-ads_banner_collection_improved-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-019200.pkl \
--bg=/export/home/projects/datasets/floors_product_samples/LaMa_stringOnly_inpainted_background_images/17_mask001.png \
--strings='10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW' \
--outdir=API_generated_single_samples/floors_product_samples/val/17_mask001

CUDA_VISIBLE_DEVICES=1 python gen_single_sample_API.py --seeds=0-4 \
--network=training-runs/layoutganpp/ads_banner_collection_improved/00002-layoutganpp-ads_banner_collection_improved-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-019200.pkl \
--bg=/export/home/projects/datasets/floors_product_samples/LaMa_stringOnly_inpainted_background_images/17_mask001.png \
--strings='10% OFF Cork Flooring 10% OFF Cork Flooring 10% OFF Cork Flooring 10% OFF Cork Flooring|Code: EXTRA10 Code: EXTRA10 Code: EXTRA10 Code: EXTRA10|SHOP NOW SHOP NOW SHOP NOW SHOP NOW' \
--outdir=API_generated_single_samples/floors_product_samples/val/17_mask001_super_long_strings

CUDA_VISIBLE_DEVICES=1 python gen_single_sample_API.py --seeds=0-4 \
--network=training-runs/layoutganpp/ads_banner_collection_improved/00002-layoutganpp-ads_banner_collection_improved-gpus8-batch8-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-019200.pkl \
--bg=/export/home/projects/datasets/floors_product_samples/interpolated_LaMa_stringOnly_inpainted_background_images/17_18_alpha_1.0.png \
--strings='10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW' \
--outdir=API_generated_single_samples/floors_product_samples/val/17_18_alpha_1.0

CUDA_VISIBLE_DEVICES=1 python gen_single_sample_API.py --seeds=0-4 \
--network=training-runs/layoutganpp/ads_banner_collection_improved_background_size_256/00000-layoutganpp-ads_banner_collection_improved-gpus8-batch16-pl0.000-gamma0.000-overlapping7-alignment17/network-snapshot-005241.pkl \
--bg=/export/home/projects/datasets/Lumber_Liquidators_Images_Brand_Styles_Emails/Lumber_Liquidators_light_flooring_living_room.png \
--strings='10% OFF Cork Flooring|Code: EXTRA10|SHOP NOW' \
--outdir=API_generated_single_samples/Lumber_Liquidators_Images_Brand_Styles_Emails/val_bg_256/Lumber_Liquidators_light_flooring_living_room