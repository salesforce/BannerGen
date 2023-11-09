# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils_layout

from metrics.metric_layoutnet import compute_iou_for_layout, compute_docsim_for_layout

#----------------------------------------------------------------------------

def compute_overlap_alignment_laywise_IoU_layerwise_DocSim(opts, max_real, num_gen):
    stats_bbox_real, stats_bbox_fake, stats_bbox_class, stats_mask, stats_overlap, stats_alignment = metric_utils_layout.compute_maxIoU_overlap_alignment_wrapper(opts=opts, rel_lo=0, rel_hi=1, max_items=max_real)
    bbox_real = stats_bbox_real.get_all().astype(np.float32)
    bbox_fake = stats_bbox_fake.get_all().astype(np.float32)
    bbox_class = stats_bbox_class.get_all().astype(np.int64)
    mask = stats_mask.get_all().astype(np.bool)
    overlap = stats_overlap.get_all().astype(np.float32)
    alignment = stats_alignment.get_all().astype(np.float32)

    if opts.rank != 0:
        return float('nan'), float('nan'), float('nan'), float('nan')

    layouts_real = []
    layouts_fake = []
    layoutwise_iou = []
    layoutwise_docsim = []
    for j in range(bbox_real.shape[0]):
        _mask = mask[j]
        b_real = bbox_real[j][_mask]
        b_fake = bbox_fake[j][_mask]
        l = bbox_class[j][_mask]
        layouts_real.append((b_real, l))
        layouts_fake.append((b_fake, l))
        layoutwise_iou.append(compute_iou_for_layout((b_real, l), (b_fake, l)))
        layoutwise_docsim.append(compute_docsim_for_layout((b_real, l), (b_fake, l)))
    return float(np.mean(overlap)), float(np.mean(alignment)), float(np.mean(np.array(layoutwise_iou))), float(np.mean(np.array(layoutwise_docsim)))

#----------------------------------------------------------------------------
