# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

from metrics.metric_layoutnet import compute_overlap, compute_alignment, compute_iou_for_layout, compute_docsim_for_layout
from util import save_image, save_real_image_with_background

import time

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def shuffle_elements(sample, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    mask = sample['mask']
    num_elements = mask.shape[0]
    num_active = int(np.sum(mask))
    order = list(range(num_active))
    rnd.shuffle(order)
    order += list(range(num_active, num_elements))
    sample['bboxes'] = np.stack([sample['bboxes'][i] for i in order])
    sample['labels'] = np.stack([sample['labels'][i] for i in order])
    sample['texts'] = [sample['texts'][i] for i in order]
    sample['patches'] = np.stack([sample['patches'][i] for i in order])
    sample['patches_orig'] = np.stack([sample['patches_orig'][i] for i in order])
    sample['patch_masks'] = np.stack([sample['patch_masks'][i] for i in order])
    return sample, order

#----------------------------------------------------------------------------

@click.command()
@click.option('--data', type=str, metavar='[ZIP]', help='Training/validation data', required=True)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--num-samples', type=int, help='Number of samples to generate', default=100)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    data: str,
    network_pkl: str,
    seeds: List[int],
    num_samples: str,
    outdir: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """
    print('Loading data from "%s"...' % data)
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_layoutganpp.LayoutDataset', path=data, use_labels=False, max_size=None, xflip=False, random_seed=0)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        time_tick = time.time()
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
        print('Network loading time:', time.time()-time_tick)

    # Generate images.
    rnd = np.random.RandomState(0)
    all_indices = list(range(len(dataset)))
    rnd.shuffle(all_indices)
    indices = [all_indices[i % len(all_indices)] for i in range(num_samples)]

    network_inference_time = []
    evaluation_time = []
    rendering_saving_time = []
    for count, i in enumerate(indices):
        sample, label = dataset[i]
        W_page = [sample['W_page']]
        H_page = [sample['H_page']]
        c = torch.from_numpy(label).to(device).unsqueeze(0)
        bbox_real_list = []
        bbox_class_list = []
        bbox_patch_orig_list = []
        padding_mask_list = []
        background_list = []
        bbox_fake_list = []
        layoutwise_iou = []
        layoutwise_docsim = []
        overlap = []
        alignment = []
        for seed in seeds:
            time_tick = time.time()
            z = torch.from_numpy(np.random.RandomState(seed).randn(sample['bboxes'].shape[0], G.z_dim)).to(device).to(torch.float32).unsqueeze(0)
            sample, _ = shuffle_elements(sample, seed)
            bbox_real = torch.from_numpy(sample['bboxes']).to(device).to(torch.float32).unsqueeze(0)
            bbox_real_list.append(bbox_real)
            bbox_class = torch.from_numpy(sample['labels']).to(device).to(torch.int64).unsqueeze(0)
            bbox_class_list.append(bbox_class)
            bbox_text = [sample['texts']]
            bbox_patch = torch.from_numpy(sample['patches']).to(device).to(torch.float32).unsqueeze(0)
            bbox_patch_orig = torch.from_numpy(sample['patches_orig']).to(device).to(torch.float32).unsqueeze(0)
            bbox_patch_orig_list.append(bbox_patch_orig)
            mask = torch.from_numpy(sample['mask']).to(device).to(torch.bool).unsqueeze(0)
            padding_mask = ~mask
            padding_mask_list.append(padding_mask)
            background = torch.from_numpy(sample['background']).to(device).to(torch.float32).unsqueeze(0)
            background_list.append(background)
            bbox_fake = G(z=z, bbox_class=bbox_class, bbox_real=bbox_real, bbox_text=bbox_text, bbox_patch=bbox_patch, padding_mask=padding_mask, background=background, c=c)
            time_tick_2 = time.time()
            network_inference_time.append(time_tick_2-time_tick)
            bbox_fake_list.append(bbox_fake)
            overlap.append(compute_overlap(bbox_fake, mask).cpu().numpy()[0])
            alignment.append(compute_alignment(bbox_fake, mask).cpu().numpy()[0])
            bbox_fake = bbox_fake[0].cpu().numpy()
            b_real = sample['bboxes'][sample['mask']]
            b_fake = bbox_fake[sample['mask']]
            l = sample['labels'][sample['mask']]
            layoutwise_iou.append(compute_iou_for_layout((b_real, l), (b_fake, l)))
            layoutwise_docsim.append(compute_docsim_for_layout((b_real, l), (b_fake, l)))
            evaluation_time.append(time.time()-time_tick_2)

        ###################################
        # Save random sample variants according to layoutwise_iou
        ###################################
        print('%d: %d: layoutwise_iou' % (len(indices), count))
        order = np.argsort(np.array(layoutwise_iou))[::-1]
        subdir = '%s/%04d/layoutwise_iou' % (outdir, count)
        os.makedirs(subdir, exist_ok=True)
        save_image(bbox_real_list[0], bbox_class_list[0], ~(padding_mask_list[0]), dataset.colors,
                   '%s/layout_0real.png' % subdir, W_page=W_page, H_page=H_page)
        save_real_image_with_background(bbox_real_list[0], bbox_real_list[0], bbox_patch_orig_list[0], ~(padding_mask_list[0]), background_list[0],
                                        '%s/image_0real.png' % subdir, W_page=W_page, H_page=H_page)
        for j, idx in enumerate(order):
            save_image(bbox_fake_list[idx], bbox_class_list[idx], ~(padding_mask_list[idx]), dataset.colors,
                       '%s/layout_fake_%04d_%f_%04d.png' % (subdir, j, layoutwise_iou[idx], idx), W_page=W_page, H_page=H_page)
            time_tick = time.time()
            save_real_image_with_background(bbox_fake_list[idx], bbox_real_list[idx], bbox_patch_orig_list[idx], ~(padding_mask_list[idx]), background_list[idx],
                                            '%s/image_fake_%04d_%f_%04d.png' % (subdir, j, layoutwise_iou[idx], idx), W_page=W_page, H_page=H_page)
            rendering_saving_time.append(time.time()-time_tick)

        ###################################
        # Save random sample variants according to layoutwise_docsim
        ###################################
        print('%d: %d: layoutwise_docsim' % (len(indices), count))
        order = np.argsort(np.array(layoutwise_docsim))[::-1]
        subdir = '%s/%04d/layoutwise_docsim' % (outdir, count)
        os.makedirs(subdir, exist_ok=True)
        save_image(bbox_real_list[0], bbox_class_list[0], ~(padding_mask_list[0]), dataset.colors,
                   '%s/layout_0real.png' % subdir, W_page=W_page, H_page=H_page)
        save_real_image_with_background(bbox_real_list[0], bbox_real_list[0], bbox_patch_orig_list[0], ~(padding_mask_list[0]), background_list[0],
                                        '%s/image_0real.png' % subdir, W_page=W_page, H_page=H_page)
        for j, idx in enumerate(order):
            save_image(bbox_fake_list[idx], bbox_class_list[idx], ~(padding_mask_list[idx]), dataset.colors,
                       '%s/layout_fake_%04d_%f_%04d.png' % (subdir, j, layoutwise_docsim[idx], idx), W_page=W_page, H_page=H_page)
            save_real_image_with_background(bbox_fake_list[idx], bbox_real_list[idx], bbox_patch_orig_list[idx], ~(padding_mask_list[idx]), background_list[idx],
                                            '%s/image_fake_%04d_%f_%04d.png' % (subdir, j, layoutwise_docsim[idx], idx), W_page=W_page, H_page=H_page)

        ###################################
        # Save random sample variants according to overlap
        ###################################
        print('%d: %d: overlap' % (len(indices), count))
        order = np.argsort(np.array(overlap))
        subdir = '%s/%04d/overlap' % (outdir, count)
        os.makedirs(subdir, exist_ok=True)
        save_image(bbox_real_list[0], bbox_class_list[0], ~(padding_mask_list[0]), dataset.colors,
                   '%s/layout_0real.png' % subdir, W_page=W_page, H_page=H_page)
        save_real_image_with_background(bbox_real_list[0], bbox_real_list[0], bbox_patch_orig_list[0], ~(padding_mask_list[0]), background_list[0],
                                        '%s/image_0real.png' % subdir, W_page=W_page, H_page=H_page)
        for j, idx in enumerate(order):
            save_image(bbox_fake_list[idx], bbox_class_list[idx], ~(padding_mask_list[idx]), dataset.colors,
                       '%s/layout_fake_%04d_%f_%04d.png' % (subdir, j, overlap[idx], idx), W_page=W_page, H_page=H_page)
            save_real_image_with_background(bbox_fake_list[idx], bbox_real_list[idx], bbox_patch_orig_list[idx], ~(padding_mask_list[idx]), background_list[idx],
                                            '%s/image_fake_%04d_%f_%04d.png' % (subdir, j, overlap[idx], idx), W_page=W_page, H_page=H_page)

        ###################################
        # Save random sample variants according to alignment
        ###################################
        print('%d: %d: alignment' % (len(indices), count))
        order = np.argsort(np.array(alignment))
        subdir = '%s/%04d/alignment' % (outdir, count)
        os.makedirs(subdir, exist_ok=True)
        save_image(bbox_real_list[0], bbox_class_list[0], ~(padding_mask_list[0]), dataset.colors,
                   '%s/layout_0real.png' % subdir, W_page=W_page, H_page=H_page)
        save_real_image_with_background(bbox_real_list[0], bbox_real_list[0], bbox_patch_orig_list[0], ~(padding_mask_list[0]), background_list[0],
                                        '%s/image_0real.png' % subdir, W_page=W_page, H_page=H_page)
        for j, idx in enumerate(order):
            save_image(bbox_fake_list[idx], bbox_class_list[idx], ~(padding_mask_list[idx]), dataset.colors,
                       '%s/layout_fake_%04d_%f_%04d.png' % (subdir, j, alignment[idx], idx), W_page=W_page, H_page=H_page)
            save_real_image_with_background(bbox_fake_list[idx], bbox_real_list[idx], bbox_patch_orig_list[idx], ~(padding_mask_list[idx]), background_list[idx],
                                            '%s/image_fake_%04d_%f_%04d.png' % (subdir, j, alignment[idx], idx), W_page=W_page, H_page=H_page)

    print('Network inference time:', np.mean(np.array(network_inference_time)))
    print('Evaluation time:', np.mean(np.array(evaluation_time)))
    print('Rendering and saving time:', np.mean(np.array(rendering_saving_time)))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
