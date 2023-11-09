# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import copy
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm

import legacy

from util import save_image, save_real_image_with_background

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 255).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

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

def gen_interp_video(dataset, G, mp4: str, seeds, shuffle_seed=None, w_frames=60*16, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, device=torch.device('cuda'), **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % grid_w != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // grid_w

    all_seeds = np.zeros(num_keyframes*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    rnd = np.random.RandomState(0)
    all_indices = list(range(len(dataset)))
    rnd.shuffle(all_indices)
    grid_indices = [all_indices[i % len(all_indices)] for i in range(grid_w)]
    samples, labels = zip(*[dataset[i] for i in grid_indices])
    zs = np.stack([np.random.RandomState(seed).randn(samples[0]['bboxes'].shape[0], G.z_dim) for seed in all_seeds])
    zs = np.reshape(zs, (grid_w, num_keyframes, samples[0]['bboxes'].shape[0], G.z_dim))
    sample_grid = []
    label_grid = []
    z_grid = []
    for xi in range(grid_w):
        sample = samples[xi]
        label = labels[xi]
        z = zs[xi]
        sample_col = []
        label_col = []
        z_col = []
        for yi in range(grid_h):
            if yi == 0:
                sample_col.append(sample)
                label_col.append(label)
                x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
                y = np.tile(z, [wraps * 2 + 1, 1, 1])
                interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
                z_col.append(interp)
            else:
                sample, order = shuffle_elements(sample, random_seed=yi)
                sample_col.append(sample)
                label_col.append(label)
                x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
                y = np.tile(np.stack([z[:,i] for i in order], axis=1), [wraps * 2 + 1, 1, 1])
                interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
                z_col.append(interp)
        sample_grid.append(sample_col)
        label_grid.append(label_col)
        z_grid.append(z_col)
    sample_grid = list(map(list, zip(*(sample_grid))))
    label_grid = list(map(list, zip(*(label_grid))))
    z_grid = list(map(list, zip(*(z_grid))))

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=30, codec='libx264', **video_kwargs)
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                sample = sample_grid[yi][xi]
                W_page = [sample['W_page']]
                H_page = [sample['H_page']]
                bbox_real = torch.from_numpy(sample['bboxes']).to(device).to(torch.float32).unsqueeze(0)
                bbox_class = torch.from_numpy(sample['labels']).to(device).to(torch.int64).unsqueeze(0)
                bbox_text = [sample['texts']]
                bbox_patch = torch.from_numpy(sample['patches']).to(device).to(torch.float32).unsqueeze(0)
                bbox_patch_orig = torch.from_numpy(sample['patches_orig']).to(device).to(torch.float32).unsqueeze(0)
                mask = torch.from_numpy(sample['mask']).to(device).to(torch.bool).unsqueeze(0)
                padding_mask = ~mask
                background = torch.from_numpy(sample['background']).to(device).to(torch.float32).unsqueeze(0)
                background_orig = torch.from_numpy(sample['background_orig']).to(device).to(torch.float32).unsqueeze(0)
                c = torch.from_numpy(label_grid[yi][xi]).to(device).unsqueeze(0)
                interp = z_grid[yi][xi]
                z = torch.from_numpy(interp(frame_idx / w_frames)).to(device).to(torch.float32).unsqueeze(0)
                bbox_fake = G(z=z, bbox_class=bbox_class, bbox_real=bbox_real, bbox_text=bbox_text, bbox_patch=bbox_patch, padding_mask=padding_mask, background=background, c=c)
                img = save_real_image_with_background(bbox_fake, bbox_real, bbox_patch_orig, ~padding_mask, background_orig,
                                                 out_path=None, W_page=W_page, H_page=H_page, size_canvas=256, return_instead_of_save=True)[0]
                layout = save_image(bbox_fake, bbox_class, ~padding_mask, dataset.colors, 
                                                 out_path=None, W_page=W_page, H_page=H_page, size_canvas=256, return_instead_of_save=True)[0]
                imgs.append(torch.cat([img, layout], dim=2))
        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))
    video_out.close()

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
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

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--data', type=str, metavar='[ZIP]', help='Training/validation data', required=True)
@click.option('--background-size', type=int, help='Background image size', default=1024)
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds', required=True)
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(1,1))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--output', help='Output .mp4 filename', type=str, required=True, metavar='FILE')
def generate_images(
    data: str,
    background_size: int,
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    output: str
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """
    print('Loading data from "%s"...' % data)
    dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset_layoutganpp.LayoutDataset', path=data, use_labels=False, max_size=None, xflip=False, background_size=background_size, random_seed=0)
    dataset = dnnlib.util.construct_class_by_name(**dataset_kwargs)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(os.path.dirname(output), exist_ok=True)

    gen_interp_video(dataset=dataset, G=G, mp4=output, bitrate='12M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames, seeds=seeds, shuffle_seed=shuffle_seed)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
