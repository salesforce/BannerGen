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
import math
from PIL import Image, ImageDraw, ImageFilter
import seaborn as sns
import torch

import legacy

from metrics.metric_layoutnet import compute_overlap, compute_alignment
from util import convert_xywh_to_ltrb

from bs4 import BeautifulSoup
import pdb
from selenium import webdriver
from io import BytesIO
import re

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

def jitter(bbox_fake, out_jittering_strength, seed): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    perturb = torch.from_numpy(np.random.RandomState(seed).uniform(low=math.log(1.0-out_jittering_strength), high=math.log(1.0+out_jittering_strength), size=bbox_fake.shape)).to(bbox_fake.device).to(torch.float32)
    bbox_fake *= perturb.exp()
    return bbox_fake

#----------------------------------------------------------------------------

def horizontal_center_aligned(bbox_fake, mask): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    xc_mean = bbox_fake[mask][:,0].mean()
    bbox_fake[:,:,0] = xc_mean
    return bbox_fake

def horizontal_left_aligned(bbox_fake, mask): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    num = bbox_fake[mask].shape[0]
    x1_sum = 0.0
    for i in range(num):
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox_fake[0,i])
        x1_sum += x1
    x1_mean = x1_sum / float(num)
    for i in range(num):
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox_fake[0,i])
        bbox_fake[0,i,0] -= x1 - x1_mean
    return bbox_fake

def de_overlap(bbox_fake, mask): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    num = bbox_fake[mask].shape[0]
    for i in range(num):
        xc1, yc1, w1, h1 = bbox_fake[0,i]
        for j in range(num):
            if i != j:
                xc2, yc2, w2, h2 = bbox_fake[0,j]
                if abs(yc2 - yc1) < h1/2 + h2/2:
                    diff = h1/2 + h2/2 - abs(yc2 - yc1)
                    if yc1 < yc2:
                        bbox_fake[0,i,1] -= diff/2
                        bbox_fake[0,j,1] += diff/2
                    else:
                        bbox_fake[0,i,1] += diff/2
                        bbox_fake[0,j,1] -= diff/2
    for i in range(num):
        xc1, yc1, w1, h1 = bbox_fake[0,i]
        for j in range(num):
            if i != j:
                xc2, yc2, w2, h2 = bbox_fake[0,j]
                if abs(yc2 - yc1) < h1/2 + h2/2:
                    diff = h1/2 + h2/2 - abs(yc2 - yc1)
                    bbox_fake[0,i,3] -= diff/2
                    bbox_fake[0,j,3] -= diff/2
    return bbox_fake

#----------------------------------------------------------------------------

def save_bboxes_with_background(boxes, masks, labels, background_orig, path):
    colors = sns.color_palette('husl', n_colors=13)
    colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]
    background_orig_temp = background_orig.copy()
    W_page, H_page = background_orig_temp.size
    draw = ImageDraw.Draw(background_orig_temp, 'RGBA')
    boxes = boxes[masks]
    labels = labels[masks]
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)), key=lambda i: area[i], reverse=True)
    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * W_page, x2 * W_page
        y1, y2 = y1 * H_page, y2 * H_page
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
    background_orig_temp.save(path, format='png', compress_level=0, optimize=False)

#----------------------------------------------------------------------------

def get_adaptive_font_color(bg):
    if isinstance(bg, str):
        start = bg.index('(')
        end = bg.index(')')
        bg = bg[start+1:end].split(', ')
        bg = [int(bg_i) for bg_i in bg]
        assert len(bg) == 4
        out = str((0, 0, 0, 255)) if sum(bg[:3]) > 255 * 3 / 2.0 else str((255, 255, 255, 255))
    else:
        img = np.array(bg)
        clr = []
        for ch in range(3):
            clr.append(np.median(img[:, :, ch]))
        out = str((0, 0, 0, 255)) if sum(clr) > 255 * 3 / 1.5 else str((255, 255, 255, 255))
    return out

#----------------------------------------------------------------------------

def get_adaptive_font_size(w_tbox, h_tbox, H_page, text, font2height=0.038422, font_aspect_ratio=0.52,
                           min_font_size=9):
    font_size = int(H_page*font2height)
    num_word = len(text)
    num_line = num_word * font_size * font_aspect_ratio / w_tbox
    if num_line < 1 or num_line * font_size < h_tbox:
        return str(font_size)
    else:  # num_word * font_size * font_aspect_ratio * font_size < w_tbox * h_tbox
        return str(max(min_font_size, int((w_tbox * h_tbox / num_word / font_aspect_ratio) ** .5)))

#----------------------------------------------------------------------------

HTML_TEMP = \
    """
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
    .container {
      position: relative;
      text-align: center;
      color: white;
    }
    .body {
      margin: 0;
      padding: 0;
    }
    </style>
    </head>
    <body class="body">
    <div class="container">
      <img src="" alt="" style="width:auto;position:absolute;top:0px;left:0px;">
    </div>
    </body>
    </html> 
    """

TEXT_CSS_TEMP = 'vertical-align:middle;background:rgba(0, 0, 0, 0);border-width:1px;border-style:solid;' \
                'border-color:black;text-align:center;position:absolute;word-wrap:break-word;'

def visualize_banner(boxes, masks, labels, text, background_orig, path,
                     browser, font_family=None, font_color=None, font_format=None, button_color=None, save_format='html+image'):
    font_family = 'font-family:Arial;' if not font_family else 'font-family:' + font_family + ';'
    # check if valid color hex code if user provides one
    if font_color:
        if not re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', font_color):
            print('Invalid font color.')
            font_color = None  # fall back to adaptive color
        else:
            font_color = 'color:' + font_color + ';'

    soup = BeautifulSoup(HTML_TEMP, "html.parser")
    # insert img src div
    img = soup.findAll('img')
    img[0]['src'] = os.path.basename(path)

    W_page, H_page = background_orig.size
    boxes = boxes[masks]
    labels = labels[masks]
    for i in range(boxes.shape[0]):
        if font_format:
            font_format_cur = 'font-style:' + font_format + ';'
        else:
            if labels[i] == 3 or labels[i] == 4: # 'body text' or 'disclaimer':
                font_format_cur = ''
            else:
                font_format_cur = "font-weight:bold;"

        x1, y1, x2, y2 = convert_xywh_to_ltrb(boxes[i])
        x1, x2 = int(x1 * W_page), int(x2 * W_page)
        y1, y2 = int(y1 * H_page), int(y2 * H_page)
        h_tbox, w_tbox = int(y2 - y1 + 1), int(x2 - x1 + 1)

        if font_color:
            font_color_cur = font_color
        else:
            if labels[i] == 5: # 'button':
                if button_color:
                    button_color_cur = 'background-color:' + button_color + ';'
                else:
                    button_color_cur = 'background-color:' + get_adaptive_font_color(background_orig.crop([x1, y1, x2, y2])) + ';'
                font_color_cur = 'color:' + get_adaptive_font_color(button_color_cur) + ';'
            else:
                button_color_cur = ''
                font_color_cur = 'color:' + get_adaptive_font_color(background_orig.crop([x1, y1, x2, y2])) + ';'

        font_size = 'font-size:' + get_adaptive_font_size(w_tbox, h_tbox, H_page, text[i]) + 'px;'

        tbox_style = TEXT_CSS_TEMP
        tbox_style = tbox_style + font_format_cur + font_color_cur + font_size + font_family + button_color_cur
        tbox_style += 'width:' + str(w_tbox) + 'px;'
        tbox_style += 'height:' + str(h_tbox) + 'px;'
        tbox_style += 'top:' + str(y1) + 'px;'
        tbox_style += 'left:' + str(x1) + 'px;'
        print(tbox_style)
        tbox_attr = {'style': tbox_style}
        new_div = soup.new_tag("div", **tbox_attr)
        new_div.string = text[i]
        soup.html.body.div.append(new_div)

    path_html = path.replace('.png', '.html')
    if 'html' in save_format:
        with open(path_html, "w") as f:
            f.write(str(soup))
        background_orig.save(path, format='png', compress_level=0, optimize=False)

    if 'image' in save_format:
        browser.get("file:///" + os.getcwd() + "/" + path_html)
        png = browser.get_screenshot_as_png()
        screenshot = Image.open(BytesIO(png))
        screenshot = screenshot.crop([0, 0, W_page, H_page])
        screenshot.save(path.replace('.png', '_vis.png'))

#----------------------------------------------------------------------------

label_list = [
        'header',
        'pre-header',
        'post-header',
        'body text',
        'disclaimer',
        'button',
        'callout',
        'logo'
        ]
label2index = dict()
for idx, label in enumerate(label_list):
    label2index[label] = idx

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--bg', type=str, help='Path of a background image', required=True)
@click.option('--bg-preprocessing', 'bg_preprocessing', help='Postprocess the background image', type=click.Choice(['256', '128', 'blur', 'jpeg', 'rec', '3x_mask', 'edge', 'none']), default='none', show_default=True)
@click.option('--strings', type=str, help="Strings to be printed on the banner. Multiple strings are separated by '|'", required=True)
@click.option('--string-labels', 'string_labels', type=str, help="String labels. Multiple labels are separated by '|'", required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--outfile', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--out-jittering-strength', 'out_jittering_strength', help='Randomly jitter the output bounding box parameters with a certain strength', type=click.FloatRange(min=0.0, max=1.0), default=0.0, show_default=True)
@click.option('--out-postprocessing', 'out_postprocessing', help='Postprocess the output layout', type=click.Choice(['horizontal_center_aligned', 'horizontal_left_aligned', 'none']), default='none', show_default=True)
def generate_images(
    network_pkl: str,
    bg: str,
    bg_preprocessing: str,
    strings: str,
    string_labels: str,
    seeds: List[int],
    outfile: str,
    out_jittering_strength: bool,
    out_postprocessing: str,
):

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    print('Loading background image from "%s"...' % bg)
    background_orig = Image.open(bg).convert('RGB')
    if bg_preprocessing == '256':
        background = np.array(background_orig.resize((256, 256), Image.ANTIALIAS))
    elif bg_preprocessing == '128':
        background = np.array(background_orig.resize((128, 128), Image.ANTIALIAS))
    elif bg_preprocessing == 'blur':
        background = background_orig.filter(ImageFilter.GaussianBlur(radius=3))
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    elif bg_preprocessing == 'jpeg':
        idx = bg.rfind('/')
        bg_new = bg[:idx] + '_jpeg' + bg[idx:].replace('.png', '.jpg')
        background = Image.open(bg_new).convert('RGB')
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    elif bg_preprocessing == 'rec':
        idx = bg.rfind('/')
        bg_new = bg[:idx] + '_rec' + bg[idx:]
        background = Image.open(bg_new).convert('RGB')
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    elif bg_preprocessing == 'edge':
        background = background_orig.convert('L').filter(ImageFilter.FIND_EDGES).convert('RGB')
        background = np.array(background.resize((1024, 1024), Image.ANTIALIAS))
    else:
        background = np.array(background_orig.resize((1024, 1024), Image.ANTIALIAS))

    if background.ndim == 2:
        background = np.dstack((background, background, background))
    background = background[:,:,:3]
    rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
    rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))
    background = (background.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
    background = background.transpose(2, 0, 1)
    background = torch.from_numpy(background).to(device).to(torch.float32).unsqueeze(0)
    bbox_text = strings.split('|')
    bbox_label = string_labels.split('|')
    bbox_label = [label2index[label] for label in bbox_label]

    print('Loading layout bboxes')
    bbox_fake_list = []
    mask_list = []
    bbox_class_list = []
    bbox_text_list = []
    overlap = []
    alignment = []
    mask = torch.from_numpy(np.array([1] * len(bbox_text) + [0] * (9-len(bbox_text)))).to(device).to(torch.bool).unsqueeze(0)
    bbox_class = torch.from_numpy(np.array(bbox_label + [0] * (9-len(bbox_label)))).to(device).to(torch.int64).unsqueeze(0)
    bbox_patch_dummy = torch.zeros((1, 9, 3, 256, 256)).to(device).to(torch.float32)
    
    z = torch.from_numpy(np.random.RandomState(0).randn(1, 9, G.z_dim)).to(device).to(torch.float32)
    order = list(range(len(bbox_text)))
    #np.random.RandomState(0).shuffle(order)
    bbox_text_temp = [bbox_text[i] for i in order]
    bbox_text_temp += [''] * (9-len(bbox_text))
    bbox_text_temp = [bbox_text_temp]
    bbox_fake = G(z=z, bbox_class=bbox_class, bbox_real=None, bbox_text=bbox_text_temp, bbox_patch=bbox_patch_dummy, padding_mask=~mask, background=background, c=None)
    if out_jittering_strength > 0.0:
        bbox_fake = jitter(bbox_fake, out_jittering_strength, seed)
    if out_postprocessing == 'horizontal_center_aligned':
        bbox_fake = horizontal_center_aligned(bbox_fake, mask)
        bbox_fake = de_overlap(bbox_fake, mask)
    elif out_postprocessing == 'horizontal_left_aligned':
        bbox_fake = horizontal_left_aligned(bbox_fake, mask)
        bbox_fake = de_overlap(bbox_fake, mask)
    bbox_fake_list.append(bbox_fake[0])
    mask_list.append(mask[0])
    bbox_class_list.append(bbox_class[0])
    bbox_text_list.append(bbox_text_temp[0])
    overlap.append(compute_overlap(bbox_fake, mask).cpu().numpy()[0])
    alignment.append(compute_alignment(bbox_fake, mask).cpu().numpy()[0])

    ###################################
    # Save rendered samples
    ###################################

    # initialize Chrome based web driver for html screenshot
    options = webdriver.ChromeOptions()
    options.add_argument('no-sandbox')
    options.add_argument('headless')
    browser = webdriver.Chrome(executable_path='/usr/bin/chromedriver', options=options)
    browser.set_window_size(4096, 4096)

    subdir = outfile[:outfile.rfind('/')]
    os.makedirs(subdir, exist_ok=True)
    order = np.argsort(np.array(overlap))
    for j, idx in enumerate(order):
        #save_bboxes_with_background(bbox_fake_list[idx], mask_list[idx], bbox_class_list[idx], background_orig, outfile)
        visualize_banner(bbox_fake_list[idx], mask_list[idx], bbox_class_list[idx], bbox_text_list[idx], background_orig, outfile,
                         browser, save_format='html+image')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
