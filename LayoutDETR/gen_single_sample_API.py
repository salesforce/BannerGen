"""
Copyright (c) 2023 Salesforce, Inc.

All rights reserved.

SPDX-License-Identifier: Apache License 2.0

For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/

By Ning Yu, ning.yu@salesforce.com and Chia-Chih Chen, chiachih.chen@salesforce.com

Modified from StyleGAN3 repo: https://github.com/NVlabs/stylegan3

Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""

"""Generate images using pretrained network pickle."""


import os
from flask import jsonify
from typing import List, Optional, Tuple, Union, Dict
import uuid
import click
import numpy as np
from PIL import Image, ImageDraw
import seaborn as sns
import torch
import LayoutDETR.legacy as legacy

from utils.util import convert_xywh_to_ltrb, convert_ltrb_to_xywh, convert_xywh_to_ltrb_pix, convert_ltrb_pix_to_xywh
from utils.util import safeMakeDirs
from bs4 import BeautifulSoup
from io import BytesIO
import re
import sys
import argparse
import json
import math
from selenium import webdriver
from selenium.webdriver import Chrome
from LayoutDETR.configs.banner_config import BannerConfig
from LayoutDETR.configs.banner_config import RendererConfig
import LayoutDETR.dnnlib as dnnlib
import base64


# Front/backend naming convention conversion
FRONT2BACKEND_NAME_MAP = {'note': 'disclaimer / footnote'}
BACK2FRONTEND_NAME_MAP = {y: x for x, y in FRONT2BACKEND_NAME_MAP.items()}


def load_model(model_path):
    print('Loading networks from "%s"...' % model_path)
    device = torch.device('cuda')
    with dnnlib.util.open_url(model_path) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    return G

#----------------------------------------------------------------------------

def compute_bounds(bbox_fake, mask, W, H):
    num = bbox_fake[mask].shape[0]
    xc_median = bbox_fake[mask][:, 0].median()
    bbox_xyxy = np.zeros((num, 4))  # x1, y1, x2, y2
    sum_y = 0
    max_y = -1
    min_y = H + 1
    for i in range(num):
        bbox_xyxy[i, :] = convert_xywh_to_ltrb_pix(bbox_fake[0, i], W, H)
        sum_y += bbox_xyxy[i, 1] + bbox_xyxy[i, 3]
        max_y = max(max_y, bbox_xyxy[i, 3])
        min_y = min(min_y, bbox_xyxy[i, 1])

    is_upper = False
    if sum_y/num/2 > H/2:  # lower
        ub = max(8, min_y)
        lb = H - 8
    else:
        ub = 8
        lb = min(H-8, max_y)
        is_upper = True

    return int(lb), int(ub), is_upper, xc_median

#----------------------------------------------------------------------------

def vertical_respacing(bbox_fake, mask, W, H, lb, ub, is_upper):
    """
     step 1: check if upper or lower bound and what's the bound value
     step 2.1: if upper bound, move top tbox below the line
     step 2.2: if lower bound, move bottom tbox above the line
     step 3: use only the average of the valid spacing to re-space (and > 16 pix)
     step 4: if not enough space, use the min 16px, if still not enough, shrink the height of tboxes by proportion
     step 5: trim left and right with 8 pix as margin
    """
    num = bbox_fake[mask].shape[0]
    bbox_xyxy = np.zeros((num, 4))  # x1, y1, x2, y2
    for i in range(num):
        bbox_xyxy[i, :] = convert_xywh_to_ltrb_pix(bbox_fake[0, i, :], W, H)

    y_max = min(H - BannerConfig.MIN_BOX_IMG_MARGIN, np.max(bbox_xyxy[:, 3]))
    y_min = max(BannerConfig.MIN_BOX_IMG_MARGIN, np.min(bbox_xyxy[:, 1]))
    h_tbox_arr = bbox_xyxy[:, 3] - bbox_xyxy[:, 1]
    yc_tbox_arr = bbox_xyxy[:, 3] + bbox_xyxy[:, 1]
    id_bbox_sorted = np.argsort(yc_tbox_arr)
    id_bbox_sorted_rev = np.argsort(id_bbox_sorted)
    h_sum = np.sum(h_tbox_arr)
    bbox_xyxy = bbox_xyxy[id_bbox_sorted, :]
    h_tbox_arr = h_tbox_arr[id_bbox_sorted]

    # compute the original average spacing
    space_sum = 0
    space_cnt = 0
    for i in range(num -1):
        space = bbox_xyxy[i+1, 1] - bbox_xyxy[i, 3]
        if space > 0:
            space_sum += space
            space_cnt += 1

    space_avg = space_sum/space_cnt if space_cnt > 0 else BannerConfig.MIN_BOX_BOX_MARGIN
    space_avg = BannerConfig.MIN_BOX_BOX_MARGIN if space_avg < BannerConfig.MIN_BOX_BOX_MARGIN else space_avg

    # make sure bbox fit into the (lb, ub) range
    h_diff = h_sum + space_avg * (num - 1) - (lb - ub)
    is_h_adjusted = False
    if h_diff > 0:
        for i in range(num):
            shrink = h_diff*h_tbox_arr[i]/h_sum
            h_tbox_arr[i] -= shrink
        y_max = lb
        y_min = ub
        is_h_adjusted = True

    if is_upper:  # stack up from y_max
        if not is_h_adjusted:
            h_diff = BannerConfig.MIN_BOX_IMG_MARGIN - (y_max - np.sum(h_tbox_arr) - space_avg * (num - 1))
            if h_diff > 0:
                for i in range(num):
                    shrink = h_diff * h_tbox_arr[i] / h_sum
                    h_tbox_arr[i] -= shrink
        y2 = y_max
        for i in range(num - 1, -1, -1):
            bbox_xyxy[i, 3] = y2
            y1 = y2 - h_tbox_arr[i]
            bbox_xyxy[i, 1] = y1
            y2 = y1 - space_avg
    else:  # stack down from y_min
        if not is_h_adjusted:
            h_diff = y_min + np.sum(h_tbox_arr) + space_avg * (num - 1) - (H - BannerConfig.MIN_BOX_IMG_MARGIN)
            if h_diff > 0:
                for i in range(num):
                    shrink = h_diff * h_tbox_arr[i] / h_sum
                    h_tbox_arr[i] -= shrink
        y1 = y_min
        for i in range(num):
            bbox_xyxy[i, 1] = y1
            y2 = y1 + h_tbox_arr[i]
            bbox_xyxy[i, 3] = y2
            y1 = y2 + space_avg

    bbox_xyxy = bbox_xyxy[id_bbox_sorted_rev, :]

    # trim x out of bounds
    left_max = 0
    right_max = 0
    for i in range(num):
        left_max = max(left_max, BannerConfig.MIN_BOX_IMG_MARGIN - bbox_xyxy[i, 0])
        right_max = max(right_max, bbox_xyxy[i, 2] - (W - BannerConfig.MIN_BOX_IMG_MARGIN))

    if left_max > 0 and right_max == 0:
        bbox_xyxy[:, 0] += left_max
        bbox_xyxy[:, 2] += left_max
        r_max = 0
        for i in range(num):
            r_max = max(r_max, bbox_xyxy[i, 2] - (W - BannerConfig.MIN_BOX_IMG_MARGIN))

        if r_max > 0:
            bbox_xyxy[:, 2] -= r_max
    elif right_max > 0 and left_max == 0:
        bbox_xyxy[:, 0] -= right_max
        bbox_xyxy[:, 2] -= right_max

        l_max = 0
        for i in range(num):
            l_max = max(l_max, BannerConfig.MIN_BOX_IMG_MARGIN - bbox_xyxy[i, 0])

        if l_max > 0:
            bbox_xyxy[:, 0] += l_max
    else:
        bbox_xyxy[:, 0] += left_max
        bbox_xyxy[:, 2] -= right_max

    for i in range(num):
        bbox_fake[0, i, :] = torch.from_numpy(convert_ltrb_pix_to_xywh(bbox_xyxy[i, :], W, H)).to(bbox_fake.device).to(torch.float32)

    return bbox_xyxy

#----------------------------------------------------------------------------

def horizontal_center_aligned(bbox_fake, mask, W, H, lb, ub, is_upper, xc_median): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    bbox_fake[:,:,0] = xc_median if np.random.rand() > 0.5 else bbox_fake[mask][:,0].mean()
    bbox_xyxy = vertical_respacing(bbox_fake, mask, W, H, lb, ub, is_upper)

    return bbox_xyxy

#----------------------------------------------------------------------------

def horizontal_left_aligned(bbox_fake, mask, W, H, lb, ub, is_upper): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    num = bbox_fake[mask].shape[0]
    x1_sum = 0.0
    for i in range(num):
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox_fake[0,i])
        x1_sum += x1
    x1_mean = x1_sum / float(num)
    for i in range(num):
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox_fake[0,i])
        bbox_fake[0,i,0] -= x1 - x1_mean
    bbox_xyxy = vertical_respacing(bbox_fake, mask, W, H, lb, ub, is_upper)
    
    return bbox_xyxy

#----------------------------------------------------------------------------

def jitter(bbox_fake, seed): # bbox_fake: [B, N, 4] (xc, yc, w, h)
    perturb = torch.from_numpy(np.random.RandomState(seed).uniform(
        low=math.log(BannerConfig.MIN_BOX_JITTER_RATIO), high=math.log(BannerConfig.MAX_BOX_JITTER_RATIO),
        size=bbox_fake.shape)).to(bbox_fake.device).to(torch.float32)
    bbox_fake *= perturb.exp()
    return bbox_fake

#----------------------------------------------------------------------------

def get_adaptive_font_color(img):
    img = np.array(img)
    clr = []
    for ch in range(3):
        clr.append(np.median(img[:, :, ch]))

    return 'rgba'+str((0, 0, 0, 255)) if sum(clr) > 255 * 3 / 1.5 else 'rgba:'+str((255, 255, 255, 255))

#----------------------------------------------------------------------------

def get_adaptive_font_button_color(img):
    img = np.array(img)
    clr = []
    for ch in range(3):
        clr.append(np.median(img[:, :, ch]))

    # adaptive font color, background color
    if sum(clr) < 255 * 2:
        return 'rgba'+str((0, 0, 0, 255)), 'rgba'+str((255, 255, 255, 255))
    else:
        return 'rgba'+str((255, 255, 255, 255)), 'rgba'+str((0, 0, 0, 255))

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

# ----------------------------------------------------------------------------

def visualize_banner(boxes, masks, styles, is_center, background_img, browser, generated_file_path):
    soup = BeautifulSoup(RendererConfig.HTML_TEMP, "html.parser")
    # insert img src div
    img = soup.findAll('img')
    background_img_io = BytesIO()
    background_img.save(background_img_io, format='PNG')
    background_img_base64 = base64.b64encode(background_img_io.getvalue()).decode()
    img[0]['src'] = 'data:image/png;base64, ' + background_img_base64
    W_page, H_page = background_img.size
    
    h_tbox_list = []
    for i in range(boxes.shape[0]):
        text = styles[i]['text']
        if not text:
            h_tbox_list.append(-1)
            continue
        # 'disclaimer / footnote' cannot be used as an id
        styles[i]['type'] = BACK2FRONTEND_NAME_MAP[styles[i]['type']] if styles[i]['type'] in BACK2FRONTEND_NAME_MAP \
            else styles[i]['type']
        x1, y1, x2, y2 = boxes[i, :]
        h_tbox, w_tbox = int(y2 - y1 + 1), int(x2 - x1 + 1)
        font_color = styles[i]['style']['color']
        font_family = styles[i]['style']['fontFamily']
        font_family = 'font-family:' + font_family + ';' if 'fontFamily' in styles[i]['style'] and \
                                                            styles[i]['style']['fontFamily'] else 'font-family:Arial;'
        if font_color:
            font_color = 'color:' + font_color + ';'
        else:
            if styles[i]['type'] == 'button':
                font_color = 'color:' + get_adaptive_font_button_color(background_img.crop([x1, y1, x2, y2]))[0] + ';'
            else:
                font_color = 'color:' + get_adaptive_font_color(background_img.crop([x1, y1, x2, y2])) + ';'

        # button resize and alignment
        if styles[i]['type'] == 'button':
            hr = 0.75
            wr = 0.75
            y_mid = (y1 + y2) / 2
            x_mid = (x1 + x2) / 2
            h = y2 - y1
            w = x2 - x1
            if is_center:
                y1 = y_mid - h*hr/2
                y2 = y_mid + h*hr/2
                x1 = x_mid - w*wr/2
                x2 = x_mid + w*wr/2
            else:
                y1 = y_mid - h*hr/2
                y2 = y_mid + h*hr/2
                x2 = x1 + w*wr
            h_tbox, w_tbox = int(y2 - y1 + 1), int(x2 - x1 + 1)

        h_tbox_list.append(h_tbox)
        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        if styles[i]['type'] == 'button' or is_center:
            tbox_style = RendererConfig.TEXT_CSS_TEMP + 'text-align:center;justify-content:center;'
        else:
            tbox_style = RendererConfig.TEXT_CSS_TEMP + 'text-align:left;'

        tbox_style = tbox_style + font_color + font_family
        tbox_style += 'width:' + str(w_tbox) + 'px;max-width:' + str(w_tbox) + 'px;'
        tbox_style += 'height:' + str(h_tbox) + 'px;max-height:' + str(h_tbox) + 'px;'
        tbox_style += 'top:' + str(y1) + 'px;'
        tbox_style += 'left:' + str(x1) + 'px;'
        if styles[i]['type'].lower() == 'button':
            if styles[i]['buttonParams']['backgroundColor']:
                tbox_style += 'background-color:' + styles[i]['buttonParams']['backgroundColor'] + ';'
            else:
                tbox_style += 'background-color:' + get_adaptive_font_button_color(background_img.crop([x1, y1, x2, y2]))[1] + ';'

            if styles[i]['buttonParams']['radius']:
                tbox_style += 'border-radius:' + str(styles[i]['buttonParams']['radius']).strip() + 'em;'

        tbox_attr = {'style': tbox_style}
        tbox_attr['id'] = styles[i]['type']
        outer_div = soup.new_tag("div", **tbox_attr)
        soup.html.body.div.append(outer_div)
        inner_div = '<div id=' + '"' + styles[i]['type'] + '_in" ' + 'style="''">' + text + '</div>'
        soup.select_one('div#'+styles[i]['type']).append(BeautifulSoup(inner_div, 'html.parser'))

    with open(generated_file_path + '.html', "w") as f:
        f.write(str(soup))

    try:
        browser.get('file://' + generated_file_path + '.html')
    except Exception as e:
        print('Failed to load banner html.')
        return

    for i in range(boxes.shape[0]):
        if not styles[i]['text']:
            continue
        if styles[i]['type'] == 'button':
            pix = browser.execute_script(
                RendererConfig.JAVASCRIPT.format(styles[i]['type'], styles[i]['type'], int(h_tbox_list[i] * 1.15), h_tbox_list[i] * 0.33, h_tbox_list[i] * 0.33))
        else:
            pix = browser.execute_script(
                RendererConfig.JAVASCRIPT.format(styles[i]['type'], styles[i]['type'], int(h_tbox_list[i] * 1.15), 0, 0))
        browser.execute_script("""document.querySelector("#{}").style.fontSize="{}";""".format(styles[i]['type'], pix))
        old_style = soup.find("div", {"id": styles[i]['type']})
        old_style["style"] = old_style["style"] + "font-size:" + str(pix) + ";"

    with open(generated_file_path + '.html', "w") as f:
        f.write(str(soup))

    screenshot = browser.get_screenshot_as_png()
    screenshot = Image.open(BytesIO(screenshot))
    screenshot = screenshot.crop([0, 0, W_page, H_page])
    if W_page > BannerConfig.MAX_TIMG_WIDTH or H_page > BannerConfig.MAX_TIMG_HEIGHT:
        screenshot.thumbnail((BannerConfig.MAX_TIMG_WIDTH, BannerConfig.MAX_TIMG_HEIGHT), Image.Resampling.LANCZOS)
    screenshot.save(generated_file_path + '.png')

    return generated_file_path + '.png', generated_file_path + '.html'

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

def generate_banners(
    G: str,
    bg: str,
    input_styles: List[Dict[str, str]],
    post_process: Dict[str, float],
    seeds: List[int],
    browser: Chrome,
    output_dir: str,
):
    device = 'cuda'
    print('Loading background image from "%s"...' % bg)
    background_orig = Image.open(bg)
    W, H = background_orig.size
    background = np.array(background_orig.resize((1024, 1024), Image.Resampling.LANCZOS))
    rgb_mean = np.reshape(np.array([0.485, 0.456, 0.406]).astype(np.float32), (1,1,3))
    rgb_std = np.reshape(np.array([0.229, 0.224, 0.225]).astype(np.float32), (1,1,3))
    if background.shape[2] > 3:
        background = background[:, :, 0:3]
    elif background.shape[2] < 3:
        background = background.repeat(1, 1, 3-background.shape[2])

    background = (background.astype(np.float32) / 255.0 - rgb_mean) / rgb_std
    background = background.transpose(2, 0, 1)

    background = torch.from_numpy(background).to(device).to(torch.float32).unsqueeze(0)
    label2index = dict()
    for idx, label in enumerate(BannerConfig.LABEL_LIST):
        label2index[label] = idx

    # construct the input text strings and the corresponding styles
    bbox_text = []
    bbox_label = []
    bbox_style = []
    sorted_input_styles = []
    note_style = None
    for style in input_styles:
        if not style['text']:
            continue
        if style['type'] in FRONT2BACKEND_NAME_MAP:
            style['type'] = FRONT2BACKEND_NAME_MAP[style['type']]
        try:
            # make sure 'disclaimer / footnote' is at the end of the input style list
            if style['type'] == 'disclaimer / footnote':
                note_style = style
            else:
                sorted_input_styles.append(style)
        except KeyError:
            continue

    if note_style:
        sorted_input_styles.append(note_style)

    for style in sorted_input_styles:
        try:
            if style['type'] == 'header' or style['type'] == 'body' or style['type'] == 'button' or \
                    style['type'] == 'disclaimer / footnote':
                bbox_text.append(style["text"])
                bbox_label.append(label2index[style["type"]])
                bbox_style.append(style)
        except KeyError:
            continue

    print('Loading layout bboxes')
    bbox_fake_list = []
    mask_list = []
    bbox_style_list = []
    is_center_list = []
    mask = torch.from_numpy(np.array([1] * len(bbox_text) + [0] * (9-len(bbox_text)))).to(device).to(torch.bool).unsqueeze(0)
    bbox_patch_dummy = torch.zeros((1, 9, 3, 256, 256)).to(device).to(torch.float32)
    lb, ub = 0, H-1
    is_upper = False
    bbox_xyxy_list = []
    xc_median = int(W/2)
    for seed in seeds:
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, 9, G.z_dim)).to(device).to(torch.float32)
        order = list(range(len(bbox_text)))
        bbox_text_temp = [bbox_text[i] for i in order]
        bbox_text_temp += [''] * (9-len(bbox_text))
        bbox_text_temp = [bbox_text_temp]
        bbox_label_temp = [bbox_label[i] for i in order]
        bbox_class_temp = torch.from_numpy(np.array(bbox_label_temp + [0] * (9-len(bbox_label_temp)))).to(device).to(torch.int64).unsqueeze(0)
        bbox_fake = G(z=z, bbox_class=bbox_class_temp, bbox_real=None, bbox_text=bbox_text_temp, bbox_patch=bbox_patch_dummy, padding_mask=~mask, background=background, c=None)
        if seed != 1:
            if 'jitter' in post_process and np.random.rand() < post_process['jitter']:
                bbox_fake = jitter(bbox_fake, seed)
            if 'horizontal_center_aligned' in post_process and np.random.rand() < post_process['horizontal_center_aligned']:
                bbox_xyxy = horizontal_center_aligned(bbox_fake, mask, W, H, lb, ub, is_upper, xc_median)
                bbox_xyxy = bbox_xyxy.astype('int32')
                is_center_list.append(True)
                bbox_xyxy_list.append(bbox_xyxy)
            else:
                bbox_xyxy = horizontal_left_aligned(bbox_fake, mask, W, H, lb, ub, is_upper)
                bbox_xyxy = bbox_xyxy.astype('int32')
                is_center_list.append(False)
                bbox_xyxy_list.append(bbox_xyxy)
        else:
            is_center_list.append(True)
            lb, ub, is_upper, xc_median = compute_bounds(bbox_fake, mask, W, H)
            bbox_xyxy = np.zeros((len(bbox_text), 4))
            for i in range(len(bbox_text)):
                bbox_xyxy[i, :] = convert_xywh_to_ltrb_pix(bbox_fake[0, i, :], W, H)
            bbox_xyxy_list.append(bbox_xyxy)

        bbox_fake_list.append(bbox_fake[0])
        mask_list.append(mask[0])

        # record the original bbox_style order
        bbox_style_temp = [bbox_style[i] for i in order]
        bbox_style_temp += [''] * (9-len(bbox_style))
        bbox_style_list.append(bbox_style_temp)
    
    ###################################
    # Save random sample variants according to overlap
    ###################################
    subdir = '%s' % output_dir
    os.makedirs(subdir, exist_ok=True)
    order = list(range(len(seeds)))
    screenshot_paths = []
    html_paths = []
    for j, idx in enumerate(order):
        generated_path = os.path.join(output_dir, 'layoutdetr_' + f'{str(uuid.uuid4())}')
        screenshot_path, html_path = \
            visualize_banner(bbox_xyxy_list[idx], mask_list[idx], bbox_style_list[idx], is_center_list[idx], background_orig,
                             browser, generated_path)

        screenshot_paths.append(screenshot_path)
        html_paths.append(html_path)

    return screenshot_paths, html_paths
