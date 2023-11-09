"""
Copyright (c) 2023 Salesforce, Inc.

All rights reserved.

SPDX-License-Identifier: Apache License 2.0

For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/

By Chia-Chih Chen, chiachih.chen@salesforce.com
"""

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union, Dict
import uuid
import cv2
import click
import dnnlib
import numpy as np
from PIL import Image, ImageDraw
import seaborn as sns
import torch
import legacy

from e2e_pipeline.utils_server import safeMakeDirs
from bs4 import BeautifulSoup
from io import BytesIO
import re
import sys
import argparse
import json
import math
from selenium import webdriver
from selenium.webdriver import Chrome
import pickle
import random
import copy
import glob
import pdb
from datetime import datetime
from SmartCropping import smart_crop_fast
import time

random.seed(datetime.now().timestamp())

HTML_TEMP = \
    """
    <!DOCTYPE html>
    <html>
    <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" type="text/css" href="css/style.css">
    <style>
    .container {
      position: relative;
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

TEXT_CSS_TEMP = 'align-items:center;position:absolute;word-wrap:break-word;overflow-wrap:' \
                'break-word;display:flex;'
BORDER_CSS_TEMP = ''
#BORDER_CSS_TEMP = 'border-width:1px;border-style:solid;border-color:black;'
TEXT_CSS_TEMP += BORDER_CSS_TEMP

LABEL_LIST = [
        'header',
        'pre-header',
        'post-header',
        'body',
        'disclaimer / footnote',
        'button',
        'callout',
        'logo'
        ]

JAVASCRIPT = \
    """    
    let inner = document.getElementById("{}_in");
    let outer = document.getElementById("{}").style;
    if (inner && outer) {{
        inner.style.fontSize = "{}px"
        resize_to_fit()
    }}
    return inner.style.fontSize;

    function resize_to_fit() {{
        let fontSize = inner.style.fontSize;
        inner.style.fontSize = (parseFloat(fontSize) - 1) + 'px';
        if (inner.clientHeight >= parseFloat(outer.height) - {}) {{
            resize_to_fit();
        }}
    }}
    """

# Front/backend naming convention conversion
FRONT2BACKEND_NAME_MAP = {'note': 'disclaimer / footnote'}
BACK2FRONTEND_NAME_MAP = {y: x for x, y in FRONT2BACKEND_NAME_MAP.items()}
WORDFORMAT2HTMLELEM_MAP = {'Bold': ['<b>', '</b>'], 'Italic': ['<i>', '</i>'], 'Strikethrough': ['<del>', '</del>'],
        'Underline': ['<ins>', '</ins>'], 'Smart Style': ['', ''], "Larger": ['', '']}

PATH_COLOR_TEMP = 'templates/color/bg_ft_clr.pkl'
PATH_FONT_TEMP = 'templates/psd_font/request2font.pkl'
PATH_FTEMP_META = 'templates/retrieve_adaptor/temp_meta.json'
PATH_FTEMP_BG = 'templates/retrieve_adaptor/'

TEMP_COLOR_BG = TEMP_COLOR_FT = TEMP_NUM_COPY2PSD_NAME = TEMP_PSD_NAME2LABEL_FONT = None
with open(PATH_COLOR_TEMP, 'rb') as fp:
    TEMP_COLOR_BG = pickle.load(fp)
    TEMP_COLOR_FT = pickle.load(fp)

with open(PATH_FONT_TEMP, 'rb') as fp:
    TEMP_NUM_COPY2PSD_NAME = pickle.load(fp)
    TEMP_PSD_NAME2LABEL_FONT = pickle.load(fp)

with open(PATH_FTEMP_META , 'r') as fp:
    FTEMP_META = json.load(fp)


#----------------------------------------------------------------------------

def get_background_color(img):
    img = np.array(img)
    clr = []
    for ch in range(3):
        clr.append(np.median(img[:, :, ch]))
    return clr

#----------------------------------------------------------------------------

def get_adaptive_font_color(img):
    clr = get_background_color(img)
    return 'rgba'+str((0, 0, 0, 255)) if sum(clr) > 255 * 3 / 1.5 else 'rgba:'+str((255, 255, 255, 255))

#----------------------------------------------------------------------------

def get_adaptive_font_button_color(img):
    clr = get_background_color(img)

    # adaptive font color, background color
    if sum(clr) < 255 * 2:
        return 'rgba'+str((0, 0, 0, 255)), 'rgba'+str((255, 255, 255, 255))
    else:
        return 'rgba'+str((255, 255, 255, 255)), 'rgba'+str((0, 0, 0, 255))

#----------------------------------------------------------------------------

def get_complementary_color(rgb):
    def hilo(a, b, c):
        if c < b:
            b, c = c, b
        if b < a:
            a, b = b, a
        if c < b:
            b, c = c, b
        return a + c

    r, g, b = rgb
    k = hilo(r, g, b)
    return tuple(k - u for u in (r, g, b))

#----------------------------------------------------------------------------

def retrieve_template_color(img, is_button):
    top_k = 5
    clr = get_background_color(img)
    clr = np.asarray(clr)
    clr = np.expand_dims(clr, axis=0)
    clr_ft = ''
    clr_btn = ''
    diff = np.abs(clr - TEMP_COLOR_BG)
    diff = np.sum(diff, 1)
    diff = np.argsort(diff)
    if not is_button:
        idx = diff[random.randint(0, top_k - 1)]
        clr_ft = TEMP_COLOR_FT[idx, :]
    else:
        # bg to btn color max dist => complementary color
        # clr_btn = get_complementary_color([clr_ft[0], clr_ft[1], clr_ft[2]])
        idx = diff[random.randint(max(diff.shape)-top_k, max(diff.shape)-1)]
        clr_btn = TEMP_COLOR_BG[idx, :]
        clr_ft = TEMP_COLOR_FT[idx, :]
        clr_btn = 'rgba' + str((clr_btn[0], clr_btn[1], clr_btn[2], 255))
    clr_ft = 'rgba' + str((clr_ft[0], clr_ft[1], clr_ft[2], 255))

    return clr_ft, clr_btn

#----------------------------------------------------------------------------

def retrieve_template_font(num_copy):
    psd = []
    for num in TEMP_NUM_COPY2PSD_NAME:
        if num_copy <= num:
            psd.extend(TEMP_NUM_COPY2PSD_NAME[num])
    psd_fname = random.choice(psd)
    return TEMP_PSD_NAME2LABEL_FONT[psd_fname]

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
def visualize_framed_banner(ftemp, old_styles, browser, output_format, generated_file_path, idx):
    background_img = cv2.cvtColor(ftemp[0]['image'], cv2.COLOR_BGR2RGB)
    background_img = Image.fromarray(background_img)
    styles = []
    for style in old_styles:
        for t in ftemp:
            if 'label' in t and 'type' in style and style['type'] == t['label']:
                style['style']['color'] = 'rgba'+str((t['font_color'][2], t['font_color'][1], t['font_color'][0], 255))
                style['style']['fontFamily'] = t['font_name']
                style['xyxy'] = t['xyxy']
                if len(style['text']) == t['char_length']:
                    style['style']['fontSize'] = t['font_size']
                else:
                    style['style']['fontSize'] = -1
                styles.append(style)

    soup = BeautifulSoup(HTML_TEMP, "html.parser")
    # insert img src div
    img = soup.findAll('img')
    img[0]['src'] = os.path.basename(generated_file_path + '.png')
    background_img.save(generated_file_path + '.png')
    W_page, H_page = background_img.size
    w_page, h_page = 600, 400  # thumbnail resolution

    h_tbox_list = []
    for i in range(len(styles)):
        text = styles[i]['text']
        if not text:
            h_tbox_list.append(-1)
            continue
        # 'disclaimer / footnote' cannot be used as an id
        styles[i]['type'] = BACK2FRONTEND_NAME_MAP[styles[i]['type']] if styles[i]['type'] in BACK2FRONTEND_NAME_MAP \
            else styles[i]['type']
        x1, y1, x2, y2 = styles[i]['xyxy']
        font_color = styles[i]['style']['color']
        font_family = styles[i]['style']['fontFamily']
        font_family = 'font-family:' + font_family + ';'
        font_color = 'color:' + font_color + ';'
        h_tbox, w_tbox = int(y2 - y1), int(x2 - x1)
        h_tbox_list.append(h_tbox)
        tbox_style = TEXT_CSS_TEMP + 'text-align:center;justify-content:center;'
        tbox_style = tbox_style + font_color + font_family
        tbox_style += 'width:' + str(w_tbox) + 'px;max-width:' + str(w_tbox) + 'px;'
        tbox_style += 'height:' + str(h_tbox) + 'px;max-height:' + str(h_tbox) + 'px;'
        tbox_style += 'top:' + str(y1) + 'px;'
        tbox_style += 'left:' + str(x1) + 'px;'
        tbox_attr = {'style': tbox_style}
        tbox_attr['id'] = styles[i]['type']
        outer_div = soup.new_tag("div", **tbox_attr)
        soup.html.body.div.append(outer_div)
        inner_div = '<div id=' + '"' + styles[i]['type'] + '_in" ' + 'style="''">' + text + '</div>'
        soup.select_one('div#'+styles[i]['type']).append(BeautifulSoup(inner_div, 'html.parser'))

    generated_image_path_vis = generated_html_path = ''
    if 'image' in output_format:
        generated_image_path_vis = generated_file_path + '_vis.png'
        with open(generated_file_path + '.html', "w") as f:
            f.write(str(soup))
        try:
            browser.get("file:///" + generated_file_path + '.html')
        except Exception as e:
            pass
        for i in range(len(styles)):
            if not styles[i]['text']:
                continue

            if styles[i]['style']['fontSize'] > 0:
                pix = str(styles[i]['style']['fontSize']) + 'px'
            else:
                pix = browser.execute_script(
                    JAVASCRIPT.format(styles[i]['type'], styles[i]['type'], int(h_tbox_list[i] * 1.0), 0, 0)) # 1.15 before
            browser.execute_script("""document.querySelector("#{}").style.fontSize="{}";""".format(styles[i]['type'], pix))

        with open(generated_file_path + '.html', 'w') as f:
            f.write(browser.page_source)

        screenshot = browser.get_screenshot_as_png()
        screenshot = Image.open(BytesIO(screenshot))
        screenshot = screenshot.crop([0, 0, W_page, H_page])
        if W_page > w_page or H_page > h_page:
            screenshot.thumbnail((w_page, h_page), Image.ANTIALIAS)
        screenshot.save(generated_image_path_vis)

    if 'html' in output_format:
        generated_html_path = generated_file_path + '.html'
        # avoid saving html twice
        if 'image' not in output_format:
            with open(generated_html_path, "w") as f:
                f.write(str(soup))

    return generated_image_path_vis, generated_html_path

#----------------------------------------------------------------------------

def retrieve_ftemp(label_list_ftemp, text_list_ftemp, num_ftemp_max=4, r_charlen_user2ftemp_max=1.25):
    ftemp_list = []
    for ftemp in FTEMP_META:
        if '4_3-white-sand' in ftemp['background']:
            continue
        print(set(label_list_ftemp))
        print(set(ftemp['label2char_length']['label_list']))

        if set(label_list_ftemp).issubset(set(ftemp['label2char_length']['label_list'])):
            is_match = True
            for label, text in zip(label_list_ftemp, text_list_ftemp):
                if len(text) > ftemp['label2char_length'][label]*r_charlen_user2ftemp_max:
                    is_match = False
                    break
            if is_match:
                ftemp_list.append(ftemp)
    print('*{} framed templates found!'.format(len(ftemp_list)))
    random.shuffle(ftemp_list)
    ftemp_list = ftemp_list[:min(len(ftemp_list), num_ftemp_max)]
    return ftemp_list

#----------------------------------------------------------------------------

def extract_ftemp(model_supres, model_saliency, model_text, model_face, ftemp_meta_retr, img):
    img_framed = np.array(img.copy())
    saliencies = smart_crop_fast.boxes_of_saliencies_onetime(img_framed, model_saliency)
    ftemp_extr = []

    for ftemp_meta in ftemp_meta_retr:
        bg_path = os.path.join(PATH_FTEMP_BG, ftemp_meta['background'])
        bg_ftemp = cv2.imread(bg_path)
        with open(bg_path.replace('.png', '.json'), 'r') as fp:
            ftemp = json.load(fp)
        x1, y1, x2, y2 = ftemp[0]['xyxy']
        w = int(max(0, x2 - x1))
        h = int(max(0, y2 - y1))
        img_crop = smart_crop_fast.smart_crop(img_framed, saliencies, w, h, True, False, False, model_supres, model_saliency)
        bg_ftemp[int(y1): int(y1+h), int(x1): int(x1+w), :3] = img_crop
        # smart crop
        ftemp[0]['image'] = bg_ftemp
        ftemp_extr.append(ftemp)
    return ftemp_extr

#----------------------------------------------------------------------------

def generate_banners(
    model_supres: str,
    model_saliency: str,
    model_text: str,
    model_face: str,
    bg: str,
    input_styles: List[Dict[str, str]],
    seeds: List[int],
    output_format: List[str],
    browser: Chrome,
    output_dir: str,
):
    device = 'cuda'
    print('Loading background image from "%s"...' % bg)
    background_orig = Image.open(bg)
    W, H = background_orig.size
    label2index = dict()
    for idx, label in enumerate(LABEL_LIST):
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

    text_list_ftemp = []
    label_list_ftemp = []
    for style in sorted_input_styles:
        try:
            if style['type'] == 'header' or style['type'] == 'body' or style['type'] == 'button' or \
                    style['type'] == 'disclaimer / footnote':
                bbox_text.append(style["text"])
                text_list_ftemp.append(style["text"])
                label_list_ftemp.append(style["type"])
                bbox_label.append(label2index[style["type"]])
                bbox_style.append(style)
        except KeyError:
            continue

    bbox_style_list = []
    for seed in seeds:
        order = list(range(len(bbox_text)))
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
    generated_image_paths = []
    generated_html_paths = []

    """
    - select templates based on ad copy type and char length
      - (optional) use color harmony rule to select templates
    - smart cropping based on selected templates
    """
    time_start = time.time()
    ftemp_meta_retr = retrieve_ftemp(label_list_ftemp, text_list_ftemp)
    ftemp = extract_ftemp(model_supres, model_saliency, model_text, model_face,ftemp_meta_retr, background_orig)
    order = list(range(min(len(seeds), len(ftemp))))
    print('***Retrieve-adapter x {} cost {}'.format(len(ftemp), time.time() - time_start))
    ctr = 0
    time_RA = 0
    for j, idx in enumerate(order):
        generated_path = os.path.join(output_dir, f'{str(uuid.uuid4())}')
        time_start = time.time()
        generated_image_path, generated_html_path = \
            visualize_framed_banner(ftemp[ctr], bbox_style_list[idx], browser, output_format, generated_path, idx)
        time_RA += time.time() - time_start
        ctr += 1

        print('***RA vis x {} cost {}'.format(ctr, time_RA))
        generated_image_paths.append(generated_image_path)
        generated_html_paths.append(generated_html_path)

    return generated_image_paths, generated_html_paths

#----------------------------------------------------------------------------

if __name__ == "__main__":
    # test generate_banners(...) from local files
    # example: python gen_single_sample_API_server.py --image=./dark_flooring.jpg --model=/export/home/ads_multi.pkl --style=./test.json
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--style', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    output_path = os.path.join(os.getcwd(), 'demo')
    safeMakeDirs(output_path)
    args = parser.parse_args()

    print('Loading networks from "%s"...' % args.model)
    device = torch.device('cuda')
    with dnnlib.util.open_url(args.model) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    with open(args.style, 'r') as f:
        data = json.load(f)
    seeds = [x+1 for x in range(int(data['numResults']))]
    options = webdriver.ChromeOptions()
    options.add_argument('no-sandbox')
    options.add_argument('headless')
    browser = Chrome(executable_path='/usr/bin/chromedriver', options=options)
    browser.set_window_size(3000, 3000)

    generate_banners(args.image, data['contentStyle']['elements'],
                     seeds, data['resultFormat'], browser, output_path)

#----------------------------------------------------------------------------
