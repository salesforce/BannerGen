"""
Copyright (c) 2023 Salesforce, Inc.

All rights reserved.

SPDX-License-Identifier: Apache License 2.0

For full license text, see LICENSE.txt file in the repo root or http://www.apache.org/licenses/

By Ning Yu, ning.yu@salesforce.com

Modified from StyleGAN3 repo: https://github.com/NVlabs/stylegan3

Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""


from __future__ import annotations

import math
import random
import sys

import k_diffusion as K
import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw
from torch.cuda.amp import autocast

sys.path.append("./stable_diffusion")

import cv2
import time
import os
import json
import copy
from utils.util import convert_xywh_to_ltrb_pix
from selenium import webdriver
from selenium.webdriver import Chrome
import uuid
from bs4 import BeautifulSoup
from io import BytesIO
import re
import sys
import argparse
import json
import math
from omegaconf import OmegaConf
from InstructPix2Pix.stable_diffusion.ldm.util import instantiate_from_config
import k_diffusion as K
import torch.nn as nn
import einops
import base64
from InstructPix2Pix.configs.banner_config import BannerConfig
from InstructPix2Pix.configs.banner_config import RendererConfig

#----------------------------------------------------------------------------

label2color = {'header': (255, 0, 255),
                'body text': (0, 255, 255),
                'button': (255, 255, 0),
                'disclaimer': (0, 128, 0),
                }
C_THRES = 96

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

label_white_list = ['header', 'body text', 'button']
# Front/backend naming convention conversion
FRONT2BACKEND_NAME_MAP = {'note': 'disclaimer / footnote'}
BACK2FRONTEND_NAME_MAP = {y: x for x, y in FRONT2BACKEND_NAME_MAP.items()}


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "n ... -> (repeat n) ...", repeat=3)
        cfg_sigma = einops.repeat(sigma, "n ... -> (repeat n) ...", repeat=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def load_model(model_path):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/generate.yaml")
    config = OmegaConf.load(f"{config_path}")
    config.model.params.ckpt_path = model_path
    vae_ckpt = None
    model_instructpix2pix = load_model_from_config(config, model_path, vae_ckpt)
    model_instructpix2pix.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model_instructpix2pix)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model_instructpix2pix.get_learned_conditioning([""])
    print('InstructPix2Pix model loaded.')

    return model_instructpix2pix, model_wrap_cfg, model_wrap, null_token


def horizontal_center_aligned(bbox_list): # bbox_list: [N, 4] (xc, yc, w, h)
    intervals = []
    for i in range(len(bbox_list)):
        xc, yc, w, h = bbox_list[i]
        l = xc - w/2
        r = xc + w/2
        intervals.append((l, r))
    intervals.sort()
    groups = []
    group_cur = []
    for interval in intervals:
        l = interval[0]
        r = interval[1]
        if len(group_cur) == 0:
            group_cur.append(interval)
            rightmost = r
        elif l < rightmost:
            group_cur.append(interval)
            rightmost = max(rightmost, r)
        else:
            groups.append(list(group_cur))
            group_cur = [interval]
            rightmost = r
    groups.append(list(group_cur))
    xc_mean_dict = {}
    for group_cur in groups:
        xc_sum = 0
        for interval in group_cur:
            xc_sum += (interval[0] + interval[1]) / 2
        xc_mean = xc_sum / float(len(group_cur))
        xc_mean_dict[tuple(group_cur)] = xc_mean
    for i in range(len(bbox_list)):
        xc, yc, w, h = bbox_list[i]
        l = xc - w/2
        r = xc + w/2
        interval = (l, r)
        xc_mean = None
        for group_cur in groups:
            if interval in group_cur:
                xc_mean = xc_mean_dict[tuple(group_cur)]
                break
        assert xc_mean is not None
        bbox_list[i][0] = xc_mean
    return bbox_list


def horizontal_left_aligned(bbox_list): # bbox_list: [N, 4] (xc, yc, w, h)
    intervals = []
    for i in range(len(bbox_list)):
        xc, yc, w, h = bbox_list[i]
        l = xc - w/2
        r = xc + w/2
        intervals.append((l, r))
    intervals.sort()
    groups = []
    group_cur = []
    for interval in intervals:
        l = interval[0]
        r = interval[1]
        if len(group_cur) == 0:
            group_cur.append(interval)
            rightmost = r
        elif l < rightmost:
            group_cur.append(interval)
            rightmost = max(rightmost, r)
        else:
            groups.append(list(group_cur))
            group_cur = [interval]
            rightmost = r
    groups.append(list(group_cur))
    l_mean_dict = {}
    for group_cur in groups:
        l_sum = 0
        for interval in group_cur:
            l_sum += interval[0]
        l_mean = l_sum / float(len(group_cur))
        l_mean_dict[tuple(group_cur)] = l_mean
    for i in range(len(bbox_list)):
        xc, yc, w, h = bbox_list[i]
        l = xc - w/2
        r = xc + w/2
        interval = (l, r)
        l_mean = None
        for group_cur in groups:
            if interval in group_cur:
                l_mean = l_mean_dict[tuple(group_cur)]
                break
        assert l_mean is not None
        bbox_list[i][0] -= l - l_mean
    return bbox_list


def de_overlap(bbox_list): # bbox_list: [N, 4] (xc, yc, w, h)
    num = len(bbox_list)
    for i in range(num):
        xc1, yc1, w1, h1 = bbox_list[i]
        for j in range(num):
            if i != j:
                xc2, yc2, w2, h2 = bbox_list[j]
                if abs(yc2 - yc1) < h1/2 + h2/2:
                    diff = h1/2 + h2/2 - abs(yc2 - yc1)
                    if yc1 < yc2:
                        bbox_list[i][1] -= diff/2
                        bbox_list[j][1] += diff/2
                    else:
                        bbox_list[i][1] += diff/2
                        bbox_list[j][1] -= diff/2
    for i in range(num):
        xc1, yc1, w1, h1 = bbox_list[i]
        for j in range(num):
            if i != j:
                xc2, yc2, w2, h2 = bbox_list[j]
                if abs(yc2 - yc1) < h1/2 + h2/2:
                    diff = h1/2 + h2/2 - abs(yc2 - yc1)
                    bbox_list[i][3] -= diff/2
                    bbox_list[j][3] -= diff/2
    return bbox_list


def extract_bbox(input_image, width, height, resolution, edited_image_list, post_process=None, margin_ratio=0.025):
    y_mar = int(height*margin_ratio)
    x_mar = int(width*margin_ratio)
    bbox_xyxy_list = []
    is_center_list = []
    factor_extr = float(resolution) / float(max(width, height))
    width_extr = int(width * factor_extr)
    height_extr = int(height * factor_extr)
    bg = np.array(input_image.resize((width_extr, height_extr), resample=Image.Resampling.LANCZOS)).astype(float)
    for edited_image in edited_image_list:
        edited_image = np.array(edited_image.resize((width_extr, height_extr), resample=Image.Resampling.LANCZOS))
        fg = edited_image.astype(float)
        diff = fg - bg
        diff = np.sqrt(np.sum(diff * diff, axis=2))
        diff = (diff > 128.0).astype(float)
        bboxes = []
        for idx, label in enumerate(label_list):
            if label in label_white_list:  # skip disclaimers due to complexity
                c = label2color[label]
                color = (fg[:,:,0] > c[0]-C_THRES) * (fg[:,:,0] < c[0]+C_THRES) * (fg[:,:,1] > c[1]-C_THRES) * (fg[:,:,1] < c[1]+C_THRES) * (fg[:,:,2] > c[2]-C_THRES) * (fg[:,:,2] < c[2]+C_THRES)
                binary = diff * color
                binary = (binary * 255.0).astype(np.ubyte)
                #contours, hierarchy, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                b3 = cv2.merge((binary,binary,binary))
                try:
                    cv2.drawContours(b3, contours, -1, (0,255,0), 3)
                except Exception as e:
                    print(e)
                    continue
                rects = [cv2.boundingRect(cnt) for cnt in contours]
                rects = sorted(rects, key=lambda x:x[2]*x[3], reverse=True)
                if len(rects) > 0:
                    x, y, w, h = rects.pop(0)
                    xc = x + w//2
                    yc = y + h//2
                    cv2.rectangle(b3, (x, y), (x+w, y+h), (255,0,0), 2)
                    bboxes.append([float(xc)/float(width_extr), float(yc)/float(height_extr), float(w)/float(width_extr), float(h)/float(height_extr)])

        if 'horizontal_center_aligned' in post_process and np.random.rand() < post_process[
            'horizontal_center_aligned']:
            bboxes = horizontal_center_aligned(bboxes)
            bboxes = de_overlap(bboxes)
            is_center_list.append(True)
        else:
            bboxes = horizontal_left_aligned(bboxes)
            bboxes = de_overlap(bboxes)
            is_center_list.append(False)

        bboxes_xyxy = []
        for bbox in bboxes:
            x1, y1, x2, y2 = convert_xywh_to_ltrb_pix(bbox, width, height)
            x1 = max(x1, x_mar)
            y1 = max(y1, y_mar)
            x2 = min(x2, width - x_mar)
            y2 = min(y2, height - y_mar)
            bboxes_xyxy.append([x1,y1,x2,y2])
        bbox_xyxy_list.append(bboxes_xyxy)
        
    return bbox_xyxy_list, is_center_list

def visualize_banner(boxes, styles, is_center, background_img, browser, generated_file_path):
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


def generate_banners(
    model,
    model_wrap_cfg,
    model_wrap,
    null_token,
    bg,
    input_styles,
    post_process,
    seeds,
    browser,
    output_dir,
    seed_rand=None, steps=25, resolution=512, cfg_text=7.5, cfg_image=0.5):

    ordered_styles = []
    for label in BannerConfig.IP2P_SUPPORTED_LABEL_LIST:
        for elem in input_styles:
            if label == elem['type']:
                ordered_styles.append(elem)
                break

    string_list = [e['text'] for e in ordered_styles]
    label_list = [e['type'] for e in ordered_styles]
    print('Loading background image from "%s"...' % bg)
    background_orig = Image.open(bg)
    input_image = background_orig.convert("RGB")
    width, height = background_orig.size

    # (strings + labels) --> instructions
    instructions = ''
    for i in range(len(label_list)):
        instruction = 'Add diverse %s texts saying \\\"%s\\\" in %d characters' % (label_list[i], string_list[i], len(string_list[i]))
        if label_list[i] == 'header' or label_list[i] == 'body':
            area_ratio = 30
        else:
            area_ratio = 10
        instruction += ' covering %d%% area. ' % area_ratio
        instructions += instruction
    instructions = instructions[:-1]

    # InstructPix2Pix image editing
    if seed_rand is None:
        seed_rand = random.randint(0, 100000)

    if resolution == 512 and max(width, height) < 512:
        resolution = 256
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width_new = int((width * factor) // 64) * 64
    height_new = int((height * factor) // 64) * 64
    input_image_new = input_image.resize((width_new, height_new), resample=Image.Resampling.LANCZOS)

    print('InstructPix2Pix bbox generation...')
    with torch.no_grad(), autocast(enabled=False), model.ema_scope():
        print('Instructions: ', instructions)
        cond = {}
        cond["c_crossattn"] = [model.get_learned_conditioning([instructions]).repeat(len(seeds), 1, 1)]
        input_image_new = 2 * torch.tensor(np.array(input_image_new)).float() / 255 - 1
        input_image_new = rearrange(input_image_new, "h w c -> 1 c h w").to(model.device)
        cond["c_concat"] = [model.encode_first_stage(input_image_new).mode().repeat(len(seeds), 1, 1, 1)]

        uncond = {}
        uncond["c_crossattn"] = [null_token.repeat(len(seeds), 1, 1)]
        uncond["c_concat"] = [torch.zeros_like(cond["c_concat"][0])]

        sigmas = model_wrap.get_sigmas(steps)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": cfg_text,
            "image_cfg_scale": cfg_image,
        }
        torch.manual_seed(seed_rand)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "n c h w -> n h w c")
        edited_image_list = [Image.fromarray(x[i].type(torch.uint8).cpu().numpy()) for i in range(x.shape[0])]

    # bbox extraction
    bbox_xyxy_list, is_center_list = extract_bbox(input_image, width, height, resolution, edited_image_list, post_process)

    subdir = '%s' % output_dir
    os.makedirs(subdir, exist_ok=True)
    order = list(range(len(seeds)))
    screenshot_strs = []
    html_strs = []
    for j, idx in enumerate(order):
        generated_path = os.path.join(output_dir, 'instructpix2pix_' + f'{str(uuid.uuid4())}')

        screenshot_str, html_str = \
            visualize_banner(np.array(bbox_xyxy_list[idx]), ordered_styles, is_center_list[idx], background_orig,
                             browser, generated_path)

        screenshot_strs.append(screenshot_str)
        html_strs.append(html_str)

    return screenshot_strs, html_strs
