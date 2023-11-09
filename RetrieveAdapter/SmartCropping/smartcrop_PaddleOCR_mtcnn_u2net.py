from __future__ import print_function
from __future__ import division

import cv2
from PIL import Image, ImageDraw
import numpy as np
import argparse
import os
import math

from facenet_pytorch import MTCNN
import torch
#from torch import autocast

from U_2_Net.model import U2NET # full size version 173.6 MB
from U_2_Net.data_loader import SalObjDataset, RescaleT, ToTensorLab
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from scipy.ndimage import label

from paddleocr import PaddleOCR

import time

#from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline

#import sys
#sys.path.insert(0, "./liif")
import models
from utils import make_coord
from test import batched_predict
import sys


def auto_resize(image, target_width, target_height):
    height = image.shape[0]
    width = image.shape[1]
    r = float(width) / float(height)

    target_ratio = float(target_width) / float(target_height)
    if target_ratio < r:
        h = target_height
        w = int(h * r)
        need_upsize = h > height
    else:
        w = target_width
        h = int(w / r)
        need_upsize = w > width

    if not need_upsize:
        print('Auto downsizing...')
        image = cv2.resize(image, (w, h))
    else: # LIIF superresolution
        print('Auto upsizing...')
        model = models.make(torch.load('/home/Claude/SmartCropping/pretrained/rdn-liif.pth')['model'], load_sd=True).cuda()
        t_superresolution_start = time.time()
        image = transforms.ToTensor()(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        coord = make_coord((h, w)).cuda()
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2 / h
        cell[:, 1] *= 2 / w
        image = batched_predict(model, ((image - 0.5) / 0.5).cuda().unsqueeze(0), coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
        image = (image * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
        image = transforms.ToPILImage()(image)
        image = np.array(image.convert('RGB'))[:, :, ::-1]
        t_superresolution_end = time.time()
        print('Image superresolution time: ', t_superresolution_end-t_superresolution_start)
        
    return image


'''
def outpaint(image, mask, process_size=512, strength=0.75, num_inference_steps=50, guidance_scale=7.5):
    text2img = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16, use_auth_token=True).to("cuda")
    inpaint = StableDiffusionInpaintPipeline(
                vae=text2img.vae,
                text_encoder=text2img.text_encoder,
                tokenizer=text2img.tokenizer,
                unet=text2img.unet,
                scheduler=text2img.scheduler,
                safety_checker=text2img.safety_checker,
                feature_extractor=text2img.feature_extractor
    ).to("cuda")

    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = Image.fromarray(mask)
    with autocast("cuda"):
        image = inpaint(
                        prompt='',
                        image=image.resize((process_size, process_size), resample=Image.Resampling.LANCZOS),
                        mask_image=mask.resize((process_size, process_size)),
                        strength=strength,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        )["sample"]
    return image
'''


def break_line(line, break_threshold=0.25):
    line_len = len(line)
    if line_len < 2:
        return [line]

    w_max = float('-inf')
    for box in line:
        w_max = max([w_max, box[2]-box[0]+1])

    lines = []
    sub_line = [line[0]]
    for idx in range(1, line_len):
        gap = line[idx][0] - sub_line[-1][2]
        if gap > w_max * break_threshold:
            lines.append(sub_line)
            sub_line = [line[idx]]
        else:
            sub_line.append(line[idx])
            sub_line.sort(key=lambda box: box[2])
    lines.append(sub_line)

    return lines


def consolidate_line(boxes, line_threshold=0.75):
    boxes_len = len(boxes)
    if boxes_len < 2:
        return boxes

    boxes.sort(key=lambda box: box[1])
    sub_lines = []
    y_cnt, y_sum, y_avg, h_max = 1, boxes[0][1], float(boxes[0][1]), max([boxes[0][3]-boxes[0][1]+1, 1])
    line = [boxes[0]]
    for idx in range(1, boxes_len):
        y_dist = float(abs(boxes[idx][1] - y_avg)) / float(h_max)
        if y_dist > line_threshold:
            line.sort(key=lambda box: box[0])
            sub_lines += break_line(line)
            y_cnt, y_sum, y_avg, h_max = 1, boxes[idx][1], float(boxes[idx][1]), max([boxes[idx][3]-boxes[idx][1]+1, 1])
            line = [boxes[idx]]
        else:
            line.append(boxes[idx])
            y_cnt += 1
            y_sum += boxes[idx][1]
            y_avg = float(y_sum)/y_cnt
            h_max = max([h_max, boxes[idx][3]-boxes[idx][1]+1])
    line.sort(key=lambda box: box[0])
    sub_lines += break_line(line)

    boxes = []
    for sub_line in sub_lines:
        x0, y0, x1, y1 = float('inf'), float('inf'), float('-inf'), float('-inf')
        for box in sub_line:
            x0 = min([x0, box[0]])
            y0 = min([y0, box[1]])
            x1 = max([x1, box[2]])
            y1 = max([y1, box[3]])
        boxes.append([x0, y0, x1, y1])
    return boxes


def boxes_of_texts(original, original_for_draw):
    ocr = PaddleOCR(ocr_version='PP-OCRv3', det=True, rec=False, use_angle_cls=False, lang='en', use_gpu=True)

    t_text_start = time.time()
    w = original.shape[1]
    h = original.shape[0]
    if w < h:
        w_new = 256
        h_new = int(float(h) * (float(w_new) / float(w)))
    else:
        h_new = 256
        w_new = int(float(w) * (float(h_new) / float(h)))
    original_temp = cv2.resize(original, (w_new, h_new))
    cv2.imwrite('temp.png', original_temp)

    result = ocr.ocr('temp.png', det=True, rec=False, cls=False)

    boxes = []
    for res in result:
        for line in res:
            #box = line[0]
            x0 = line[0][0] / float(w_new) * float(w)
            y0 = line[0][1] / float(h_new) * float(h)
            x1 = line[2][0] / float(w_new) * float(w)
            y1 = line[2][1] / float(h_new) * float(h)
            boxes.append([int(x0), int(y0), int(x1), int(y1)])

    boxes_new = consolidate_line(boxes)
    boxes = []
    draw = ImageDraw.Draw(original_for_draw)
    for box in boxes_new:
        x0 = box[0]
        y0 = box[1]
        x1 = box[2]
        y1 = box[3]
        boxes.append([int((x0+x1)/2), int((y0+y1)/2), int(x1-x0), int(y1-y0)])
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=(255, 0, 0), width=3)
        draw.rectangle([int((x0+x1)/2), int((y0+y1)/2), int((x0+x1)/2)+1, int((y0+y1)/2)+1], outline=(255, 0, 0), width=3)
    t_text_end = time.time()
    print('Text detection time: ', t_text_end-t_text_start)
    return boxes, original_for_draw


def boxes_of_faces(original, original_for_draw):
    device = torch.device('cuda:0')
    mtcnn = MTCNN(keep_all=True, device=device)

    t_face_start = time.time()
    height, width, depth = original.shape
    if height < width:
        h = 160
        ratio = float(h) / float(height)
        w = int(float(width) * ratio)
    else:
        w = 160
        ratio = float(w) / float(width)
        h = int(float(height) * ratio)
    original = cv2.resize(original, (w, h))
    original = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))

    faces, _ = mtcnn.detect(original)
    boxes = []
    draw = ImageDraw.Draw(original_for_draw)
    if faces is not None:
        for face in faces:
            x0, y0, x1, y1 = face.tolist()
            x0 /= ratio
            y0 /= ratio
            x1 /= ratio
            y1 /= ratio
            boxes.append([int((x0+x1)/2), int((y0+y1)/2), int(x1-x0), int(y1-y0)])
            draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=(0, 0, 255), width=3)
            draw.rectangle([int((x0+x1)/2), int((y0+y1)/2), int((x0+x1)/2)+1, int((y0+y1)/2)+1], outline=(0, 0, 255), width=3)
    t_face_end = time.time()
    print('Face detection time: ', t_face_end-t_face_start)
    return boxes, original_for_draw


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def boxes_of_saliencies(image, original, original_for_draw):
    net = U2NET(3,1)
    net.load_state_dict(torch.load('/home/Claude/SmartCropping/U_2_Net/saved_models/u2net/u2net.pth'))
    net.cuda()
    net.eval()

    t_saliency_start = time.time()
    test_salobj_dataset = SalObjDataset(img_name_list = [image],
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    for data_test in test_salobj_dataloader:
        inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)
    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, _, _, _, _, _, _= net(inputs_test)
    pred = d1[:,0,:,:]
    pred = normPRED(pred).squeeze().cpu().data.numpy()
    #print('saliency ratio: ', np.sum(pred>=0.5)/float(pred.shape[0])/float(pred.shape[1]))

    structure = np.ones((3, 3), dtype='int')
    labeled, n_components = label(pred>=0.5, structure)
    #print('n_components: ', n_components)

    boxes = []
    draw = ImageDraw.Draw(original_for_draw)
    for idx in range(1, n_components+1):
        y_label, x_label = np.where(labeled==idx)
        x0 = np.amin(x_label) / 320.0 * float(original.shape[1])
        y0 = np.amin(y_label) / 320.0 * float(original.shape[0])
        x1 = np.amax(x_label) / 320.0 * float(original.shape[1])
        y1 = np.amax(y_label) / 320.0 * float(original.shape[0])
        boxes.append([int((x0+x1)/2), int((y0+y1)/2), int(x1-x0), int(y1-y0)])
        draw.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=(0, 255, 0), width=3)
        draw.rectangle([int((x0+x1)/2), int((y0+y1)/2), int((x0+x1)/2)+1, int((y0+y1)/2)+1], outline=(0, 255, 0), width=3)
    t_saliency_end = time.time()
    print('Saliency detection time: ', t_saliency_end-t_saliency_start)
    return boxes, original_for_draw, cv2.resize(pred, (original.shape[1], original.shape[0]))


def boxes_center_max(boxes):
    weight = 0
    x = 0
    y = 0
    area_max = 0
    for box in boxes:
        cx, cy, w, h = box
        area = w * h
        weight += area
        x += cx * area
        y += cy * area
        if area > area_max:
            area_max = area
            boxes_max = list(box)
    boxes_center = [int(x/float(weight)), int(y/float(weight))]
    return boxes_center, boxes_max


def is_valid_center(boxes_center, boxes_max, target_width, target_height):
    return boxes_center[0] - target_width/2 <= boxes_max[0] - boxes_max[2]/2 \
           and boxes_center[0] + target_width/2 >= boxes_max[0] + boxes_max[2]/2 \
           and boxes_center[1] - target_height/2 <= boxes_max[1] - boxes_max[3]/2 \
           and boxes_center[1] + target_height/2 >= boxes_max[1] + boxes_max[3]/2


def auto_center(image, original, target_width, target_height, text_prioritized, face_prioritized):
    height, width, _ = original.shape
    original_for_draw = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    texts = faces = []
    #texts, original_for_draw = boxes_of_texts(original, original_for_draw)
    #faces, original_for_draw = boxes_of_faces(original, original_for_draw)
    saliencies, original_for_draw, _ = boxes_of_saliencies(image, original, original_for_draw)

    if len(texts) == 0 and len(faces) == 0 and len(saliencies) == 0:
        center = [width//2, height//2]
    else:
        count = 0
        center = np.array([0.0, 0.0])
        if len(texts) > 0:
            count += 1
            texts_center, texts_max = boxes_center_max(texts)
            center += np.array(texts_center)
        if len(faces) > 0:
            count += 1
            faces_center, faces_max = boxes_center_max(faces)
            center += np.array(faces_center)
        if len(saliencies) > 0:
            count += 1
            saliencies_center, saliencies_max = boxes_center_max(saliencies)
            center += np.array(saliencies_center)
        center /= float(count)
        center = [int(center[0]), int(center[1])]

        if text_prioritized and len(texts) > 0:
            if not is_valid_center(center, texts_max, target_width, target_height):
                center = list(texts_center)
                if not is_valid_center(center, texts_max, target_width, target_height):
                    center = texts_max[:2]
        elif face_prioritized and len(faces) > 0:
            if not is_valid_center(center, faces_max, target_width, target_height):
                center = list(faces_center)
                if not is_valid_center(center, faces_max, target_width, target_height):
                    center = faces_max[:2]
        elif len(saliencies) > 0:
            if not is_valid_center(center, saliencies_max, target_width, target_height):
                center = list(saliencies_center)
                if not is_valid_center(center, saliencies_max, target_width, target_height):
                    center = saliencies_max[:2]

    return center, original_for_draw


def exact_crop(center, original_width, original_height, target_width, target_height):
    top = max(center[1] - math.floor(target_height / 2), 0)
    offset_h = top + target_height
    if offset_h > original_height:
        # overflowing
        # print("Top side over by ", offsetH - original_height)
        top = top - (offset_h - original_height)
    top = max(top, 0)
    bottom = min(offset_h, original_height)

    left = max(center[0] - math.floor(target_width / 2), 0)
    offset_w = left + target_width
    if offset_w > original_width:
        # overflowing
        # print("Left side over by ", offsetW - original_width)
        left = left - (offset_w - original_width)
    left = max(left, 0)
    right = min(left + target_width, original_width)

    return {
        'left': left,
        'right': right,
        'top': top,
        'bottom': bottom
    }


def smart_crop(image, target_width, target_height, destination, do_resize, text_prioritized, face_prioritized, draw_bboxes):
    original = np.array(Image.open(image).convert('RGB'))[:, :, ::-1].copy() 

    if original is None:
        print("Could not read source image")
        exit(1)

    target_height = int(target_height)
    target_width = int(target_width)

    if do_resize:
        original = auto_resize(original, target_width, target_height)

    height, width, _ = original.shape

    '''
    height_outpaint, width_outpaint = int(height*1.25), int(width*1.25)
    image_outpaint = np.zeros((height_outpaint, width_outpaint, 3)).astype('uint8')
    image_outpaint[height_outpaint//2-height//2:height_outpaint//2-height//2+height, width_outpaint//2-width//2:width_outpaint//2-width//2+width, :] = original.copy()
    mask = np.ones((height_outpaint, width_outpaint)).astype('uint8') * 255
    mask[height_outpaint//2-height//2:height_outpaint//2-height//2+height, width_outpaint//2-width//2:width_outpaint//2-width//2+width] = 0
    height_outpaint = outpaint(image_outpaint, mask)
    cv2.imwrite(destination, image_outpaint)
    '''

    if target_height > height:
        print('Warning: target higher than image')

    if target_width > width:
        print('Warning: target wider than image')

    center, original_for_draw = auto_center(image, original, target_width, target_height, text_prioritized, face_prioritized)
    crop_pos = exact_crop(center, width, height, target_width, target_height)
    if draw_bboxes:
        original_for_draw = np.array(original_for_draw.convert('RGB'))[:, :, ::-1].copy() 
        cropped = original_for_draw[int(crop_pos['top']): int(crop_pos['bottom']), int(crop_pos['left']): int(crop_pos['right'])]
    else:
        cropped = original[int(crop_pos['top']): int(crop_pos['bottom']), int(crop_pos['left']): int(crop_pos['right'])]
    cv2.imwrite(destination, cropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-W", "--width", required=True, help="Target width")
    ap.add_argument("-H", "--height", required=True, help="Target height")
    ap.add_argument("-i", "--image", required=True, help="Image to crop")
    ap.add_argument("-o", "--output", required=True, help="Output")
    ap.add_argument("-n", "--no-resize", required=False, default=False, action="store_true",
                    help="Don't resize image before treating it")
    ap.add_argument("-t", "--text-prioritized", required=False, default=False, action="store_true",
                    help="Text regions are prioritized over other face or salient regions")
    ap.add_argument("-f", "--face-prioritized", required=False, default=False, action="store_true",
                    help="Face regions are prioritized over other salient regions")
    ap.add_argument("-draw", "--draw-bboxes", required=False, default=False, action="store_true",
                    help="Draw bboxes of texts, faces, and salient objects")
    print(sys.argv[1:])
    args = vars(ap.parse_args())
    print(args)

    smart_crop(args["image"], args["width"], args["height"], args["output"], not args["no_resize"], args["text_prioritized"], args["face_prioritized"], args["draw_bboxes"])


if __name__ == '__main__':
    main()
