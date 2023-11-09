from __future__ import print_function
from __future__ import division

import cv2
from PIL import Image
import numpy as np
import argparse
import os
import math

from facenet_pytorch import MTCNN
import torch

# Algorithm parameters
FEATURE_DETECT_MAX_CORNERS = 50
FEATURE_DETECT_QUALITY_LEVEL = 0.1
FEATURE_DETECT_MIN_DISTANCE = 10


def boxes_from_faces(original):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(keep_all=True, device=device)

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
    if faces is not None:
        for face in faces:
            x0, y0, x1, y1 = face.tolist()
            boxes.append([int((x0+x1)/2/ratio), int((y0+y1)/2/ratio), int((x1-x0)/ratio), int((y1-y0)/ratio)])

    return boxes


def center_from_good_features(original):
    x, y = (0, 0)
    weight = 0
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(original, FEATURE_DETECT_MAX_CORNERS, FEATURE_DETECT_QUALITY_LEVEL,
                                      FEATURE_DETECT_MIN_DISTANCE)

    for point in corners:
        weight += 1
        x += point[0][0]
        y += point[0][1]

    return [int(x/float(weight)), int(y/float(weight))]


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


def auto_resize(image, target_width, target_height):
    height, width, _ = image.shape

    ratio = target_width / width
    w, h = width * ratio, height * ratio
    p = 1

    # if there is still height or width to compensate, let's do it
    if w - target_width < 0 or h - target_height < 0:
        ratio = max(target_width / w, target_height / h)
        w, h = w * ratio, h * ratio
        p = 2

    image = cv2.resize(image, (int(w), int(h)))
    print("Image resized by", w - width, "*", h - height, "in", p, "pass(es)")

    return image

def auto_center(original, target_width, target_height):
    faces = boxes_from_faces(original)

    if len(faces) == 0:
        print('Using Good Feature Tracking method')
        center = center_from_good_features(original)
    else:
        weight = 0
        x = 0
        y = 0
        area_max = 0
        for face in faces:
            cx, cy, w, h = face
            area = w * h
            weight += area
            x += cx * area
            y += cy * area
            if area > area_max:
                area_max = area
                face_max = list(face)
        faces_center = [int(x/float(weight)), int(y/float(weight))]
        print('Combining with Good Feature Tracking method')
        features_center = center_from_good_features(original)
        center = [(faces_center[0]+features_center[0])//2, (faces_center[1]+features_center[1])//2]

        if center[0] - target_width/2 > face_max[0] - face_max[2]/2 or center[0] + target_width/2 < face_max[0] + face_max[2]/2 or center[1] - target_height/2 > face_max[1] - face_max[3]/2 or center[1] + target_height/2 < face_max[1] + face_max[3]/2:
            center = faces_center
        print('Face center', faces_center)
        print('Feat center', features_center)

    return center


def smart_crop(image, target_width, target_height, destination, do_resize):
    original = np.array(Image.open(image).convert('RGB'))[:, :, ::-1].copy() 

    if original is None:
        print("Could not read source image")
        exit(1)

    target_height = int(target_height)
    target_width = int(target_width)

    if do_resize:
        original = auto_resize(original, target_width, target_height)

    height, width, _ = original.shape

    if target_height > height:
        print('Warning: target higher than image')

    if target_width > width:
        print('Warning: target wider than image')

    center = auto_center(original, target_width, target_height)
    crop_pos = exact_crop(center, width, height, target_width, target_height)
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

    args = vars(ap.parse_args())

    smart_crop(args["image"], args["width"], args["height"], args["output"], not args["no_resize"])


if __name__ == '__main__':
    main()
