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

from U_2_Net.model import U2NET # full size version 173.6 MB
from U_2_Net.data_loader import SalObjDataset, RescaleT, ToTensorLab
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from scipy.ndimage.measurements import label

import time


# normalize the predicted SOD probability map
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn


def boxes_from_faces(original):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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


def center_from_saliency(image, original, target_width, target_height):
    net = U2NET(3,1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load('U_2_Net/saved_models/u2net/u2net.pth'))
        net.cuda()
    else:
        net.load_state_dict(torch.load('U_2_Net/saved_models/u2net/u2net.pth', map_location='cpu'))
    net.eval()

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
    print('saliency ratio: ', np.sum(pred>=0.5)/float(pred.shape[0])/float(pred.shape[1]))

    structure = np.ones((3, 3), dtype='int')
    labeled, n_components = label(pred>=0.5, structure)
    print('n_components: ', n_components)
    if n_components == 0:
        return None

    weight = 0
    x = 0
    y = 0
    area_max = 0
    for idx in range(1, n_components+1):
        y_label, x_label = np.where(labeled==idx)
        x0 = np.amin(x_label) / 320.0 * original.shape[1]
        y0 = np.amin(y_label) / 320.0 * original.shape[0]
        x1 = np.amax(x_label) / 320.0 * original.shape[1]
        y1 = np.amax(y_label) / 320.0 * original.shape[0]
        cx = int((x0 + x1) / 2)
        cy = int((y0 + y1) / 2)
        w = int(x1 - x0)
        h = int(y1 - y0)
        box = [cx, cy, w, h]
        area = len(y_label)
        weight += area
        x += cx * weight
        y += cy * weight
        if area > area_max:
            area_max = area
            box_max = list(box)
    center = [int(x/float(weight)), int(y/float(weight))]
    if center[0] - target_width/2 > box_max[0] - box_max[2]/2 or center[0] + target_width/2 < box_max[0] + box_max[2]/2 or center[1] - target_height/2 > box_max[1] - box_max[3]/2 or center[1] + target_height/2 < box_max[1] + box_max[3]/2:
        center = box_max[:2]

    return center


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
    height = image.shape[0]
    width = image.shape[1]

    ratio = target_width / width
    w, h = width * ratio, height * ratio
    p = 1

    # if there is still height or width to compensate, let's do it
    if w - target_width < 0 or h - target_height < 0:
        ratio = max(target_width / w, target_height / h)
        w, h = w * ratio, h * ratio
        p = 2

    image = cv2.resize(image, (int(w), int(h)))
    return image

def auto_center(image, original, target_width, target_height, face_prioritized):
    height, width, _ = original.shape

    t_face_start = time.time()
    faces = boxes_from_faces(original)
    t_face_end = time.time()
    print('Face detection time: ', t_face_end-t_face_start)

    if len(faces) == 0:
        center = center_from_saliency(image, original, target_width, target_height)
        if center is None:
            center = [width//2, height//2]
    else:
        t_others_start = time.time()
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

        t_saliency_start = time.time()
        features_center = center_from_saliency(image, original, target_width, target_height)
        t_saliency_end = time.time()
        print('Saliency detection time: ', t_saliency_end-t_saliency_start)
        if features_center is None:
            center = faces_center
        else:
            center = [(faces_center[0]+features_center[0])//2, (faces_center[1]+features_center[1])//2]

        if face_prioritized:
            if center[0] - target_width/2 > face_max[0] - face_max[2]/2 or center[0] + target_width/2 < face_max[0] + face_max[2]/2 or center[1] - target_height/2 > face_max[1] - face_max[3]/2 or center[1] + target_height/2 < face_max[1] + face_max[3]/2:
                center = faces_center
                if center[0] - target_width/2 > face_max[0] - face_max[2]/2 or center[0] + target_width/2 < face_max[0] + face_max[2]/2 or center[1] - target_height/2 > face_max[1] - face_max[3]/2 or center[1] + target_height/2 < face_max[1] + face_max[3]/2:
                    center = face_max[:2]
        t_others_end = time.time()
        print('Others time: ', t_others_end-t_others_start-(t_saliency_end-t_saliency_start))
        print('Face center', faces_center)
        print('Feat center', features_center)

    return center


def smart_crop(image, target_width, target_height, destination, do_resize, face_prioritized):
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

    center = auto_center(image, original, target_width, target_height, face_prioritized)
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
    ap.add_argument("-f", "--face-prioritized", required=False, default=False, action="store_true",
                    help="Face regions are prioritized over other salient regions")

    args = vars(ap.parse_args())

    smart_crop(args["image"], args["width"], args["height"], args["output"], not args["no_resize"], args["face_prioritized"])


if __name__ == '__main__':
    main()
