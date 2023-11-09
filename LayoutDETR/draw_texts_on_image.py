# ref https://www.alpharithms.com/fit-custom-font-wrapped-text-image-python-pillow-552321/

import argparse
import json
import math
import os.path
import textwrap
from string import ascii_letters

from PIL import Image, ImageDraw, ImageFont


def readJSON(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image_path', help='path to image file')
    parser.add_argument('--output', help='output dir')
    return parser.parse_args()


def get_font_area_to_size_dict(font_path, min_size=1, max_size=101):
    area_to_size = {}

    for cur_size in range(min_size, max_size):
        font = ImageFont.truetype(font=font_path, size=cur_size)
        avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
        avg_char_height = sum(font.getsize(char)[1] for char in ascii_letters) / len(ascii_letters)

        area_to_size[avg_char_width * avg_char_height] = cur_size
    return area_to_size


# https://www.geeksforgeeks.org/find-closest-number-array/
# Python3 program to find element
# closest to given target.

# Returns element closest to target in arr[]
def findClosest(arr, n, target):
    # Corner cases
    if (target <= arr[0]):
        return arr[0]
    if (target >= arr[n - 1]):
        return arr[n - 1]

    # Doing binary search
    i = 0;
    j = n;
    mid = 0
    while (i < j):
        mid = (i + j) // 2

        if (arr[mid] == target):
            return arr[mid]

        # If target is less than array
        # element, then search in left
        if (target < arr[mid]):

            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > arr[mid - 1]):
                return getClosest(arr[mid - 1], arr[mid], target)

            # Repeat for left half
            j = mid

        # If target is greater than mid
        else:
            if (mid < n - 1 and target < arr[mid + 1]):
                return getClosest(arr[mid], arr[mid + 1], target)

            # update i
            i = mid + 1

    # Only single element left after search
    return arr[mid]


# Method to compare which one is the more close.
# We find the closest by taking the difference
# between the target and both values. It assumes
# that val2 is greater than val1 and target lies
# between these two.
def getClosest(val1, val2, target):
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1


def draw_text(areas, area2size, font_path, draw, text, bbox, color, threshold):
    num_chars = len(text)

    target_area = math.floor(bbox[2] * bbox[3] / float(num_chars) * threshold)
    closest_area = findClosest(areas, len(areas), target_area)
    best_font_size = area2size[closest_area]
    font = ImageFont.truetype(font=font_path, size=best_font_size)

    avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
    max_char_count = int(float(bbox[2]) / avg_char_width)

    scaled_wrapped_text = textwrap.fill(text=text, width=max_char_count)
    draw.text(xy=(bbox[0], bbox[1]), text=scaled_wrapped_text, font=font, fill=color)  # COLOR


def xyxy2bbox(xyxy_list):
    x1, y1, x2, y2 = xyxy_list
    return [x1, y1, x2 - x1 + 1, y2 - y1 + 1]


def safeMakeDirs(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir)
        except:
            print('Failed to make dirs at {}'.format(dir))


def main():
    args = parse_args()
    output_dir = args.output

    safeMakeDirs(output_dir)

    background_dir = "/export/share/ning/projects/datasets/ads_banner_collection/LaMa_stringOnly_inpainted_background_images"
    detection_dir = "/export/share/datasets/vision/AI4SW/content_generation/filtered_ads/pseudo_labels/Fortune_500"
    background_img_path = os.path.join(background_dir, "CBRE Group_0013_mask001.png")

    detection_json_path = os.path.join(detection_dir, "out_CBRE Group_0013.json")
    detection_image_path = os.path.join(detection_dir, "out_CBRE Group_0013.png")

    font_path = "/export/share/zeyuan/sample_images/cg/Source_Code_Pro/static/arial.ttf"

    json_data = readJSON(detection_json_path)

    detection_image = Image.open(fp=detection_image_path, mode='r')
    tgt_width, tgt_height = detection_image.size

    # read & resize the background image
    img = Image.open(fp=background_img_path, mode='r')
    img = img.resize((tgt_width, tgt_height))

    bbox = xyxy2bbox(json_data[0]['xyxy'])
    area2size = get_font_area_to_size_dict(font_path)
    areas = [area for area in area2size]

    draw = ImageDraw.Draw(im=img)
    threshold = 0.65

    text1 = "CBRE INDUSTRIAL & LOGISTICS SELEZIONE INTELLIGENTE DEL SITO"
    bbox1 = xyxy2bbox(json_data[1]['xyxy'])
    draw_text(areas, area2size, font_path, draw, text1, bbox1, color='#ffffff', threshold=threshold)

    text2 = "ALIMENTARE L'IMMOBILIARE CHE ALIMENTA IL BUSINESS"
    bbox2 = xyxy2bbox(json_data[0]['xyxy'])
    draw_text(areas, area2size, font_path, draw, text2, bbox2, color='#bfd958', threshold=threshold)

    save_path = os.path.join(output_dir,
                             f'CBRE_Italian_inpaint_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}_{threshold}.jpg')
    img.save(save_path)
    print(f'Saved image to {save_path}')


if __name__ == '__main__':
    main()
