import os
import json
import argparse
from selenium import webdriver
from selenium.webdriver import Chrome

from LayoutDETR.gen_single_sample_API import load_model as load_model_layoutdetr
from utils.util import safeMakeDirs
from InstructPix2Pix.gen_single_sample_API import load_model as load_model_instructpix2pix
from RetrieveAdapter.gen_single_sample_API import load_model as load_model_retrieveadapter
from LayoutDETR.gen_single_sample_API import generate_banners as generate_banners_layoutdetr
from InstructPix2Pix.gen_single_sample_API import generate_banners as generate_banners_instructpix2pix
from RetrieveAdapter.gen_single_sample_API import generate_banners as generate_banners_retrieveadapter



BANNER_GEN_MODEL_MAPPER = {'LayoutDETR': {'layout': 'ads_multi.pkl'},
                           'InstructPix2Pix': {'layout': 'instructpix2pix.ckpt'},
                           'RetrieveAdapter': {'superes': 'rdn-liif.pth', 'saliency': 'u2net.pth'}}

BROWSER_CONFIG = ['no-sandbox', 'disable-infobars', 'disable-dev-shm-usage', 'disable-browser-side-navigation',
                  'disable-gpu', 'disable-features=VizDisplayCompositor', 'headless']

def main():
    parser = argparse.ArgumentParser(description='Generate banners using one of the generation models')
    parser.add_argument('-mn', '--model_name', help='Model of choice: LayoutDETR, InstructPix2Pix, '
                                              'RetrieveAdapter', default='LayoutDETR')
    parser.add_argument('-mp', '--model_path', help='Path to all the model files', required=True)
    parser.add_argument('-im', '--image_path', help='Image used as the banner background or foreground',
                        default='test/data/example1/burning.jpg')
    parser.add_argument('-bcp', '--banner_content_path', help='Detailed specification of the banner content in json,'
                                                             'e.g. ad copy type, font family, font color, etc.',
                        default='test/data/example1/banner_content.json')
    parser.add_argument('-hdt', '--header_text', help='Banner header text.',
                        default='')
    parser.add_argument('-bdt', '--body_text', help='Banner body text..',
                        default='')
    parser.add_argument('-btt', '--button_text', help='Banner button text..',
                        default='')
    parser.add_argument('-pp', '--post_process', help='the dictionary of post-process method to its probability',
                        default={'jitter': 5 / 6, 'horizontal_center_aligned': 2 / 3, 'horizontal_left_aligned': 1 / 3})
    parser.add_argument('-nr', '--num_result', help='Number of banner images to be generated', default=3)
    parser.add_argument('-op', '--output_path', help='Relative path to store the generated banners.',
                        default='result')

    args = vars(parser.parse_args())
    cfd = os.path.dirname(os.path.abspath(__file__)) # current file directory
    # different test case for RetrieveAdapter
    if (args['model_name'] == 'RetrieveAdapter' and args['image_path'] == 'test/data/example1/burning.jpg'
            and args['banner_content_path'] == 'test/data/example1/banner_content.json'):
        args['image_path'] = 'test/data/example2/4_3-green-purple.png'
        args['banner_content_path'] = 'test/data/example2/banner_content.json'



    if args['model_name'] not in BANNER_GEN_MODEL_MAPPER.keys():
        print('Invalid model name')
        return

    if not os.path.exists(args['output_path']):
        safeMakeDirs(args['output_path'])
        # copy the css and fonts symbolic link to the output_path
        os.symlink(os.path.join(cfd, 'BannerGen/RetrieveAdapter/templates/css'),
                   os.path.join(args['output_path'], 'css'))

    if not os.path.exists(args['image_path']):
        print(args['image_path'])
        print('Please provide a valid image path.')
        return

    if not os.path.isdir(args['model_path']):
        print('Please provide a valid model path.')
        return

    if not os.path.exists(args['banner_content_path']) or os.path.splitext(args['banner_content_path'])[1] != '.json':
        print('Please provide a valid banner content json file.')
        return

    with open(args['banner_content_path'], 'r') as fp:
        banner_content = json.load(fp)

    if args['header_text'] or args['body_text'] or args['button_text']:
        for ele in banner_content['contentStyle']['elements']:
            ele['text'] = ''
            if ele['type'] == 'header' and args['header_text']:
                ele['text'] = args['header_text']
            elif ele['type'] == 'body' and args['body_text']:
                ele['text'] = args['body_text']
            elif ele['type'] == 'button' and args['button_text']:
                ele['text'] = args['button_text']

    options = webdriver.ChromeOptions()
    for opt in BROWSER_CONFIG:
        options.add_argument(opt)

    browser = Chrome(executable_path='/usr/bin/chromedriver', options=options)
    browser.set_window_size(2000, 2000)  # depending on hardware config, larger browser resolution may cause OOM
    safeMakeDirs(args['output_path'])
    seeds = [x + 1 for x in range(int(args['num_result']))]
    print(os.path.join(os.path.abspath(os.getcwd()), args['output_path']))
    screenshot_paths = html_paths = []

    if args['model_name'] == 'LayoutDETR':
        model = load_model_layoutdetr(os.path.join(args['model_path'], BANNER_GEN_MODEL_MAPPER[args['model_name']]
            ['layout']))
        screenshot_paths, html_paths = generate_banners_layoutdetr(model, args['image_path'],
                                                                   banner_content['contentStyle']['elements'],
                                                                   args['post_process'],
                                                                   seeds,browser,
                                                                   os.path.join(cfd, args['output_path']))

    elif args['model_name'] == 'InstructPix2Pix':
        model_instructpix2pix, model_wrap_cfg, model_wrap, null_token = load_model_instructpix2pix(
            os.path.join(args['model_path'], BANNER_GEN_MODEL_MAPPER[args['model_name']]['layout']))
        screenshot_paths, html_paths = generate_banners_instructpix2pix(model_instructpix2pix,
                                                                        model_wrap_cfg, model_wrap,
                                                                        null_token, args['image_path'],
                                                                        banner_content['contentStyle']['elements'],
                                                                        args['post_process'],
                                                                        seeds, browser,
                                                                        os.path.join(cfd, args['output_path']))
    elif args['model_name'] == 'RetrieveAdapter':
        model_superes, model_saliency, model_text, model_face = load_model_retrieveadapter(
            os.path.join(args['model_path'], BANNER_GEN_MODEL_MAPPER[args['model_name']]['superes']),
            os.path.join(args['model_path'], BANNER_GEN_MODEL_MAPPER[args['model_name']]['saliency']))
        screenshot_paths, html_paths = generate_banners_retrieveadapter(model_superes, model_saliency, model_text,
                                                                        model_face, args['image_path'],
                                                                        banner_content['contentStyle']['elements'],
                                                                        seeds, browser,
                                                                        os.path.join(cfd, args['output_path']))
    else:
        print('Unknown model name')
        return

    print(f'Generated banner screenshot paths:\n {screenshot_paths}')
    print(f'Generated banner html paths:\n {html_paths}')


if __name__ == '__main__':
    main()
