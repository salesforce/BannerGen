import os
import json
import argparse

from LayoutDETR.e2e_pipeline.api_server import load_model as load_model_layoutdetr
#from InstructPix2Pix.e2e_pipeline.api_server import load_model as load_model_instructpix2pix
#from RetrieveAdapter.e2e_pipeline.api_server import load_model as load_model_retrieveadapter
from LayoutDETR.gen_single_sample_API_server import generate_banners as generate_banners_layoutdetr
from InstructPix2Pix.gen_single_sample_API_server import generate_banners as generate_banners_instructpix2pix
from RetrieveAdapter.gen_single_sample_API_server import generate_banners as generate_banners_retrieveadapter


banner_gen_function_mapper = {'LayoutDETR': generate_banners_layoutdetr,
                              'InstructPix2Pix': generate_banners_instructpix2pix,
                              'RetrieveAdapter': generate_banners_retrieveadapter}

banner_gen_model_mapper = {'LayoutDETR': {'layout': 'ads_multi.pkl'},
                           'InstructPix2Pix': {'layout': 'instructpix2pix.ckpt',
                                               'config': 'InstructPix2Pix/configs/generate.yaml'},
                           'RetrieveAdapter': {'superres': 'rdn-liif.pth', 'face': 'u2net.pth'}}

def main():
    parser = argparse.ArgumentParser(description='Generate banners using one of the generation models')
    parser.add_argument('-mn', '--model_name', help='Model of choice: LayoutDETR, InstructPix2Pix, '
                                              'RetrieveAdapter', default='LayoutDETR')
    parser.add_argument('-im', '--image', help='Image used as the banner background or foreground',
                        default='test/data/darkflooring.jpg')
    parser.add_argument('-ns', '--num_sample', help='Number of banner images to be generated', default=3)
    # default = '/export/share/chiachihchen/BANNERS/
    parser.add_argument('-mp', '--model_path', help='Path to all the model files', required=True)
    parser.add_argument('-bc', '--banner_content', help='Detailed specification of the banner content in json,'
                                                        'e.g. ad copy type, font family, font color, etc.',
                        default='test/data/banner_content.json')
    parser.add_argument('-op', '--output_path', help='Path to store the generated banners',
                        default='./result')
    args = vars(parser.parse_args())

    if args['model_name'] not in banner_gen_model_mapper.keys():
        print('Invalid model name')
        return

    if not os.path.exists(args['output_path']):
        os.makedirs(args['output_path'])

    if not os.path.exists(args['image']):
        print('Please provide a valid image path.')
        return

    if not os.path.isdir(args['model_path']):
        print('Please provide a valid model path.')
        return

    if not os.path.exists(args['banner_content']) or os.path.splitext(args['banner_content'])[1] != 'json':
        print('Please provide a valid banner content json file.')
        return

    with open(args['banner_content'], 'r') as fp:
        content = json.load(fp)



    if args['model_name'] == 'LayoutDETR':
        print(os.path.join(args['model_path']))
        model = load_model_layoutdetr(os.path.join())
        print(model)
    elif args['model_name'] == 'InstructPix2Pix':
        pass
    elif args['model_name'] == 'RetrieveAdapter':
        pass
    else:
        print('Unknown model name')
        return

    banner_gen_mapper['model'](content)


if __name__ == '__main__':
    main()