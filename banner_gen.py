from LayoutDETR import gen_single_sample_API_server as layout_detr
from InstructPix2Pix import gen_single_sample_API_server as instruct_pix2pix
from RetrieveAdapter import gen_single_sample_API_server as retrieve_adapter
import os
import json
import argparse

banner_gen_mapper = {'layout_detr': layout_detr,
                     'instruct_pix2pix': instruct_pix2pix,
                     'retrieve_adapter': retrieve_adapter}

def main():
    parser = argparse.ArgumentParser(description='Generate banners using one of the generation models')
    parser.add_argument('-m', '--model', help='Model of choice: layout_detr, instruct_pix2pix, '
                                              'retrieve_adapter', default='layout_detr')
    parser.add_argument('-im', '--image', help='Image used as banner background or foreground', default='layout_detr')
    parser.add_argument('-ns', '--num_sample', help='Number of banner images to be generated', default=3)
    parser.add_argument('-mp', '--model_path', help='Path to all the model files',
                        default='/export/share/chiachihchen/BANNER/')
    parser.add_argument('-c', '--content', help='Detailed specification of the banner content,'
                                                'e.g. text type, font family, font color, etc.',
                        default='test/data/banner_content.json')
    parser.add_argument('-op', '--output_path', help='Path to store the generated banners',
                        default='./result')
    args = vars(parser.parse_args())
    if not os.path.exists(args['output_path']):
        os.makedirs(args['output_path'])

    with open(args['content'], 'r') as fp:
        content = json.load(fp)

    if args['model'] == 'layout_detr':
        # change banner content to layout_detr specs
        pass
    elif args['model'] == 'instruct_pix2pix':
        pass
    elif args['model'] == 'retrieve_adapter':
        pass
    else:
        print('Unknown model name')
        return

    banner_gen_mapper['model'](content)


if __name__ == '__main__':
    main()