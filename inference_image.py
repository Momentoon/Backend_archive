import os
import argparse
from inference import Transformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='/content/checkpoints')
    parser.add_argument('--src', type=str, default='/content/checkpoints', help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default='/content/images', help='destination dir to save generated images')

    return parser.parse_args()


def animeGAN(data, src, dest):
    transformer = Transformer(data)

    if os.path.exists(src) and not os.path.isfile(src):
        transformer.transform_in_dir(src, dest)
    else:
        transformer.transform_file(src, dest)
