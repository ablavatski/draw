from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle
import numpy as np
import scipy as sc
from PIL import Image, ImageDraw
from svhn import SVHN
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_file", help="filename of a pickled Locator model")
    parser.add_argument("--channels", type=int,
                        default=1, help="number of channels")
    parser.add_argument("--size", type=int,
                        default=28, help="Output image size (width and height)")
    args = parser.parse_args()

    with open('Locator-2015-09-14/Locator_model_71.pkl', "rb") as f:
        model = pickle.load(f)

    svhn = SVHN(which_sets=['extra'], height=28, width=58, N=10, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))
    svhn = Flatten(DataStream.default_stream(svhn, iteration_scheme=SequentialScheme(svhn.num_examples, 1)))
    svhn.get_epoch_iterator()
    for i in range(1, 102):
        image = svhn.get_data()

    im = Image.fromarray(image[4].reshape(28, 58, 3), 'RGB')
    draw = ImageDraw.Draw(im)
    half = image[3] * 10 / 2
    draw.rectangle([(image[1] - half, image[2] - half), (image[1] + half, image[2] + half)], outline=(0, 255, 0))

    x = T.matrix("features")
    batch_size = T.iscalar('batch_size')

    locator = model.get_top_bricks()[0]
    center_y, center_x, delta = locator.find(x, batch_size)

    do_sample = theano.function([x, batch_size], outputs=[center_y, center_x, delta], allow_input_downcast=True)
    center_y, center_x, delta = do_sample(image[0], 1)

    for c_y, c_x, d in zip(center_y[-1:], center_x[-1:], delta[-1:]):
        draw.rectangle([(c_x - d/2, c_y - d/2), (c_x + d/2, c_y + d/2)], outline=(255, 0, 0))
    del draw

    fraction = 5
    im.resize((im.size[0] * fraction, im.size[1] * fraction)).show()
