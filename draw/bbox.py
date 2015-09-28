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
from fuel.datasets.cifar10 import CIFAR10
from attention import ZoomableAttentionWindow

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_file", help="filename of a pickled Locator model")
    args = parser.parse_args()

    with open('Locator-Absolute-2015-09-28/Locator-Cross-Entropy_model_10.pkl', "rb") as f:
        model = pickle.load(f)
    locator = model.get_top_bricks()[0]

    img_height, img_width = locator.reader.img_height, locator.reader.img_width
    N = locator.reader.N
    n_iter = locator.n_iter

    svhn = SVHN(which_sets=['train'], height=img_height, width=img_width, N=N, n_iter=n_iter, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))
    svhn = Flatten(DataStream.default_stream(svhn, iteration_scheme=SequentialScheme(svhn.num_examples, 1)))
    svhn.get_epoch_iterator()

    for i in range(1, 105):
        image = svhn.get_data()

        im = Image.fromarray(image[4].reshape(img_height, img_width, 3), 'RGB')
        draw = ImageDraw.Draw(im)
        half = image[3][n_iter - 1] / 2 * (N - 1) * max(img_width, img_height)
        draw.rectangle([(image[1][n_iter - 1] * img_width - half, image[2][n_iter - 1] * img_height - half), (image[1][n_iter - 1] * img_width + half, image[2][n_iter - 1] * img_height + half)],
                       outline=(0, 255, 0))

        x = T.matrix("features")
        batch_size = T.iscalar('batch_size')

        center_y, center_x, delta = locator.find(x, batch_size)

        do_sample = theano.function([x, batch_size], outputs=[center_y, center_x, delta], allow_input_downcast=True)
        center_y, center_x, delta = do_sample(image[0], 1)

        for c_y, c_x, d in zip(center_y[-1:], center_x[-1:], delta[-1:]):
            half = d / 2 * max(img_width, img_height) * (N - 1)
            draw.rectangle([(c_x * img_width - half, c_y * img_height - half), (c_x * img_width + half, c_y * img_height + half)], outline=(255, 0, 0))
        del draw
        #
        fraction = 5
        im.resize((im.size[0] * fraction, im.size[1] * fraction)).save('Locator-Absolute-2015-09-28/%d.jpg' % i)
