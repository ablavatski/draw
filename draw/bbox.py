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
from evaluation import BoundingBox
import time

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model_file", help="filename of a pickled Locator model")
    args = parser.parse_args()

    with open('Locator-Absolute-2015-10-09/Locator-Absolute_model_203.pkl', "rb") as f:
        model = pickle.load(f)
    locator = model.get_top_bricks()[0]

    img_height, img_width = locator.reader.img_height, locator.reader.img_width
    N = locator.reader.N
    n_iter = locator.n_iter

    svhn = SVHN(which_sets=['train'], height=img_height, width=img_width, N=N, n_iter=n_iter, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))

    batch_size = 1000
    num_examples = int(svhn.num_examples / batch_size) + 1
    evaluation = True

    # num_examples = 100
    # batch_size = 1
    # evaluation = False

    svhn_stream = Flatten(DataStream.default_stream(svhn, iteration_scheme=SequentialScheme(svhn.num_examples, batch_size)))
    svhn_stream.get_epoch_iterator()

    x = T.fmatrix("features")
    batch_size = T.iscalar('batch_size')

    center_y, center_x, deltaY, deltaX = locator.find(x, batch_size)

    do_sample = theano.function([x, batch_size], outputs=[center_y, center_x, deltaY, deltaX], allow_input_downcast=True)

    overlap = .0
    distance = .0

    for i in range(0, num_examples):
        image = svhn_stream.get_data()

        half_x = image[3][n_iter - 1] / 2 * (N - 1) * (img_width - 1)
        half_y = image[4][n_iter - 1] / 2 * (N - 1) * (img_height - 1)
        x1 = image[1][n_iter - 1] * (img_width - 1) - half_x
        y1 = image[2][n_iter - 1] * (img_height - 1) - half_y
        w1 = 2 * half_x
        h1 = 2 * half_y

        if not evaluation:
            im = image[0].reshape([3, img_height, img_width]) * 255
            im = im.transpose([1, 2, 0]).astype('uint8')
            im = Image.fromarray(im, 'RGB')
            draw = ImageDraw.Draw(im)
            draw.rectangle([(x1, y1), (x1 + w1, y1 + h1)], outline=(0, 255, 0))

        center_y, center_x, deltaY, deltaX = do_sample(image[0], len(image[0]))

        for c_y, c_x, dy, dx in zip(center_y[-1:], center_x[-1:], deltaY[-1:], deltaX[-1:]):
            half_x = dx / 2 * (img_width - 1) * (N - 1)
            half_y = dy / 2 * (img_height - 1) * (N - 1)
            x2 = c_x * img_width - half_x
            y2 = c_y * img_height - half_y
            w2 = 2 * half_x
            h2 = 2 * half_y

        if not evaluation:
            draw.rectangle([(x2, y2), (x2 + w2, y2 + h2)], outline=(255, 0, 0))
            del draw
            fraction = 5
            im.resize((im.size[0] * fraction, im.size[1] * fraction)).save('Locator-Absolute-2015-10-09/%d.jpg' % i)

        else:
            for xg, yg, wg, hg, xt, yt, wt, ht in zip(x1, y1, w1, h1, x2, y2, w2, h2):
                gt = BoundingBox(int(xg), int(yg), int(wg), int(hg))
                test = BoundingBox(int(xt), int(yt), int(wt), int(ht))

                overlap += gt.relative_overlap(test)
                distance += gt.center_distance(test)

    if evaluation:
        print('Mean precision = %.2f' % (overlap * 100 / svhn.num_examples))
        print('Mean center distance = %.2f' % (distance / svhn.num_examples))
