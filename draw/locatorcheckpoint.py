from __future__ import division, print_function

import logging
import os
from PIL import Image, ImageDraw

from fuel.transformers import Flatten
import theano
import theano.tensor as T
import numpy as np
from blocks.config import config
from blocks.extensions.saveload import Checkpoint
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme

from svhn import SVHN

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


class LocatorCheckpoint(Checkpoint):
    def __init__(self, save_subdir, img_height, img_width, N, batch_size, **kwargs):
        super(LocatorCheckpoint, self).__init__(path=None, **kwargs)
        self.save_subdir = save_subdir
        self.img_height = img_height
        self.img_width = img_width
        self.N = N
        self.batch_size = batch_size
        svhn = SVHN(which_sets=['extra'], height=img_height, width=img_width, N=N, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))
        self.svhn = Flatten(DataStream.default_stream(svhn, iteration_scheme=SequentialScheme(svhn.num_examples, batch_size)))
        self.svhn.get_epoch_iterator()
        self.iteration = 1

    def do(self, callback_name, *args):
        locator = self.main_loop.model.get_top_bricks()[0]
        # reset the random generator
        # del locator._theano_rng
        # del locator._theano_seed
        # locator.seed_rng = np.random.RandomState(config.default_seed)

        # ------------------------------------------------------------
        logging.info("Compiling locator function...")

        x = T.matrix("features")
        batch_size = T.iscalar('batch_size')

        center_y, center_x, delta = locator.find(x, batch_size)

        do_sample = theano.function([x, batch_size], outputs=[center_y, center_x, delta], allow_input_downcast=True)

        # ------------------------------------------------------------
        logging.info("Locating and saving images...")

        image = self.svhn.get_data()
        center_y, center_x, delta = do_sample(image[0], self.batch_size)

        im = Image.fromarray(image[4].reshape(self.img_height, self.img_width, 3), 'RGB')
        draw = ImageDraw.Draw(im)
        half = image[3] * self.N / 2
        draw.rectangle([(image[1] - half, max(image[2] - half, 0)), (image[1] + half, min(image[2] + half, self.img_height - 1))], outline=(0, 255, 0))

        for c_y, c_x, d in zip(center_y[-1:], center_x[-1:, ], delta[-1,]):
            draw.rectangle([(c_x - d / 2, c_y - d / 2), (c_x + d / 2, c_y + d / 2)], outline=(255, 0, 0))
        del draw

        fraction = 5
        im.resize((im.size[0] * fraction, im.size[1] * fraction)).save(os.path.join(self.save_subdir, '%d.jpg' % self.iteration))

        self.iteration += 1
