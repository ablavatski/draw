#!/usr/bin/env python

from __future__ import division, print_function

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import fuel
import os
import time
import cPickle as pickle
from argparse import ArgumentParser

from fuel.streams import DataStream
from fuel.transformers import Flatten, FilterSources
from fuel.schemes import SequentialScheme
from svhn import SVHN

from blocks.algorithms import *
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop

from blocks.bricks.cost import *
from blocks.bricks.recurrent import LSTM

from model import *
from draw import *
from partsonlycheckpoint import PartsOnlyCheckpoint
from locatorcheckpoint import LocatorCheckpoint
from plot import Plot
import numpy as np
from PIL import Image, ImageDraw
import scipy as sc

fuel.config.floatX = theano.config.floatX


# ----------------------------------------------------------------------------
def main(name, epochs, batch_size, learning_rate, read_N, n_iter, enc_dim, dec_dim, oldmodel):
    channels, img_height, img_width = 3, 54, 54

    rnninits = {
        # 'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    inits = {
        # 'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.001),
        'biases_init': Constant(0.),
    }
    x_dim = channels * img_height * img_width

    read_dim = channels * read_N ** 2

    subdir = name + "-" + time.strftime("%Y-%m-%d")
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    print("\nRunning experiment")
    print("          subdirectory: %s" % subdir)
    print("         learning rate: %g" % learning_rate)
    print("           attention N: %d" % read_N)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("     decoder dimension: %d" % dec_dim)
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    reader = LocatorReader(x_dim=x_dim, dec_dim=dec_dim, channels=channels, width=img_width, height=img_height, N=read_N, **inits)
    encoder_mlp = MLP([Identity()], [(read_dim + dec_dim), 4 * enc_dim], name="MLP_enc", **inits)
    decoder_mlp = MLP([Identity()], [enc_dim, 4 * dec_dim], name="MLP_dec", **inits)
    encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
    decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
    simple_locator = SimpleLocatorModel2LSTM(n_iter, reader=reader, encoder_mlp=encoder_mlp, encoder_rnn=encoder_rnn, decoder_mlp=decoder_mlp, decoder_rnn=decoder_rnn)
    simple_locator.initialize()

    # reader = LocatorReader(x_dim=x_dim, dec_dim=dec_dim, channels=channels, width=img_width, height=img_height, N=read_N, **inits)
    # location_mlp = MLP([Identity()], [5, dec_dim], name="MLP_loc", **inits)
    # representation_mlp = MLP([Identity()], [read_dim, dec_dim], name="MLP_repr", **inits)
    #
    # representer = Representer(representation_mlp, **inits)
    # locater = Locater(location_mlp, **inits)
    #
    # decoder_mlp = MLP([Identity()], [2 * dec_dim, 4 * dec_dim], name="MLP_dec", **inits)
    # decoder_rnn = LSTM(activation=Tanh(), dim=dec_dim, name="RNN_dec", **rnninits)
    # simple_locator = SimpleLocatorModel1LSTM(n_iter, reader=reader, locater=locater, representer=representer, decoder_mlp=decoder_mlp, decoder_rnn=decoder_rnn)
    # simple_locator.initialize()

    # ------------------------------------------------------------------------
    x = tensor.fmatrix('features')

    # def detect_nan(i, node, fn):
    #     for output in fn.outputs:
    #         if (not isinstance(output[0], np.random.RandomState) and np.isnan(output[0]).any()):
    #             print('*** NaN detected ***')
    #             theano.printing.debugprint(node)
    #             print('Inputs : %s' % [input[0] for input in fn.inputs])
    #             print('Outputs: %s' % [output[0] for output in fn.outputs])
    #             break
    #
    # svhn = SVHN(which_sets=['train'], height=img_height, width=img_width, N=read_N, n_iter=n_iter, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))
    # svhn = Flatten(DataStream.default_stream(svhn, iteration_scheme=SequentialScheme(svhn.num_examples, 3)))
    # svhn.get_epoch_iterator()
    # image = svhn.get_data()
    #
    # batch_size = T.iscalar('batch_size')
    #
    # center_y, center_x, delta = simple_locator.find(x, batch_size)
    #
    # do_sample = theano.function([x, batch_size], outputs=[center_y, center_x, delta], mode=theano.compile.MonitorMode(post_func=detect_nan))
    # center_y, center_x, delta = do_sample(image[0], 3)
    #
    # im = image[0].reshape([channels, img_height, img_height])
    # im = im.transpose([1, 2, 0])
    # # sc.misc.imsave('1.png', im / im.max())
    # import pylab
    # pylab.figure()
    # pylab.gray()
    # pylab.imshow(im / im.max(), interpolation='nearest')
    # pylab.show(block=True)
    # return

    center_y, center_x, delta = simple_locator.calculate(x)

    orig_y = tensor.fmatrix('bbox_tops')
    orig_x = tensor.fmatrix('bbox_lefts')
    orig_d = tensor.fmatrix('bbox_widths')

    cost = BinaryCrossEntropy().apply(tensor.concatenate([center_y, center_x, delta]), tensor.concatenate([orig_y, orig_x, orig_d]))
    cost.name = "loss_function"

    # ------------------------------------------------------------

    cg = ComputationGraph([cost])

    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost,
        on_unused_sources='warn',
        parameters=params,
        step_rule=CompositeRule([
            RemoveNotFinite(),
            StepClipping(10.),
            Adam(learning_rate),
            # RMSProp(learning_rate=learning_rate),
            # Momentum(learning_rate=learning_rate, momentum=0.95)
        ]),
        # theano_func_kwargs={'mode': theano.compile.MonitorMode(post_func=detect_nan)}
    )


    # ------------------------------------------------------------------------
    # Setup monitors
    monitors = [cost]
    # for v in [center_y, center_x, log_delta, log_sigma, log_gamma]:
    #    v_mean = v.mean()
    #    v_mean.name = v.name
    #    monitors += [v_mean]
    #    monitors += [aggregation.mean(v)]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]

    # Live plotting...
    plot_channels = [
        ["train_loss_function", "test_loss_function"],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    # ------------------------------------------------------------
    plotting_extensions = [
        Plot(name, channels=plot_channels, start_server=False, server_url='http://localhost:5006/')
    ]

    # ------------------------------------------------------------
    svhn_train = SVHN(which_sets=['extra'], height=img_height, width=img_width, N=read_N, n_iter=n_iter, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))
    svhn_test = SVHN(which_sets=['test'], height=img_height, width=img_width, N=read_N, n_iter=n_iter, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))

    main_loop = MainLoop(
        model=Model(cost),
        data_stream=Flatten(
            FilterSources(DataStream.default_stream(svhn_train, iteration_scheme=SequentialScheme(svhn_train.num_examples, batch_size)), ('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths'))
        ),
        algorithm=algorithm,
        extensions=[
                       Timing(),
                       FinishAfter(after_n_epochs=epochs),
                       TrainingDataMonitoring(
                           train_monitors,
                           prefix="train",
                           after_epoch=True),
                       DataStreamMonitoring(
                           monitors,
                           Flatten(
                               FilterSources(DataStream.default_stream(svhn_test, iteration_scheme=SequentialScheme(svhn_test.num_examples, batch_size)),
                                             ('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths'))
                           ),
                           prefix="test"),
                       PartsOnlyCheckpoint("{}/{}".format(subdir, name), before_training=False, after_epoch=True, save_separately=['model']),
                       # LocatorCheckpoint(save_subdir=subdir, img_height=img_height, img_width=img_width, N=read_N, batch_size=1, before_training=False, after_epoch=True),
                       ProgressBar(),
                       Printing()] + plotting_extensions)
    if oldmodel is not None:
        print("Initializing parameters with old model %s" % oldmodel)
        with open(oldmodel, "rb") as f:
            oldmodel = pickle.load(f)
            main_loop.model.set_parameter_values(oldmodel.get_param_values())
        del oldmodel
    main_loop.run()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                        default='Locator-Cross-Entropy', help="Name for this experiment")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=500, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=240, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-3, help="Learning rate")
    parser.add_argument("--read_N", "-a", type=int,
                        default=36, help="Use attention mechanism")
    parser.add_argument("--niter", type=int, dest="n_iter",
                        default=8, help="No. of iterations")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                        default=256, help="Encoder RNN state dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                        default=256, help="Decoder  RNN state dimension")
    parser.add_argument("--oldmodel", type=str, help="Use a model pkl file created by a previous run as a starting point for all parameters")
    args = parser.parse_args()

    main(**vars(args))
