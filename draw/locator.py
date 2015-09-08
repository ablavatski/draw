#!/usr/bin/env python

from __future__ import division, print_function

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import fuel
import os
import time

from argparse import ArgumentParser

from fuel.streams import DataStream
from fuel.transformers import Flatten, FilterSources
from fuel.schemes import SequentialScheme
from svhn import SVHN

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, Adam, RemoveNotFinite
from blocks.initialization import Constant, IsotropicGaussian
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
from plot import Plot

fuel.config.floatX = theano.config.floatX


# ----------------------------------------------------------------------------
def main(name, epochs, batch_size, n_iter, learning_rate):
    channels, img_height, img_width = 1, 28, 58

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

    read_N = 12

    read_dim = channels * read_N ** 2

    enc_dim = 256
    dec_dim = 256
    z_dim = 100

    subdir = name + "-" + time.strftime("%Y-%m-%d")
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    print("\nRunning experiment")
    print("          subdirectory: %s" % subdir)
    print("         learning rate: %g" % learning_rate)
    print("           attention N: %d" % read_N)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    reader = LocatorReader(x_dim=x_dim, dec_dim=dec_dim, channels=channels, width=img_width, height=img_height, N=read_N, **inits)
    writer = LocatorWriter(input_dim=dec_dim, output_dim=3, channels=channels, width=img_width, height=img_height, N=read_N, **inits)

    encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
    decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
    encoder_mlp = MLP([Identity()], [(read_dim + dec_dim), 4 * enc_dim], name="MLP_enc", **inits)
    decoder_mlp = MLP([Identity()], [z_dim, 4 * dec_dim], name="MLP_dec", **inits)
    l_sampler = LocatorSampler(input_dim=enc_dim, output_dim=z_dim, **inits)

    locator = LocatorModel(
        n_iter,
        reader=reader,
        encoder_mlp=encoder_mlp,
        encoder_rnn=encoder_rnn,
        sampler=l_sampler,
        decoder_mlp=decoder_mlp,
        decoder_rnn=decoder_rnn,
        writer=writer)
    locator.initialize()


    # ------------------------------------------------------------------------
    x = tensor.matrix('features')

    center_y, center_x, delta = locator.calculate(x)

    orig_y = tensor.col('bbox_tops')
    orig_x = tensor.col('bbox_lefts')
    orig_d = tensor.col('bbox_widths')

    cost = BinaryCrossEntropy().apply(tensor.concatenate([center_y, center_x, delta]), tensor.concatenate([orig_y, orig_x, orig_d]))
    cost.name = "squared_error"

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
        ])
        # step_rule=RMSProp(learning_rate),
        # step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
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
        ["train_squared_error", "test_squared_error"],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    # ------------------------------------------------------------
    plotting_extensions = [
        Plot(name, channels=plot_channels, server_url='http://localhost:5006')
    ]

    # ------------------------------------------------------------
    svhn_train = SVHN(which_sets=['train'], height=img_height, width=img_width, N=read_N, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))
    svhn_test = SVHN(which_sets=['test'], height=img_height, width=img_width, N=read_N, sources=('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights'))

    main_loop = MainLoop(
        model=Model(cost),
        data_stream=Flatten(
            FilterSources(DataStream.default_stream(svhn_train, iteration_scheme=SequentialScheme(svhn_train.num_examples, batch_size)), ('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths'))
        ),
        algorithm=algorithm,
        extensions=[
                       Timing(),
                       FinishAfter(after_n_epochs=epochs),
                       DataStreamMonitoring(
                           monitors,
                           Flatten(
                               FilterSources(DataStream.default_stream(svhn_test, iteration_scheme=SequentialScheme(svhn_test.num_examples, batch_size)),
                                             ('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths'))
                           ),
                           prefix="test"),
                       TrainingDataMonitoring(
                           train_monitors,
                           prefix="train",
                           after_epoch=True),
                       PartsOnlyCheckpoint("{}/{}".format(subdir, name), before_training=True, after_epoch=True, save_separately=['log', 'model']),
                       # SampleCheckpoint(image_size=img_height, channels=channels, save_subdir=subdir, before_training=True, after_epoch=True),
                       ProgressBar(),
                       Printing()] + plotting_extensions)
    main_loop.run()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                        default=None, help="Name for this experiment")
    parser.add_argument("--epochs", type=int, dest="epochs",
                        default=25, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                        default=100, help="Size of each mini-batch")
    parser.add_argument("--niter", type=int, dest="n_iter",
                        default=10, help="No. of iterations")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                        default=1e-3, help="Learning rate")
    args = parser.parse_args()

    main(**vars(args))
