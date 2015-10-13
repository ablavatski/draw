from __future__ import division, print_function

import logging

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)
try:
    from bokeh.plotting import (curdoc, cursession, figure, output_server,
                                push, show)

    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False
from argparse import ArgumentParser
from os import listdir
from os.path import isfile, join
import cPickle as pickle
from blocks.main_loop import MainLoop
from blocks.log.log import TrainingLogBase
import re

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
channels = [
    ["train_loss_function", "test_loss_function"],
    ["train_total_gradient_norm", "train_total_step_norm"]
]


def main(document, server_url):
    plots = {}

    output_server(document, url=server_url)

    # Create figures for each group of channels
    p = []
    p_indices = {}
    for i, channel_set in enumerate(channels):
        channel_set_opts = {}
        if isinstance(channel_set, dict):
            channel_set_opts = channel_set
            channel_set = channel_set_opts.pop('channels')
        channel_set_opts.setdefault('title',
                                    '{} #{}'.format(document, i + 1))
        p.append(figure(**channel_set_opts))
        for channel in channel_set:
            p_indices[channel] = i

    files = [f for f in listdir(document) if isfile(join(document, f)) and '_log_' in f]

    for file_name in sorted(files, key=lambda s: int(re.findall(r'\d+', s)[1])):
        with open(join(document, file_name), "rb") as f:
            log = pickle.load(f)

        iteration = log.status['iterations_done']
        i = 0
        for key, value in log.current_row.items():
            if key in p_indices:
                if key not in plots:
                    fig = p[p_indices[key]]
                    fig.line([iteration], [value], legend=key,
                             x_axis_label='iterations',
                             y_axis_label='value', name=key,
                             line_color=colors[i % len(colors)])
                    i += 1
                    renderer = fig.select(dict(name=key))
                    plots[key] = renderer[0].data_source
                else:
                    plots[key].data['x'].append(iteration)
                    plots[key].data['y'].append(value)

                    cursession().store_objects(plots[key])
        push()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="document",
                        default='Locator-16-Iter-2015-10-01', help="Folder name")
    parser.add_argument("-server_url", type=str, dest="server_url",
                        default="http://155.69.151.184:80/", help="Server url")
    args = parser.parse_args()

    main(**vars(args))
