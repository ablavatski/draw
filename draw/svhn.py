import numpy as np
import scipy as sc
from fuel.datasets import H5PYDataset
from fuel.utils import find_in_data_path
from fuel.transformers import *
from scipy.misc import toimage


class SVHN(H5PYDataset):
    N_global = None
    height_global = None
    width_global = None
    n_iter_global = None

    def fix_representation(data):
        max_value = 0.999
        x, left, top, width, height = data

        new_x = []
        new_t = []
        new_l = []
        new_dt = []
        new_dl = []

        start_l = 0.5
        start_t = 0.5
        start_dt = 1. / N_global
        start_dl = 1. / N_global

        # start_t = 0.5 * height_global
        # start_l = 0.5 * width_global
        # start_d = 1. / N_global * max(width_global, height_global)

        for idx, image in enumerate(x):
            ratio_y = float(height_global) / image.shape[1]
            ratio_x = float(width_global) / image.shape[2]

            t = max(np.min(top[idx].astype(np.int16)), 0)
            l = max(np.min(left[idx].astype(np.int16)), 0)
            h = np.max([np.sum(coord) for coord in zip(top[idx].astype(np.int16), height[idx].astype(np.int16))]) - t
            w = np.max([np.sum(coord) for coord in zip(left[idx].astype(np.int16), width[idx].astype(np.int16))]) - l

            gt_t = min((t + (h + 1) / 2) * ratio_y / height_global, max_value)
            # gt_t = (t + h / 2) * ratio_y
            step = (start_t - gt_t) / n_iter_global
            new_t.append([(start_t - i * step) for i in range(1, n_iter_global + 1)])

            gt_l = min((l + (w + 1) / 2) * ratio_x / width_global, max_value)
            # gt_l = (l + w / 2) * ratio_x
            step = (start_l - gt_l) / n_iter_global
            new_l.append([(start_l - i * step) for i in range(1, n_iter_global + 1)])

            gt_dt = min(h * ratio_y / (N_global - 1) / (height_global - 1), max_value)
            # gt_d = float(max(h * ratio_y, w * ratio_x)) / (N_global)
            step = (start_dt - gt_dt) / n_iter_global
            new_dt.append([(start_dt - i * step) for i in range(1, n_iter_global + 1)])

            gt_dl = min(w * ratio_x / (N_global - 1) / (width_global - 1), max_value)
            # gt_d = float(max(h * ratio_y, w * ratio_x)) / (N_global)
            step = (start_dl - gt_dl) / n_iter_global
            new_dl.append([(start_dl - i * step) for i in range(1, n_iter_global + 1)])

            resized = sc.misc.imresize(image, (height_global, width_global))
            new_x.append([resized.transpose([2, 0, 1])])

        return np.array(new_x), np.array(new_l).transpose(), np.array(new_t).transpose(), np.array(new_dl).transpose(), np.array(new_dt).transpose()

    default_transformers = (
        (Mapping, [fix_representation], {}),
        (ScaleAndShift, [1 / 255.0, 0], {'which_sources': ('features',)}),
        (Cast, ['floatX'], {'which_sources': ('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths', 'bbox_heights')}),
    )

    def __init__(self, which_sets, height, width, N, n_iter, **kwargs):
        super(SVHN, self).__init__(file_or_path=find_in_data_path('svhn_format_1.hdf5'), which_sets=which_sets, **kwargs)
        global N_global, height_global, width_global, n_iter_global
        height_global = height
        width_global = width
        N_global = N
        n_iter_global = n_iter
