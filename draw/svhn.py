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

    def fix_representation(data):
        x, left, top, width, height = data

        new_x = []
        new_t = []
        new_l = []
        new_d = []
        orig = []

        def rgb2gray(rgb):
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

        for idx, image in enumerate(x):
            ratio_y = float(height_global) / image.shape[1]
            ratio_x = float(width_global) / image.shape[2]

            t = np.min(top[idx])
            l = np.min(left[idx])
            h = np.max([np.sum(coord) for coord in zip(top[idx], height[idx])]) - t
            w = np.max([np.sum(coord) for coord in zip(left[idx], width[idx])]) - l

            new_t.append((t + h / 2) * ratio_y)
            new_l.append((l + w / 2) * ratio_x)
            new_d.append(float(max(h * ratio_y, w * ratio_x)) / N_global)

            resized = sc.misc.imresize(image, (height_global, width_global))
            new_x.append([rgb2gray(resized)])
            orig.append(resized)

        return np.array(new_x), np.array(new_l), np.array(new_t), np.array(new_d), np.array(orig)

    default_transformers = (
        (Mapping, [fix_representation], {}),
        (ScaleAndShift, [1 / 255.0, 0], {'which_sources': ('features',)}),
        (Cast, ['floatX'], {'which_sources': ('features', 'bbox_lefts', 'bbox_tops', 'bbox_widths')}),
    )

    def __init__(self, which_sets, height, width, N, **kwargs):
        super(SVHN, self).__init__(file_or_path=find_in_data_path('svhn_format_1.hdf5'), which_sets=which_sets, **kwargs)
        global N_global, height_global, width_global
        height_global = height
        width_global = width
        N_global = N
