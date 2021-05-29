# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import sys
import os
import torchvision
import torch
import numpy as np



def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        pass


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def mean_accuracy(ground_truths, predictions):
    ground_truths = np.array(ground_truths)
    predictions = np.array(predictions)

    class_acc0 = np.sum(
        ground_truths[ground_truths == 0] == predictions[ground_truths == 0]) / np.sum(ground_truths == 0)
    class_acc1 = np.sum(
        ground_truths[ground_truths == 1] == predictions[ground_truths == 1]) / np.sum(ground_truths == 1)
    return class_acc0, class_acc1, (class_acc0+class_acc1) / 2


def load_pretrain_vgg16(model,pretrain=False):
    vgg16_bn = torchvision.models.vgg16_bn(pretrained=pretrain)
    model.copy_params_from_vgg16_bn(vgg16_bn)


def imsave(file_name, img):
    """
    save a torch tensor as an image
    :param file_name: 'image/folder/image_name'
    :param img: c*h*w torch tensor
    :return: nothing
    """
    assert(type(img) == torch.FloatTensor,
           'img must be a torch.FloatTensor')
    ndim = len(img.size())
    assert(ndim == 2 or ndim == 3,
           'img must be a 2 or 3 dimensional tensor')

    img = img.numpy()

    if ndim == 3:
        plt.imsave(file_name, np.transpose(img, (1, 2, 0)))
    else:
        plt.imsave(file_name, img, cmap='gray')


class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            # self.file = open(fpath, 'w')
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


