# -*- coding: utf-8 -*-
import numpy as np


# Configuration Class
class Config(object):

    MEAN_AND_STD = {'mean_rgb':np.array([0.485,0.456,0.406]),
                    'std_rgb':np.array([0.229,0.224,0.225])}

    # save visual feature map in this path
    SAVE_FEATURE_MAP = ''

    # set the size of test image
    SCALE_SIZE = 256

    # set your optimizer ['adam','sgd','rmsprop']
    OPTIMIZERS = 'adam'

    # initial learning rate
    LR = 0.0001

    # stepsize to decay learning rate (>0 means this is enabled)
    STEP_SIZE = -1

    # learning rate decay
    GAMMA = 0.1

    # weight decay (default: 5e-04)
    WEIGHT_DECAY = 5e-04


    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
