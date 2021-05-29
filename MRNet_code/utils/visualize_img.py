# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from utils.utils import imsave
from config import Config


feature_map_out_path = Config.SAVE_FEATURE_MAP


def plot_img_and_mask(img, mask):
    classes = mask.shape[2] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i+1].set_title(f'Output mask (class {i+1})')
            ax[i+1].imshow(mask[:, :, i])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()

    return 0



def plot_feature_map_channel(features,is_show=False,is_save=True):
    n, c, h, w = features.size()
    fig, ax = plt.subplots(1, c)
    for i in range(c):
        out = features[0][i].cpu().data.resize_(h, w)
        imsave(os.path.join(feature_map_out_path, 'channel_' + str(i) + '.png'), out) if is_save else None
        if is_show:
            ax[i].set_title(f'Feature map (channel {i})')
            ax[i].imshow(features[0][i])
            # ax[i].imshow(features[0][i], cmap='Blues')

    if is_show:
        plt.xticks([]), plt.yticks([])
        plt.show()

    return 0











