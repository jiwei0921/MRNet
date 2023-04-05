import torchvision.transforms as transforms
import torch
from PIL import Image, ImageOps
import random
import utils.utils2.transforms as local_transforms

"""
As mentioned in http://pytorch.org/docs/master/torchvision/models.html

All pre-trained models expect input images normalized in the same way, i.e. mini-batches 
of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 
224. The images have to be loaded in to a range of [0, 1] and then normalized using 
ean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 

NOTE: transforms.ToTensor() transforms the incoming data to range of [0, 1]. It also
converts [H x W x C] to [C x H x W], which is expected by PyTorch models.
"""

# For now we will use PyTorch model zoo models
pytorch_zoo_normaliser = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

# inception_normaliser = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


class MyRandomCrop(object):

    def __init__(self, size):
        """
        This is a variant of torchvision's RandomCrop. This one pads image only if
        the image is smaller than the intended size. Image will be padded to the
        right and bottom.

        :param size: tuple (width, height)
        """
        self.size = size

    def __call__(self, img):

        width, height = img.size
        target_width, target_height = self.size

        pad_width = 0
        pad_height = 0
        do_padding = False

        if width < target_width:
            pad_width = target_width - width
            do_padding = True

        if height < target_height:
            pad_height = target_height - height
            do_padding = True

        #
        pad = (0, 0, pad_width, pad_height)

        if do_padding:
            img = ImageOps.expand(img, border=pad, fill=0)
            width, height = img.size

        if width == target_width and height == target_height:
            return img

        x1 = random.randint(0, width - target_width)
        y1 = random.randint(0, height - target_height)

        return img.crop((x1, y1, x1 + target_width, y1 + target_height))


def get_transformer_crop(crop_img_size,  # 224 or more expected by PyTorch model zoo
                    scale_img_size,
                    normaliser = pytorch_zoo_normaliser,
                    do_augment=False):

    if do_augment:
        # This is a augmented transformation,
        return transforms.Compose([transforms.Scale(scale_img_size),
                                   MyRandomCrop((crop_img_size, crop_img_size)),
                                   transforms.RandomHorizontalFlip(),
                                   local_transforms.ColorJitter(0.4, 0.4, 0.4, 0),
                                   # TODO - Add more transformations
                                   transforms.ToTensor(),
                                   normaliser])
    else:
        # This is a vanilla transformation
        return transforms.Compose([transforms.Scale(scale_img_size),
                                   MyRandomCrop((crop_img_size, crop_img_size)),
                                   transforms.ToTensor(),
                                   normaliser])


def get_transformer(img_size,  # 224 or more expected by PyTorch model zoo
                    normaliser = pytorch_zoo_normaliser,
                    do_augment=False):

    if do_augment:
        # This is a augmented transformation,
        return transforms.Compose([transforms.Scale((img_size, img_size)),
                                   transforms.RandomHorizontalFlip(),
                                   local_transforms.ColorJitter(0.4, 0.4, 0.4, 0),
                                   transforms.ToTensor(),
                                   normaliser])
    else:
        # This is a vanilla transformation
        return transforms.Compose([transforms.Scale((img_size, img_size)),
                                   transforms.ToTensor(),
                                   normaliser])




def get_test_valid_transformer_crop(crop_img_size,
                               scale_img_size,
                               normaliser=pytorch_zoo_normaliser):
    """Transformation for Validation and Test set"""

    # TODO, implement TTA

    # NOTE: With the below logic, one might want to do multiple inference on the same
    # image, because there is some randomness, we do not know how big the image is
    return transforms.Compose([transforms.Resize(scale_img_size),
                               MyRandomCrop((crop_img_size, crop_img_size)),
                               transforms.ToTensor(),
                               normaliser])


def get_test_valid_transformer(img_size,
                               normaliser=pytorch_zoo_normaliser):
    """Transformation for Validation and Test set"""

    # TODO, implement TTA

    # NOTE: With the below logic, one might want to do multiple inference on the same
    # image, because there is some randomness, we do not know how big the image is
    return transforms.Compose([transforms.Resize((img_size, img_size)),
                               transforms.ToTensor(),
                               normaliser])