# encoding: utf-8
import os
import math
import random
import pickle

from PIL import Image
import numpy as np
import cv2


def build_random_replace_background(cfg):
    masks_path = cfg.DATASETS.MASKS_PATH
    if masks_path is None or masks_path == "":
        return None
    masks = load_masks(cfg.DATASETS.MASKS_PATH)
    return RandomReplaceBackground(masks, probability=cfg.DATASETS.BACKGROUND_REPLACE_PROBABILITY)


def load_masks(masks_path):
    """
    masks file format: dict
      - key: image_name
      - value: mask
      - example
        {
            "4627_c8_f0118398.jpg": numpy.ndarray([[True, True, ...], ..., [True, True, ...]]),
            "0580_c1_f0154127.jpg": numpy.ndarray([[True, True, ...], ..., [True, True, ...]]),
            ...
        }
    """
    with open(masks_path, "rb") as f:
        masks = pickle.load(f)
    return masks


class RandomReplaceBackground(object):
    """
    Args:
         masks: dataset image person masks
         probability: The probability that the Random Erasing operation will be performed.
    """

    def __init__(self, masks, probability=0.5):
        self.masks = masks
        self.probability = probability

    def __call__(self, image, image_name):
        if random.uniform(0, 1) > self.probability:
            return image
        if image_name not in self.masks:
            return image

        mask = self.masks[image_name]
        image = np.asarray(image)
        height, width = image.shape[:2]

        background = self.create_random_background(width, height)
        image = self.replace_background(mask, image, background)
        image = Image.fromarray(image)
        return image

    def create_random_background(self, width, height):
        background = np.zeros((height, width, 3), dtype=np.uint8)
        r = int(random.uniform(0, 256))
        g = int(random.uniform(0, 256))
        b = int(random.uniform(0, 256))
        background[:, :] = (r, g, b)
        return background

    def replace_background(self, mask, image, background):
        assert mask.shape == image.shape[:2] == background.shape[:2]

        mask = np.stack((mask,) * 3, axis=-1)  # convert to 3-channel
        mask = mask.astype(np.float)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        image = image / 255
        background = background / 255
        fg = image * mask
        bg = background * (1 - mask)
        output = fg + bg
        output *= 255
        output = output.astype(np.uint8)

        return output

