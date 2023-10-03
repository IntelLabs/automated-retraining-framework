# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: MIT License

import pandas as pd
from PIL import Image
from torchvision import transforms


class DistributionCompose(transforms.Compose):
    """Subclass of torchvision transforms.Compose which redefines transforms that are
    used to augment data samples.
    """

    def __call__(
        self,
        img: Image.Image,
        distribution_transforms: dict,
        info_df: pd.DataFrame,
        index: int,
    ):
        for t in self.transforms:
            if t.__class__.__name__ in distribution_transforms.keys():
                t.__init__(info_df[t.__class__.__name__].iloc[index])
            img = t(img)
        return img


class ColorAugmentation(object):
    """Custom data transform used to alter the color channel (R, G, B) of an image.
    The augmentation used a color tuple specifying the respective multiplier to apply to
    each color channel. To use an unmodified version of the image, the color multiplier
    values would be (1, 1, 1). A red image would be created by (1, 0, 0). Note that
    (0, 0, 0) will return an all black image. Mixtures of color multipliers may also be
    used.
    """

    def __init__(self, color=[1, 1, 1]):
        self.name = "ColorAugmentation"
        self.color = color

    def __call__(self, img):
        red, green, blue = self.color
        color_transform = (red, 0, 0, 0, 0, green, 0, 0, 0, 0, blue, 0)
        img = img.convert("RGB", color_transform)
        img = transforms.ToTensor()(img)
        return img

    def __repr__(self):
        red, green, blue = self.color
        return f"{self.name}(red={red}, green={green}, blue={blue})"
