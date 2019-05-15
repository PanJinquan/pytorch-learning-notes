# -*-coding: utf-8 -*-
"""
    @Project: UNet
    @File   : PairRandomCrop.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-04-23 10:29:25
"""


class PairRandomCrop(object):
    """Crop the given PIL.Image at a random location.
    ** This is a MODIFIED version **, which supports identical random crop for
    both image and target map in Semantic Segmentation.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    image_crop_position = {}

    def __init__(self, size, padding=0):
        import random
        import os
        import numbers

        from PIL import ImageOps

        self.os = os
        self.random = random
        self.ImageOps = ImageOps

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = self.ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        pid = self.os.getpid()
        if pid in self.image_crop_position:
            x1, y1 = self.image_crop_position.pop(pid)
        else:
            x1 = self.random.randint(0, w - tw)
            y1 = self.random.randint(0, h - th)
            self.image_crop_position[pid] = (x1, y1)
        return img.crop((x1, y1, x1 + tw, y1 + th))