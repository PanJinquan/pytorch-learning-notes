# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : dataset.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-03-07 18:45:06
"""
import os
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import sys
import PIL.Image as Image
import numpy as np
import cv2
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from utils import image_processing, file_processing


def make_weights_for_balanced_classes(images, nclasses):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]

    return weight


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader=datasets.folder.default_loader, transform=None, target_transform=None,
                 is_valid_file=None):
        super(DatasetFolder, self).__init__(root)
        extensions = IMG_EXTENSIONS if is_valid_file is None else None
        self.transform = transform
        self.target_transform = target_transform
        # classes, class_to_idx = self._find_classes(self.root)
        classes, class_to_idx = self._find_classes_idx(self.root)

        imgs = datasets.folder.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.imgs = imgs
        self.targets = [s[1] for s in imgs]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _find_classes_idx(self, dir, class_prefix="", start_idx=0):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    image_dir1 = "/media/dm/dm1/project/InsightFace_Pytorch/custom_insightFace/data/facebank"
    # image_dir2 = "/media/dm/dm1/project/InsightFace_Pytorch/custom_insightFace/data/faces_emore/imgs"
    # 图像预处理Rescale，RandomCrop，ToTensor
    input_size = [112, 112]
    image_dir_list = [image_dir1]
    train_transform = transforms.Compose([
        transforms.Resize((input_size[0], input_size[1])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    PIN_MEMORY = True
    NUM_WORKERS = 2
    DROP_LAST = True
    dataset_train = DatasetFolder(root=image_dir1, transform=train_transform)
    print("num classs:{}".format(dataset_train.classes))
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    # 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    dataloader = DataLoader(dataset_train, batch_size=8, sampler=sampler, pin_memory=PIN_MEMORY,
                            num_workers=NUM_WORKERS, drop_last=DROP_LAST, shuffle=False)
    for batch_image, batch_label in iter(dataloader):
        image = batch_image[0, :]
        # image = image.numpy()  #
        image = np.array(image, dtype=np.float32)
        image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        print("batch_image.shape:{},batch_label:{}".format(batch_image.shape, batch_label))
        image_processing.cv_show_image("image", image)
        # batch_x, batch_y = Variable(batch_x), Variable(batch_y)
