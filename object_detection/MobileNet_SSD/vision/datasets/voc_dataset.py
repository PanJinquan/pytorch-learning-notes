import numpy as np
import pathlib
import xml.etree.ElementTree as ET
import cv2


class VOCDataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "ImageSets/Main/test.txt"
        else:
            image_sets_file = self.root / "ImageSets/Main/trainval.txt"
        self.ids = VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        self.class_names = ('BACKGROUND',
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train', 'tvmonitor'
        )
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_id)
        # # 测试读取的图片
        # print(" boxes.shape:{},boxes:{}".format(boxes.shape,boxes))
        # print("labels.shape:{},labels:{}".format(labels.shape,labels))
        # image_processing.show_image_boxes("image",image,boxes)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.root / f"Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            bbox = object.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            boxes.append([x1, y1, x2, y2])
            labels.append(self.class_dict[class_name])
            is_difficult_str = object.find('difficult').text
            is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self.root / f"JPEGImages/{image_id}.jpg"
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


from torch.utils.data import DataLoader, ConcatDataset
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.ssd import MatchPrior
from utils import image_processing
device='cpu'
if __name__=='__main__':
    config = mobilenetv1_ssd_config
    dataset_path='E:\git\VOC0712\VOC2007'
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    train_dataset = VOCDataset(dataset_path, transform=train_transform,target_transform=target_transform)
    train_loader = DataLoader(train_dataset, batch_size=2,num_workers=0,shuffle=False)
    for i, data in enumerate(train_loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        # image = images[0, :]
        # image = image.numpy()  # image=np.array(image)
        # image = image.transpose(1, 2, 0)  # 通道由[c,h,w]->[h,w,c]
        # image_processing.cv_show_image("image", image)
        # print("boxes:{},labels:{}".format(boxes,labels))
