# -*-coding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(__file__))
import copy
import numpy as np
import cv2
import torch
import PIL.Image as Image
from net import box_utils, onet_landmark_det


class ONetLandmarkDet():
    def __init__(self, onet_path, device):
        '''
        :param onet_path: model path
        :param device: cuda or cpu
        '''
        self.onet = onet_landmark_det.ONet(onet_path).to(device)
        self.onet.eval()
        self.device = device

    def get_faces_landmarks(self, faces):
        '''
        :param faces: face list
        :return:
        '''
        input_tensor = []
        for face in faces:
            height, width, depths = np.shape(face)
            xmin, ymin = 0, 0
            # resize image for net inputs
            input_face = cv2.resize(face, (48, 48))
            input_face = box_utils._preprocess(input_face)
            input_face = torch.FloatTensor(input_face)
            input_tensor.append(input_face)

        input_tensor = torch.cat(input_tensor)
        input_tensor = torch.FloatTensor(input_tensor).to(self.device)
        output = self.onet(input_tensor)
        landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        # compute landmark points in src face
        landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]
        # adapter landmarks
        landmarks_list = []
        for landmark in landmarks:
            face_landmarks = [[landmark[j], landmark[j + 5]] for j in range(5)]
            landmarks_list.append(face_landmarks)
        return landmarks_list


def show_landmark(win_name, img, landmarks_list):
    '''
    显示landmark
    :param win_name:
    :param image:
    :param landmarks_list: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    image = copy.deepcopy(img)
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    for landmarks in landmarks_list:
        for landmark in landmarks:
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, point_size, point_color, thickness)
    image = Image.fromarray(image)
    image.show(win_name)


def show_image_boxes(win_name, image, boxes_list):
    '''
    :param win_name:
    :param image:
    :param boxes_list:[[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    for box in boxes_list:
        x1, y1, x2, y2 = box
        point1 = (int(x1), int(y1))
        point2 = (int(x2), int(y2))
        cv2.rectangle(image, point1, point2, (0, 0, 255), thickness=2)
    image = Image.fromarray(image)
    image.show(win_name)


def main():
    image_path = "./test.jpg"
    onet_path = "./XMC2-landmark-detection.pth.tar"
    image = Image.open(image_path)
    image = np.array(image)
    # bbox=[xmin,ymin,xmax,ymax]
    face_bbox = [69, 58, 173, 201]
    xmin, ymin, xmax, ymax = face_bbox
    # cut face ROI
    face = image[ymin:ymax, xmin:xmax]
    # show face
    Image.fromarray(face).show("face")
    # show face bbox
    show_image_boxes("image", image, [face_bbox])
    # init landmark Detection
    lmdet = ONetLandmarkDet(onet_path, device="cuda:0")
    landmarks = lmdet.get_faces_landmarks([face])
    # show face landmarks
    show_landmark("face-landmark", face, landmarks)
    print("landmarks:{}".format(landmarks))


if __name__ == "__main__":
    main()
