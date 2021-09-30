# -*-coding: utf-8 -*-
import os, sys

sys.path.append(os.path.dirname(__file__))
import numpy as np
import cv2
import PIL.Image as Image
from alignment import align_trans


def show_landmark_boxes(win_name, image, landmarks_list, boxes):
    '''
    显示landmark和boxes
    :param win_name:
    :param image:
    :param landmarks_list: [[x1, y1], [x2, y2]]
    :param boxes:     [[ x1, y1, x2, y2],[ x1, y1, x2, y2]]
    :return:
    '''
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8
    for landmarks in landmarks_list:
        for landmark in landmarks:
            # 要画的点的坐标
            point = (int(landmark[0]), int(landmark[1]))
            cv2.circle(image, point, point_size, point_color, thickness)
    show_image_boxes(win_name, image, boxes)


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


if __name__ == "__main__":
    image_path = "./test.jpg"
    image = Image.open(image_path)
    # face detection from MTCNN
    bbox_score = np.asarray([[69.48486808, 58.12609892, 173.92575279, 201.95947894, 0.99979943]])
    landmarks = np.asarray([[[103.97721, 119.6718],
                             [152.35837, 113.06249],
                             [136.67535, 142.62952],
                             [112.62607, 171.1305],
                             [154.60092, 165.12515]]])

    bboxes = bbox_score[:, :4]
    scores = bbox_score[:, 4:]
    # show bbox_score and bounding boxes
    show_landmark_boxes("image", np.array(image), landmarks, bboxes)
    output_size = [112, 112]
    # get reference facial points
    refrence = align_trans.get_reference_facial_points(output_size, default_square=True)

    # face alignment and crop face roi
    faces_list = []
    for landmark in landmarks:
        warped_face = align_trans.warp_and_crop_face(np.array(image), landmark, refrence, crop_size=output_size)
        faces_list.append(warped_face)

    # show alignment face
    for face in faces_list:
        face = Image.fromarray(face)
        face.show("alignment-face")
