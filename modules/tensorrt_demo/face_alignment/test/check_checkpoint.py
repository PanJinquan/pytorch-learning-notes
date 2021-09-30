# -*-coding: utf-8 -*-

import sys

sys.path.append('..')
sys.path.append('../alignment')
import torch
import logging
import numpy as np
import cv2
import torch
import PIL.Image as Image
import demo
from alignment import align_trans


def check_model_key(checkpoint, default_dict):
    flag = True
    for k in default_dict.keys():
        if k not in checkpoint:
            logging.critical("Missing key `{}` in.".format(k))
            flag = False
        if k != 'state_dict':
            if default_dict[k] != checkpoint.get(k):
                logging.critical(
                    "Wrong Value `{}` of `{}`, expected `{}`.".format(checkpoint.get(k), k, default_dict[k]))
                flag = False

    return flag


def check_weight_type(checkpoint):
    flag = True
    state_dict = checkpoint.get('state_dict')
    for s in state_dict:
        if state_dict[s].dtype not in [torch.int64, torch.float16, torch.float32]:
            logging.critical(
                "Wrong type `{}` of `{}`, expected `torch.int64 or torch.float16,torch.float32`.".format(
                    state_dict[s].dtype, s))
            flag = False
    return flag


def pass_all_test(flag_list):
    for flag in flag_list:
        if flag == False:
            raise SystemExit('Can not Past All Tests!')


def main():
    try:
        image_path = "../test.jpg"
        image = Image.open(image_path)
        # face detection from MTCNN
        bbox_score = np.asarray([[69.48486808, 58.12609892, 173.92575279, 201.95947894, 0.99979943]])
        landmarks = np.asarray([[[103.97721, 119.6718],
                                 [152.35837, 113.06249],
                                 [136.67535, 142.62952],
                                 [112.62607, 171.1305],
                                 [154.60092, 165.12515]]])
        logging.info("landmarks  :\n{}".format(landmarks))
        logging.info("bbox_score :\n{}".format(bbox_score))

        bboxes = bbox_score[:, :4]
        scores = bbox_score[:, 4:]
        # show bbox_score and bounding boxes
        # show_landmark_boxes("image", np.array(image), landmarks, bboxes)
        output_size = [112, 112]
        # get reference facial points
        refrence = align_trans.get_reference_facial_points(output_size, default_square=True)

        # face alignment and crop face roi
        faces_list = []
        for landmark in landmarks:
            warped_face = align_trans.warp_and_crop_face(np.array(image), landmark, refrence, crop_size=output_size)
            faces_list.append(warped_face)

        logging.info("warped_face :{}".format(len(faces_list)))
    except:
        logging.info("Fail to pass all test!")
        sys.exit(1)
    else:
        logging.info("Succeed to pass all test!")
        sys.exit(0)


if __name__ == "__main__":
    # LOG_PATH = str(sys.argv[1])
    LOG_PATH = "log"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    logging.basicConfig(filename=LOG_PATH, filemode='a', level=logging.DEBUG, format=LOG_FORMAT)
    logging.info("Check face alignment:")
    main()

    # log_file.close()
