# -*-coding: utf-8 -*-

import sys

sys.path.append('..')
sys.path.append('../alignment')
import os
import torch
import logging
import numpy as np
import cv2
import torch
import PIL.Image as Image
import demo


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
        onet_path = "../XMC2-landmark-detection.pth.tar"
        if not os.path.exists(onet_path):
            logging.info("no model path:{}".format(onet_path))
        image = Image.open(image_path)
        image = np.array(image)
        # bbox=[xmin,ymin,xmax,ymax]
        face_bbox = [69, 58, 173, 201]
        xmin, ymin, xmax, ymax = face_bbox
        # cut face ROI
        face = image[ymin:ymax, xmin:xmax]
        # show face
        # Image.fromarray(face).show("face")
        # show face bbox
        # demo.show_image_boxes("image", image, [face_bbox])
        # init landmark Detection
        lmdet = demo.ONetLandmarkDet(onet_path, device="cuda:0")
        logging.info("Succeed to load model:{}".format(onet_path))

        landmarks = lmdet.get_faces_landmarks([face])
        # show face landmarks
        # demo.show_landmark("face-landmark", face, landmarks)
        logging.info("landmarks:{}".format(landmarks))
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
    logging.info("Check Model File:")
    main()

    # log_file.close()
