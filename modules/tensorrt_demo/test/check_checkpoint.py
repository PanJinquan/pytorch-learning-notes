# -*-coding: utf-8 -*-

import sys

sys.path.append('..')

import torch
import logging
from net.model_irse import IR_18


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
        if state_dict[s].dtype not in [torch.int64, torch.float16,torch.float32]:
            logging.critical(
                "Wrong type `{}` of `{}`, expected `torch.int64 or torch.float16,torch.float32`.".format(state_dict[s].dtype, s))
            flag = False
    return flag


def pass_all_test(flag_list):
    for flag in flag_list:
        if flag == False:
            raise SystemExit('Can not Past All Tests!')


def main():
    ## Args
    CHECKPOINT_PATH = '../XMC2-Rec_face_recognition.pth.tar'
    DEFAULT_DICT = {'arch': 'IR_18',
                    'backbone_name': 'IR_18',
                    'state_dict': '',
                    'input_shape': (-1, 3, 64, 64),
                    'output_shape': (-1, 256)
                    }
    device = "cuda:0"
    input_shape = DEFAULT_DICT["input_shape"]
    output_shape = DEFAULT_DICT["output_shape"]
    input_size = [input_shape[2], input_shape[3]]  # 模型输入大小
    embedding_size = output_shape[1]
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model = IR_18(input_size, embedding_size)
    model = model.to(device)
    model.eval()
    flag_list = list()
    flag = check_model_key(checkpoint, DEFAULT_DICT)

    if flag:
        logging.info("Pass the test of model keys!")
    flag_list.append(flag)

    flag = check_weight_type(checkpoint)
    if flag:
        logging.info("Pass the test of weight type!")
    flag_list.append(flag)

    try:
        pass_all_test(flag_list)
    except:
        logging.info("Fail to pass all test!")
        sys.exit(1)
    else:
        logging.info("Succeed to pass all test!")
        sys.exit(0)



if __name__ == "__main__":
    # LOG_PATH = str(sys.argv[1])
    LOG_PATH="log"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

    logging.basicConfig(filename=LOG_PATH, filemode='a', level=logging.DEBUG, format=LOG_FORMAT)
    logging.info("Check Model File:")
    main()

    # log_file.close()
