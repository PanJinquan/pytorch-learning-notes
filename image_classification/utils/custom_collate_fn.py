# -*-coding: utf-8 -*-
"""
    @Project: pytorch-learning-tutorials
    @File   : SimpleCustomBatch.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-06-14 18:51:31
"""


def dict_value_legal(dict_data):
    '''
    判断dict_data中是的value是否合法，比如是否存在tuple、list、dict为空、None的情况
    PS：判断tuple、list、dict类型是否为空，可以直接使用if not v，单输入的数据v可能是numpy或者其他类型，因此不能直接if not v
    :param data:
    :return:存在为True,否则为False
    '''
    for v in list(dict_data.values()):
        if v is None or v == [] :  # fix a numpy bug:if not v
            return False
    return True


def collate_fn_dict(batch):
    '''
    通过dict返回batch数据
    :param batch:
    :return:
    '''
    out_batch = {}
    for data in batch:
        # 判断value数据中是否合法，否则过滤掉
        if not dict_value_legal(data):
            continue
        # 按照dict的key进行拼接数据
        for key, value in data.items():
            if key not in out_batch.keys():
                out_batch[key] = [value]
            else:
                out_batch[key] += [value]
    return out_batch


def collate_fn_raw(batch):
    '''
    不进行任何拼接处理，直接返回原始数据
    :param batch:
    :return:
    '''
    return batch
