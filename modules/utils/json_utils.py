# -*- coding:utf-8 -*-
"""
提供工具函数的模块
"""

import logging
import yaml

log = logging.getLogger(__name__)


class Dict2Obj:
    '''
    dict转类对象
    '''

    def __init__(self, args):
        self.__dict__.update(args)


def load_config(config_file='config.yaml'):
    """
    读取配置文件，并返回一个python dict 对象
    :param config_file: 配置文件路径
    :return: python dict 对象
    """
    with open(config_file, 'r', encoding="UTF-8") as stream:
        try:
            config = yaml.load(stream, Loader=yaml.FullLoader)
            # config = Dict2Obj(config)
        except yaml.YAMLError as e:
            print(e)
            return None
    return config


if __name__ == '__main__':
    pass
