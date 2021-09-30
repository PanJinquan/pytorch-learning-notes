# -*-coding: utf-8 -*-
"""
    @Project: tools
    @File   : debug.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-05-10 16:24:49
"""
import datetime
import logging
from logging.handlers import TimedRotatingFileHandler


def set_logging(name, level="info", logfile=None):
    '''
    url:https://cuiqingcai.com/6080.html
    level级别：debug>info>warning>error>critical
    :param level: 设置log输出级别
    :param logfile: log保存路径，如果为None，则在控制台打印log
    :return:
    '''
    if logfile is None:
        handler = logging.StreamHandler()  # 创建一个handler，用于输出到控制台
    else:
        # 创建一个handler，用于写入日志文件, logging.FileHandler('test.log')
        handler = TimedRotatingFileHandler(logfile, when="midnight", interval=1)
    # setup logging format and logger
    handler.suffix = "%Y%m%d"
    logFormatter = logging.Formatter("%(asctime)s %(filename)s %(funcName)s %(lineno)s \
          %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(logFormatter)
    # 指定name，返回一个名称为name的Logger实例。如果再次使用相同的名字，是实例化一个对象。
    # 未指定name，返回Logger实例，名称是root，即根Logger。
    logger = logging.getLogger(name)
    logger.addHandler(handler)

    if level == 'debug':
        logger.setLevel(logging.DEBUG)
    if level == 'info':
        logger.setLevel(logging.INFO)
    if level == 'warning':
        logger.setLevel(logging.WARN)
    if level == 'critical':
        logger.setLevel(logging.CRITICAL)
    if level == 'fatal':
        logger.setLevel(logging.FATAL)
    logger.info("Init log in %s level", level)
    return logger


def RUN_TIME(deta_time):
    '''
    计算时间差，返回毫秒,deta_time.seconds获得秒数=1000ms，deta_time.microseconds获得微妙数=1/1000ms
    :param deta_time: ms
    :return:
    '''
    time_ = deta_time.seconds * 1000 + deta_time.microseconds / 1000.0
    return time_


def TIME():
    '''
    获得当前时间
    :return:
    '''
    return datetime.datetime.now()


def run_time_decorator(title=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            T0 = TIME()
            result = func(*args, **kwargs)
            T1 = TIME()
            logger.debug("{}-- function : {}-- rum time : {}ms ".format(title, func.__name__, RUN_TIME(T1 - T0)))
            return result

        return wrapper

    return decorator


logger = set_logging(name="FBPyramidBox", level="debug", logfile=None)

if __name__ == '__main__':
    T0 = TIME()
    # do something
    T1 = TIME()
    print("rum time:{}ms".format(RUN_TIME(T1 - T0)))
    t_logger = set_logging(name=__name__, level="info", logfile=None)
    t_logger.debug('debug')
    t_logger.info('info')
    t_logger.warning('Warning exists')
    t_logger.error('Finish')
