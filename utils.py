# -*- coding: utf-8 -*-

"""
   author: ZuoXiang
   date: 2017-01-17
"""

import threading
import tensorflow as tf
import numpy as np
import os


def get_incoming_shape(incoming):
    """
    获取输入数据的形状
    :param incoming: 输入数据
    :return: 输入数据形状
    """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, np.ndarray, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


class DataFeeder(threading.Thread):
    """ 通过后台线程将多个batch的数据传输到队列中 """

    def __init__(self, coordinator, metadata_filename, hparams):
        super(DataFeeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._offset = 0

        # load data
        self._datadir = os.path.dirname(metadata_filename)
        with open(metadata_filename) as f:
            self.metadata = [line.strip().split('|') for line in f]

        pass
    pass

