# -*- coding: utf-8 -*-

"""
   author: ZuoXiang
   date: 2017-01-17
"""

import tensorflow as tf
from utils import get_incoming_shape
from tensorflow.contrib.rnn import GRUCell


def conv_2d_layer(incoming, nb_filter, filter_size, strides, kernel_init,
                  is_training, bn=True, padding='same', activation=None, bias='True',
                  bias_init=None, scope=None):
    """
    构建二维卷积层函数
    :param incoming: 输入
    :param nb_filter: 卷积核（feature map）个数
    :param filter_size: 卷积核大小
    :param strides: 步长
    :param kernel_init: 卷积核参数初始化方法
    :param is_training: 是否是训练
    :param bn: 是否需要batch normalization
    :param padding: padding方法
    :param activation: 激活函数
    :param bias: 是否需要偏移量
    :param bias_init: 偏移量初始化方法
    :param scope:
    :return: 返回经过卷积的输出
    """

    input_shape = get_incoming_shape(incoming)
    # 输入必须为4维矩阵
    assert len(input_shape) == 4

    with tf.variable_scope(scope):
        con2d_output = tf.layers.conv2d(
            incoming,
            filters=nb_filter,
            kernel_size=filter_size,
            strides=strides,
            padding=padding,
            activation=activation,
            kernel_initializer=kernel_init,
            use_bias=bias,
            bias_initializer=bias_init
        )
        if bn:
            return tf.layers.batch_normalization(
                con2d_output,
                axis=-1,
                training=is_training)
        else:
            return con2d_output


def max_pooling(incoming, pool_size=2, strides=2, padding='same', name=None):
    """
    max pooling层
    :param incoming: 输入
    :param pool_size: pooling框的大小
    :param strides: 移动步长
    :param padding: padding类型
    :param name: 层的名称
    :return: 经过max pooling结果
    """
    pooling_out = tf.layers.max_pooling2d(
        incoming,
        pool_size=[pool_size, 2],
        strides=strides,
        padding=padding,
        name=name
    )
    return pooling_out


def map_to_sequence(incoming):
    """
    把输入的向量序列化
    :param incoming: 输入向量
    :return: 序列化后的向量
    """
    sequence = tf.reshape(incoming, [-1], name='seq_len')
    return sequence


def bi_rnn_layers(incoming, units, input_length):
    """
    双向RNN层，RNN的单元为GRU
    :param incoming: RNN的输入
    :param units: 双向RNN的单元数
    :param input_length: 输入的长度
    :return: bi-RNN的输出
    """
    outputs, state = tf.nn.bidirectional_dynamic_rnn(
        GRUCell(units),
        GRUCell(units),
        incoming,
        sequence_length=input_length,
        dtype=tf.float32
    )
    return tf.concat(outputs, axis=2)


def conv_module(incoming, params, is_training):
    conv_init = tf.contrib.layers.xavier_initializer()
    output = incoming
    for param in params:
        nb_filter = param[0]
        filter_size = param[1]
        stride = param[2]
        padding = param[3]
        name = param[4]
        bn = param[5]
        if name[:4] == "conv":
            output = conv_2d_layer(
                output,
                nb_filter=nb_filter,
                filter_size=filter_size,
                strides=stride,
                kernel_init=conv_init,
                is_training=is_training,
                bn=bn,
                padding=padding,
                activation=tf.nn.relu,
                bias_init=tf.constant_initializer(0.0)
            )
        else:
            output = max_pooling(output, filter_size, stride, padding, name)
    return output


def calculate_mean_edit_distance_and_loss(model_output, y, use_warpctc):
    batch_seq_len = model_output.get_shape().as_list()[1]
    # 是否使用warp ctc loss
    if use_warpctc:
        total_loss = tf.contrib.warpctc.warp_ctc_loss(label=y, inputs=model_output, sequence_length=batch_seq_len)
    else:
        total_loss = tf.nn.ctc_loss(labels=y, inputs=model_output, sequence_length=batch_seq_len)

    # 计算一个batch的平均损失
    avg_loss = tf.reduce_mean(total_loss)

    # Beam search decode the batch
    decoded, _ = tf.nn.ctc_beam_search_decoder(model_output, batch_seq_len, merge_repeated=True)

    # 计算编辑距离
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), model_output)

    # 计算平均编辑距离
    mean_edit_distance = tf.reduce_mean(distance)

    return total_loss, avg_loss, distance, mean_edit_distance, decoded


# TODO: 语言模型


'''
custom_op_module = tf.load_op_library(FLAGS.decoder_library_path)


def decode_with_lm(inputs, sequence_length, beam_width=100,
                   top_paths=1, merge_repeated=True):
  decoded_ixs, decoded_vals, decoded_shapes, log_probabilities = (
      custom_op_module.ctc_beam_search_decoder_with_lm(
          inputs, sequence_length, beam_width=beam_width,
          model_path=FLAGS.lm_binary_path, trie_path=FLAGS.lm_trie_path, alphabet_path=FLAGS.alphabet_config_path,
          lm_weight=FLAGS.lm_weight, word_count_weight=FLAGS.word_count_weight, valid_word_count_weight=FLAGS.valid_word_count_weight,
          top_paths=top_paths, merge_repeated=merge_repeated))

  return (
      [tf.SparseTensor(ix, val, shape) for (ix, val, shape)
       in zip(decoded_ixs, decoded_vals, decoded_shapes)],
      log_probabilities)


def calculate_mean_edit_distance_and_loss(model_feeder, tower, dropout):

    # Compute the CTC loss using either TensorFlow's `ctc_loss` or Baidu's `warp_ctc_loss`.
    if FLAGS.use_warpctc:
        total_loss = tf.contrib.warpctc.warp_ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)
    else:
        total_loss = tf.nn.ctc_loss(labels=batch_y, inputs=logits, sequence_length=batch_seq_len)

    # Calculate the average loss across the batch
    avg_loss = tf.reduce_mean(total_loss)

    # Beam search decode the batch
    decoded, _ = decode_with_lm(logits, batch_seq_len, merge_repeated=False, beam_width=FLAGS.beam_width)

    # Compute the edit (Levenshtein) distance
    distance = tf.edit_distance(tf.cast(decoded[0], tf.int32), batch_y)

    # Compute the mean edit distance
    mean_edit_distance = tf.reduce_mean(distance)

    # Finally we return the
    # - calculated total and
    # - average losses,
    # - the Levenshtein distance,
    # - the recognition mean edit distance,
    # - the decoded batch and
    # - the original batch_y (which contains the verified transcriptions).
    return total_loss, avg_loss, distance, mean_edit_distance, decoded, batch_y
'''