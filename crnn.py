# -*- coding: utf-8 -*-

"""
   author: ZuoXiang
   date: 2017-01-17
"""

import tensorflow as tf
from modules import conv_module
from modules import map_to_sequence
from modules import bi_rnn_layers
from modules import calculate_mean_edit_distance_and_loss


class Crnn():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, content, is_training):
        with tf.variable_scope('inference') as scope:
            batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            # 7层卷积
            conv_out = conv_module(inputs, params=hp.layer_params, is_training=is_training)
            # 将卷积结果序列化
            conv_sequence_out = map_to_sequence(conv_out)
            # 输入到双向GRU中
            bi_rnn_out = bi_rnn_layers(conv_sequence_out, units=128, input_length=input_lengths)
            # 全连接层输出结果
            mdoel_output = tf.layers.dense(bi_rnn_out, 128,
                                           activation=tf.nn.relu,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                           name='logits')

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.content = content
            self.is_training = is_training
            self.model_output = mdoel_output

            print('Initialized CRNN model. Dimensions: ')
            print('  conv7:               %d' % conv_out.shape[-1])
            print('  sequence out:        %d' % conv_sequence_out.shape[-1])
            print('  bi rnn out:          %d' % bi_rnn_out.shape[-1])
            print('  mdoel out:           %d' % mdoel_output.shape[-1])

    def add_loss(self):
        with tf.variable_scope('loss'):
            hp = self._hparams
            self.total_loss, self.avg_loss, \
            self.distance, self.mean_edit_distance, \
            self.decoded = calculate_mean_edit_distance_and_loss(self.model_output, self.content, self.is_training)

    def add_optimizer(self, global_step):
        with tf.variable_scope('optimizer'):
            hp = self._hparams
            if hp.decay_learning_rate:
                self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.avg_loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)