# -*- coding: utf-8 -*-

"""
   author: ZuoXiang
   date: 2017-01-17
"""

import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Layer params: Filts K stride Padding Name BatchNorm
    layer_params=[[64, 3, 1, 'valid', 'conv1', False],  # conv1
                  [64, 2, 2, 'same', 'pool1', False],  # pool1
                  [128, 3, 1, 'same', 'conv2', False],  # conv2
                  [128, 2, 2, 'same', 'pool2', True],  # pool2
                  [256, 3, 1, 'same', 'conv3', False],  # conv3
                  [256, 3, 1, 'same', 'conv4', False],  # conv4
                  [256, 1, 2, 'same', 'pool4', False],  # pool4
                  [512, 3, 1, 'same', 'conv5', True],  # conv5
                  [512, 3, 1, 'same', 'conv6', True],  # conv6
                  [512, 1, 2, 'same', 'pool6', False],  # pool6
                  [512, 2, 1, 'same', 'conv7', False]],  # conv7

    dropout_rate=0.5,
    rnn_size=2 ** 9,
    # 是否使用百度的warp ctc loss
    # 如果为True，则使用，否则使用tensorflow实现的ctc loss
    use_warpctc=True,
    # adam参数
    adam_beta1=0.9,
    adam_beta2=0.999,
    # 初始学习率
    initial_learning_rate=0.002,
    # 指数衰减学习率
    decay_learning_rate=True,

    ##################
    # language model #
    ##################

    # path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.
    decoder_library_path='native_client/libctc_decoder_with_kenlm.so',

    # path to the language model binary file created with KenLM
    lm_binary_path='data/lm/lm.binary',

    # path to the language model trie file created with native_client/generate_trie
    lm_trie_path='data/lm/trie',

    # path to the configuration file specifying the alphabet used by the network.
    # See the comment in data/alphabet.txt for a description of the format.
    alphabet_config_path='data/alphabet.txt',

    # the alpha hyperparameter of the CTC decoder. Language Model weight.
    lm_weight=1.75,

    # the beta hyperparameter of the CTC decoder. Word insertion weight (penalty).
    word_count_weight=1.00,

    # valid word insertion weight.
    # This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.
    valid_word_count_weight=1.00,

    # beam width used in the CTC decoder when building candidate transcriptions
    beam_width=1024
    # TODO: Add more configurable hparams
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
