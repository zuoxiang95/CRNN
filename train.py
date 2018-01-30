# -*- coding: utf-8 -*-

"""
   author: ZuoXiang
   date: 2017-01-17
"""
import os
import time
import math
import argparse
import traceback
import tensorflow as tf

from crnn import Crnn
from utils import DataFeeder
from value_window import ValueWindow
from hparams import hparams, hparams_debug_string

from datetime import datetime
from os.path import join


def add_stats(model):
    with tf.variable_scope('stats'):
        tf.summary.scalar('total_loss', model.total_loss)
        tf.summary.scalar('avg_loss', model.avg_loss)
        tf.summary.scalar('decoded', model.decoded)
        tf.summary.scalar('distance', model.distance)
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram('gradient_norm', gradient_norms)
        tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))
        return tf.summary.merge_all()


def time_string():
    return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, args):
    checkpoint_path = join(log_dir, 'model.ckpt')
    input_path = join(args.base_dir, args.input)
    print('Checkpoint path: %s' % checkpoint_path)
    print('Loading training data from: %s' % input_path)
    print('Using model: %s' % args.model)
    print(hparams_debug_string())

    # 创建DataFeeder：
    coord = tf.train.Coordinator()
    with tf.variable_scope('data_feeder'):
        feeder = DataFeeder(coord, input_path, hparams)

    # 启动模型
    global_step = tf.Variable(0, name='global_step', trainable=False)
    with tf.variable_scope('model'):
        model = Crnn(hparams)
        model.initialize(feeder.inputs, feeder.input_lengths, feeder.content, True)
        model.add_loss()
        model.add_optimizer(global_step)
        stats = add_stats(model)

    # 保存模型
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(tf.global_variables())

    # 开始训练
    with tf.Session() as sess:
        try:
            summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())
            sess.run(tf.global_variables_initializer())
            # 是否读档 ^-^~~
            if args.restore_step:
                restore_path = '%s-%d' % (checkpoint_path, args.restore_step)
                saver.restore(sess, restore_path)
                print('Resume from checkpoint: %s' % restore_path)
            else:
                print("Starting new training process.")

            feeder.start_in_session(sess)

            while not coord.should_stop():
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.avg_loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f' % (
                    step, time_window.average, loss, loss_window.average
                )
                print(message)

                # 检测梯度爆炸
                if loss > 100 or math.isnan(loss):
                    print('Loss exploded to %.05f at step %d!' % (loss, step))
                    raise Exception('Loss exploded')

                # 写入到tensorboard中
                if step % args.summary_interval == 0:
                    print('Writing summary at step: %d' % step)
                    summary_writer.add_summary(sess.run(stats), step)

                # 保存checkpoint
                if step % args.checkpoint_interval == 0:
                    print('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                    saver.save(sess, checkpoint_path, global_step=step)

        except Exception as e:
            print('Exiting due to exception: %s' % e)
            traceback.print_exc()
            coord.request_stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('~/crnn'))
    parser.add_argument('--input', default='training/train.txt')
    parser.add_argument('--model', default='crnn')
    parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
    parser.add_argument('--summary_interval', type=int, default=100)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    args = parser.parse_args()
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
    os.makedirs(log_dir, exist_ok=True)
    hparams.parse(args.hparams)
    train(log_dir, args)


if __name__ == '__main__':
    main()
