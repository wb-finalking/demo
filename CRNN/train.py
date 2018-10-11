# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
import cv2
from PIL import Image
import pandas as pd
import csv
import json
import numpy as np

import logging
logging.basicConfig(level=10, filename='train.log')
logger = logging.getLogger(__name__)

from prepare import *
from AFG_Net import AFGNet

#####################
# General Flags #
#####################

tf.app.flags.DEFINE_integer('image_size', 224, 'The input image size')

tf.app.flags.DEFINE_integer('num_classes', 22, 'The class number')

tf.app.flags.DEFINE_integer('num_epochs', 22, 'The epochs of train data.')

tf.app.flags.DEFINE_integer('batch_size', 5, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('train_images_num', 2800, 'The number of images in train data.')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'VGG dropout keep prob')

tf.app.flags.DEFINE_string(
    'model_dir', 'model',
    'Directory where checkpoints and event logs are written to.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'adadelta',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.00001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_bool(
    'sync_replicas', False,
    'Whether or not to synchronize the replicas during training.')

tf.app.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'The Number of gradients to collect before updating params.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

FLAGS = tf.app.flags.FLAGS


def configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.
    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / FLAGS.batch_size *
                      FLAGS.num_epochs_per_decay)
    if FLAGS.sync_replicas:
        decay_steps /= FLAGS.replicas_to_aggregate

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)

def configure_optimizer(learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer

def train(trainList, stage='landmark', init=False):
    afg_net = AFGNet()

    if stage.lower() == 'landmark':
        augment = False
    else:
        augment = True
    images, label, heatmaps, landmarks = input_fn(True, trainList, params={
        'augment': augment,
        'num_epochs': FLAGS.num_epochs,
        'num_classes': FLAGS.num_classes,
        'batch_size': FLAGS.batch_size,
        'buffer_size': FLAGS.train_images_num,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'height': FLAGS.image_size,
        'width': FLAGS.image_size,
    })
    # heatmaps = tf.image.resize_images(heatmaps, [28, 28],
    #                                   method=tf.image.ResizeMethod.BILINEAR)

    # build net
    ground_heatmaps = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 9))
    images_input = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
    label_input = tf.placeholder(tf.int32, shape=(None, FLAGS.num_classes))
    net = afg_net.buildNet(images_input, FLAGS.num_classes, weight_decay=FLAGS.weight_decay,
                           is_training=True, dropout_keep_prob=FLAGS.dropout_keep_prob,
                           stage=stage)

    # set optimizer
    global_step = tf.train.get_or_create_global_step()
    learning_rate = configure_learning_rate(FLAGS.train_images_num, global_step)
    optimizer = configure_optimizer(learning_rate)

    # loss definition
    # slim.losses.add_loss(pose_loss)
    if stage.lower() == 'landmark':
        net = tf.image.resize_images(net, [FLAGS.image_size, FLAGS.image_size],
                                     method=tf.image.ResizeMethod.BILINEAR)
        slim.losses.mean_squared_error(net, ground_heatmaps)
    else:
        slim.losses.sigmoid_cross_entropy(net, label, label_smoothing=0.0000001)
    loss = slim.losses.get_total_loss()
    tf.summary.scalar('loss', loss)

    if stage.lower() == 'landmark':
        variables_to_train = tf.global_variables()
    else:
        exclude = ['vgg_16', 'BCRNN', 'LandmarkAttention']
        variables_to_train = [v for v in tf.trainable_variables()
                              if v.name.split('/')[0] not in exclude]
    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainOp = optimizer.minimize(loss, global_step=global_step, var_list=variables_to_train)

    merge_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        if init:
            sess.run(tf.global_variables_initializer())

            if stage.lower() == 'landmark':
                exclude = ['BCRNN', 'LandmarkAttention', 'ClothingAttention',
                           'Classification', 'global_step', 'save']
            else:
                exclude = ['ClothingAttention', 'Classification', 'global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)

            # saver = tf.train.Saver(variables_to_restore)
            # ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            # if ckpt and ckpt.model_checkpoint_path:
            #     saver.restore(sess, ckpt.model_checkpoint_path)
            #     logger.info("Model restored...")
            init = slim.assign_from_checkpoint_fn(FLAGS.model_dir+'/model.ckpt', variables_to_restore,
                                                  ignore_missing_vars=True)
            init(sess)

            saver = tf.train.Saver()
            saver.save(sess, FLAGS.model_dir+'/model.ckpt', 0)

            return
        else:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                logger.info("Model restored...")

        train_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
        while True:
            try:
                images_res, label_res, heatmaps_res, landmarks_res = \
                    sess.run([images, label, heatmaps, landmarks])
                # heatmaps = []
                # for item in landmarks_res:
                #     heatmaps.append(convertLandmark2Heatmap(item, FLAGS.image_size, FLAGS.image_size))
                # heatmaps = np.array(heatmaps)

                tensors = [trainOp, merge_summary, global_step, loss]
                _, train_summary, itr, res_loss = sess.run(tensors,
                                                           feed_dict={ground_heatmaps: heatmaps_res,
                                                                      images_input: images_res,
                                                                      label_input: label_res})
                logger.info("itr: {}".format(itr))

                train_writer.add_summary(train_summary, itr)
                logger.info("loss: {}, summary written...".format(res_loss))

            except tf.errors.OutOfRangeError:
                logger.warning('Maybe OutOfRangeError...')
                break

            if (itr % 500 == 0):
                saver.save(sess, FLAGS.model_dir + '/model.ckpt', itr)

def freeze():
    afg_net = AFGNet()

    # build net
    ground_heatmaps = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 9))
    images_input = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
    label_input = tf.placeholder(tf.int32, shape=(None, FLAGS.num_classes))
    net = afg_net.buildNet(images_input, FLAGS.num_classes, weight_decay=FLAGS.weight_decay,
                           is_training=True, dropout_keep_prob=FLAGS.dropout_keep_prob,
                           stage='classification')

    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info("Model restored...")

        graph_def = sess.graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            ['Classification/Predictions']
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile('frozen.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


if __name__ == '__main__':
    train(['clothing.record'], stage='landmark', init=False)
