# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util
from classifyPreprocess import *
import cv2
from PIL import Image
import pandas as pd
import csv
import logging
logging.basicConfig(level=10, filename='train.log')
logger = logging.getLogger(__name__)

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'color_num', 11,
    'The color class number')

tf.app.flags.DEFINE_float('batch_norm_decay', 0.95, 'batch norm decay')

tf.app.flags.DEFINE_integer(
    'material_num', 5,
    'The material class number')

tf.app.flags.DEFINE_integer(
    'style_num', 5,
    'The style class number')

tf.app.flags.DEFINE_integer(
    'train_data_num', 830,
    'The style class number')

tf.app.flags.DEFINE_boolean('freeze_batch_norm', True,
                            'freeze batch norm.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'model_dir', 'model',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer('num_clones', 1,
                            'Number of model clones to deploy.')

tf.app.flags.DEFINE_boolean('clone_on_cpu', False,
                            'Use CPUs to deploy clones.')

tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')

tf.app.flags.DEFINE_integer(
    'num_ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'task', 0, 'Task id of the replica running the training.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

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

tf.app.flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')

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

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v4', 'The name of the architecture to train.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_integer(
    'batch_size', 5, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'train_image_size', None, 'Train image size')

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

#####################
# Fine-Tuning Flags #
#####################

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'checkpoint_exclude_scopes', None,
    'Comma-separated list of scopes of variables to exclude when restoring '
    'from a checkpoint.')

tf.app.flags.DEFINE_string(
    'trainable_scopes', None,
    'Comma-separated list of scopes to filter the set of variables to train.'
    'By default, None would train all the variables.')

tf.app.flags.DEFINE_boolean(
    'ignore_missing_vars', False,
    'When restoring a checkpoint would ignore missing variables.')

FLAGS = tf.app.flags.FLAGS

class Inceptionv4:

    def __init__(self):
        self.default_image_size = 299
        self.inception_v4_arg_scope = self.inception_arg_scope

    def buildNet(self, images, color_classes, material_classes,
                 style_classes, weight_decay=0.0, is_training=False, **kwargs):
        arg_scope = self.inception_arg_scope(weight_decay=weight_decay)
        with slim.arg_scope(arg_scope):
            end_points = self.inception_multilabel(images, color_classes,
                                                   material_classes,
                                                   style_classes,
                                                   is_training=is_training, **kwargs)
        return end_points

    def block_inception_a(self, inputs, scope=None, reuse=None):
        """Builds Inception-A block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionA', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 96, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 96, [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs, 64, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 96, [1, 1], scope='Conv2d_0b_1x1')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    def block_reduction_a(self, inputs, scope=None, reuse=None):
        """Builds Reduction-A block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockReductionA', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 384, [3, 3], stride=2, padding='VALID',
                                           scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, 256, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

    def block_inception_b(self, inputs, scope=None, reuse=None):
        """Builds Inception-B block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionB', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 224, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 256, [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 192, [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, 224, [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, 224, [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, 256, [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    def block_reduction_b(self, inputs, scope=None, reuse=None):
        """Builds Reduction-B block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockReductionB', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 192, [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, 192, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, 256, [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, 320, [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, 320, [3, 3], stride=2,
                                           padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(inputs, [3, 3], stride=2, padding='VALID',
                                               scope='MaxPool_1a_3x3')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2])

    def block_inception_c(self, inputs, scope=None, reuse=None):
        """Builds Inception-C block for Inception v4 network."""
        # By default use stride=1 and SAME padding
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d, slim.max_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope(scope, 'BlockInceptionC', [inputs], reuse=reuse):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(inputs, 256, [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_1, 256, [1, 3], scope='Conv2d_0b_1x3'),
                        slim.conv2d(branch_1, 256, [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(inputs, 384, [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, 448, [3, 1], scope='Conv2d_0b_3x1')
                    branch_2 = slim.conv2d(branch_2, 512, [1, 3], scope='Conv2d_0c_1x3')
                    branch_2 = tf.concat(axis=3, values=[
                        slim.conv2d(branch_2, 256, [1, 3], scope='Conv2d_0d_1x3'),
                        slim.conv2d(branch_2, 256, [3, 1], scope='Conv2d_0e_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(inputs, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, 256, [1, 1], scope='Conv2d_0b_1x1')
                return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    def inception_v4_base(self, inputs, final_endpoint='Mixed_7d', scope=None):
        """Creates the Inception V4 network up to the given final endpoint.
        Args:
          inputs: a 4-D tensor of size [batch_size, height, width, 3].
          final_endpoint: specifies the endpoint to construct the network up to.
            It can be one of [ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
            'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
            'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
            'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
            'Mixed_7d']
          scope: Optional variable_scope.
        Returns:
          logits: the logits outputs of the model.
          end_points: the set of end_points from the inception model.
        Raises:
          ValueError: if final_endpoint is not set to one of the predefined values,
        """
        end_points = {}

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        with tf.variable_scope(scope, 'InceptionV4', [inputs]):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
                # 299 x 299 x 3
                net = slim.conv2d(inputs, 32, [3, 3], stride=2,
                                  padding='VALID', scope='Conv2d_1a_3x3')
                if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points
                # 149 x 149 x 32
                net = slim.conv2d(net, 32, [3, 3], padding='VALID',
                                  scope='Conv2d_2a_3x3')
                if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
                # 147 x 147 x 32
                net = slim.conv2d(net, 64, [3, 3], scope='Conv2d_2b_3x3')
                if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
                # 147 x 147 x 64
                with tf.variable_scope('Mixed_3a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_0a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 96, [3, 3], stride=2, padding='VALID',
                                               scope='Conv2d_0a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                    if add_and_check_final('Mixed_3a', net): return net, end_points

                # 73 x 73 x 160
                with tf.variable_scope('Mixed_4a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
                        branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
                        branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
                        branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                    if add_and_check_final('Mixed_4a', net): return net, end_points

                # 71 x 71 x 192
                with tf.variable_scope('Mixed_5a'):
                    with tf.variable_scope('Branch_0'):
                        branch_0 = slim.conv2d(net, 192, [3, 3], stride=2, padding='VALID',
                                               scope='Conv2d_1a_3x3')
                    with tf.variable_scope('Branch_1'):
                        branch_1 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                                   scope='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1])
                    if add_and_check_final('Mixed_5a', net): return net, end_points

                # 35 x 35 x 384
                # 4 x Inception-A blocks
                for idx in range(4):
                    block_scope = 'Mixed_5' + chr(ord('b') + idx)
                    net = self.block_inception_a(net, block_scope)
                    if add_and_check_final(block_scope, net): return net, end_points

                # 35 x 35 x 384
                # Reduction-A block
                net = self.block_reduction_a(net, 'Mixed_6a')
                if add_and_check_final('Mixed_6a', net): return net, end_points

                # 17 x 17 x 1024
                # 7 x Inception-B blocks
                for idx in range(7):
                    block_scope = 'Mixed_6' + chr(ord('b') + idx)
                    net = self.block_inception_b(net, block_scope)
                    if add_and_check_final(block_scope, net): return net, end_points

                # 17 x 17 x 1024
                # Reduction-B block
                net = self.block_reduction_b(net, 'Mixed_7a')
                if add_and_check_final('Mixed_7a', net): return net, end_points

                # 8 x 8 x 1536
                # 3 x Inception-C blocks
                for idx in range(3):
                    block_scope = 'Mixed_7' + chr(ord('b') + idx)
                    net = self.block_inception_c(net, block_scope)
                    if add_and_check_final(block_scope, net): return net, end_points

        raise ValueError('Unknown final endpoint %s' % final_endpoint)

    def inception_arg_scope(self, weight_decay=0.00004,
                            use_batch_norm=True,
                            batch_norm_decay=FLAGS.batch_norm_decay,
                            batch_norm_epsilon=0.001,
                            activation_fn=tf.nn.relu):
        """Defines the default arg scope for inception models.
        Args:
          weight_decay: The weight decay to use for regularizing the model.
          use_batch_norm: "If `True`, batch_norm is applied after each convolution.
          batch_norm_decay: Decay for batch norm moving average.
          batch_norm_epsilon: Small float added to variance to avoid dividing by zero
            in batch norm.
          activation_fn: Activation function for conv2d.
        Returns:
          An `arg_scope` to use for the inception models.
        """
        batch_norm_params = {
            # Decay for the moving averages.
            'decay': batch_norm_decay,
            # epsilon to prevent 0s in variance.
            'epsilon': batch_norm_epsilon,
            # collection containing update_ops.
            'updates_collections': tf.GraphKeys.UPDATE_OPS,
            # use fused batch norm if possible.
            'fused': None,
        }
        if use_batch_norm:
            normalizer_fn = slim.batch_norm
            normalizer_params = batch_norm_params
        else:
            normalizer_fn = None
            normalizer_params = {}
        # Set weight_decay for weights in Conv and FC layers.
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay)):
            with slim.arg_scope(
                    [slim.conv2d],
                    weights_initializer=slim.variance_scaling_initializer(),
                    activation_fn=activation_fn,
                    normalizer_fn=normalizer_fn,
                    normalizer_params=normalizer_params) as sc:
                return sc

    def configure_learning_rate(self, num_samples_per_epoch, global_step):
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

    def configure_optimizer(self, learning_rate):
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

    def inception_multilabel(self, inputs, color_classes=1001,
                             material_classes=1001,
                             style_classes=1001,
                             is_training=True,
                             dropout_keep_prob=0.8,
                             reuse=None,
                             scope='InceptionV4'):
        end_points = {}
        with tf.variable_scope(scope, 'InceptionV4', [inputs], reuse=reuse) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
                net, end_points = self.inception_v4_base(inputs, scope=scope,
                                                         final_endpoint='Mixed_7d')

                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                    stride=1, padding='SAME'):
                    # with tf.variable_scope('color'):
                    #     color_logits = end_points['Mixed_3a']
                    #     # with tf.variable_scope('Branch_0'):
                    #     #     branch_0 = slim.conv2d(color_logits, 64, [1, 1], scope='Conv2d_0a_1x1')
                    #     #     branch_0 = slim.conv2d(branch_0, 96, [3, 3], padding='VALID',
                    #     #                            scope='Conv2d_1a_3x3')
                    #     # with tf.variable_scope('Branch_1'):
                    #     #     branch_1 = slim.conv2d(color_logits, 64, [1, 1], scope='Conv2d_0a_1x1')
                    #     #     branch_1 = slim.conv2d(branch_1, 64, [1, 7], scope='Conv2d_0b_1x7')
                    #     #     branch_1 = slim.conv2d(branch_1, 64, [7, 1], scope='Conv2d_0c_7x1')
                    #     #     branch_1 = slim.conv2d(branch_1, 96, [3, 3], padding='VALID',
                    #     #                            scope='Conv2d_1a_3x3')
                    #     # color_logits = tf.concat(axis=3, values=[branch_0, branch_1])
                    #     color_logits = slim.avg_pool2d(color_logits, [5, 5], stride=3,
                    #                                    padding='VALID',
                    #                                    scope='AvgPool_1a_5x5')
                    #     color_logits = slim.conv2d(color_logits, 128, [1, 1],
                    #                                scope='Conv2d_1b_1x1')
                    #     color_logits = slim.conv2d(color_logits, 768,
                    #                                color_logits.get_shape()[1:3],
                    #                                padding='VALID', scope='Conv2d_2a')
                    #     color_logits = slim.flatten(color_logits)
                    #     color_logits = slim.fully_connected(color_logits, color_classes,
                    #                                         activation_fn=None,
                    #                                         scope='color_logits')
                    #     end_points['colorLogits'] = color_logits
                    #     end_points['colorPredictions'] = tf.nn.softmax(color_logits,
                    #                                                    name='colorPredictions')
                    #
                    with tf.variable_scope('material'):
                        material_logits = end_points['Mixed_6h']

                        # material_logits = slim.avg_pool2d(material_logits, [5, 5], stride=3,
                        #                                   padding='VALID',
                        #                                   scope='AvgPool_1a_5x5')
                        # material_logits = slim.conv2d(material_logits, 128, [1, 1],
                        #                               scope='Conv2d_1b_1x1')
                        # material_logits = slim.conv2d(material_logits, 768,
                        #                               material_logits.get_shape()[1:3],
                        #                               padding='VALID', scope='Conv2d_2a')

                        kernel_size = material_logits.get_shape()[1:3]
                        if kernel_size.is_fully_defined():
                            material_logits = slim.avg_pool2d(material_logits, kernel_size,
                                                              padding='VALID',
                                                              scope='AvgPool_1a')
                        else:
                            material_logits = tf.reduce_mean(material_logits, [1, 2],
                                                             keepdims=True,
                                                             name='global_pool')
                        # 1 x 1 x 1536
                        material_logits = slim.dropout(material_logits, dropout_keep_prob,
                                                       scope='Dropout_1b')
                        material_logits = slim.flatten(material_logits, scope='materialLogitsFlatten')
                        material_logits = slim.fully_connected(material_logits, material_classes,
                                                               activation_fn=None,
                                                               scope='materialLogits')
                        end_points['materialLogits'] = material_logits
                        end_points['materialPredictions'] = tf.nn.softmax(material_logits,
                                                                          name='materialPredictions')

                    # with tf.variable_scope('style'):
                    #     style_logits = end_points['Mixed_6h']
                    #
                    #     # style_logits = slim.avg_pool2d(style_logits, [5, 5], stride=3,
                    #     #                                padding='VALID',
                    #     #                                scope='AvgPool_1a_5x5')
                    #     # style_logits = slim.conv2d(style_logits, 128, [1, 1],
                    #     #                            scope='Conv2d_1b_1x1')
                    #     # style_logits = slim.conv2d(style_logits, 768,
                    #     #                            style_logits.get_shape()[1:3],
                    #     #                            padding='VALID', scope='Conv2d_2a')
                    #
                    #     kernel_size = style_logits.get_shape()[1:3]
                    #     if kernel_size.is_fully_defined():
                    #         style_logits = slim.avg_pool2d(style_logits, kernel_size,
                    #                                        padding='VALID',
                    #                                        scope='AvgPool_1a')
                    #     else:
                    #         style_logits = tf.reduce_mean(style_logits, [1, 2],
                    #                                       keepdims=True,
                    #                                       name='global_pool')
                    #     # 1 x 1 x 1536
                    #     style_logits = slim.dropout(style_logits, dropout_keep_prob,
                    #                                 scope='Dropout_1b')
                    #     style_logits = slim.flatten(style_logits, scope='styleLogitsFlatten')
                    #     style_logits = slim.fully_connected(style_logits, style_classes,
                    #                                         activation_fn=None,
                    #                                         scope='styleLogits')
                    #     end_points['styleLogits'] = style_logits
                    #     end_points['stylePredictions'] = tf.nn.softmax(style_logits,
                    #                                                    name='stylePredictions')

            return end_points

def test():

    img = Image.open('/home/lingdi/project/fabricImages/fabric/632.jpg')
    img = img.convert('RGB')
    img = resize(img, 299, 299)
    img.show('')
    img = np.array(img)
    (R, G, B) = cv2.split(img)
    R = R - 123.68
    G = G - 116.779
    B = B - 103.939
    img = cv2.merge([R, G, B])

    # img = mean_image_subtraction(img)
    img = np.expand_dims(img, axis=0)

    inceptionv4 = Inceptionv4()

    inputs = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

    end_points = inceptionv4.buildNet(inputs,
                                      color_classes=FLAGS.color_num,
                                      material_classes=FLAGS.material_num,
                                      style_classes=FLAGS.style_num,
                                      weight_decay=FLAGS.weight_decay,
                                      is_training=False)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info("Model restored...")

        # tensors = [end_points['colorPredictions'], end_points['materialPredictions'],
        #            end_points['stylePredictions']]
        # color_pre, material_pre, style_pre = sess.run(tensors, feed_dict={inputs: img})
        # logger.debug('color class: {}, material class: {}, style class: {}'
        #              .format(np.argmax(color_pre), np.argmax(material_pre), np.argmax(style_pre)))
        tensors = [end_points['stylePredictions']]
        style_pre = sess.run(tensors,feed_dict={inputs:img})
        logger.debug('style class: {}'.format(np.argmax(style_pre)))

        colorList = ['black', 'gray', 'white', 'red', 'yellow', 'blue', 'pink', 'purple', 'green',
                     'orange', 'brown']
        materialList = ['cotton', 'linen', 'silk', 'wool', 'blended']
        styleList = ['solidcolor', 'stripe', 'geometricpatterns', 'lattice', 'other']
        # print(color_pre)
        # print(colorList[np.argmax(color_pre)])
        # print(material_pre)
        # print(materialList[np.argmax(material_pre)])
        print(style_pre)
        print(styleList[np.argmax(style_pre)])

def train(train):
    inceptionv4 = Inceptionv4()

    # inputs = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    # color_labels = tf.placeholder(tf.int32, shape=(None, FLAGS.color_num))
    # material_labels = tf.placeholder(tf.int32, shape=(None, FLAGS.material_num))
    # style_labels = tf.placeholder(tf.int32, shape=(None, FLAGS.style_num))

    # training data
    inputsTrain, colorLabelsTrain, \
    materialLabelsTrain, styleLabelsTrain = input_fn(True, train, params={
        'num_epochs': 400,
        'color_num': FLAGS.color_num,
        'material_num': FLAGS.material_num,
        'style_num': FLAGS.style_num,
        'batch_size': FLAGS.batch_size,
        'buffer_size': FLAGS.train_data_num,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'height': 299,
        'width': 299,
        'ignore_label': 255,
    })

    end_points = inceptionv4.buildNet(inputsTrain,
                                      color_classes=FLAGS.color_num,
                                      material_classes=FLAGS.material_num,
                                      style_classes=FLAGS.style_num,
                                      weight_decay=FLAGS.weight_decay,
                                      is_training=True)

    # color_loss = tf.losses.softmax_cross_entropy(colorLabelsTrain, end_points['colorLogits'],
    #                                              label_smoothing=FLAGS.label_smoothing)
    material_loss = tf.losses.softmax_cross_entropy(materialLabelsTrain, end_points['materialLogits'],
                                                    label_smoothing=FLAGS.label_smoothing)
    # style_loss = tf.losses.softmax_cross_entropy(styleLabelsTrain, end_points['styleLogits'],
    #                                              label_smoothing=FLAGS.label_smoothing)
    loss = material_loss

    # tf.summary.scalar('color_loss', color_loss)
    tf.summary.scalar('material_loss', material_loss)
    # tf.summary.scalar('style_loss', style_loss)
    tf.summary.scalar('loss', loss)

    # set optimizer
    global_step = tf.train.get_or_create_global_step()
    learning_rate = inceptionv4.configure_learning_rate(FLAGS.train_data_num, global_step)
    optimizer = inceptionv4.configure_optimizer(learning_rate)

    # Batch norm requires update ops to be added as a dependency to the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainOp = optimizer.minimize(loss, global_step=global_step)

    # if True:
    #     exclude = ['InceptionV4/Logits/Logits', 'global_step']
    #     variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
    # print([v.name.split(':')[0] for v in variables_to_restore])
    # tf.train.init_from_checkpoint(FLAGS.model_dir+'/model.ckpt',
    #                               {v.name.split(':')[0]: v for v in variables_to_restore})

    merge_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        # sess.run(tf.global_variables_initializer())
        # tf.local_variables_initializer().run()
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info("Model restored...")
            # itr = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

        # saver = tf.train.Saver()
        # saver.save(sess, 'model/model.ckpt', 0)
        # print(global_step)

        itr = 0
        train_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
        while True:

            try:
                # inputsnNp, colorlabelsNp,\
                # materialLabelsNp, styleLabelsNp = sess.run([inputsTrain, colorLabelsTrain,
                #                                             materialLabelsTrain, styleLabelsTrain])

                tensors = [trainOp, merge_summary, global_step, loss]
                _, train_summary, itr, total_loss = sess.run(tensors)
                logger.info("itr: {}".format(itr))

                train_writer.add_summary(train_summary, itr)
                logger.info("loss {}, summary written...".format(total_loss))

            except tf.errors.OutOfRangeError:
                logger.warning('Maybe OutOfRangeError...')
                break

            if (itr % 500 == 0):
                saver.save(sess, FLAGS.model_dir + '/model.ckpt', itr)

def initializingModel(tfrecord):
    inceptionv4 = Inceptionv4()

    # inputs = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    # labels = tf.placeholder(tf.int32, shape=(None, FLAGS.num_classes))
    inputs, color_labels, material_labels, style_labels = input_fn(True, [tfrecord],
                                                                   params={
                                                                       'num_epochs': 1,
                                                                       'color_num': FLAGS.color_num,
                                                                       'material_num': FLAGS.material_num,
                                                                       'style_num': FLAGS.style_num,
                                                                       'batch_size': FLAGS.batch_size,
                                                                       'buffer_size': 30,
                                                                       'min_scale': 0.8,
                                                                       'max_scale': 1.2,
                                                                       'height': 299,
                                                                       'width': 299,
                                                                       'ignore_label': 255,
                                                                   })

    end_points = inceptionv4.buildNet(inputs,
                                      color_classes=FLAGS.color_num,
                                      material_classes=FLAGS.material_num,
                                      style_classes=FLAGS.style_num,
                                      weight_decay=FLAGS.weight_decay,
                                      is_training=True)

    # color_loss = tf.losses.softmax_cross_entropy(color_labels, end_points['colorLogits'],
    #                                              label_smoothing=FLAGS.label_smoothing)
    material_loss = tf.losses.softmax_cross_entropy(material_labels, end_points['materialLogits'],
                                                    label_smoothing=FLAGS.label_smoothing)
    # style_loss = tf.losses.softmax_cross_entropy(style_labels, end_points['styleLogits'],
    #                                              label_smoothing=FLAGS.label_smoothing)
    loss = material_loss

    if True:
        exclude = ['InceptionV4/color',
                   'InceptionV4/material',
                   'InceptionV4/style',
                   'global_step']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
    # print([v.name.split(':')[0] for v in variables_to_restore])
    # tf.train.init_from_checkpoint(FLAGS.model_dir+'/model.ckpt',
    #                               {v.name.split(':')[0]: v for v in variables_to_restore})

    with tf.Session() as sess:
        # set optimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = inceptionv4.configure_learning_rate(FLAGS.train_data_num, global_step)
        optimizer = inceptionv4.configure_optimizer(learning_rate)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            trainOp = optimizer.minimize(loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(variables_to_restore)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info("Model restored...")
        saver = tf.train.Saver()
        saver.save(sess, 'model/model.ckpt', 0)
        # print(global_step)

def freezingModel():
    inceptionv4 = Inceptionv4()

    inputs = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))


    end_points = inceptionv4.buildNet(inputs,
                                      color_classes=FLAGS.color_num,
                                      material_classes=FLAGS.material_num,
                                      style_classes=FLAGS.style_num,
                                      weight_decay=FLAGS.weight_decay,
                                      is_training=False)


    with tf.Session() as sess:

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info("Model restored...")

        itr = 0
        train_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)




        graph_def = sess.graph.as_graph_def()
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            [#'InceptionV4/color/colorPredictions',
             #'InceptionV4/material/materialPredictions',
             'InceptionV4/style/stylePredictions']
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile('frozen.pb', "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

def printGraph(pbName):

    segmentDetectionGraph = tf.Graph()

    with segmentDetectionGraph.as_default():
        odGraphDef = tf.GraphDef()

        with tf.gfile.GFile(pbName, 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

    # names = [op.name for op in segmentDetectionGraph.get_operations()]
    index = 0
    for op in segmentDetectionGraph.get_operations():
        # if 'detect' in op.name:
        print(op.name)

        # index = index + 1
        # if index >20:
        #     break


if __name__ == '__main__':

    # test model
    # test()
    # testSemir()

    # init model
    # initializingModel('fabric.record')

    # train model
    train(['fabric.record'])

    # freeze model
    # freezingModel()

    # printGraph('frozen.pb')
