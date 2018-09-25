# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util
from classifyPreprocess import *
import cv2
from PIL import Image
import pandas as pd
import csv
import json
import logging

#####################
# General Flags #
#####################
tf.app.flags.DEFINE_integer('num_classes', 22, 'The class number')

tf.app.flags.DEFINE_float('vgg_weight_decay', 0.0005, 'VGG weight decay')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'VGG dropout keep prob')

FLAGS = tf.app.flags.FLAGS

class VGG(object):
    def __init__(self):
        self.default_image_size = 224

    def buildNet(self, netName, images, num_classes, is_training=False,
                 final_endpoint='conv5'):
        arg_scope = self.vgg_arg_scope(weight_decay=FLAGS.vgg_weight_decay)

        networks_map = {'VGG_11': self.vgg_a,
                        'VGG_16': self.vgg_16,
                        'VGG_19': self.vgg_19,
                        }
        with slim.arg_scope(arg_scope):
            func = networks_map[netName]
            logits, end_points = func(images, num_classes,
                                      is_training=is_training,
                                      final_endpoint=final_endpoint)
        return logits, end_points

    def vgg_arg_scope(self, weight_decay=0.0005):
        """Defines the VGG arg scope.

        Args:
          weight_decay: The l2 regularization coefficient.

        Returns:
          An arg_scope.
        """
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
                return arg_sc

    def vgg_a(self, inputs,
              num_classes=FLAGS.num_classes,
              is_training=True,
              dropout_keep_prob=FLAGS.dropout_keep_prob,
              spatial_squeeze=True,
              scope='vgg_a',
              fc_conv_padding='VALID',
              global_pool=False,
              final_endpoint='conv5'):
        """Oxford Net VGG 11-Layers version A Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
          global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)

        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the input to the logits layer (if num_classes is 0 or None).
          end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(scope, 'vgg_a', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points

    def vgg_16(self, inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_16',
               fc_conv_padding='VALID',
               global_pool=False,
               final_endpoint='conv5'):
        """Oxford Net VGG 16-Layers version D Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
          global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)

        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the input to the logits layer (if num_classes is 0 or None).
          end_points: a dict of tensors with intermediate activations.
        """

        def add_and_check_final(name, net):
            end_points[name] = net
            return name == final_endpoint

        end_points = {}
        with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                if add_and_check_final('conv1', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                if add_and_check_final('conv2', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                if add_and_check_final('conv3', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                if add_and_check_final('conv4', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                if add_and_check_final('conv5', net): return net, end_points
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points

    def vgg_19(self, inputs,
               num_classes=1000,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_19',
               fc_conv_padding='VALID',
               global_pool=False,
               final_endpoint='conv5'):
        """Oxford Net VGG 19-Layers version E Example.

        Note: All the fully_connected layers have been transformed to conv2d layers.
              To use in classification mode, resize input to 224x224.

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer is
            omitted and the input features to the logits layer are returned instead.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.
          global_pool: Optional boolean flag. If True, the input to the classification
            layer is avgpooled to size 1x1, for any input size. (This is not part
            of the original VGG architecture.)

        Returns:
          net: the output of the logits layer (if num_classes is a non-zero integer),
            or the non-dropped-out input to the logits layer (if num_classes is 0 or
            None).
          end_points: a dict of tensors with intermediate activations.
        """
        with tf.variable_scope(scope, 'vgg_19', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                                outputs_collections=end_points_collection):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                # Convert end_points_collection into a end_point dict.
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='dropout7')
                    net = slim.conv2d(net, num_classes, [1, 1],
                                      activation_fn=None,
                                      normalizer_fn=None,
                                      scope='fc8')
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                    end_points[sc.name + '/fc8'] = net
                return net, end_points

class AFGNet(object):
    def __init__(self):
        self.default_image_size = 224
        self.vgg = VGG()

    def buildNet(self, images, num_classes, weight_decay=0.0, is_training=False,
                 vgg_final_endpoint='conv5'):
        net, end_points = self.vgg.buildNet('VGG_16', images, num_classes,
                                            is_training=is_training,
                                            final_endpoint=vgg_final_endpoint)

        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            padding='SAME'):
            net = slim.conv2d(net, 8, [1, 1], scope='heatmaps')

        output = self.multiLayerBidirectionalRnn(128, 3, net, 8)



    def multiLayerBidirectionalRnn(self, num_units, num_layers, inputs, seq_lengths):
        """multi layer bidirectional rnn
        Args:
            num_units: int, hidden unit of RNN cell
            num_layers: int, the number of layers
            inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]
            seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths,
                        the length of the list is batch_size
        Returns:
            the output of last layer bidirectional rnn with concatenating
        """
        # TODO: add time_major parameter
        _inputs = inputs
        if len(_inputs.get_shape().as_list()) != 3:
            raise ValueError("the inputs must be 3-dimentional Tensor")
        batch_size = tf.shape(inputs)[0]

        for _ in range(num_layers):
            # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
            # 恶心的很.如果不在这加的话,会报错的.
            with tf.variable_scope(None, default_name="bidirectional-rnn"):
                rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units)
                rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units)
                initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
                initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
                (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                                  initial_state_fw, initial_state_bw, dtype=tf.float32)
                _inputs = tf.concat(output, 2)
        return _inputs


