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
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors, LSTMStateTuple
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import graph_util

class VGG(object):
    def __init__(self):
        self.default_image_size = 224

    def buildNet(self, netName, images, num_classes, dropout_keep_prob=0.5,
                 is_training=False, weight_decay=0.5, final_endpoint='conv5'):
        arg_scope = self.vgg_arg_scope(weight_decay=weight_decay)

        networks_map = {'VGG_11': self.vgg_a,
                        'VGG_16': self.vgg_16,
                        'VGG_19': self.vgg_19,
                        }
        with slim.arg_scope(arg_scope):
            func = networks_map[netName]
            logits, end_points = func(images, num_classes,
                                      dropout_keep_prob=dropout_keep_prob,
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
              num_classes=1000,
              is_training=True,
              dropout_keep_prob=0.5,
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
            # print('{}_shape:{}'.format(name, net.shape))
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

def _conv(args, filter_size, num_features, bias, bias_start=0.0):
    """Convolution.
    Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter height and width.
    num_features: int, number of features.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.
    Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    shape_length = len(shapes[0])
    for shape in shapes:
        if len(shape) not in [3, 4, 5]:
            raise ValueError("Conv Linear expects 3D, 4D "
                             "or 5D arguments: %s" % str(shapes))
        if len(shape) != len(shapes[0]):
            raise ValueError("Conv Linear expects all args "
                             "to be of same Dimension: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[-1]
        dtype = [a.dtype for a in args][0]

    # determine correct conv operation
    if shape_length == 3:
        conv_op = nn_ops.conv1d
        strides = 1
    elif shape_length == 4:
        conv_op = nn_ops.conv2d
        strides = shape_length * [1]
    elif shape_length == 5:
        conv_op = nn_ops.conv3d
        strides = shape_length * [1]

    # Now the computation.
    kernel = variable_scope.get_variable(
        "kernel", filter_size + [total_arg_size_depth, num_features], dtype=dtype)
    if len(args) == 1:
        res = conv_op(args[0], kernel, strides, padding="SAME")
    else:
        res = conv_op(
            array_ops.concat(axis=shape_length - 1, values=args),
            kernel,
            strides,
            padding="SAME")
    if not bias:
        return res
    bias_term = variable_scope.get_variable(
        "biases", [num_features],
        dtype=dtype,
        initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return res + bias_term

class CRNN(tf.contrib.rnn.RNNCell):
    def __init__(self, conv_ndims, input_shape, output_channels, kernel_shape,
                 use_bias=True, initializers=None, name="crnn_cell"):
        """Construct CRNN.
        Args:
          conv_ndims: Convolution dimensionality (1, 2 or 3).
          input_shape: Shape of the input as int tuple, excluding the batch size, time steps and channel.
          output_channels: int, number of output channels of the conv.
          kernel_shape: Shape of kernel as in tuple (of size 1,2 or 3).
          use_bias: (bool) Use bias in convolutions.
          skip_connection: If set to `True`, concatenate the input to the
            output of the conv LSTM. Default: `False`.
          forget_bias: Forget bias.
          initializers: Unused.
          name: Name of the module.
        Raises:
          ValueError: If `skip_connection` is `True` and stride is different from 1
            or if `input_shape` is incompatible with `conv_ndims`.
        """
        super(CRNN, self).__init__(name=name)

        if conv_ndims != len(input_shape) - 1:
            raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
                input_shape, conv_ndims))

        self._input_shape = input_shape
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._use_bias = use_bias

        self._state_size = input_shape
        self._output_size = input_shape

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    # def zero_state(self, batch_size, dtype):
    #     state_size = self.state_size
    #     # return _zero_state_tensors(state_size, batch_size, dtype)
    #
    #     def expand(x, dim, N):
    #         return tf.concat([tf.expand_dims(x, dim) for _ in range(N)], axis=dim)
    #
    #     with tf.variable_scope('init', reuse=False):
    #         state = expand(tf.get_variable('init_state', self.state_size,
    #                                        initializer=tf.random_normal_initializer(mean=0.0, stddev=0.5)), dim=0,N=batch_size)
    #
    #     return state

    def __call__(self, inputs, state, scope=None):
        cell, hidden = state[0], state[1]
        print(inputs.shape, hidden.shape, len(state))
        new_hidden = _conv([inputs, hidden], self._kernel_shape,
                           self._output_channels, self._use_bias)

        output = math_ops.tanh(new_hidden)

        return output, output

class AFGNet(object):
    def __init__(self):
        self.default_image_size = 224
        self.vgg = VGG()

    def buildNet(self, images, num_classes, weight_decay=0.0005,
                 is_training=False, dropout_keep_prob=0.5, stage='landmark'):

        # construct VGG base net
        net, end_points = self.vgg.buildNet('VGG_16', images, num_classes,
                                            is_training=is_training,
                                            weight_decay=weight_decay,
                                            dropout_keep_prob=dropout_keep_prob,
                                            final_endpoint='conv4')

        with tf.variable_scope('BCRNN'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                padding='SAME'):
                # 8 landmarks and 1 background
                heat_maps = slim.conv2d(net, 9, [1, 1], scope='ConstructHeatMaps')
                heat_maps = tf.sigmoid(heat_maps, name='sigmoid')

            # heat-maps l-collar l-sleeve l-waistline l-hem r-...
            heat_maps = tf.transpose(heat_maps, (3, 0, 1, 2))
            # grammar:
            # RK:
            #         l.collar <-> l.waistline <-> l.hem;
            #         l.collar <-> l.sleeve;
            #         r.collar <-> r.waistline <-> r.hem;
            #         r.collar <-> r.sleeve:
            # RS:
            #         l.collar <-> r.collar;
            #         l.sleeve <-> r.sleeve;
            #         l.waistline <-> r.waistline;
            #         l.hem <-> r.hem:
            RK1_refined_heatmaps = self.BCRNNBlock(heat_maps, 3, [0, 2, 3], 'RK_1')
            RK2_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [0, 1], 'RK_2')
            RK3_refined_heatmaps = self.BCRNNBlock(heat_maps, 3, [4, 6, 7], 'RK_3')
            RK4_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [4, 5], 'RK_4')

            RS1_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [0, 4], 'RS_1')
            RS2_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [1, 5], 'RS_2')
            RS3_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [2, 6], 'RS_3')
            RS4_refined_heatmaps = self.BCRNNBlock(heat_maps, 2, [3, 7], 'RS_4')

            background = heat_maps[8]

            # max merge heatmaps
            l_collar= tf.reduce_max([RK1_refined_heatmaps[0], RK2_refined_heatmaps[0], RS1_refined_heatmaps[0]], axis=3)
            l_sleeve = tf.reduce_max([RK2_refined_heatmaps[1], RS2_refined_heatmaps[0]], axis=3)
            l_waistline = tf.reduce_max([RK1_refined_heatmaps[1], RS3_refined_heatmaps[0]], axis=3)
            l_hem = tf.reduce_max([RK1_refined_heatmaps[2], RS4_refined_heatmaps[0]], axis=3)

            r_collar = tf.reduce_max([RK3_refined_heatmaps[0], RK4_refined_heatmaps[0], RS1_refined_heatmaps[1]], axis=3)
            r_sleeve = tf.reduce_max([RK4_refined_heatmaps[1], RS2_refined_heatmaps[1]], axis=3)
            r_waistline = tf.reduce_max([RK3_refined_heatmaps[1], RS3_refined_heatmaps[1]], axis=3)
            r_hem = tf.reduce_max([RK3_refined_heatmaps[2], RS4_refined_heatmaps[1]], axis=3)

            refined_heatmaps = tf.stack([l_collar, l_sleeve, l_waistline, l_hem,
                                         r_collar, r_sleeve, r_waistline, r_hem,
                                         background], axis=3)

            # landmarks predictions
            output = tf.nn.softmax(refined_heatmaps, name='RefinedHeatMaps')

        if stage.lower() == 'landmark':
            return output

        with tf.variable_scope('LandmarkAttention'):
            output = output[:, :, :, :-1]
            AL = tf.reduce_mean(output, axis=-1, keep_dims=True)
            tile_shape = tf.ones_like(output.shape)
            tile_shape[-1] = output.shape[-1]
            AL = tf.tile(AL, tile_shape)
            GL = tf.multiply(AL, net)

        with tf.variable_scope('ClothingAttention'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer(),
                                scope='ClothingAttention'):
                AC = slim.max_pool2d(net, [2, 2], scope='AC_pool1')
                AC = slim.conv2d(AC, 512, [2, 2], scope='AC_conv1')
                AC = slim.max_pool2d(AC, [2, 2], scope='AC_pool2')
                AC = slim.conv2d(AC, 512, [2, 2], scope='AC_conv2')
                AC = slim.conv2d_transpose(AC, num_outputs=512,
                                           strides=4, kernel_size=[2, 2],
                                           padding='SAME',
                                           scope='AC_upsample')
                AC = tf.sigmoid(AC, 'sigmoid')
                GC = tf.multiply(AC, net)

        with tf.variable_scope('Classification'):
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer()):
                net = net + GL + GC
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1], scope='fc8')
                net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

        return net

    def BCRNNBlock(self, heat_maps, maps_num, maps_idxs, scope):
        with tf.variable_scope(scope):
            if maps_num == 2:
                grammar_serial = tf.stack([heat_maps[maps_idxs[0]],
                                           heat_maps[maps_idxs[1]]], axis=3)
            else:
                grammar_serial = tf.stack([heat_maps[maps_idxs[0]],
                                           heat_maps[maps_idxs[1]],
                                           heat_maps[maps_idxs[2]]], axis=3)
            # grammar_serial_RK1 shape (batch_size, time_steps, row, col)
            grammar_serial = tf.transpose(grammar_serial, (0, 3, 1, 2))
            refined_heatmaps = self.multiLayerBidirectionalRnn(1, 3, grammar_serial, [maps_num])
            return tf.transpose(refined_heatmaps, (1, 0, 2, 3))

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
        if len(_inputs.get_shape().as_list()) < 3:
            raise ValueError("the inputs must be 3-dimentional Tensor")
        batch_size = tf.shape(inputs)[0]

        for T in range(num_layers):
            # 为什么在这加个variable_scope,被逼的,tf在rnn_cell的__call__中非要搞一个命名空间检查
            # 恶心的很.如果不在这加的话,会报错的.
            # with tf.variable_scope(None, default_name="bidirectional-rnn"):
            # rnn_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units)
            # rnn_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units)
            rnn_cell_fw = CRNN(1, [28, 28], 1, [2, 2])
            rnn_cell_bw = CRNN(1, [28, 28], 1, [2, 2])
            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
            output, state = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw,
                                                            _inputs, seq_lengths,
                                                            initial_state_fw, initial_state_bw,
                                                            dtype=tf.float32,
                                                            scope="BCRNN_"+str(T))
            # generate input for next bcrnn layer
            # _inputs = tf.concat(output, 2)
            output_fw, output_bw = output[0], output[1]
            # _inputs shape (batch_size, time_steps, row, col)
            _inputs = _inputs + output_fw + output_bw

        return _inputs

if __name__ == '__main__':
    a = tf.constant([[2, 2], [2, 2]])
    b = tf.constant([[1, 2, 3], [3, 4, 5]])

    # c = tf.reduce_max([a[0], b[1]], axis=1)
    with tf.Session() as sess:
        b_result = sess.run(b[1])
        print(b_result, b.shape)