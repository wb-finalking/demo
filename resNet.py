# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import graph_util
import collections
from classifyPreprocess import *
import cv2
from PIL import Image
import pandas as pd
import csv
import json
import logging
logging.basicConfig(level=10, filename='train.log')
logger = logging.getLogger(__name__)

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'num_classes', 22,
    'The class number')

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

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

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

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """

class ResNetV2(object):

    def __init__(self):
        self.default_image_size = 224

    def buildNet(self, netName, images, num_classes, weight_decay=0.0, is_training=False, **kwargs):
        arg_scope = self.resnet_arg_scope(weight_decay=weight_decay)

        networks_map = {'resnet_v2_50': self.resnet_v2_50,
                        'resnet_v2_101': self.resnet_v2_101,
                        'resnet_v2_152': self.resnet_v2_152,
                        'resnet_v2_200': self.resnet_v2_200,
                        }
        with slim.arg_scope(arg_scope):
            func = networks_map[netName]
            logits, end_points = func(images,num_classes,
                                      is_training=is_training, **kwargs)
        return logits, end_points

    def resnet_v2_50(self, inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     output_stride=None,
                     spatial_squeeze=True,
                     reuse=None,
                     scope='resnet_v2_50'):
        """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            self.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            self.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
            self.resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
            self.resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ]
        return self.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                         global_pool=global_pool, output_stride=output_stride,
                         include_root_block=True, spatial_squeeze=spatial_squeeze,
                         reuse=reuse, scope=scope)

    def resnet_v2_101(self, inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      spatial_squeeze=True,
                      reuse=None,
                      scope='resnet_v2_101'):
        """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            self.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            self.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
            self.resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
            self.resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ]
        return self.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                              global_pool=global_pool, output_stride=output_stride,
                              include_root_block=True, spatial_squeeze=spatial_squeeze,
                              reuse=reuse, scope=scope)

    def resnet_v2_152(self, inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      spatial_squeeze=True,
                      reuse=None,
                      scope='resnet_v2_152'):
        """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
        blocks = [
            self.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            self.resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
            self.resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
            self.resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ]
        return self.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                              global_pool=global_pool, output_stride=output_stride,
                              include_root_block=True, spatial_squeeze=spatial_squeeze,
                              reuse=reuse, scope=scope)

    def resnet_v2_200(self, inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      spatial_squeeze=True,
                      reuse=None,
                      scope='resnet_v2_200'):
        """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
        blocks = [
            self.resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
            self.resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
            self.resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
            self.resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
        ]
        return self.resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                              global_pool=global_pool, output_stride=output_stride,
                              include_root_block=True, spatial_squeeze=spatial_squeeze,
                              reuse=reuse, scope=scope)

    def resnet_v2(self, inputs,
                  blocks,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  include_root_block=True,
                  spatial_squeeze=True,
                  reuse=None,
                  scope=None):
        """Generator for v2 (preactivation) ResNet models.

        This function generates a family of ResNet v2 models. See the resnet_v2_*()
        methods for specific model instantiations, obtained by selecting different
        block instantiations that produce ResNets of various depths.

        Training for image classification on Imagenet is usually done with [224, 224]
        inputs, resulting in [7, 7] feature maps at the output of the last ResNet
        block for the ResNets defined in [1] that have nominal stride equal to 32.
        However, for dense prediction tasks we advise that one uses inputs with
        spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
        this case the feature maps at the ResNet output will have spatial shape
        [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
        and corners exactly aligned with the input image corners, which greatly
        facilitates alignment of the features to the image. Using as input [225, 225]
        images results in [8, 8] feature maps at the output of the last ResNet block.

        For dense prediction tasks, the ResNet needs to run in fully-convolutional
        (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
        have nominal stride equal to 32 and a good choice in FCN mode is to use
        output_stride=16 in order to increase the density of the computed features at
        small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

        Args:
          inputs: A tensor of size [batch, height_in, width_in, channels].
          blocks: A list of length equal to the number of ResNet blocks. Each element
            is a resnet_utils.Block object describing the units in the block.
          num_classes: Number of predicted classes for classification tasks.
            If 0 or None, we return the features before the logit layer.
          is_training: whether batch_norm layers are in training mode.
          global_pool: If True, we perform global average pooling before computing the
            logits. Set to True for image classification, False for dense prediction.
          output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
          include_root_block: If True, include the initial convolution followed by
            max-pooling, if False excludes it. If excluded, `inputs` should be the
            results of an activation-less convolution.
          spatial_squeeze: if True, logits is of shape [B, C], if false logits is
              of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
              To use this parameter, the input images must be smaller than 300x300
              pixels, in which case the output logit layer does not contain spatial
              information and can be removed.
          reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
          scope: Optional variable_scope.


        Returns:
          net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
            If global_pool is False, then height_out and width_out are reduced by a
            factor of output_stride compared to the respective height_in and width_in,
            else both height_out and width_out equal one. If num_classes is 0 or None,
            then net is the output of the last ResNet block, potentially after global
            average pooling. If num_classes is a non-zero integer, net contains the
            pre-softmax activations.
          end_points: A dictionary from components of the network to the corresponding
            activation.

        Raises:
          ValueError: If the target output_stride is not valid.
        """
        with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'
            with slim.arg_scope([slim.conv2d, self.bottleneck,
                                 self.stack_blocks_dense],
                                outputs_collections=end_points_collection):
                with slim.arg_scope([slim.batch_norm], is_training=is_training):
                    net = inputs
                    if include_root_block:
                        if output_stride is not None:
                            if output_stride % 4 != 0:
                                raise ValueError('The output_stride needs to be a multiple of 4.')
                            output_stride /= 4
                        # We do not include batch normalization or activation functions in
                        # conv1 because the first ResNet unit will perform these. Cf.
                        # Appendix of [2].
                        with slim.arg_scope([slim.conv2d],
                                            activation_fn=None, normalizer_fn=None):
                            net = self.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                        net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                    net = self.stack_blocks_dense(net, blocks, output_stride)
                    # This is needed because the pre-activation variant does not have batch
                    # normalization or activation functions in the residual unit output. See
                    # Appendix of [2].
                    net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
                    # Convert end_points_collection into a dictionary of end_points.
                    end_points = slim.utils.convert_collection_to_dict(
                        end_points_collection)

                    if global_pool:
                        # Global average pooling.
                        net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                        end_points['global_pool'] = net
                    if num_classes:
                        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                          normalizer_fn=None, scope='logits')
                        end_points[sc.name + '/logits'] = net
                        if spatial_squeeze:
                            net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                            end_points[sc.name + '/spatial_squeeze'] = net
                        end_points['predictions'] = slim.softmax(net, scope='predictions')
                    return net, end_points

    def resnet_v2_block(self, scope, base_depth, num_units, stride):
        """Helper function for creating a resnet_v2 bottleneck block.

        Args:
          scope: The scope of the block.
          base_depth: The depth of the bottleneck layer for each unit.
          num_units: The number of units in the block.
          stride: The stride of the block, implemented as a stride in the last unit.
            All other units have stride=1.

        Returns:
          A resnet_v2 bottleneck block.
        """
        return Block(scope, self.bottleneck, [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1
        }] * (num_units - 1) + [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': stride
        }])

    @slim.add_arg_scope
    def stack_blocks_dense(self, net, blocks, output_stride=None,
                           store_non_strided_activations=False,
                           outputs_collections=None):
        """Stacks ResNet `Blocks` and controls output feature density.

        First, this function creates scopes for the ResNet in the form of
        'block_name/unit_1', 'block_name/unit_2', etc.

        Second, this function allows the user to explicitly control the ResNet
        output_stride, which is the ratio of the input to output spatial resolution.
        This is useful for dense prediction tasks such as semantic segmentation or
        object detection.

        Most ResNets consist of 4 ResNet blocks and subsample the activations by a
        factor of 2 when transitioning between consecutive ResNet blocks. This results
        to a nominal ResNet output_stride equal to 8. If we set the output_stride to
        half the nominal network stride (e.g., output_stride=4), then we compute
        responses twice.

        Control of the output feature density is implemented by atrous convolution.

        Args:
          net: A `Tensor` of size [batch, height, width, channels].
          blocks: A list of length equal to the number of ResNet `Blocks`. Each
            element is a ResNet `Block` object describing the units in the `Block`.
          output_stride: If `None`, then the output will be computed at the nominal
            network stride. If output_stride is not `None`, it specifies the requested
            ratio of input to output spatial resolution, which needs to be equal to
            the product of unit strides from the start up to some level of the ResNet.
            For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,
            then valid values for the output_stride are 1, 2, 6, 24 or None (which
            is equivalent to output_stride=24).
          store_non_strided_activations: If True, we compute non-strided (undecimated)
            activations at the last unit of each block and store them in the
            `outputs_collections` before subsampling them. This gives us access to
            higher resolution intermediate activations which are useful in some
            dense prediction problems but increases 4x the computation and memory cost
            at the last unit of each block.
          outputs_collections: Collection to add the ResNet block outputs.

        Returns:
          net: Output tensor with stride equal to the specified output_stride.

        Raises:
          ValueError: If the target output_stride is not valid.
        """
        # The current_stride variable keeps track of the effective stride of the
        # activations. This allows us to invoke atrous convolution whenever applying
        # the next residual unit would result in the activations having stride larger
        # than the target output_stride.
        current_stride = 1

        # The atrous convolution rate parameter.
        rate = 1

        for block in blocks:
            with tf.variable_scope(block.scope, 'block', [net]) as sc:
                block_stride = 1
                for i, unit in enumerate(block.args):
                    if store_non_strided_activations and i == len(block.args) - 1:
                        # Move stride from the block's last unit to the end of the block.
                        block_stride = unit.get('stride', 1)
                        unit = dict(unit, stride=1)

                    with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                        # If we have reached the target output_stride, then we need to employ
                        # atrous convolution with stride=1 and multiply the atrous rate by the
                        # current unit's stride for use in subsequent layers.
                        if output_stride is not None and current_stride == output_stride:
                            net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                            rate *= unit.get('stride', 1)

                        else:
                            net = block.unit_fn(net, rate=1, **unit)
                            current_stride *= unit.get('stride', 1)
                            if output_stride is not None and current_stride > output_stride:
                                raise ValueError('The target output_stride cannot be reached.')

                # Collect activations at the block's end before performing subsampling.
                net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

                # Subsampling of the block's output activations.
                if output_stride is not None and current_stride == output_stride:
                    rate *= block_stride
                else:
                    net = self.subsample(net, block_stride)
                    current_stride *= block_stride
                    if output_stride is not None and current_stride > output_stride:
                        raise ValueError('The target output_stride cannot be reached.')

        if output_stride is not None and current_stride != output_stride:
            raise ValueError('The target output_stride cannot be reached.')

        return net

    @slim.add_arg_scope
    def bottleneck(self, inputs, depth, depth_bottleneck, stride, rate=1,
                   outputs_collections=None, scope=None):
        """Bottleneck residual unit variant with BN before convolutions.

        This is the full preactivation residual unit variant proposed in [2]. See
        Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
        variant which has an extra bottleneck layer.

        When putting together two consecutive ResNet blocks that use this unit, one
        should use stride = 2 in the last unit of the first block.

        Args:
          inputs: A tensor of size [batch, height, width, channels].
          depth: The depth of the ResNet unit output.
          depth_bottleneck: The depth of the bottleneck layers.
          stride: The ResNet unit's stride. Determines the amount of downsampling of
            the units output compared to its input.
          rate: An integer, rate for atrous convolution.
          outputs_collections: Collection to add the ResNet unit output.
          scope: Optional variable_scope.

        Returns:
          The ResNet unit's output.
        """
        with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
            depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
            preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
            if depth == depth_in:
                shortcut = self.subsample(inputs, stride, 'shortcut')
            else:
                shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                       normalizer_fn=None, activation_fn=None,
                                       scope='shortcut')

            residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                                   scope='conv1')
            residual = self.conv2d_same(residual, depth_bottleneck, 3, stride,
                                                rate=rate, scope='conv2')
            residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='conv3')

            output = shortcut + residual

            return slim.utils.collect_named_outputs(outputs_collections,
                                                    sc.name,
                                                    output)

    def subsample(self, inputs, factor, scope=None):
        """Subsamples the input along the spatial dimensions.

        Args:
          inputs: A `Tensor` of size [batch, height_in, width_in, channels].
          factor: The subsampling factor.
          scope: Optional variable_scope.

        Returns:
          output: A `Tensor` of size [batch, height_out, width_out, channels] with the
            input, either intact (if factor == 1) or subsampled (if factor > 1).
        """
        if factor == 1:
            return inputs
        else:
            return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

    def conv2d_same(self, inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
        """Strided 2-D convolution with 'SAME' padding.

        When stride > 1, then we do explicit zero-padding, followed by conv2d with
        'VALID' padding.

        Note that

           net = conv2d_same(inputs, num_outputs, 3, stride=stride)

        is equivalent to

           net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')
           net = subsample(net, factor=stride)

        whereas

           net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')

        is different when the input's height or width is even, which is why we add the
        current function. For more details, see ResnetUtilsTest.testConv2DSameEven().

        Args:
          inputs: A 4-D tensor of size [batch, height_in, width_in, channels].
          num_outputs: An integer, the number of output filters.
          kernel_size: An int with the kernel_size of the filters.
          stride: An integer, the output stride.
          rate: An integer, rate for atrous convolution.
          scope: Scope.

        Returns:
          output: A 4-D tensor of size [batch, height_out, width_out, channels] with
            the convolution output.
        """
        if stride == 1:
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                               padding='SAME', scope=scope)
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            inputs = tf.pad(inputs,
                            [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
            return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                               rate=rate, padding='VALID', scope=scope)

    def resnet_arg_scope(self, weight_decay=0.0001,
                         batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,
                         batch_norm_scale=True,
                         activation_fn=tf.nn.relu,
                         use_batch_norm=True,
                         batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
        """Defines the default ResNet arg scope.

        TODO(gpapan): The batch-normalization related default values above are
          appropriate for use in conjunction with the reference ResNet models
          released at https://github.com/KaimingHe/deep-residual-networks. When
          training ResNets from scratch, they might need to be tuned.

        Args:
          weight_decay: The weight decay to use for regularizing the model.
          batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
          batch_norm_epsilon: Small constant to prevent division by zero when
            normalizing activations by their variance in batch normalization.
          batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
            activations in the batch normalization layer.
          activation_fn: The activation function which is used in ResNet.
          use_batch_norm: Whether or not to use batch normalization.
          batch_norm_updates_collections: Collection for the update ops for
            batch norm.

        Returns:
          An `arg_scope` to use for the resnet models.
        """
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'scale': batch_norm_scale,
            'updates_collections': batch_norm_updates_collections,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope(
                [slim.conv2d],
                weights_regularizer=slim.l2_regularizer(weight_decay),
                weights_initializer=slim.variance_scaling_initializer(),
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm if use_batch_norm else None,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                # The following implies padding='SAME' for pool1, which makes feature
                # alignment easier for dense prediction tasks. This is also used in
                # https://github.com/facebook/fb.resnet.torch. However the accompanying
                # code of 'Deep Residual Learning for Image Recognition' uses
                # padding='VALID' for pool1. You can switch to that choice by setting
                # slim.arg_scope([slim.max_pool2d], padding='VALID').
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc

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

def test():
    catagory = ['茄克', '毛衫', '短袖T恤', '长袖T恤', '棉服', '长袖衬衫', '中袖衬衫',
                '卫衣', '风衣', '休闲西服', '马甲', '羽绒服', '开衫', '牛仔长裤', '牛仔中裤',
                '牛仔短裤', '连衣裙', '裙子', '休闲长裤', '休闲中裤', '休闲短裤', '打底裤']

    # img = Image.open('/home/lingdi/project/test/21.jpg')
    img = Image.open('/home/lingdi/project/semir_data/dataset/18/休闲长裤_bottom/26.jpg')
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

    resNetv2 = ResNetV2()

    inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))

    logits, end_points = resNetv2.buildNet(inputs,
                                           num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
                                           weight_decay=FLAGS.weight_decay,
                                           is_training=False)

    print(tf.contrib.slim.get_variables_to_restore(exclude=[]))
    w = tf.get_default_graph().get_tensor_by_name('InceptionV4/Conv2d_1a_3x3/BatchNorm/beta:0')
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info("Model restored...")

        tensors = [end_points['Predictions'], w]
        pre, w = sess.run(tensors,feed_dict={inputs:img})
        logger.debug('class :{}'.format(np.argmax(pre)))

        print(catagory[np.argmax(pre)])
        print(pre)

def train(trainList, eval=None, is_eval=False):
    resNetv2 = ResNetV2()

    inputs = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    labels = tf.placeholder(tf.int32, shape=(None, FLAGS.num_classes))

    # training data
    inputsTrain, labelsTrain = input_fn(True, trainList, params={
        'num_epochs': 10,
        'class_num': FLAGS.num_classes,
        'batch_size': FLAGS.batch_size,
        'buffer_size': 22000,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'height': 299,
        'width': 299,
        'ignore_label': 255,
    })

    # evaluating data
    if is_eval:
        inputsEval, labelsEval = input_fn(True, [eval], params={
            'num_epochs': 10,
            'class_num': FLAGS.num_classes,
            'batch_size': FLAGS.batch_size,
            'buffer_size': 17858,
            'min_scale': 0.8,
            'max_scale': 1.2,
            'height': 299,
            'width': 299,
            'ignore_label': 255,
        })

    logits, end_points = resNetv2.buildNet(inputs,
                                           num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
                                           weight_decay=FLAGS.weight_decay,
                                           is_training=True)

    loss = tf.losses.softmax_cross_entropy(labels, logits,
                                           label_smoothing=FLAGS.label_smoothing)

    train_var_list=[]
    if not FLAGS.freeze_batch_norm:
        train_var_list = [v for v in tf.trainable_variables()]
    else:
        train_var_list = [v for v in tf.trainable_variables()
                          if 'beta' not in v.name and 'gamma' not in v.name]

    # Add weight decay to the loss.
    # with tf.variable_scope("total_loss"):
    #     loss = loss + FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
    tf.summary.scalar('loss', loss)

    # set optimizer
    global_step = tf.train.get_or_create_global_step()
    learning_rate = resNetv2.configure_learning_rate(10000, global_step)
    optimizer = resNetv2.configure_optimizer(learning_rate)

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
        train_writer = tf.summary.FileWriter(FLAGS.model_dir+'/log22', sess.graph)
        if is_eval:
            eval_writer = tf.summary.FileWriter(FLAGS.model_dir+'/log22/eval', sess.graph)
        while True:

            try:
                inputsnNp, labelsNp = sess.run([inputsTrain, labelsTrain])

                tensors = [trainOp, merge_summary, global_step, loss]
                _, train_summary, itr, res_loss = sess.run(tensors,feed_dict={inputs:inputsnNp,
                                                                    labels:labelsNp})
                logger.info("itr: {}".format(itr))

                train_writer.add_summary(train_summary, itr)
                logger.info("loss: {}, summary written...".format(res_loss))

                if is_eval and itr % 10 == 0:
                    inputsnNp, labelsNp = sess.run([inputsEval, labelsEval])
                    _, eval_summary = sess.run([loss, merge_summary],feed_dict={inputs:inputsnNp,
                                                                                labels:labelsNp})
                    eval_writer.add_summary(eval_summary, itr)

            except tf.errors.OutOfRangeError:
                logger.warning('Maybe OutOfRangeError...')
                break

            if (itr % 500 == 0):
                saver.save(sess, FLAGS.model_dir + '/model.ckpt', itr)

def initializingModel(tfrecord):
    resNetv2 = ResNetV2()

    # inputs = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
    # labels = tf.placeholder(tf.int32, shape=(None, FLAGS.num_classes))
    inputs, labels = input_fn(True, [tfrecord], params={
        'num_epochs': 1,
        'class_num': FLAGS.num_classes,
        'batch_size': FLAGS.batch_size,
        'buffer_size': 30,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'height': 299,
        'width': 299,
        'ignore_label': 255,
    })

    logits, end_points = resNetv2.buildNet(inputs,
                                           num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
                                           weight_decay=FLAGS.weight_decay,
                                           is_training=True)

    loss = tf.losses.softmax_cross_entropy(labels, logits,
                                           label_smoothing=FLAGS.label_smoothing)

    train_var_list = []
    if not FLAGS.freeze_batch_norm:
        train_var_list = [v for v in tf.trainable_variables()]
    else:
        train_var_list = [v for v in tf.trainable_variables()
                          if 'beta' not in v.name and 'gamma' not in v.name]

    # Add weight decay to the loss.
    with tf.variable_scope("total_loss"):
        loss = loss + FLAGS.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])
    tf.summary.scalar('loss', loss)


    if True:
        exclude = ['InceptionV4/Logits/Logits', 'global_step']
        variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
    # print([v.name.split(':')[0] for v in variables_to_restore])
    # tf.train.init_from_checkpoint(FLAGS.model_dir+'/model.ckpt',
    #                               {v.name.split(':')[0]: v for v in variables_to_restore})

    merge_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        # set optimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = resNetv2.configure_learning_rate(22000, global_step)
        optimizer = resNetv2.configure_optimizer(learning_rate)

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
    resNetv2 = ResNetV2()

    inputs = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))

    logits, end_points = resNetv2.buildNet(inputs,
                                           num_classes=(FLAGS.num_classes - FLAGS.labels_offset),
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
            ['InceptionV4/Logits/Predictions']  # We split on comma for convenience
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

def accAndRecall():
    import matplotlib.pyplot as plt
    d = {}
    data = pd.read_csv('res.csv')
    keys = data.keys()
    for key in keys:
        d[key] = data[key].values

    total_num = len(d['prob'])
    acc = []
    recall = []
    for idx in range(50):
        threshold = idx * 0.02
        acc_num = 0
        num = 0
        for i, prob in enumerate(d['prob']):
            if prob > threshold and d['acc'][i]:
                acc_num += 1
            if prob > threshold:
                num += 1
            # if prob > 0.5 and not d['acc'][i]:
            # print('{}, {}, {}, {}, {}'.format(d['name'][i], d['catagory'][i],
            #                                   d['pre'][i], d['prob'][i],
            #                                   d['vector'][i]))
        acc.append(1.0 * acc_num / (num + 1e-12))
        recall.append(1.0 * num / total_num)

    x = [i for i in range(50)]
    plt.plot(x, acc)
    print(acc)
    plt.plot(x, recall)
    plt.show()

if __name__ == '__main__':

    # test model
    # test()
    # testSemir('top')
    # testSemirWithJson('top')
    # accAndRecall()

    # init model
    # initializingModel('miniTrain.record')

    # train model
    # train(['miniTrain.record', 'shirtTrain.record', 'shirtTest.record', 'semirTrain.record'],
    #       'fashionDataEval.record', False)
    # train(['bingGoogle.record'])

    # freeze model
    freezingModel()

    # printGraph('frozen.pb')



