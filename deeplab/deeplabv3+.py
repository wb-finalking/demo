import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import graph_util
import tensorflow.contrib.slim as slim

import cv2
import numpy as np
from PIL import Image
import argparse
from preImage import *
import logging
logging.basicConfig(level=10, filename='train.log')
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

###################
# general parser  #
###################

parser.add_argument('--num_classes', type=int, default=21,
                    help='Number of image classes.')

parser.add_argument('--buffer_size', type=int, default=30,
                    help='Number of buffer size.')

parser.add_argument('--image_size', type=int, default=513,
                    help='Size of image width and height.')

parser.add_argument('--min_scale', type=float, default=0.5,
                    help='Minimum scale in image augment.')

parser.add_argument('--max_scale', type=float, default=2.0,
                    help='Maximum scale in image augment.')

parser.add_argument('--ignore_label', type=int, default=255,
                    help='Value of ignored label.')

parser.add_argument('--num_images', type=int, default=2097,
                    help='Number of training images.')

#####################
# directory parser  #
#####################

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

parser.add_argument('--data_dir', type=str, default='./dataset/',
                    help='Path of the directory containing the data tfrecord.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--pre_trained_model', type=str, default='./ini_checkpoints/resnet_v2_101/resnet_v2_101.ckpt',
                    help='Path of the pre-trained model checkpoint.')

#################
# model parser  #
#################

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

######################
# Optimization parser#
######################

parser.add_argument('--batch_size', type=int, default=2,
                    help='Number of examples per batch.')

parser.add_argument('--train_epochs', type=int, default=52,
                    help='Number of training epochs.')

parser.add_argument('--batch_norm_decay', type=float, default=0.9997,
                    help='The weight decay to use for batch normalization.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--power', type=float, default=0.9,
                    help='Learning rate power.')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='Learning rate momentum.')

parser.add_argument('--max_iter', type=int, default=30000,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')


_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

# colour map
label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
    mask: result of inference after taking argmax.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).

    Returns:
    A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                            % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

class DeepLabv3Plus:

    def __init__(self):
        pass

    def atrous_spatial_pyramid_pooling(self, inputs, output_stride,
                                       batch_norm_decay, is_training, depth=256):
        """Atrous Spatial Pyramid Pooling.

        Args:
          inputs: A tensor of size [batch, height, width, channels].
          output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
            the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
          batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
          is_training: A boolean denoting whether the input is for training.
          depth: The depth of the ResNet unit output.

        Returns:
          The atrous spatial pyramid pooling output.
        """
        with tf.variable_scope("aspp"):
            if output_stride not in [8, 16]:
                raise ValueError('output_stride must be either 8 or 16.')

            atrous_rates = [6, 12, 18]
            if output_stride == 8:
                atrous_rates = [2 * rate for rate in atrous_rates]

            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    inputs_size = tf.shape(inputs)[1:3]
                    # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                    # the rates are doubled when output stride = 8.
                    conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
                    conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0],
                                                   scope='conv_3x3_1')
                    conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1],
                                                   scope='conv_3x3_2')
                    conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2],
                                                   scope='conv_3x3_3')

                    # (b) the image-level features
                    with tf.variable_scope("image_level_features"):
                        # global average pooling
                        image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling',
                                                              keepdims=True)
                        # 1x1 convolution with 256 filters( and batch normalization)
                        image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1,
                                                                 scope='conv_1x1')
                        # bilinearly upsample features
                        image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size,
                                                                        name='upsample')

                    net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3,
                                    name='concat')
                    net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

                    return net

    def net(self, num_classes, output_stride, base_architecture,
            pre_trained_model, batch_norm_decay, data_format='channels_last'):
        """net for DeepLab v3 plus models.

        Args:
        num_classes: The number of possible classes for image classification.
        output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
            the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
        base_architecture: The architecture of base Resnet building block.
        pre_trained_model: The path to the directory that contains pre-trained models.
        batch_norm_decay: The moving average decay when estimating layer activation
            statistics in batch normalization.
        data_format: The input format ('channels_last', 'channels_first', or None).
            If set to None, the format is dependent on whether a GPU is available.
            Only 'channels_last' is supported currently.

        Returns:
        The model function that takes in `inputs` and `is_training` and
        returns the output tensor of the DeepLab v3 model.
        """

        if data_format is None:
            # data_format = (
            #     'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
            pass

        if base_architecture not in ['resnet_v2_50', 'resnet_v2_101']:
            raise ValueError("'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_50'.")

        if base_architecture == 'resnet_v2_50':
            base_model = resnet_v2.resnet_v2_50
        else:
            base_model = resnet_v2.resnet_v2_101

        def model(inputs, is_training):
            """Constructs the ResNet model given the inputs."""
            if data_format == 'channels_first':
                # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
                # This provides a large performance boost on GPU. See
                # https://www.tensorflow.org/performance/performance_guide#data_formats
                inputs = tf.transpose(inputs, [0, 3, 1, 2])

            # tf.logging.info('net shape: {}'.format(inputs.shape))
            # encoder
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                logits, end_points = base_model(inputs,
                                                num_classes=None,
                                                is_training=is_training,
                                                global_pool=False,
                                                output_stride=output_stride)

            # if is_training:
            #     exclude = [base_architecture + '/logits', 'global_step']
            #     variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)
            #     tf.train.init_from_checkpoint(pre_trained_model,
            #                                   {v.name.split(':')[0]: v for v in variables_to_restore})

            inputs_size = tf.shape(inputs)[1:3]
            net = end_points[base_architecture + '/block4']
            encoder_output = self.atrous_spatial_pyramid_pooling(net, output_stride, batch_norm_decay, is_training)

            with tf.variable_scope("decoder"):
                with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
                    with arg_scope([layers.batch_norm], is_training=is_training):
                        with tf.variable_scope("low_level_features"):
                            low_level_features = end_points[base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
                            low_level_features = layers_lib.conv2d(low_level_features, 48,
                                                                   [1, 1], stride=1, scope='conv_1x1')
                            low_level_features_size = tf.shape(low_level_features)[1:3]

                        with tf.variable_scope("upsampling_logits"):
                            net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name='upsample_1')
                            net = tf.concat([net, low_level_features], axis=3, name='concat')
                            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_1')
                            net = layers_lib.conv2d(net, 256, [3, 3], stride=1, scope='conv_3x3_2')
                            net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None,
                                                    scope='conv_1x1')
                            logits = tf.image.resize_bilinear(net, inputs_size, name='upsample_2')

            return logits

        return model

    def trainAndInference(self, features, labels, params, is_train=True):
        """Model function for PASCAL VOC."""
        if isinstance(features, dict):
            features = features['feature']

        network = self.net(params['num_classes'],
                           params['output_stride'],
                           params['base_architecture'],
                           params['pre_trained_model'],
                           params['batch_norm_decay'])

        logits = network(features, is_train)

        pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3, name='pred_classes')

        pred_decoded_labels = tf.py_func(decode_labels,
                                         [pred_classes, params['batch_size'], params['num_classes']],
                                         tf.uint8)

        predictions = {
            'classes': pred_classes,
            'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
            'decoded_labels': pred_decoded_labels
        }

        if not is_train:
            return predictions

        gt_decoded_labels = tf.py_func(decode_labels,
                                       [labels, params['batch_size'], params['num_classes']], tf.uint8)

        labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.

        logits_by_num_classes = tf.reshape(logits, [-1, params['num_classes']])
        labels_flat = tf.reshape(labels, [-1, ])

        valid_indices = tf.to_int32(labels_flat <= params['num_classes'] - 1)
        valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
        valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

        preds_flat = tf.reshape(pred_classes, [-1, ])
        valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
        confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=params['num_classes'])

        predictions['valid_preds'] = valid_preds
        predictions['valid_labels'] = valid_labels
        predictions['confusion_matrix'] = confusion_matrix

        cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=valid_logits, labels=valid_labels)

        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        if not params['freeze_batch_norm']:
            train_var_list = [v for v in tf.trainable_variables()]
        else:
            train_var_list = [v for v in tf.trainable_variables()
                              if 'beta' not in v.name and 'gamma' not in v.name]

        # Add weight decay to the loss.
        with tf.variable_scope("total_loss"):
            loss = cross_entropy + params['weight_decay'] * tf.add_n(
                [tf.nn.l2_loss(v) for v in train_var_list])
        # loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        global_step = tf.train.get_or_create_global_step()

        if params['learning_rate_policy'] == 'piecewise':
            # Scale the learning rate linearly with the batch size. When the batch size
            # is 128, the learning rate should be 0.1.
            initial_learning_rate = 0.1 * params['batch_size'] / 128
            batches_per_epoch = params['num_train'] / params['batch_size']
            # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
            boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 150, 200]]
            values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
            learning_rate = tf.train.piecewise_constant(
                tf.cast(global_step, tf.int32), boundaries, values)
        elif params['learning_rate_policy'] == 'poly':
            learning_rate = tf.train.polynomial_decay(
                params['initial_learning_rate'],
                tf.cast(global_step, tf.int32) - params['initial_global_step'],
                params['max_iter'], params['end_learning_rate'], power=params['power'])
        else:
            raise ValueError('Learning rate policy must be "piecewise" or "poly"')

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=params['momentum'])

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step, var_list=train_var_list)

        return train_op

def train(record_list):
    FLAGS, unparsed = parser.parse_known_args()

    net = DeepLabv3Plus()
    inputs, labels = input_fn(True, record_list, params={
        'num_epochs': FLAGS.train_epochs,
        'batch_size': FLAGS.batch_size,
        'buffer_size': FLAGS.buffer_size,
        'min_scale': FLAGS.min_scale,
        'max_scale': FLAGS.max_scale,
        'height': FLAGS.image_size,
        'width': FLAGS.image_size,
        'ignore_label': FLAGS.ignore_label,
    })
    # inputs = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    # labels = tf.placeholder(tf.int32, shape=(None, None, None, 1))

    tainOp = net.trainAndInference(inputs, labels, params={
        'output_stride': FLAGS.output_stride,
        'batch_size': FLAGS.batch_size,
        'base_architecture': FLAGS.base_architecture,
        'pre_trained_model': FLAGS.pre_trained_model,
        'batch_norm_decay': FLAGS.batch_norm_decay,
        'num_classes': FLAGS.num_classes,
        'weight_decay': FLAGS.weight_decay,
        'learning_rate_policy': FLAGS.learning_rate_policy,
        'num_train': FLAGS.num_images,
        'initial_learning_rate': FLAGS.initial_learning_rate,
        'max_iter': FLAGS.max_iter,
        'end_learning_rate': FLAGS.end_learning_rate,
        'power': FLAGS.power,
        'momentum': FLAGS.momentum,
        'freeze_batch_norm': FLAGS.freeze_batch_norm,
        'initial_global_step': FLAGS.initial_global_step
    })

    merge_summary = tf.summary.merge_all()
    with tf.Session() as sess:
        # saver = tf.train.Saver()
        # model_file = tf.train.latest_checkpoint(FLAGS.model_dir)
        # saver.restore(sess, model_file)
        # saver.save(sess, 'deepLab/model.ckpt')

        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info("Model restored...")
            itr = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])
        else:
            itr = 0

        train_writer = tf.summary.FileWriter(FLAGS.model_dir, sess.graph)
        while itr < 500 * FLAGS.train_epochs:
            logger.info("itr: {}".format(itr))
            itr = itr + 1

            try:
                tensors = [tainOp, merge_summary, inputs]
                _, train_summary, res_input = sess.run(tensors)

                logger.debug(res_input.shape)
                train_writer.add_summary(train_summary, itr)
                logger.info("summary written...")

            except tf.errors.OutOfRangeError:
                logger.warning('Maybe OutOfRangeError...')
                continue

            if (itr % 100 == 0):
                saver.save(sess, FLAGS.model_dir + '/model.ckpt', itr)

def pbDecorator(func):
    def wrapper():
        global sess
        global net
        global image_tensor
        global pred_tensor
        global FLAGS

        net = tf.Graph()
        with net.as_default():
            graphDef = tf.GraphDef()

            with tf.gfile.GFile('frozen.pb', 'rb') as fid:
                serializedGraph = fid.read()
                graphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(graphDef, name='')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44
        res = []

        with tf.Session(graph=net, config=config) as sess:
            # get tensor
            image_tensor = net.get_tensor_by_name('inputs:0')
            pred_tensor = net.get_tensor_by_name('pred_classes:0')

            func()

    return wrapper

def ckptDecorator(func):
    def wrapper(is_train=False):
        global sess
        global net
        global image_tensor
        global pred_tensor
        global FLAGS

        image_tensor = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='inputs')
        labels = tf.placeholder(tf.int32, shape=(None, None, None, 1), name='labels')

        FLAGS, unparsed = parser.parse_known_args()

        net = DeepLabv3Plus()
        pred_tensor = net.trainAndInference(image_tensor, labels, is_train=is_train, params={
            'output_stride': FLAGS.output_stride,
            'batch_size': FLAGS.batch_size,
            'base_architecture': FLAGS.base_architecture,
            'pre_trained_model': FLAGS.pre_trained_model,
            'batch_norm_decay': FLAGS.batch_norm_decay,
            'num_classes': FLAGS.num_classes,
            'weight_decay': FLAGS.weight_decay,
            'learning_rate_policy': FLAGS.learning_rate_policy,
            'num_train': FLAGS.num_images,
            'initial_learning_rate': FLAGS.initial_learning_rate,
            'max_iter': FLAGS.max_iter,
            'end_learning_rate': FLAGS.end_learning_rate,
            'power': FLAGS.power,
            'momentum': FLAGS.momentum,
            'freeze_batch_norm': FLAGS.freeze_batch_norm,
            'initial_global_step': FLAGS.initial_global_step
        })

        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                logger.info("Model restored...")

            func()

    return wrapper

@ckptDecorator
def initModel():
    sess.run(tf.global_variables_initializer())

    exclude = [FLAGS.base_architecture + '/logits', 'global_step']
    variables_to_restore = tf.contrib.slim.get_variables_to_restore(exclude=exclude)

    init = slim.assign_from_checkpoint_fn(FLAGS.model_dir + '/model.ckpt', variables_to_restore,
                                          ignore_missing_vars=True)
    init(sess)

    saver = tf.train.Saver()
    saver.save(sess, FLAGS.model_dir + '/model.ckpt', 0)

@ckptDecorator
def freezing():
    graph_def = sess.graph.as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        ['pred_classes']
    )

    # Finally we serialize and dump the output graph to the filesystem
    with tf.gfile.GFile('frozen.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph." % len(output_graph_def.node))

@pbDecorator
def evaluating():
    image = Image.open('1.png')
    # image = resizeImageKeepScale(image, 513, 513)
    image = image.resize((513, 513))
    imgArray = np.array(image)
    (R, G, B) = cv2.split(imgArray)
    R = R - _R_MEAN
    G = G - _G_MEAN
    B = B - _B_MEAN
    imgArray = cv2.merge([R, G, B])
    imageInput = np.expand_dims(imgArray, axis=0)

    mask = sess.run(pred_tensor, feed_dict={image_tensor: imageInput})

    res = decode_labels(mask)
    cv2.imshow('', res[0])
    cv2.waitKey()

if __name__ == '__main__':
    # initModel(is_train=True)
    # train(['dataset/train.record'])
    evaluating()



