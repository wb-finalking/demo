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

tf.app.flags.DEFINE_integer('train_images_num', 5, 'The number of images in train data.')

tf.app.flags.DEFINE_float('dropout_keep_prob', 0.5, 'VGG dropout keep prob')

tf.app.flags.DEFINE_string(
    'model_dir', 'model',
    'Directory where checkpoints and event logs are written to.')

FLAGS = tf.app.flags.FLAGS

def predByImage(image):
    image_array = np.array(image)
    image_input = np.expand_dims(image_array, axis=0)

    afg_net = AFGNet()
    images_input = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
    net = afg_net.buildNet(images_input, FLAGS.num_classes, weight_decay=FLAGS.weight_decay,
                           is_training=True, dropout_keep_prob=FLAGS.dropout_keep_prob,
                           stage='classification')

    with tf.Session() as sess:
        pred = sess.run(net, feed_dict={images_input: image_input})

    print(pred)

if __name__ == '__main__':
    predByImage()
