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

tf.app.flags.DEFINE_float(
    'weight_decay', 0.0005, 'The weight decay on the model weights.')

FLAGS = tf.app.flags.FLAGS

def modelDecorator(func):
    def wrapper():
        global sess
        global net
        global images_input

        afg_net = AFGNet()
        images_input = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3))
        net = afg_net.buildNet(images_input, FLAGS.num_classes, weight_decay=FLAGS.weight_decay,
                               is_training=False, dropout_keep_prob=FLAGS.dropout_keep_prob,
                               stage='landmark')

        with tf.Session() as sess:
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                logger.info("Model restored...")

            func()

    return wrapper

@modelDecorator
def testImage():
    for i in range(6, 32):
        filenames = '/home/lingdi/project/test/' + str(i) + '.jpg'
        image = Image.open(filenames)
        image = resizeImage(image, targetW=224, targetH=224)
        image.show('')

        image_array = np.array(image)
        image_input = np.expand_dims(image_array, axis=0)

        pred = sess.run(net, feed_dict={images_input: image_input})

        heatmaps = pred[0, :, :, :8]
        landmark = cv2.resize(np.max(heatmaps, axis=2), (224, 224))
        print(np.max(pred[0,:,:,0]))
        cv2.imshow('', landmark)
        cv2.waitKey()

@modelDecorator
def testTrainData():
    images, label, heatmaps, landmarks = input_fn(True, ['clothing.record'], params={
        'num_epochs': FLAGS.num_epochs,
        'num_classes': FLAGS.num_classes,
        'batch_size': FLAGS.batch_size,
        'buffer_size': FLAGS.train_images_num,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'height': FLAGS.image_size,
        'width': FLAGS.image_size,
    })

    for i in range(1, 32):
        images_inputs, heatmaps_inputs = sess.run([images, heatmaps])

        # tmp_tensor = tf.get_default_graph().get_tensor_by_name('BCRNN/ConstructHeatMaps:0')
        pred = tf.image.resize_images(net, [FLAGS.image_size, FLAGS.image_size],
                                      method=tf.image.ResizeMethod.BILINEAR)
        pred_res = sess.run(pred, feed_dict={images_input: images_inputs})

        heatmaps_res = np.max(pred_res[0, :, :, :8], axis=2)
        loss_res = np.mean((heatmaps_inputs-pred_res)**2)
        # landmark = cv2.resize(np.max(heatmaps_res, axis=2), (224, 224))
        print(np.max(heatmaps_res), loss_res)
        print(heatmaps_res)
        cv2.imshow('0', images_inputs[0]/255)
        cv2.imshow('1', heatmaps_res)
        cv2.imshow('2', np.max(heatmaps_inputs[0, :, :, :8], axis=2))
        cv2.waitKey()


if __name__ == '__main__':
    # testImage()
    testTrainData()
