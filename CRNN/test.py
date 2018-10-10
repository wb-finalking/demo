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

from prepare import *
from AFG_Net import AFGNet

def testTfExample():
    with tf.python_io.TFRecordWriter('test.tfrecord') as writer:
        for i in range(2):
            filenames = 'C:/project/fabricImages/fabric/'+str(1)+'.jpg'
            tf_example = dict_to_tf_example(filenames, 0, [[10, 10], [100, 100]])
            writer.write(tf_example.SerializeToString())

def testTfParser():
    dataset = tf.data.Dataset.from_tensor_slices(['test.tfrecord'])
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    dataset = dataset.map(parse_record)
    dataset = dataset.prefetch(5)
    dataset = dataset.repeat(5)
    dataset = dataset.batch(1)

    iterator = dataset.make_one_shot_iterator()
    images, id, heatmaps = iterator.get_next()

    # heatmaps = tf.transpose(heatmaps, (3,0,1,2))
    with tf.Session() as sess:
        heatmaps_r = sess.run(heatmaps)

    print(heatmaps_r.shape)
    print(heatmaps_r[heatmaps_r>0])
    g = heatmaps_r[0, :, :, 0]*255
    cv2.imshow('', cv2.merge([g,g,g]))
    cv2.waitKey()

def testAFG_Net():
    afg_net = AFGNet()
    image_size = 224
    num_classes = 22
    ground_heatmaps = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 9))
    images_input = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
    label_input = tf.placeholder(tf.int32, shape=(None, num_classes))
    net = afg_net.buildNet(images_input, num_classes, weight_decay=0.0005,
                           is_training=True, dropout_keep_prob=0.5,
                           stage='landmark')

    print(tf.trainable_variables())


if __name__ == '__main__':
    # testTfExample()
    # testTfParser()
    testAFG_Net()

    # img = convertLandmark2Heatmap([[71, 71]], 600, 600)
    # print(img.shape)
    # g = img[:,:,0] * 255
    # cv2.imshow('', cv2.merge([g, g, g]))
    # cv2.waitKey()