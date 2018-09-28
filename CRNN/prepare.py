# -*- coding: utf-8 -*-
import io
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import json
from scipy import misc
from io import BytesIO
import pandas as pd

slim = tf.contrib.slim

"""
    create tfrecorde
    dict_to_tf_example:
    createTFRecord:
"""


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def contrastLimitedAHE(image):
    # image_arrary = np.array(image)
    # b, g, r = cv2.split(image_arrary)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # b = clahe.apply(b)
    # g = clahe.apply(g)
    # r = clahe.apply(r)
    # image = cv2.merge([b, g, r])

    image = image.convert('L')
    image_arrary = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_arrary = clahe.apply(image_arrary)
    image = cv2.merge([image_arrary, image_arrary, image_arrary])

    return Image.fromarray(image)

def convertLandmark2Heatmap(landmarks, height, width):
    heatmaps = []
    for landmark in landmarks:
        img = np.zeros((height, width))
        if landmark[0] >= 0 and landmark[1] >= 0:
            img[landmark[0], landmark[1]] = 1
        img = cv2.GaussianBlur(img, (31, 31), 0)
        heatmaps.append(img)
    heatmaps = np.array(heatmaps)

    background = 1 - np.max(heatmaps, axis=0)
    heatmaps = np.concatenate([background, heatmaps], axis=0)

    heatmaps = heatmaps.transpose([1, 2, 0])
    return np.array(heatmaps)

def dict_to_tf_example(image_path, labelID, landmarks):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    # image preproess and convert to jpeg serials
    width, height = image.size
    # image = contrastLimitedAHE(image)
    # with BytesIO() as output:
    #     image.save(output, 'JPEG')
    #     data = output.getvalue()
    # heatmaps = convertLandmark2Heatmap(landmarks, height, width)
    # heatmaps = heatmaps.reshape(-1)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpg'.encode('utf8')),
        'landmarks': int64_list_feature(landmarks.reshape(-1)),
        'labelID': int64_feature(labelID),
    }))
    return example

def createTFRecord(output):
    itr = 0

    catagory = ['solidcolor', 'stripe', 'lattice', 'flower', 'printing', 'other']
    numDict = {}
    for num in catagory:
        numDict[num] = 0

    path = '/home/lingdi/project/fabricImages/style/'

    threshold = 100
    with tf.python_io.TFRecordWriter(output) as writer:
        for idx, item in enumerate(catagory):
            for root, dirs, files in os.walk(path + item):
                for fn in files:
                    filenames = root + os.sep + fn
                    try:
                        tf_example = dict_to_tf_example(filenames, styleID=idx)
                    except:
                        print("{} error...".format(filenames))
                        continue
                    writer.write(tf_example.SerializeToString())
                    numDict[item] += 1
                    if numDict[item] > threshold:
                        break
                if numDict[item] > threshold:
                    break

    for num in catagory:
        print('{}: {}'.format(num, numDict[num]))

def createLandmarkRecord(output):
    def location(idx):
        if idx == -1:
            return [-1, -1]
        return [row['landmark_location_y_'+str(idx)], row['landmark_location_x_'+str(idx)]]

    data = pd.read_csv('/home/lingdi/Downloads/Img/Anno/landmarks.txt', sep=' ')

    path = '/home/lingdi/Downloads/Img/'

    threshold = 100
    with tf.python_io.TFRecordWriter(output) as writer:
        for row in data.rows:
            filenames = path + row['image_name']
            labelID = row['clothes_type']
            if labelID == 1:
                # collar sleeve waistline hem
                landmarks = [location(0), location(2), location(-1), location(4),
                             location(1), location(3), location(-1), location(5)]
            elif labelID == 2:
                # collar sleeve waistline hem
                landmarks = [location(-1), location(-1), location(0), location(2),
                             location(-1), location(-1), location(1), location(3)]
            elif labelID == 3:
                # collar sleeve waistline hem
                landmarks = [location(0), location(2), location(4), location(6),
                             location(1), location(3), location(5), location(7)]

            try:
                tf_example = dict_to_tf_example(filenames, labelID=labelID, landmarks=landmarks)
                print("{} completed...".format(filenames))
            except:
                print("{} error...".format(filenames))
                continue
            writer.write(tf_example.SerializeToString())

"""
    read tfrecorde

"""


def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
        'image/height':
            tf.FixedLenFeature((), tf.int64),
        'image/width':
            tf.FixedLenFeature((), tf.int64),
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'landmarks':
            tf.VarLenFeature(tf.int64),
        'labelID':
            tf.FixedLenFeature((), tf.int64),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    labelID = tf.cast(parsed['labelID'], tf.int32)
    width = tf.cast(parsed['image/width'], tf.int32)
    height = tf.cast(parsed['image/height'], tf.int32)

    landmarks = tf.sparse_tensor_to_dense(parsed['landmarks'], default_value=0)
    landmarks = tf.reshape(landmarks, [-1, 2])

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), 3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    return image, labelID, landmarks


def random_rescale_image_and_label(image, min_scale, max_scale):
    """Rescale an image and label with in target scale.

    Rescales an image and label within the range of target scale.

    Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    min_scale: Min target scale.
    max_scale: Max target scale.

    Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
    If `labels` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, 1]`.
    """
    if min_scale <= 0:
        raise ValueError('\'min_scale\' must be greater than 0.')
    elif max_scale <= 0:
        raise ValueError('\'max_scale\' must be greater than 0.')
    elif min_scale >= max_scale:
        raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

    shape = tf.shape(image)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.random_uniform([], minval=min_scale, maxval=max_scale, dtype=tf.float32)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(image, [new_height, new_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

    return image


def random_rotate_image(image, lowAngle, highAngle):
    # tf.set_random_seed(666)

    def random_rotate_image_func(image):
        angle = np.random.uniform(low=lowAngle, high=highAngle)
        return misc.imrotate(image, angle, 'bicubic')

    image_rotate = tf.py_func(random_rotate_image_func, [image], tf.uint8)

    return image_rotate


def random_crop_or_pad_image_and_label(image, crop_height, crop_width):
    """Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by rondomly
    cropping the image or padding it evenly with zeros.

    Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    crop_height: The new height.
    crop_width: The new width.

    Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
    """

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_pad = tf.image.pad_to_bounding_box(
        image, 0, 0,
        tf.maximum(crop_height, image_height),
        tf.maximum(crop_width, image_width))
    image_pad = tf.random_crop(
        image_pad, [crop_height, crop_width, 3])

    image_crop = image_pad[:, :, :3]

    return image_crop


def resize_and_random_crop_image(image, crop_height, crop_width):
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    w_scale = tf.maximum(float(crop_width) / tf.to_float(image_width), 1.0)
    h_scale = tf.maximum(float(crop_height) / tf.to_float(image_height), 1.0)
    scale = tf.maximum(w_scale, h_scale)

    new_height = tf.to_int32(tf.to_float(image_height) * scale)
    new_width = tf.to_int32(tf.to_float(image_width) * scale)
    image = tf.image.resize_images(image, [new_height, new_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

    image_pad = tf.image.pad_to_bounding_box(
        image, 0, 0,
        tf.maximum(crop_height, new_height),
        tf.maximum(crop_width, new_width))
    image_crop = tf.random_crop(image_pad, [crop_height, crop_width, 3])
    image_crop = image_crop[:, :, :3]

    return image_crop


def random_flip_left_right_image_and_label(image):
    """Randomly flip an image and label horizontally (left to right).

    Args:
    image: A 3-D tensor of shape `[height, width, channels].`

    Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
    """
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)

    return image


def mean_image_subtraction(image, means=(123.68, 116.779, 103.939)):
    """Subtracts the given means from each image channel.

    For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

    Returns:
    the centered image.

    Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def resizeImageKeepScale(image, targetW=299, targetH=299):
    shape = tf.shape(image)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.maximum(float(targetW) / width, float(targetH) / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(image, [new_height, new_width],
                                   method=tf.image.ResizeMethod.BILINEAR)
    # image_pad = tf.image.pad_to_bounding_box(
    #     image, 0, 0,
    #     targetH, targetW)

    return image


def preprocess_image(image, labelID, landmarks, is_training, params):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training and params['augment']:
        # # Randomly scale the image and label.
        image = random_rescale_image_and_label(
            image, params['min_scale'], params['max_scale'])

        # Randomly rotate the image between lowAngel and highAngel
        # image = random_rotate_image(image, -45.0, 45.0)

        # resize Image and keep scale
        image = resizeImageKeepScale(image, targetW=params['width'], targetH=params['height'])

        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image = resize_and_random_crop_image(
            image, params['height'], params['width'])

        # Randomly flip the image and label horizontally.
        image = random_flip_left_right_image_and_label(image)

        # image.set_shape([params['height'], params['width'], 3])

    # image = mean_image_subtraction(image)
    label = slim.one_hot_encoding(labelID, params['num_classes'])

    return image, label, landmarks


def input_fn(is_training, recordFilename, params):
    """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

    Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    num_epochs: The number of epochs to repeat the dataset.

    Returns:
    A tuple of images and labels.
    """
    dataset = tf.data.Dataset.from_tensor_slices(recordFilename)
    dataset = dataset.flat_map(tf.data.TFRecordDataset)

    if is_training:
        # When choosing shuffle buffer sizes, larger sizes result in better
        # randomness, while smaller sizes have better performance.
        # is a relatively small dataset, we choose to shuffle the full epoch.
        dataset = dataset.shuffle(buffer_size=params['buffer_size'])

    dataset = dataset.map(parse_record)
    dataset = dataset.map(
        lambda image, labelID, landmarks:
        preprocess_image(image, labelID, landmarks, is_training, params=params))
    dataset = dataset.prefetch(params['batch_size'])

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(params['num_epochs'])
    dataset = dataset.batch(params['batch_size'])

    iterator = dataset.make_one_shot_iterator()
    images, label, landmarks = iterator.get_next()

    return images, label, landmarks


def testTfrecord(trainList):

    images, labelID, landmarks = input_fn(True, trainList, params={
        'num_epochs': 1,
        'class_num': 1,
        'batch_size': 5,
        'buffer_size': 1000,
        'min_scale': 0.8,
        'max_scale': 1.2,
        'height': 244,
        'width': 244,
    })

    with tf.Session() as sess:
        for i in range(1):
            input, label, landmarks_input = sess.run([images, labelID, landmarks])

            # if i < 998:
            #     continue
            print('label: {}'.format(label))
            img = input[0, :, :, :]
            # (R, G, B) = cv2.split(img)
            # R = R + 123.68
            # G = G + 116.779
            # B = B + 103.939
            # img = cv2.merge([R, G, B])
            im = Image.fromarray(np.uint8(np.array(img)))
            # print(np.array(im))
            im.show('')
            # cv2.imshow('',img)
            # cv2.waitKey(0)


def resize(im, targetW=300, targetH=300):
    # targetW = 300
    # targetH = 300

    ratio = 1

    im = Image.fromarray(np.uint8(np.array(im)))
    w = im.size[0]
    h = im.size[1]

    if w < targetW or h < targetH:
        pass
    else:
        ratio = min(float(targetW) / w, float(targetH) / h)
        w = int(w * ratio)
        h = int(h * ratio)
        im = im.resize((w, h), Image.ANTIALIAS)

    new_im = Image.new("RGB", (targetW, targetH))
    new_im.paste(im, ((targetW - w) // 2,
                      (targetH - h) // 2))
    return new_im


def scaleAndCrop(im, targetW=300, targetH=300):
    ratio = 1

    im = Image.fromarray(np.uint8(np.array(im)))
    w = im.size[0]
    h = im.size[1]

    ratio = max(float(targetW) / w, float(targetH) / h)
    w = int(w * ratio)
    h = int(h * ratio)
    im = im.resize((w, h), Image.ANTIALIAS)

    cood = [(w - targetW) // 2, (h - targetH) // 2,
            targetW + (w - targetW) // 2, targetH + (h - targetH) // 2]

    return im.crop(cood)


if __name__ == '__main__':
    # createTFRecord('clothing.record')
    createLandmarkRecord('clothing.record')
    # testTfrecord()
