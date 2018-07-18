import io
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2

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

def dict_to_tf_example(image_path, catagory_label):
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width, height = image.size

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('jpg'.encode('utf8')),
        'label/catagory': int64_feature(catagory_label),
    }))
    return example

def createTFRecord(output, data_path, list_category_img, list_partition, partition):
    itr = 0
    num = 0
    with tf.python_io.TFRecordWriter(output) as writer:
        with open(list_category_img, "r") as f_c:
            with open(list_partition, "r") as f_p:
                line_c = f_c.next()
                total_num = int(line_c.strip())
                f_c.next()

                f_p.next()
                f_p.next()
                for line_c, line_p in zip(f_c, f_p):
                    line_c = line_c.strip()
                    line_p = line_p.strip()

                    image_path = os.path.join(data_path, line_c.split(' ')[0])
                    catagory_label = int(line_c.split(' ')[-1]) - 1

                    if line_p.split(' ')[-1] == partition:
                        num = num + 1
                        try:
                            tf_example = dict_to_tf_example(image_path, catagory_label)
                        except:
                            print("{} error...".format(image_path))
                            continue
                        writer.write(tf_example.SerializeToString())

                    itr = itr + 1
                    if itr % 500 == 0:
                        print('On image %d of %d', num, total_num)

def createNewTFRecord(output, data_path, list_category_img):
    itr = 0
    num = 0

    path = '/home/lingdi/project/Img/'

    with tf.python_io.TFRecordWriter(output) as writer:
        with open(list_category_img, "r") as f_c:
            line_c = f_c.next()
            total_num = int(line_c.strip())
            f_c.next()

            for line_c in f_c:
                line_c = line_c.strip()


                image_path = os.path.join(data_path, line_c.split(' ')[0])
                catagory_label = int(line_c.split(' ')[-1])
                if catagory_label in [1, 11]:
                    num = num + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path+'jacket/'+str(num)+'.jpg')
                        tf_example = dict_to_tf_example(image_path, 0)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [16]:
                    num = num + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'sweater/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 1)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [18]:
                    num = num + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'T/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 2)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [26,27]:
                    num = num + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'jeans/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 3)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [32]:
                    num = num + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'shorts/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 4)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [41]:
                    num = num + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'dress/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 5)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [33]:
                    num = num + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'skirt/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 6)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())

                itr = itr + 1
                if itr % 500 == 0:
                    print('On image {} of {} @ {}'.format(num, total_num, itr))

    # with open(list_category_img, "r") as f_c:
    #
    #     line_c = f_c.next()
    #     total_num = int(line_c.strip())
    #     f_c.next()
    #
    #     for line_c in f_c:
    #         line_c = line_c.strip()
    #
    #         image_path = os.path.join(data_path, line_c.split(' ')[0])
    #         catagory_label = int(line_c.split(' ')[-1])
    #         if catagory_label in [1]:
    #             num = num + 1
    #             try:
    #                 img = Image.open(image_path)
    #                 img.save(path+'jacket/'+str(num)+'.jpg')
    #             except:
    #                 print("{} error...".format(image_path))
    #                 continue
    #         elif catagory_label in [6,16,20]:
    #             num = num + 1
    #             try:
    #                 img = Image.open(image_path)
    #                 img.save(path + 'sweater/' + str(num) + '.jpg')
    #             except:
    #                 print("{} error...".format(image_path))
    #                 continue
    #         elif catagory_label in [18]:
    #             num = num + 1
    #             try:
    #                 img = Image.open(image_path)
    #                 img.save(path + 'T/' + str(num) + '.jpg')
    #             except:
    #                 print("{} error...".format(image_path))
    #                 continue
    #         # elif catagory_label in [26,27]:
    #         #     num = num + 1
    #         #     try:
    #         #         img = Image.open(image_path)
    #         #         img.save(path + 'jeans/' + str(num) + '.jpg')
    #         #     except:
    #         #         print("{} error...".format(image_path))
    #         #         continue
    #         elif catagory_label in [32]:
    #             num = num + 1
    #             try:
    #                 img = Image.open(image_path)
    #                 img.save(path + 'shorts/' + str(num) + '.jpg')
    #             except:
    #                 print("{} error...".format(image_path))
    #                 continue
    #         # elif catagory_label in [41]:
    #         #     num = num + 1
    #         #     try:
    #         #         img = Image.open(image_path)
    #         #         img.save(path + 'dress/' + str(num) + '.jpg')
    #         #     except:
    #         #         print("{} error...".format(image_path))
    #         #         continue
    #         # elif catagory_label in [33]:
    #         #     num = num + 1
    #         #     try:
    #         #         img = Image.open(image_path)
    #         #         img.save(path + 'skirt/' + str(num) + '.jpg')
    #         #     except:
    #         #         print("{} error...".format(image_path))
    #         #         continue
    #
    #         itr = itr + 1
    #         if itr % 500 == 0:
    #             print('On image %d of %d', num, total_num)

def createMiniTFRecord(output, data_path, list_category_img):
    itr = 0
    num = 0

    jacketNum = 0
    sweaterNum = 0
    TNum = 0
    jeansNum = 0
    shortsNum = 0
    dressNum = 0
    skirtNum = 0
    threshold = 10000

    path = '/home/lingdi/project/Img/'

    with tf.python_io.TFRecordWriter(output) as writer:
        with open(list_category_img, "r") as f_c:
            line_c = f_c.next()
            total_num = int(line_c.strip())
            f_c.next()

            for line_c in f_c:
                line_c = line_c.strip()

                image_path = os.path.join(data_path, line_c.split(' ')[0])
                catagory_label = int(line_c.split(' ')[-1])
                if catagory_label in [1, 11]:
                    num = num + 1
                    if jacketNum > threshold:
                        continue
                    jacketNum = jacketNum + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path+'jacket/'+str(num)+'.jpg')
                        tf_example = dict_to_tf_example(image_path, 0)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [16]:
                    num = num + 1
                    if sweaterNum > threshold:
                        continue
                    sweaterNum = sweaterNum + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'sweater/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 1)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [18]:
                    num = num + 1
                    if TNum > threshold:
                        continue
                    TNum = TNum + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'T/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 2)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [26,27]:
                    num = num + 1
                    if jeansNum > threshold:
                        continue
                    jeansNum = jeansNum + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'jeans/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 3)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [32]:
                    num = num + 1
                    if shortsNum > threshold:
                        continue
                    shortsNum = shortsNum + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'shorts/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 4)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [41]:
                    num = num + 1
                    if dressNum > threshold:
                        continue
                    dressNum = dressNum + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'dress/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 5)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())
                elif catagory_label in [33]:
                    num = num + 1
                    if skirtNum > threshold:
                        continue
                    skirtNum = skirtNum + 1
                    try:
                        # img = Image.open(image_path)
                        # img.save(path + 'skirt/' + str(num) + '.jpg')
                        tf_example = dict_to_tf_example(image_path, 6)
                    except:
                        print("{} error...".format(image_path))
                        continue
                    writer.write(tf_example.SerializeToString())

                itr = itr + 1
                if itr % 500 == 0:
                    print('On image {} of {} @ {}'.format(num, total_num, itr))
                    print('jacket: {}, sweater: {}, Tee: {}, jeans: {}, shorts: {}, dress: {}, skirt: {}'
                          .format(jacketNum, sweaterNum, TNum, jeansNum, shortsNum, dressNum, skirtNum))

def createFashionDataRecord(output, data_path, list_category_img):
    '''
    0:  'blouses':896,
    1:  'cloak':7061,
    2:  'coat':9061,
    3:  'jacket':9366,
    4:  'long dress':10090,
    5:  'polo shirt, sport shirt':780,
    6:  'robe':5799,
    7:  'shirt':1425,
    8:  'short dress':4285,
    9:  'suit, suit of clothes':6054,
    10: 'sweater':5209,
    11: 'jersey, T-shirt, tee shirt':1426,
    12: 'undergarment, upper body':5538,
    13: 'uniform':3353,
    14: 'vest, waistcoat':750,
    '''

    itr = 0

    numSet = ['blousesNum','cloakNum','oatNum','jacketNum','dressNum','poloShirtNum',
           'robeNum','shirtNum','skirtNum','suitNum','sweaterNum','TNum',
           'undergarmentNum','uniformNum','waistcoatNum']

    numDict = {}
    for num in numSet:
        numDict[num] = 0

    path = '/home/lingdi/project/fashion-data/'

    with tf.python_io.TFRecordWriter(output) as writer:
        with open(list_category_img, "r") as f_c:
            for line_c in f_c:
                line_c = line_c.strip()

                image_path = os.path.join(data_path, line_c.split(' ')[0])
                image_path += '.jpg'
                catagory_label = int(line_c.split('/')[0])
                try:
                    tf_example = dict_to_tf_example(image_path, catagory_label)
                    numDict[numSet[catagory_label]] += 1

                except:
                    print("{} error...".format(image_path))
                    continue
                writer.write(tf_example.SerializeToString())

                itr = itr + 1
                if itr % 500 == 0:
                    print('On image {} of {}'.format(itr, 17858))

    for num in numSet:
        print('{}: {}'.format(num, numDict[num]))


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
        'label/catagory':
        tf.FixedLenFeature((), tf.int64),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    catagory = tf.cast(parsed['label/catagory'], tf.int32)
    width = tf.cast(parsed['image/width'], tf.int32)
    height = tf.cast(parsed['image/height'], tf.int32)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), 3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    return image, catagory

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
    scale = tf.minimum(float(targetW) / width, float(targetH) / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(image, [new_height, new_width],
                                   method=tf.image.ResizeMethod.BILINEAR)
    image_pad = tf.image.pad_to_bounding_box(
        image, 0, 0,
        targetH,targetW)

    return image_pad

def preprocess_image(image, catagory, is_training, params):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # # Randomly scale the image and label.
        # image = random_rescale_image_and_label(
        #     image, params['min_scale'], params['max_scale'])

        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        # image = random_crop_or_pad_image_and_label(
        #     image, params['height'], params['width'])

        # Randomly flip the image and label horizontally.
        image = random_flip_left_right_image_and_label(image)

        # image.set_shape([params['height'], params['width'], 3])

        # resize Image and keep scale
        image = resizeImageKeepScale(image, targetW=params['width'], targetH=params['height'])

    image = mean_image_subtraction(image)
    label = slim.one_hot_encoding(catagory, params['class_num'])

    return image, label

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
        lambda image, catagory: preprocess_image(image, catagory, is_training, params=params))
    dataset = dataset.prefetch(params['batch_size'])

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(params['num_epochs'])
    dataset = dataset.batch(params['batch_size'])

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels

def testTfrecord():

    inputs, labels = input_fn(True, ['miniTrain.record'], params={
        'num_epochs': 20,
        'class_num': 7,
        'batch_size': 5,
        'buffer_size': 70000,
        'min_scale': 0.9,
        'max_scale': 1.1,
        'height': 299,
        'width': 299,
        'ignore_label': 255,
    })

    with tf.Session() as sess:
        for i in range(10):
            input, label = sess.run([inputs, labels])

            # if i < 998:
            #     continue
            print('label: {}'.format(label))
            img = input[0,:,:,:]
            (R, G, B) = cv2.split(img)
            R = R + 123.68
            G = G + 116.779
            B = B + 103.939
            img = cv2.merge([R, G, B])
            im = Image.fromarray(np.uint8(np.array(img)))
            im.show('')
            #cv2.imshow('',img)
            #cv2.waitKey(0)

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

if __name__ == '__main__':
    # createTFRecord('val.record',
    #                '/home/lingdi/Downloads/Img',
    #                '/home/lingdi/Downloads/Img/Anno/list_category_img.txt',
    #                '/home/lingdi/Downloads/Img/Eval/list_eval_partition.txt',
    #                'val')

    # createMiniTFRecord('miniTrain.record',
    #                '/home/lingdi/Downloads/Img',
    #                '/home/lingdi/Downloads/Img/Anno/list_category_img.txt')

    # createFashionDataRecord('fashionDataEval.record',
    #                    '/home/lingdi/project/fashion-data/images',
    #                    '/home/lingdi/project/fashion-data/test.txt')
    testTfrecord()
