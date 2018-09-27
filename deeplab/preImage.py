import io
import os
import sys
import numpy as np
from PIL import Image
import tensorflow as tf

"""
    Image preprocess
    getImageList:
    merge:
"""
def getImageList(image_path, txtName):
    with open(txtName, "a+") as f:
        for root, dirs, files in os.walk(image_path):
            for fn in files:
                fn = fn.split('.')[0]
                f.write(fn + '\n')


label_colours = [(0, 0, 0),  # 0=background
                 # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                 (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                 # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                 (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                 # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                 (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                 # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                 (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]

def merge(im):
    im[im != 0] = 15
    return im

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

def prepareLabel(label_path, txtName, output_path):
    with open(txtName, "r") as f:
        for line in f:
            line = line.strip()
            print(line)

            im = Image.open(label_path+'/'+line+'.png')
            im = np.array(im)
            im = merge(im)
            im = im[:,:,0]
            im = Image.fromarray(im)
            im.save(output_path+'/'+line+'.png')

"""
    create tfrecorde contain:
    int64_feature, int64_list_feature, bytes_feature, bytes_list_feature, float_list_feature
    decode_labels, dict_to_tf_example, create_tf_record
"""
def read_examples_list(path):
  """Read list of training or validation examples.

  The file is assumed to contain a single example per line where the first
  token in the line is an identifier that allows us to find the image and
  annotation xml for that example.

  For example, the line:
  xyz 3
  would allow us to find files xyz.jpg and xyz.xml (the 3 would be ignored).

  Args:
    path: absolute path to examples list file.

  Returns:
    list of example identifiers (strings).
  """
  with tf.gfile.GFile(path) as fid:
    lines = fid.readlines()
  return [line.strip().split(' ')[0] for line in lines]

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

def dict_to_tf_example(image_path,label_path):
    """Convert image and label to tf.Example proto.

    Args:
    image_path: Path to a single PASCAL image.
    label_path: Path to its corresponding label.

    Returns:
    example: The converted tf.Example.

    Raises:
    ValueError: if the image pointed to by image_path is not a valid JPEG or
                if the label pointed to by label_path is not a valid PNG or
                if the size of image does not match with that of label.
    """
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # if image.format != 'JPEG':
    #     raise ValueError('Image format not JPEG')
    if image.format != 'PNG':
        raise ValueError('Label format not PNG')

    with tf.gfile.GFile(label_path, 'rb') as fid:
        encoded_label = fid.read()
    encoded_label_io = io.BytesIO(encoded_label)
    label = Image.open(encoded_label_io)
    if label.format != 'PNG':
        raise ValueError('Label format not PNG')

    if image.size != label.size:
        raise ValueError('The size of image does not match with that of label.')

    width, height = image.size

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/encoded': bytes_feature(encoded_jpg),
        'image/format': bytes_feature('png'.encode('utf8')),
        'label/encoded': bytes_feature(encoded_label),
        'label/format': bytes_feature('png'.encode('utf8')),
    }))
    return example

def create_tf_record(output_filename,image_dir,label_dir,examples):
    """Creates a TFRecord file from examples.

    Args:
    output_filename: Path to where output file is saved.
    image_dir: Directory where image files are stored.
    label_dir: Directory where label files are stored.
    examples: Examples to parse and save to tf record.
    """
    writer = tf.python_io.TFRecordWriter(output_filename)
    for idx, example in enumerate(examples):
        if idx % 500 == 0:
            tf.logging.info('On image %d of %d', idx, len(examples))
        image_path = os.path.join(image_dir, example + '.png')
        label_path = os.path.join(label_dir, example + '.png')

        if not os.path.exists(image_path):
            tf.logging.warning('Could not find %s, ignoring example.', image_path)
            continue
        elif not os.path.exists(label_path):
            tf.logging.warning('Could not find %s, ignoring example.', label_path)
            continue

        try:
            tf_example = dict_to_tf_example(image_path, label_path)
            writer.write(tf_example.SerializeToString())
        except ValueError:
            tf.logging.warning('Invalid example: %s, ignoring.', example)

    writer.close()

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
        'label/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format':
        tf.FixedLenFeature((), tf.string, default_value='png'),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    # height = tf.cast(parsed['image/height'], tf.int32)
    # width = tf.cast(parsed['image/width'], tf.int32)

    image = tf.image.decode_image(
        tf.reshape(parsed['image/encoded'], shape=[]), 3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(
        tf.reshape(parsed['label/encoded'], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    return image, label

def random_rescale_image_and_label(image, label, min_scale, max_scale):
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
    # Since label classes are integers, nearest neighbor need to be used.
    label = tf.image.resize_images(label, [new_height, new_width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, label

def random_crop_or_pad_image_and_label(image, label, crop_height, crop_width, ignore_label):
    """Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by rondomly
    cropping the image or padding it evenly with zeros.

    Args:
    image: 3-D Tensor of shape `[height, width, channels]`.
    label: 3-D Tensor of shape `[height, width, 1]`.
    crop_height: The new height.
    crop_width: The new width.
    ignore_label: Label class to be ignored.

    Returns:
    Cropped and/or padded image.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
    """
    label = label - ignore_label  # Subtract due to 0 padding.
    label = tf.to_float(label)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_and_label = tf.concat([image, label], axis=2)
    image_and_label_pad = tf.image.pad_to_bounding_box(
        image_and_label, 0, 0,
        tf.maximum(crop_height, image_height),
        tf.maximum(crop_width, image_width))
    image_and_label_crop = tf.random_crop(
        image_and_label_pad, [crop_height, crop_width, 4])

    image_crop = image_and_label_crop[:, :, :3]
    label_crop = image_and_label_crop[:, :, 3:]
    label_crop += ignore_label
    label_crop = tf.to_int32(label_crop)

    return image_crop, label_crop

def random_flip_left_right_image_and_label(image, label):
    """Randomly flip an image and label horizontally (left to right).

    Args:
    image: A 3-D tensor of shape `[height, width, channels].`
    label: A 3-D tensor of shape `[height, width, 1].`

    Returns:
    A 3-D tensor of the same type and shape as `image`.
    A 3-D tensor of the same type and shape as `label`.
    """
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, .5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    label = tf.cond(mirror_cond, lambda: tf.reverse(label, [1]), lambda: label)

    return image, label

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

def preprocess_image(image, label, is_training, params):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Randomly scale the image and label.
        image, label = random_rescale_image_and_label(
            image, label, params['min_scale'], params['max_scale'])

        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image, label = random_crop_or_pad_image_and_label(
            image, label, params['height'], params['width'], params['ignore_label'])

        # Randomly flip the image and label horizontally.
        image, label = random_flip_left_right_image_and_label(
            image, label)

        image.set_shape([params['height'], params['width'], 3])
        label.set_shape([params['height'], params['width'], 1])

    image = mean_image_subtraction(image)

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
        lambda image, label: preprocess_image(image, label, is_training, params=params))
    dataset = dataset.prefetch(params['batch_size'])

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(params['num_epochs'])
    dataset = dataset.batch(params['batch_size'])

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels

if __name__ == '__main__':
    image_path = '/home/lingdi/project/person/P1/person__ds10/img'
    label_path = '/home/lingdi/project/person/P1/person__ds10/masks_machine'
    image_path = 'img'
    # get image list
    #getImageList(image_path, "val_image.txt")

    # prepare label file
    #prepareLabel(label_path, "val_image.txt",
                 #'/home/lingdi/project/tensorflow-deeplab-v3-plus-master/label')

    # prepare tf recorde
    tf.logging.set_verbosity(tf.logging.INFO)
    examples = read_examples_list("train_image.txt")
    create_tf_record('train.record', image_path, 'label', examples)

    # with open("train_image.txt", "r") as f:
    #     for line in f:
    #         # line = f.readline()
    #         line = line.strip()
    #         print(line)
    #
    #         im = Image.open(label_path+'/'+line+'.png')
    #         im = np.array(im)
    #         im = np.expand_dims(im, axis=0)
    #         im = merge(im)
    #         im = decode_labels(im)
    #         # print(im)
    #         im = Image.fromarray(im[0])
    #         im.save('./est/'+line+'.jpg')

    # filenames = []
    # txtName = "test_image.txt"
    # with open(txtName, "a+") as f:
    #     for root, dirs, files in os.walk(image_path):
    #         for fn in files:
    #             fn = fn.split('.')[0]
    #             print(fn)
    #             f.write(fn+'\n')
    #             # filenames.append(fn)
    #             # filenames.append(root + os.sep + fn)
    #     # f.writelines(filenames)
