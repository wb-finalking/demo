from PIL import Image
import os, sys
sys.path.append('/home/lingdi/Downloads/models/research')
sys.path.append('/home/lingdi/project/fast-rcnn/caffe-fast-rcnn/python')
import cv2
import time
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from tensorflow.python.framework import graph_util
from nets.inception_v4 import *
from nets import nets_factory
from fastRCNN import *
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from sklearn.externals import joblib
from skimage import color
import matplotlib.pyplot as plt
import xlwt, xlrd

# img = Image.open('1.jpg')
# img = img.crop((0,0,100,100))
# print(img)
# img.show()
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def detect(filenames, savePath, pbName):

    labelMap = label_map_util.load_labelmap('label_map.pbtxt')
    categories = label_map_util.convert_label_map_to_categories(labelMap, max_num_classes=3, use_display_name=True)
    segmentCategoryIndex = label_map_util.create_category_index(categories)

    segmentDetectionGraph = tf.Graph()

    with segmentDetectionGraph.as_default():
        odGraphDef = tf.GraphDef()


        with tf.gfile.GFile(pbName, 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        with tf.Session(graph=segmentDetectionGraph, config=config) as sess:
            for k in range(len(filenames)):
                start = time.time()
                rgbImage = Image.open(filenames[k])

                res = {'top': {}, 'bottom': {}, 'full': {}}

                # print('image shape: {}'.format(np.array(rgbImage).shape))
                imgTmp = np.array(rgbImage)
                if(len(imgTmp.shape)==2):
                    imgTmp = np.expand_dims(imgTmp, axis=2)
                    imgTmp = np.tile(imgTmp,(1,1,3))
                im = cv2.cvtColor(imgTmp, cv2.COLOR_RGB2BGR)
                imExpanded = np.expand_dims(im, axis=0)
                image_tensor = segmentDetectionGraph.get_tensor_by_name('image_tensor:0')
                boxes = segmentDetectionGraph.get_tensor_by_name('detection_boxes:0')
                scores = segmentDetectionGraph.get_tensor_by_name('detection_scores:0')
                classes = segmentDetectionGraph.get_tensor_by_name('detection_classes:0')
                numDetections = segmentDetectionGraph.get_tensor_by_name('num_detections:0')

                (boxes, scores, classes, _) = sess.run(
                    [boxes, scores, classes, numDetections],
                    feed_dict={image_tensor: imExpanded})

                end = time.time()
                print("{}  {} / {}, takes {} seconds".format(filenames[k], k, len(filenames),
                                                             end - start))
                # determine boxes
                maxBoxesToDraw = 10
                minScoreThresh = .3

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                for i in range(min(maxBoxesToDraw, boxes.shape[0])):
                    if scores is None or scores[i] > minScoreThresh:

                        # get bounding box
                        box = tuple(boxes[i].tolist())

                        # normalize coodinates to PIL image
                        ymin, xmin, ymax, xmax = box
                        w, h = rgbImage.size
                        cood = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))

                        # get detection class
                        if classes[i] in segmentCategoryIndex.keys():
                            className = str(segmentCategoryIndex[classes[i]]['name'])
                        else:
                            className = 'N/A'

                        # get calculated score
                        score = int(100 * scores[i])

                        res[className.lower()]['pos'] = cood
                        res[className.lower()]['score'] = score

                filename = filenames[k].split('/')[-2]+filenames[k].split('/')[-1]
                label(savePath+'/'+filename, np.array(im), res)

def label(filename, im, res):

    font = cv2.FONT_ITALIC

    if(res['top']):
        topLeft = (res['top']['pos'][0], res['top']['pos'][1])
        bottomRight = (res['top']['pos'][2], res['top']['pos'][3])
        cv2.rectangle(im, topLeft, bottomRight, (0, 255, 0), 4)
        text = 'top' + str(res['top']['score'])
        cv2.putText(im, text, topLeft, font, 1, (0, 0, 255), 1)

    if (res['bottom']):
        topLeft = (res['bottom']['pos'][0], res['bottom']['pos'][1])
        bottomRight = (res['bottom']['pos'][2], res['bottom']['pos'][3])
        cv2.rectangle(im, topLeft, bottomRight, (0, 255, 0), 4)
        text = 'bottom' + str(res['bottom']['score'])
        cv2.putText(im, text, topLeft, font, 1, (0, 0, 255), 1)

    if (res['full']):
        topLeft = (res['full']['pos'][0], res['full']['pos'][1])
        bottomRight = (res['full']['pos'][2], res['full']['pos'][3])
        cv2.rectangle(im, topLeft, bottomRight, (0, 255, 0), 4)
        text = 'full' + str(res['full']['score'])
        cv2.putText(im, text, topLeft, font, 1, (0, 0, 255), 1)

    cv2.imwrite(filename, im)

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

def featureExtractor(pbName, inputTensor,tensor, rgbImage):
    segmentDetectionGraph = tf.Graph()
    # tensor = 'FeatureExtractor/InceptionV2/Mixed_5c_2_Conv2d_5_3x3_s2_128/Relu6:0'
    # tensor = 'FeatureExtractor/InceptionV2/InceptionV2/Mixed_5b/concat:0'

    with segmentDetectionGraph.as_default():
        odGraphDef = tf.GraphDef()

        with tf.gfile.GFile(pbName, 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44
        # config.log_device_placement=True

    start = time.time()
    with tf.Session(graph=segmentDetectionGraph, config=config) as sess:
        im = cv2.cvtColor(np.array(rgbImage), cv2.COLOR_RGB2BGR)
        # imExpanded = np.expand_dims(im, axis=0)
        imExpanded = im
        imExpanded = np.expand_dims(im, axis=0)
        imExpanded = np.tile(imExpanded,(2,1,1,1))

        image_tensor = segmentDetectionGraph.get_tensor_by_name(inputTensor)
        feature = segmentDetectionGraph.get_tensor_by_name(tensor)
        res = sess.run(
            feature,
            feed_dict={image_tensor: imExpanded})

    end = time.time()
    print('takes {} recordes'.format(end - start))

    return res

def inception():

    inception_v4_graph = tf.Graph()
    with inception_v4_graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, 300, 300, 3))
        # inception_v4(inputs, 3)
        network_fn = nets_factory.get_network_fn(
            'inception_v4',
            num_classes=3,
            is_training=True)
        logits, end_points = network_fn(inputs)

        with tf.Session() as sess:
            # saver = tf.train.import_meta_graph('./inception_v4.ckpt')
            saver = tf.train.Saver()
            # model_file = tf.train.latest_checkpoint('./inception_v4.ckpt')
            saver.restore(sess, './inception_v4.ckpt')

            graph_def = inception_v4_graph.as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['InceptionV4/Logits/Predictions']  # We split on comma for convenience
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile('frozen.pb', "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

def testFastRCNN():

    im = cv2.imread('3.jpg')
    # im = cv2.resize(im, (227,227))

    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml')
    # rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    start = time.time()
    edges = edge_detection.detectEdges(np.float32(im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(100)
    edge_boxes.setMinScore(0.01)
    edge_boxes.setAlpha(0.65)
    edge_boxes.setBeta(0.65)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)
    boxes[:,2:4] = boxes[:,:2] + boxes[:,2:4]

    # end = time.time()
    # print('takes {} seconds'.format(end - start))

    # for b in boxes:
    #     x, y, w, h = b
    #     cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    # cv2.imshow("edges", edges)
    # cv2.imshow("edgeboxes", im)
    # cv2.waitKey(0)
    #
    #
    # cv2.destroyAllWindows()
    file_def_frcn = 'fashion_detector.prototxt'
    file_net_frcn = 'fashion_detector.caffemodel'
    model_frcn = fast_rcnn_load_net(file_def_frcn, file_net_frcn, False)
    bbox_pred = fast_rcnn_im_detect(model_frcn, im, boxes)

    print(bbox_pred[0][:,4])
    x, y, w, h, score = np.int32(bbox_pred[0][0,:])
    cv2.rectangle(im, (x, y), (w, h), (0, 255, 0), 1, cv2.LINE_AA)

    x, y, w, h, score = np.int32(bbox_pred[1][0, :])
    cv2.rectangle(im, (x, y), (w, h), (0, 255, 0), 1, cv2.LINE_AA)

    x, y, w, h, score = np.int32(bbox_pred[2][0, :])
    cv2.rectangle(im, (x, y), (w, h), (0, 255, 0), 1, cv2.LINE_AA)

    end = time.time()
    print('takes {} seconds'.format(end - start))

    # cv2.imshow("edges", edges)
    cv2.imshow("edgeboxes", im)
    cv2.waitKey(0)

def testFastrcnn(filenames, savePath):
    file_def_frcn = 'fashion_detector.prototxt'
    file_net_frcn = 'fashion_detector.caffemodel'
    model_frcn = fast_rcnn_load_net(file_def_frcn, file_net_frcn, False)
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection('model.yml')

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('test', cell_overwrite_ok=True)

    for index, filename in enumerate(filenames):
        ext = filename.split('.')[-1]
        if ext.lower() in ['jpg', 'png']:
            im = cv2.imread(filename)

            start = time.time()

            edges = edge_detection.detectEdges(np.float32(im) / 255.0)

            orimap = edge_detection.computeOrientation(edges)
            edges = edge_detection.edgesNms(edges, orimap)

            edge_boxes = cv2.ximgproc.createEdgeBoxes()
            edge_boxes.setMaxBoxes(100)
            edge_boxes.setMinScore(0.01)
            edge_boxes.setAlpha(0.65)
            edge_boxes.setBeta(0.65)
            boxes = edge_boxes.getBoundingBoxes(edges, orimap)
            boxes[:, 2:4] = boxes[:, :2] + boxes[:, 2:4]

            bbox_pred = fast_rcnn_im_detect(model_frcn, im, boxes)

            end = time.time()
            print("{}  {} / {}, takes {} seconds".format(filename, index, len(filenames),
                                                         end - start))

            font = cv2.FAST_FEATURE_DETECTOR_FAST_N

            sheet.write(index, 0, bbox_pred[0][0, 4])
            sheet.write(index, 1, bbox_pred[1][0, 4])
            sheet.write(index, 2, bbox_pred[2][0, 4])

            pred = np.int32(bbox_pred[0][0, :4])
            score = bbox_pred[0][0, 4]
            if score > 0.75:
                topLeft = (pred[0], pred[1])
                bottomRight = (pred[2], pred[3])
                cv2.rectangle(im, topLeft, bottomRight, (125, 0, 125), 4)
                text = 'top' + str(score*100)
                cv2.putText(im, text, topLeft, font, 1, (0, 0, 255), 2)

            pred = np.int32(bbox_pred[1][0, :4])
            score = bbox_pred[1][0, 4]
            if score > 0.75:
                topLeft = (pred[0], pred[1])
                bottomRight = (pred[2], pred[3])
                cv2.rectangle(im, topLeft, bottomRight, (0, 125, 125), 4)
                text = 'bottom' + str(score*100)
                cv2.putText(im, text, topLeft, font, 1, (0, 0, 255), 2)

            pred = np.int32(bbox_pred[2][0, :4])
            score = bbox_pred[2][0, 4]
            if score > 0.75:
                topLeft = (pred[0], pred[1])
                bottomRight = (pred[2], pred[3])
                cv2.rectangle(im, topLeft, bottomRight, (125, 125, 0), 4)
                text = 'full' + str(score*100)
                cv2.putText(im, text, topLeft, font, 1, (0, 0, 255), 2)

            filename_save = filename.split('/')[-2] + filename.split('/')[-1]
            cv2.imwrite(savePath + '/' + filename_save, im)

    book.save(r'test1.xls')

def excel():
    workbook = xlrd.open_workbook(u'test1.xls')

    sheet1 = workbook.sheet_by_name('test')
    sheet2 = workbook.sheet_by_name('test2')

    cols1 = np.array(sheet1.col_values(0))
    cols2 = np.array(sheet1.col_values(1))
    cols3 = np.array(sheet1.col_values(2))

    print(np.sum(cols1>0.75)*1.0/len(cols1))
    print(np.sum(cols2 > 0.75) * 1.0 / len(cols2))
    print(np.sum(cols3 > 0.75) * 1.0 / len(cols3))
    # book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    # sheet = book.add_sheet('test', cell_overwrite_ok=True)
    #
    # for i in range(len(cols1)):
    #     sheet2.write(i, 0, cols1[i])
    #     sheet2.write(i, 1, cols2[i])
    #     sheet2.write(i, 2, cols3[i])

def testRPN():
    im = cv2.imread('1.jpg')
    im = cv2.resize(im,(960, 720))

    rpnGraph = tf.Graph()

    with rpnGraph.as_default():
        odGraphDef = tf.GraphDef()

        with tf.gfile.GFile('rpn.pb', 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44
        # config.log_device_placement=True

    start = time.time()
    with tf.Session(graph=rpnGraph, config=config) as sess:
        im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        imExpanded = np.expand_dims(im, axis=0)
        # imExpanded = im
        # imExpanded = np.expand_dims(im, axis=0)
        # imExpanded = np.tile(imExpanded, (2, 1, 1, 1))

        input_tensor = rpnGraph.get_tensor_by_name('Placeholder:0')
        bbox = rpnGraph.get_tensor_by_name('content_rpn/bbox:0')
        prob = rpnGraph.get_tensor_by_name('content_rpn/prob:0')
        resBbox, resProb = sess.run(
            [bbox, prob],
            feed_dict={input_tensor: imExpanded})

        print(resBbox,resProb)
    end = time.time()
    print('takes {} recordes'.format(end - start))

def testBodyDetector():
    im = cv2.imread('1.jpg')
    im = imutils.resize(im, width=min(400, im.shape[1]))
    min_wdw_sz = (64, 128)
    step_size = (10, 10)
    downscale = 1.25

    clf = joblib.load('svm.model')

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0

    def sliding_window(image, window_size, step_size):
        '''
        This function returns a patch of the input 'image' of size
        equal to 'window_size'. The first image returned top-left
        co-ordinate (0, 0) and are increment in both x and y directions
        by the 'step_size' supplied.

        So, the input parameters are-
        image - Input image
        window_size - Size of Sliding Window
        step_size - incremented Size of Window

        The function returns a tuple -
        (x, y, im_window)
        '''
        for y in xrange(0, image.shape[0], step_size[1]):
            for x in xrange(0, image.shape[1], step_size[0]):
                yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

    for im_scaled in pyramid_gaussian(im, downscale=downscale):
        # The list contains detections at the current scale
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window = color.rgb2gray(im_window)
            fd = hog(im_window, 9, [6, 6], [2, 2])

            fd = fd.reshape(1, -1)
            pred = clf.predict(fd)

            if pred == 1:

                if clf.decision_function(fd) > 0.5:
                    detections.append(
                        (int(x * (downscale ** scale)), int(y * (downscale ** scale)), clf.decision_function(fd),
                         int(min_wdw_sz[0] * (downscale ** scale)),
                         int(min_wdw_sz[1] * (downscale ** scale))))

        scale += 1

    clone = im.copy()

    for (x_tl, y_tl, _, w, h) in detections:
        cv2.rectangle(im, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 255, 0), thickness=2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
    sc = [score[0] for (x, y, score, w, h) in detections]
    print "sc: ", sc
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
    print "shape, ", pick.shape

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    plt.axis("off")
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.title("Raw Detection before NMS")
    plt.show()

    plt.axis("off")
    plt.imshow(cv2.cvtColor(clone, cv2.COLOR_BGR2RGB))
    plt.title("Final Detections after applying NMS")
    plt.show()

def testBodyDetectorByOpencv():
    def inside(r, q):
        rx, ry, rw, rh = r
        qx, qy, qw, qh = q
        return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

    def draw_detections(img, rects, thickness=1):
        for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15 * w), int(0.05 * h)
            cv2.rectangle(img, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), thickness)

    im = cv2.imread('3.jpg')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(im, winStride=(8, 8), padding=(32, 32), scale=1.05)
    print(len(found), w)
    draw_detections(im, found)
    cv2.imshow('feed', im)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def testTime():
    ct = time.time()
    timeHead = time.strftime("%Y%m%d%H%M%S", time.localtime(ct))
    timeSec = (ct - long(ct)) * 1000
    curTime = "%s%03d" % (timeHead, timeSec)
    print(curTime)

def testPIL():
    im = Image.open('3.jpg')
    # im = cv2.imread('3.jpg')
    im = np.array(im)
    im = im[:,:,[2,1,0]]
    cv2.imshow('',im)
    cv2.waitKey(0)

    im = Image.fromarray(im)
    im.show('')

def testTry():
    im = []

    try:
        im.append(1)
        raise Exception("Invalid level!", 2)
    except:
        print('error')

    print im

def testMaskRCNN(filenames, savePath):
    maskGraph = tf.Graph()

    with maskGraph.as_default():
        odGraphDef = tf.GraphDef()

        with tf.gfile.GFile('maskRcnn.pb', 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        with tf.Session(graph=maskGraph, config=config) as sess:
            for k in range(len(filenames)):
                start = time.time()
                rgbImage = Image.open(filenames[k])

                # print('image shape: {}'.format(np.array(rgbImage).shape))
                imgTmp = np.array(rgbImage)
                if (len(imgTmp.shape) == 2):
                    imgTmp = np.expand_dims(imgTmp, axis=2)
                    imgTmp = np.tile(imgTmp, (1, 1, 3))
                im = cv2.cvtColor(imgTmp, cv2.COLOR_RGB2BGR)
                imExpanded = np.expand_dims(im, axis=0)

                image_tensor = maskGraph.get_tensor_by_name('image_tensor:0')
                boxes = maskGraph.get_tensor_by_name('detection_boxes:0')
                scores = maskGraph.get_tensor_by_name('detection_scores:0')
                classes = maskGraph.get_tensor_by_name('detection_classes:0')
                numDetections = maskGraph.get_tensor_by_name('num_detections:0')
                mask = maskGraph.get_tensor_by_name('detection_masks:0')

                (boxes, scores, classes, numDetections, mask) = sess.run(
                    [boxes, scores, classes, numDetections, mask],
                    feed_dict={image_tensor: imExpanded})

                end = time.time()
                print("{}  {} / {}, takes {} seconds".format(filenames[k], k, len(filenames),
                                                             end - start))

                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)

                print(boxes[0])
                print(mask.shape)

                cv2.imshow('', mask[0,0,:,:])
                cv2.waitKey(0)
                filename = filenames[k].split('/')[-2] + filenames[k].split('/')[-1]

def testDeepLab(filenames):
    label_colours = [(0, 0, 0)
        # 0=background
        , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
        # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
        , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
        # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
        , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
        # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
        , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

    def decode_labels(mask, num_images=1, num_classes=21):
        """Decode batch of segmentation masks.

        Args:
          mask: result of inference after taking argmax.
          num_images: number of images to decode from the batch.
          num_classes: number of classes to predict (including background).

        Returns:
          A batch with num_images RGB images of the same size as the input.
        """
        print(mask.shape)
        n, h, w, c= mask.shape
        assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
        n, num_images)
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

    deepLabGraph = tf.Graph()

    with deepLabGraph.as_default():
        odGraphDef = tf.GraphDef()

        with tf.gfile.GFile('deepLabv3.pb', 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        with tf.Session(graph=deepLabGraph, config=config) as sess:
            # for k in range(len(filenames)):
            start = time.time()
            rgbImage = Image.open('15.jpg')
            #rgbImage = rgbImage.resize((900, 900), Image.ANTIALIAS)


            # print('image shape: {}'.format(np.array(rgbImage).shape))
            imgTmp = np.array(rgbImage)
            if (len(imgTmp.shape) == 2):
                imgTmp = np.expand_dims(imgTmp, axis=2)
                imgTmp = np.tile(imgTmp, (1, 1, 3))
            im = cv2.cvtColor(imgTmp, cv2.COLOR_RGB2BGR)
            h, w = im.shape[0], im.shape[1]
            tmp = np.tile(IMG_MEAN, (h, w, 1))
            im = im - tmp
            imExpanded = np.expand_dims(im, axis=0)

            image_tensor = deepLabGraph.get_tensor_by_name('Input:0')
            sematic = deepLabGraph.get_tensor_by_name('Predictions:0')

            sematic = sess.run(
                sematic,
                feed_dict={image_tensor: imExpanded})

            end = time.time()
            # print("{}  {} / {}, takes {} seconds".format(filenames[k], k, len(filenames),end - start))
            print("takes {} seconds".format(end - start))

            # np.set_printoptions(threshold=np.inf)
            print(sematic[sematic != 0])
            res = decode_labels(sematic)
	    
	    cv2.imwrite('res2.jpg', res[0])
            #cv2.imshow('', res[0])
            #cv2.waitKey(0)
                # filename = filenames[k].split('/')[-2] + filenames[k].split('/')[-1]

def testStyle():
    # from tensorflow.core.framework import graph_pb2
    # import copy
    #
    # INPUT_GRAPH_DEF_FILE = 'style.pb'
    # OUTPUT_GRAPH_DEF_FILE = 'style2.pb'
    #
    # # load our graph
    # def load_graph(filename):
    #     graph_def = tf.GraphDef()
    #     with tf.gfile.FastGFile(filename, 'rb') as f:
    #         graph_def.ParseFromString(f.read())
    #     return graph_def
    #
    # graph_def = load_graph(INPUT_GRAPH_DEF_FILE)
    #
    # target_node_name = 'data:0'
    # c = tf.placeholder(tf.float32, (None, 244, 244, 3), name= 'input')
    #
    # # Create new graph, and rebuild it from original one
    # # replacing phase train node def with constant
    # new_graph_def = graph_pb2.GraphDef()
    # for node in graph_def.node:
    #     if node.name == target_node_name:
    #         new_graph_def.node.extend([c.op.node_def])
    #     else:
    #         new_graph_def.node.extend([copy.deepcopy(node)])
    #
    # # save new graph
    # with tf.gfile.GFile(OUTPUT_GRAPH_DEF_FILE, "wb") as f:
    #     f.write(new_graph_def.SerializeToString())




    deepLabGraph = tf.Graph()

    with deepLabGraph.as_default():
        odGraphDef = tf.GraphDef()

        with tf.gfile.GFile('inceptionv1.pb', 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        with tf.Session(graph=deepLabGraph, config=config) as sess:
            # for k in range(len(filenames)):
            start = time.time()
            rgbImage = Image.open('15.jpg')
            rgbImage = rgbImage.resize((224, 224), Image.ANTIALIAS)

            # print('image shape: {}'.format(np.array(rgbImage).shape))
            imgTmp = np.array(rgbImage)
            if (len(imgTmp.shape) == 2):
                imgTmp = np.expand_dims(imgTmp, axis=2)
                imgTmp = np.tile(imgTmp, (1, 1, 3))
            im = cv2.cvtColor(imgTmp, cv2.COLOR_RGB2BGR)
            h, w = im.shape[0], im.shape[1]
            tmp = np.tile(IMG_MEAN, (h, w, 1))
            im = im - tmp
            imExpanded = np.expand_dims(im, axis=0)

            image_tensor = deepLabGraph.get_tensor_by_name('Placeholder:0')
            print(tf.shape(image_tensor))
            # image_tensor = tf.reshape(image_tensor, (None, 244, 244, 3))
            sematic = deepLabGraph.get_tensor_by_name('InceptionV1/Logits/AvgPool_0a_7x7/AvgPool:0')

            sematic = sess.run(
                sematic,
                feed_dict={image_tensor: imExpanded})

            print(sematic.shape)
            print(sematic)

def inception_v1():

    inception_v1_graph = tf.Graph()
    with inception_v1_graph.as_default():
        inputs = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        # inception_v4(inputs, 3)
        network_fn = nets_factory.get_network_fn(
            'inception_v1',
            num_classes=1001,
            is_training=True)
        logits, end_points = network_fn(inputs)

        with tf.Session() as sess:
            # saver = tf.train.import_meta_graph('./inception_v4.ckpt')
            saver = tf.train.Saver()
            # model_file = tf.train.latest_checkpoint('./inception_v4.ckpt')
            saver.restore(sess, './inception_v1.ckpt')

            graph_def = inception_v1_graph.as_graph_def()
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['InceptionV1/Logits/SpatialSqueeze']  # We split on comma for convenience
            )

            # Finally we serialize and dump the output graph to the filesystem
            with tf.gfile.GFile('inceptionv1.pb', "wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("%d ops in the final graph." % len(output_graph_def.node))

def ssd_inceptionv2():
    rgbImage = Image.open('/home/lingdi/project/test/30.jpg')

    modelNet = tf.Graph()
    with modelNet.as_default():
        odGraphDef = tf.GraphDef()

        with tf.gfile.GFile('person.pb', 'rb') as fid:
            serializedGraph = fid.read()
            odGraphDef.ParseFromString(serializedGraph)
            tf.import_graph_def(odGraphDef, name='')

    with modelNet.as_default():
        # optimization for CPU detection
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        with tf.Session(graph=modelNet, config=config) as sess:
            im = cv2.cvtColor(np.array(rgbImage), cv2.COLOR_RGB2BGR)
            imExpanded = np.expand_dims(im, axis=0)
            image_tensor = modelNet.get_tensor_by_name('image_tensor:0')
            boxes = modelNet.get_tensor_by_name('detection_boxes:0')
            scores = modelNet.get_tensor_by_name('detection_scores:0')
            classes = modelNet.get_tensor_by_name('detection_classes:0')
            numDetections = modelNet.get_tensor_by_name('num_detections:0')

            (boxes, scores, classes, _) = sess.run(
                [boxes, scores, classes, numDetections],
                feed_dict={image_tensor: imExpanded})

            # determine boxes
            maxBoxesToDraw = 10
            minScoreThresh = .3

            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)
            idx = np.argsort(-np.array(classes))
            print(idx)

            for i in idx:
                if classes[i] == 1 and scores[i] > minScoreThresh:

                    # get bounding box
                    box = tuple(boxes[i].tolist())

                    # normalize coodinates to PIL image
                    ymin, xmin, ymax, xmax = box
                    w, h = rgbImage.size
                    cood = (int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h))
                    cv2.rectangle(im, (cood[0], cood[1]), (cood[2], cood[3]), (0, 255, 0), 2)

                    break

    cv2.imshow('After', im)
    cv2.waitKey(0)


if __name__ == '__main__':

    # test boxes and score
    # filenames = []
    # for root, dirs, files in os.walk('/home/lingdi/project/test/images'):
    #     for fn in files:
    #         filenames.append(root + os.sep + fn)
    #         # print(root + os.sep + fn)
    #
    # # filenames = ['/home/lingdi/Downloads/Img/img/Striped_Mandarin_Collar_Blouse/img_00000055.jpg']
    # detect(filenames, '/home/lingdi/project/test/test/new_res',
    #        'inception.pb')


    # print tensor name
    # printGraph('rpn.pb')
    # printGraph('classify_image_graph_def.pb')
    # printGraph('maskRcnn.pb')
    # printGraph('inception_v4.pb')
    # printGraph('tensorflow_inception_graph.pb')
    # printGraph('inceptionv1.pb')

    # extract feature
    # im = Image.open('1.jpg')
    # im = im.resize((300,300))
    # feature = featureExtractor('inception_v4.pb','Placeholder:0',
    #                            'InceptionV4/Logits/AvgPool_1a/AvgPool:0', im)
    # print(feature.shape)
    # print(np.array(feature).shape)

    # freeze inception_v4
    # inception()

    # test fast RCNN
    # testFastRCNN()
    # filenames = []
    # for root, dirs, files in os.walk('/home/lingdi/project/test/images'):
    #     for fn in files:
    #         filenames.append(root + os.sep + fn)
    # testFastrcnn(filenames, '/home/lingdi/project/test/test/fastrcnn')

    # test RPN people detection
    # testRPN()
    # testBodyDetectorByOpencv()

    # test time format
    # testTime()
    # testPIL()

    # test try
    # testTry()

    # test mask rcnn
    # filenames = []
    # for root, dirs, files in os.walk('/home/lingdi/project/test/images'):
    #     for fn in files:
    #         filenames.append(root + os.sep + fn)
    # testMaskRCNN(filenames, '')

    # test deepLab v3
    #filenames = []
    #for root, dirs, files in os.walk('/home/lingdi/project/test/images'):
    #    for fn in files:
    #        filenames.append(root + os.sep + fn)
    #testDeepLab(filenames)

    # test excel
    # excel()

    # test style
    # testStyle()

    # inception_v1
    # inception_v1()

    # test person detecting
    ssd_inceptionv2()

    # im = Image.open('1.jpg')
    # # im = cv2.imread('1.jpg')
    # print(type(im))
    # res = detect(im)
    #
    # im = cv2.imread('1.jpg')
    # topLeft = (res['top']['pos'][0],res['top']['pos'][1])
    # bottomRight = (res['top']['pos'][2], res['top']['pos'][3])
    # cv2.rectangle(im, topLeft, bottomRight, (0, 255, 0), 4)
    #
    # font = cv2.FONT_ITALIC
    # text = 'top' + str(res['top']['score'])
    # cv2.putText(im, text, topLeft, font, 1, (0, 0, 255), 1)
    # cv2.imshow('1',im)
    # cv2.waitKey()
    # print(res)


