# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from PIL import Image
import sys
import caffe
import cv2
import math

def fast_rcnn_im_detect(model, im, boxes):
    # Perform detection a Fast R-CNN network given an image and
    # object proposals.

    im_batch, scales = image_pyramid(im, model['pixel_means'], False)

    feat_pyra_boxes, feat_pyra_levels = project_im_rois(boxes, scales)
    rois = np.concatenate((feat_pyra_levels, feat_pyra_boxes),axis=1)
    # Adjust to 0-based indexing and make roi info the fastest dimension
    rois = rois
    # rois = rois.transpose((1, 0))

    input_blobs = []
    input_blobs.append(im_batch)
    input_blobs.append(rois)

    # blobs_out = caffe('forward', input_blobs)
    model['net'].blobs['data'].reshape(1,3,600,600)
    model['net'].blobs['data'].data[...] = im_batch.transpose((3,2,1,0))
    num_rois = rois.shape[0]
    model['net'].blobs['rois'].reshape(num_rois, 5)
    model['net'].blobs['rois'].data[...] = rois
    blobs_out = model['net'].forward()

    # print('im_batch shape : {}'.format(im_batch.shape))

    bbox_deltas = np.squeeze(blobs_out['bbox_pred'])
    probs = np.squeeze(blobs_out['cls_prob'])
    # print('probs shape : {}'.format(probs))

    num_classes = probs.shape[1]
    dets = {}
    NMS_THRESH = 0.3
    # class index 1 is __background__, so we don't return it
    for j in range(num_classes):
        if j==0:
            continue
        cls_probs = probs[:, j]
        cls_deltas = bbox_deltas[:, (j * 4):((j+1) * 4)]
        pred_boxes = bbox_pred(boxes, cls_deltas)
        # print('pred_boxes shape : {}'.format(pred_boxes.shape))
        # print('cls_probs shape : {}'.format(cls_probs.shape))
        cls_dets = np.concatenate((pred_boxes, cls_probs.reshape((-1,1))), axis=1)
        keep = nms(cls_dets, NMS_THRESH)
        cls_dets = cls_dets[np.int32(keep), :]
        dets[j - 1]= cls_dets


    return dets

def image_pyramid(im, pixel_means, multiscale):
    # Construct an image pyramid that's ready for feeding directly into caffe
    if not multiscale:
      SCALES = np.array([600])
      MAX_SIZE = 1000
    else:
      SCALES = np.array([1200,864,688,576,480])
      MAX_SIZE = 2000

    num_levels = SCALES.size

    # im = single(im)
    # Convert to BGR
    im = im[:,:,[2,1,0]]
    # print('image shape {}'.format(im.shape))
    # Subtract mean (mean of the image mean--one mean per channel)
    # im = bsxfun(@minus, im, pixel_means);
    h, w = im.shape[0], im.shape[1]
    tmp = np.tile(pixel_means, (h,w,1))
    im = im - tmp

    im_orig = im
    im_size = min(im_orig.shape[0],im_orig.shape[1])
    im_size_big = max(im_orig.shape[0],im_orig.shape[1])
    scale_factors = np.double(SCALES) / im_size
    scale_factors = scale_factors.reshape((-1,1))

    max_size = np.array([0,0,0])
    ims = {}

    for i in range(num_levels):
        if round(im_size_big * scale_factors[i]) > MAX_SIZE:
            scale_factors[i] = MAX_SIZE / im_size_big

        # print('im_orig shape {}'.format(im_orig.shape))
        width, height = im_orig.shape[0], im_orig.shape[1]
        # print('new shape {} {}'.format(np.int32(width*scale_factors[i]),
        #                                np.int32(height*scale_factors[i])))
        ims[i] = cv2.resize(im_orig,(np.int32(width*scale_factors[i]),
                                      np.int32(height*scale_factors[i])))
        # ims[i] = imresize(im_orig, scale_factors(i), 'bilinear',\
        #                 'antialiasing', false)
        # print('ims shape {}'.format(ims[i].shape))
        max_size[0] = max(max_size[0], ims[i].shape[0])
        max_size[1] = max(max_size[1], ims[i].shape[1])
        max_size[2] = max(max_size[2], ims[i].shape[2])

    batch = np.zeros((max_size[0], max_size[1], 3, num_levels))
    for i in range(num_levels):
        im = ims[i]
        im_sz = im.shape
        im_sz = im_sz[0:2]
        # Make width the fastest dimension (for caffe)
        im = im.transpose((1,0,2))

        # print('im_sz:{}, batch:{}'.format(im_sz, batch.shape))
        batch[0:im_sz[0], 0:im_sz[1], :, i] = im

    scales = scale_factors
    return (batch, scales)

def project_im_rois(boxes, scales):
    # ------------------------------------------------------------------------
    boxes = np.array(boxes)
    widths = boxes[:,2] - boxes[:,0]
    heights = boxes[:,3] - boxes[:,1]

    areas = np.array(widths)*np.array(heights)
    areas = areas.reshape((-1,1))
    scalesTmp = (np.array(scales)**2).reshape((1,-1))
    scaled_areas = areas * scalesTmp
    # print('scaled_areas shape {}'.format(scaled_areas.shape))
    diff_areas = abs(scaled_areas - (224 * 224))
    levels = np.argmin(diff_areas, axis=1)

    boxes = boxes * scales[levels].reshape((-1,1))

    return (boxes, levels.reshape((-1,1)))

def bbox_pred(boxes, bbox_deltas):

    if len(boxes)==0:
      pred_boxes = []
      return pred_boxes

    Y = bbox_deltas
    # print('bbox_deltas {}'.format(bbox_deltas.shape))
    # print('boxes {}'.format(boxes.shape))

    # Read out predictions
    dst_ctr_x = Y[:, 0]
    dst_ctr_y = Y[:, 1]
    dst_scl_x = Y[:, 2]
    dst_scl_y = Y[:, 3]

    src_w = boxes[:, 2] - boxes[:, 0]
    src_h = boxes[:, 3] - boxes[:, 1]
    src_ctr_x = boxes[:, 0] + 0.5 * src_w
    src_ctr_y = boxes[:, 1] + 0.5 * src_h

    pred_ctr_x = dst_ctr_x * src_w + src_ctr_x
    pred_ctr_y = dst_ctr_y * src_h + src_ctr_y
    pred_w = np.exp(dst_scl_x) * src_w
    pred_h = np.exp(dst_scl_y) * src_h

    npTmp1 = (pred_ctr_x - 0.5 * pred_w).reshape((-1, 1))
    npTmp2 = (pred_ctr_y - 0.5 * pred_h).reshape((-1, 1))
    npTmp3 = (pred_ctr_x + 0.5 * pred_w).reshape((-1, 1))
    npTmp4 = (pred_ctr_y + 0.5 * pred_h).reshape((-1, 1))
    pred_boxes = np.concatenate((npTmp1,npTmp2,npTmp3,npTmp4), axis=1)

    return pred_boxes

def nms(boxes, overlap):
    # top = nms(boxes, overlap)
    # Non-maximum suppression. (FAST VERSION)
    # Greedily select high-scoring detections and skip detections
    # that are significantly covered by a previously selected
    # detection.
    #
    # NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
    # but an inner loop has been eliminated to significantly speed it
    # up in the case of a large number of boxes

    # Copyright (C) 2011-12 by Tomasz Malisiewicz
    # All rights reserved.
    #
    # This file is part of the Exemplar-SVM library and is made
    # available under the terms of the MIT license (see COPYING file).
    # Project homepage: https://github.com/quantombone/exemplarsvm

    if len(boxes) == 0:
        return boxes

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]

    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    # print('probs shape : {}'.format(boxes[I][:,4]))

    pick = s*0
    counter = 0
    while len(I)!=0:
        last = len(I)
        i = I[last-1]
        # print('probs : {}'.format(boxes[i, 4]))
        pick[counter] = i
        counter = counter + 1

        xx1 = np.maximum(x1[i], x1[I[:last]])
        yy1 = np.maximum(y1[i], y1[I[:last]])
        xx2 = np.minimum(x2[i], x2[I[:last]])
        yy2 = np.minimum(y2[i], y2[I[:last]])

        w = np.maximum(0.0, xx2-xx1)
        h = np.maximum(0.0, yy2-yy1)

        inter = w*h
        o = inter / (area[i] + area[I[:last]] - inter)

        # I = I(find(o<=overlap))
        I = [I[i] for i in range(len(o)) if o[i]<overlap]


    pick = pick[0:(counter-2)]

    return pick

def fast_rcnn_load_net(mdoel_def, model_net, use_gpu):
    # Load a Fast R-CNN network.
    model = {}

    model['net'] = caffe.Net(mdoel_def, model_net, caffe.TEST)
    # init_key = caffe('init', mdoel_def, model_net, 'test')
    if use_gpu:
        caffe.set_mode_gpu()
    else:
        caffe.set_mode_cpu()


    # model['init_key'] = init_key
    # model.stride is correct for the included models, but may not be correct
    # for other models!
    model['stride'] = 16;
    model['pixel_means'] = np.array([102.9801, 115.9465, 122.7717]).reshape((1,1,3))
    return model