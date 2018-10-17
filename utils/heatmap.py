import numpy as np
import cv2
from PIL import Image
from io import BytesIO

def np_draw_labelmap(pt, heatmap_sigma, heatmap_size, type='Gaussian'):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    img = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)
    if pt[0] < 1 or pt[1] < 1:
        return (img, 0)

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * heatmap_sigma), int(pt[1] - 3 * heatmap_sigma)]
    br = [int(pt[0] + 3 * heatmap_sigma + 1), int(pt[1] + 3 * heatmap_sigma + 1)]
    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return (img, 0)

    # Generate gaussian
    size = 6 * heatmap_sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    if type == 'Gaussian':
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * heatmap_sigma ** 2))
    elif type == 'Cauchy':
        g = heatmap_sigma / (((x - x0) ** 2 + (y - y0) ** 2 + heatmap_sigma ** 2) ** 1.5)


    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return (img, 1)

def convertLandmark2Heatmap(landmarks, height, width):
    def heatmapCompress(heatmap):
        heatmap = heatmap * 255 * 255
        heatmap[heatmap > 255] = 255
        image = Image.fromarray(np.uint8(heatmap), 'L')
        # image.show('')
        # g = np.asarray(image)
        # print('heatmapCompress:{}'.format(g[0:20, 0:20]))
        # cv2.imshow('', heatmap)
        # cv2.waitKey()

        with BytesIO() as output:
            image.save(output, 'JPEG')
            data = output.getvalue()
        return data

    heatmaps = []
    for landmark in landmarks:
        img = np.zeros((height, width))
        if landmark[0] >= 0 and landmark[1] >= 0:
            # print('convertLandmark2Heatmap:{}'.format(img.shape))
            img[int(landmark[0]), int(landmark[1])] = 1
        img = cv2.GaussianBlur(img, (31, 31), 0)
        # img = cv2.resize(img, (28, 28))
        heatmaps.append(img)
    heatmaps = np.array(heatmaps)

    # background = 1 - np.max(heatmaps, axis=0)
    # background = np.expand_dims(background, axis=0)
    #
    # heatmaps = np.concatenate([heatmaps, background], axis=0)

    # heatmaps = heatmaps.transpose([1, 2, 0])
    # return np.array(heatmaps)

    compressedHeatmaps = []
    num_heatmaps = heatmaps.shape[0]
    for heatmap in heatmaps:
        compressedHeatmaps.append(heatmapCompress(heatmap))

    return compressedHeatmaps