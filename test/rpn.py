class BodyDetector:
    def __init__(self):

        self.image_height = 720
        self.image_width = 960

        self.convmap_height = int(np.ceil(self.image_height / 16.))
        self.convmap_width = int(np.ceil(self.image_width / 16.))

        self.anchor_size = 9

        self.bbox_normalize_scale = 5
        self.wandh = [[100.0, 100.0], [300.0, 300.0], [500.0, 500.0],
          [200.0, 100.0], [370.0, 185.0], [440.0, 220.0],
          [100.0, 200.0], [185.0, 370.0], [220.0, 440.0]]
        self.proposal_prepare()

        self.modelRoot = getConfig('deep_learning', 'model_root')
        self.graph = self.getGraph(self.modelRoot + '/bodyDetector.pb')

    def detectBody(self, im):
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        start = time.time()
        with tf.Session(graph=self.graph, config=config) as sess:

            input_tensor = self.graph.get_tensor_by_name('Placeholder:0')
            bbox = self.graph.get_tensor_by_name('content_rpn/bbox:0')
            prob = self.graph.get_tensor_by_name('content_rpn/prob:0')

            im = np.array(im)
            im = cv2.resize(im, (960, 720))
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            imExpanded = np.expand_dims(im, axis=0)

            test_bbox_pred, test_prob = sess.run([bbox, prob],
                                                 feed_dict={input_tensor: imExpanded})

            bboxes = self.rpn_nms(test_prob, test_bbox_pred)

            containBody = False
            for bbox in bboxes:
                if bbox[4] > 0.99:
                    containBody = True
                    break

        end = time.time()
        logger.info('detect body takes: {} seconds'.format(end - start))

        return containBody


    def getGraph(self, pbName):
        rpnGraph = tf.Graph()

        with rpnGraph.as_default():
            odGraphDef = tf.GraphDef()

            with tf.gfile.GFile(pbName, 'rb') as fid:
                serializedGraph = fid.read()
                odGraphDef.ParseFromString(serializedGraph)
                tf.import_graph_def(odGraphDef, name='')

        return rpnGraph

    def rpn_nms(self, prob, bbox_pred):
        prob = prob[:, 0]
        bbox_pred /= self.bbox_normalize_scale
        anchors = self.proposals.copy()
        anchors[:, 2] -= anchors[:, 0]
        anchors[:, 3] -= anchors[:, 1]
        anchors[:, 0] = bbox_pred[:, 0] * anchors[:, 2] + anchors[:, 0]
        anchors[:, 1] = bbox_pred[:, 1] * anchors[:, 3] + anchors[:, 1]
        anchors[:, 2] = np.exp(bbox_pred[:, 2]) * anchors[:, 2]
        anchors[:, 3] = np.exp(bbox_pred[:, 3]) * anchors[:, 3]
        bbox = np.zeros([anchors.shape[0], 5])

        bbox[:, :4] = anchors
        bbox[:, 4] = prob
        bbox = self.filter_bbox(bbox)
        bbox = self.non_max_suppression_fast(bbox, 0.7)

        keep_prob = np.sort(bbox[:, 4])[max(-50, -1 * bbox.shape[0])]

        index = np.where(bbox[:, 4] >= keep_prob)[0]
        bbox = bbox[index]
        return bbox

    def proposal_prepare(self):

        anchors = self.generate_anchors()
        proposals = np.zeros([self.anchor_size * self.convmap_width * self.convmap_height, 4])

        for i in range(self.convmap_height):
            h = i * 16 + 8
            for j in range(self.convmap_width):
                w = j * 16 + 8
                for k in range(self.anchor_size):
                    index = i * self.convmap_width * self.anchor_size + j * self.anchor_size + k
                    anchor = anchors[k, :]
                    proposals[index, :] = anchor + np.array([w, h, w, h])

        self.proposals = proposals

    def generate_anchors(self):
        anchors = np.zeros([self.anchor_size, 4])

        for i in range(self.anchor_size):
            anchor_width = self.wandh[i][0]
            anchor_height = self.wandh[i][1]
            anchors[i, :] = np.array(
                [-0.5 * anchor_width, -0.5 * anchor_height, 0.5 * anchor_width, 0.5 * anchor_height])
        return anchors

    def non_max_suppression_fast(self, boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return boxes

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        xw = boxes[:, 2]
        yh = boxes[:, 3]
        x2 = x1 + xw
        y2 = y1 + yh

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = xw * yh
        idxs = np.argsort(boxes[:, 4])

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            # compute the ratio of overlap
            overlap = (w * h) / np.minimum(area[idxs[:last]], area[i])

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        return boxes[pick]

    def filter_bbox(self, bbox):
        xrng = [0.1, self.image_width-0.1]
        yrng = [0.1, self.image_height-0.1]

        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 0] + bbox[:, 2]
        y2 = bbox[:, 1] + bbox[:, 3]
        keep = np.where((x1 > xrng[0]) & (x2 < xrng[1]) & (y1 > yrng[0]) & (y2 < yrng[1]))[0]

        return bbox[keep, :]