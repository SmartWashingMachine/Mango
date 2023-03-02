import numpy as np
import albumentations as A
import logging
import copy

from gandy.onnx_models.detrg import DETRGONNX
from gandy.text_detection.base_image_detection import BaseImageDetection

# See: https://github.com/pytorch/vision/issues/942
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def tnms(dets, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores ia  nump array : num_dets,
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


logger = logging.getLogger('Gandy')

class DETRGImageDetectionApp(BaseImageDetection):
    def __init__(self):
        """
        This app uses a custom DETR (pretrained with resnet50, finetuned on a custom dataset).

        Why the "G" at the end? The default IoU loss was replaced with the complete IoU loss (combining IoU, distance, as well as aspect ratio differences) when finetuned.
        """
        super().__init__()

        self.confidence_threshold = None
        self.transform = self.get_image_transform()

    def get_image_transform(self):
        transforms = [A.ToGray(always_apply=True)]

        return A.Compose(transforms)

    def load_model(self):
        if not hasattr(self, 'model'):
            raise RuntimeError('DETRGImageDetectionApp is a base app intended to be used by other classes. Instead, use DETRGBig.')

        self.loaded = True

    def detect_bboxes(self, image):
        image = image.convert('RGB') # Needs 3 channels.

        logger.info('Transforming image before passing it into object detection model...')

        t_x = self.transform(image=np.array(image))['image']
        # No need for normalization - the feature extractor will take care of that, as well as resizing.
        return [self.model.full_pipe(t_x)]

    # Sorts a group of bbox tensors such that the tensors with the highest top-right values are first.
    # Note: This does not modify in place.
    def sort_bboxes(self, bboxes):
        # Create copy of bboxes C
        # Set top rights to order of maxes
        # Set C second : == A[max]
        new_bboxes = copy.deepcopy(bboxes)

        if new_bboxes.shape[0] == 0:
            # No boxes to sort.
            return new_bboxes

        # x2 - y1 = topright
        # Since argsort sorts by ascending order, we invert the top rights so that the "closest" points are the most negative.
        top_rights = -(bboxes[:, 2] - bboxes[:, 1])
        # argsort returns the indices that would sort the top_rights list in ascending order.
        # for example, [1, 5, 3] would become [0, 2, 1] as 0-index corresponds to 1, 2-index corresponds to 3, and 1-index corresponds to 5. ([1, 3, 5])
        sorted_indices = np.argsort(top_rights)

        li = len(sorted_indices)
        for i in range(li):
            sorted_index = sorted_indices[i]

            new_bboxes[i, :] = bboxes[sorted_index, :]

        return new_bboxes

    def fuse_boxes(self, all_dict_outputs, image_width, image_height):
        iou_thr = 0.25

        logger.debug(f'Using non maximum suppression to postprocess object detections... (Boxes #: {all_dict_outputs[0]["boxes"].shape[0]})')
        # NMS
        keep = tnms(
            all_dict_outputs[0]['boxes'],
            all_dict_outputs[0]['scores'],
            thresh=iou_thr,
        )

        processed_boxes = all_dict_outputs[0]['boxes'][keep, :]
        scores = all_dict_outputs[0]['scores'][keep] # TODO: Val.

        if self.confidence_threshold is not None:
            boxes = []
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    boxes.append(processed_boxes[i])

            boxes = np.stack(boxes, axis=0)
        else:
            boxes = processed_boxes

        logger.debug(f'Done using NMS! (Filtered Boxes #: {boxes.shape[0]})')

        return boxes

    def process(self, image):
        image_width, image_height = image.size

        logger.debug('Detecting boxes...')
        dict_output = self.detect_bboxes(image) # No .copy? Stable? We'll see later on...

        logger.debug('Fusing boxes...')
        bboxes = self.fuse_boxes(dict_output, image_width, image_height)

        logger.debug('Sorting boxes...')
        bboxes = self.sort_bboxes(bboxes)

        return bboxes.tolist()

class DETRGBigImageDetectionApp(DETRGImageDetectionApp):
    def __init__(self, model_name = 'big', confidence_threshold = None):
        """
        Slower than the RCNNImageDetectionApp, but more precise.
        """
        super().__init__()

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

    def load_model(self):
        logger.info('Loading object detection model...')
        self.model = DETRGONNX(f'models/detrg/{self.model_name}.onnx', 'models/detrg/big_fe', use_cuda=self.use_cuda)

        logger.info('Done loading object detection model!')
        return super().load_model()

class DETRVnImageDetectionApp(DETRGBigImageDetectionApp):
    def __init__(self):
        super().__init__(model_name='vn', confidence_threshold=0.35)

# Quantized models performs poorly. VERY poorly.
