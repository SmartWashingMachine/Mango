import numpy as np
import albumentations as A
from ensemble_boxes import weighted_boxes_fusion, nms
import albumentations as A
import cv2
import logging
from random import random
import copy

from gandy.text_detection.base_image_detection import BaseImageDetection
from gandy.onnx_models.rcnn50 import RCnnONNX

logger = logging.getLogger('Gandy')

class RCNNImageDetectionApp(BaseImageDetection):
    def __init__(self, use_tta = False, do_resize = None):
        """
        This app uses a custom R-CNN model (pretrained with resnet50, finetuned on a custom dataset).

        It also supports test time augmentation, albeit with questionable accuracy improvements.

        It also supports weighted box fusion.

        If do_resize is a tuple of (height, width), then the image is scaled to that size before detecting text, and the boxes are rescaled afterwards.
        """
        self.use_tta = use_tta

        super().__init__()

        self.do_resize = do_resize
        self.transform = self.get_image_transform(do_resize)

        self.fuse_mode = 'wbf'

    def channels_last_to_channels_first(self, n):
        # expects unbatched input.
        return np.transpose(n, (2, 0, 1))

    def get_image_transform(self, resize):
        transforms = [A.ToGray(always_apply=True)]
        if resize is not None:
            transforms = [A.Resize(resize[0], resize[1], interpolation=cv2.INTER_AREA), A.ToGray(always_apply=True)]

        return A.Compose(transforms)

    def load_model(self):
        if not hasattr(self, 'model'):
            raise RuntimeError('RCNNImageDetectionApp is a base app intended to be used by other classes. Instead, use ResNetImageDetectionApp or ConvNextImageDetectionApp.')

        self.loaded = True

    def detect_bboxes(self, image):
        # Transform the image into a tensor first.
        image_width, image_height = image.size

        transform = self.transform

        def _use_model(img):
            # Model expects a batch, but we only want to send a single image. Unsqueeze will make it a batch with one image.
            batch = img[np.newaxis, ...]

            # The model will only return the first item (as a dict with tensors).
            return self.model(batch)

        # Test time augmentation may boost accuracy at the cost of slower inference.
        if self.use_tta:
            logger.info('Using TTA for object detection. This may improve accuracy at the cost of a much slower runtime.')

            t_x = np.array(image) # Albumentations requires PIL images to be converted to a Numpy array first.
            all_outputs = []

            total_runs = 5
            # TO-DO: Optimize this. The images shouldn't be sent one at a time like this...
            for i in range(total_runs):
                do_unaugmented_run = i == 0 # The first run will be unaugmented.

                if do_unaugmented_run:
                    extra_transforms = []
                else:
                    # Pixelwise transforms.
                    extra_transforms = [
                        A.Sharpen(p=0.5),
                        A.InvertImg(p=0.5),
                    ]

                flip_prob = random()
                if flip_prob <= 0.5 and not do_unaugmented_run:
                    extra_transforms.append(A.HorizontalFlip(p=1, always_apply=True))

                if len(extra_transforms) > 0:
                    composed_transform = A.Compose(extra_transforms)

                    done_transform = composed_transform(image=t_x)
                    augmented_image = done_transform['image']
                    # TO-DO: Not efficient.
                    augmented_image = transform(augmented_image)
                else:
                    augmented_image = transform(t_x)

                augmented_image = self.channels_last_to_channels_first(augmented_image)
                augmented_image = augmented_image / 255.
                # This contains the bounding boxes of the augmented image.
                output = _use_model(augmented_image)

                if flip_prob <= 0.5 and not do_unaugmented_run:
                    # Unflip the bboxes.
                    # Code borrowed from torchvision.
                    # NOTE: This may break if we use transforms that change the image size.
                    output['boxes'][:, [0, 2]] = image_width - output['boxes'][:, [2, 0]]
                    # TO-DO: Optimize this.
                    output['boxes'] = self.sort_bboxes(output['boxes'])

                all_outputs.append(output)

            return all_outputs
        else:
            logger.info('Normalizing image before passing it into object detection model...')

            t_x = transform(image=np.array(image))['image']
            t_x = self.channels_last_to_channels_first(t_x)
            t_x = t_x / 255.
            return [_use_model(t_x)]

    # Sorts a group of bbox tensors such that the tensors with the highest top-right values are first.
    # Note: This does not modify in place.
    def sort_bboxes(self, bboxes):
        # Create copy of bboxes C
        # Set top rights to order of maxes
        # Set C second : == A[max]
        new_bboxes = copy.deepcopy(bboxes)

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
        iou_thr = 0.15
        skip_box_thr = 0.75 # confidence threshold

        boxes_per_iter = []

        def _get_output_per_iter(dict_output):
            boxes = dict_output['boxes'].tolist()
            boxes = [ [b[0] / image_width, b[1] / image_height, b[2] / image_width, b[3] / image_height] for b in boxes ]

            scores = dict_output['scores'].tolist()
            labels = dict_output['labels'].tolist()

            return boxes, scores, labels

        boxes_per_iter, scores_per_iter, labels_per_iter = [], [], []
        for o in all_dict_outputs:
            output = _get_output_per_iter(o)
            boxes_per_iter.append(output[0])
            scores_per_iter.append(output[1])
            labels_per_iter.append(output[2])

        if self.fuse_mode == 'wbf':
            logger.debug('Using weighted box fusion to postprocess object detections...')

            # WBF
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_per_iter,
                scores_per_iter,
                labels_per_iter,
                weights=None,
                iou_thr=iou_thr,
                skip_box_thr=skip_box_thr,
            )
        else:
            logger.debug('Using non maximum suppression to postprocess object detections...')

            # NMS
            _boxes, scores, labels = nms(
                boxes_per_iter,
                scores_per_iter,
                labels_per_iter,
                weights=None,
                iou_thr=iou_thr,
            )
            boxes = []
            for i in range(len(scores)):
                if scores[i] >= skip_box_thr:
                    boxes.append(_boxes[i])

        boxes = [ [b[0] * image_width, b[1] * image_height, b[2] * image_width, b[3] * image_height] for b in boxes ]

        if len(boxes) == 0:
            # TODO: Test this case. May crash? Inefficient at the very least.
            logger.warning('No boxes were found after filtering for the image.')
            return np.zeros((1, 4))

        return boxes

    def rescale_bboxes(self, bboxes, image_width, image_height):
        if self.do_resize is None:
            return bboxes

        logger.info('Rescaling boxes...')

        width_ratio = image_width / self.do_resize[1]
        height_ratio = image_height / self.do_resize[0]
        for bb in bboxes:
            bb[0] *= width_ratio
            bb[1] *= height_ratio
            bb[2] *= width_ratio
            bb[3] *= height_ratio

        return bboxes

    def process(self, image):
        image_width, image_height = image.size

        logger.debug('Detecting boxes...')
        dict_output = self.detect_bboxes(image)

        logger.debug('Fusing boxes...')
        bboxes = self.fuse_boxes(dict_output, image_width, image_height)

        logger.debug('Sorting boxes...')
        bboxes = self.sort_bboxes(bboxes)

        bboxes = self.rescale_bboxes(bboxes, image_width, image_height)
        return bboxes.tolist()

class ResNetImageDetectionApp(RCNNImageDetectionApp):
    def __init__(self, use_tta=False):
        super().__init__(use_tta, do_resize=None)

    def load_model(self):
        logger.info('Loading object detection model...')
        # It's actually a mobilenet lol. Use to be ResNet.
        # NOTE: Is this actually the ResF? O.o I forgot...
        self.model = RCnnONNX('models/rcnn/v5_rese.onnx', use_cuda=self.use_cuda)

        logger.info('Done loading object detection model!')
        return super().load_model()
