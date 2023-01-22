from transformers.models.detr.feature_extraction_detr import DetrFeatureExtractor
from scipy.special import softmax
import numpy as np

# See: https://stackoverflow.com/questions/64097426/is-there-unstack-in-numpy (or: https://github.com/google/jax/discussions/11028)
def unstack(a, axis=0):
    return np.moveaxis(a, axis, 0)

def center_to_corners_format(x):
    # Old: x_c, y_c, w, h = x.unbind(-1)
    # No unpacking in numpy? TODO. Or maybe just squeeze in unstack.
    x_c, y_c, w, h = unstack(x, axis=-1)

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1)

class DetrFeatureExtractorNumpy(DetrFeatureExtractor):
    def post_process(self, outputs, target_sizes):
        """
        Converts the output of [`DetrForObjectDetection`] into the format expected by the COCO api. Only supports
        PyTorch.

        Args:
            outputs ([`DetrObjectDetectionOutput`]):
                Raw outputs of the model.
            target_sizes (`torch.Tensor` of shape `(batch_size, 2)`):
                Tensor containing the size (h, w) of each image of the batch. For evaluation, this must be the original
                image size (before any data augmentation). For visualization, this should be the image size after data
                augment, but before padding.

        Returns:
            `List[Dict]`: A list of dictionaries, each dictionary containing the scores, labels and boxes for an image
            in the batch as predicted by the model.
        """
        out_logits, out_bbox = outputs.logits, outputs.pred_boxes

        if len(out_logits) != len(target_sizes):
            raise ValueError("Make sure that you pass in as many target sizes as the batch dimension of the logits")
        if target_sizes.shape[1] != 2:
            raise ValueError("Each element of target_sizes must contain the size (h, w) of each image of the batch")

        # Old: prob = nn.functional.softmax(out_logits, -1)
        prob = softmax(out_logits, -1)

        # Old: scores, labels = prob[..., :-1].max(-1)
        labels = np.argmax(prob[..., :-1], axis=-1)
        scores = prob[..., :-1][labels]

        # convert to [x0, y0, x1, y1] format
        boxes = center_to_corners_format(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # Old: img_h, img_w = target_sizes.unbind(1)
        img_h, img_w = unstack(target_sizes, axis=1)

        # Old: scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)

        boxes = boxes * scale_fct[:, None, :]

        results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]

        return results
