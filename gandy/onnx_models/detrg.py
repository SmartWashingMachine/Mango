import numpy as np
from gandy.utils.tokenizer_utils.detr_image_processor import DetrFeatureExtractorNumpy
from scipy.special import softmax

from gandy.onnx_models.base_onnx_model import BaseONNXModel

# Hacky class for huggingface's feature extractor.
class Outputs():
    def __init__(self, a, b):
        self.logits = np.array(a)
        self.pred_boxes = np.array(b)

class DETRGONNX(BaseONNXModel):
    def __init__(self, onnx_path, dataloader_path, use_cuda):
        """
        This model uses a transformer to detect text. (NOT ocr)
        """
        super().__init__(use_cuda=use_cuda)

        self.load_session(onnx_path)
        self.load_dataloader(dataloader_path)

    def load_dataloader(self, fe_path):
        self.feature_extractor = DetrFeatureExtractorNumpy.from_pretrained(fe_path)

    def forward(self, inp_data):
        encoding, image = inp_data

        input_name = self.ort_sess.get_inputs()[0].name

        ort_inputs = { input_name: encoding['pixel_values'], }
        ort_outs = self.ort_sess.run(None, ort_inputs)

        logits, pred_boxes = ort_outs[0], ort_outs[1]

        outp_data = (logits, pred_boxes, image)
        return outp_data

    def preprocess(self, image):
        encoding = self.feature_extractor(image, return_tensors="np")
        encoding['pixel_mask'] = None # Batch of 1. This is unused.

        inp_data = (encoding, image)
        return inp_data

    def postprocess(self, outp_data):
        logits, pred_boxes, image = outp_data
        # logits = [batch, 100, 2]
        # pred_boxes = [batch, 100, 4]

        # Confidence filter mask. 75%.
        probas = softmax(logits, axis=-1)[0, :, :-1]
        v_probs = probas.max(axis=-1)
        keep = v_probs > 0.75

        outputs = Outputs(logits, pred_boxes)

        # Rescale.
        hw = image.shape[:2] # [height, width]
        # Old: target_sizes = torch.tensor(hw).unsqueeze(0)
        target_sizes = np.array(hw)[np.newaxis, ...]

        postprocessed_outputs = self.feature_extractor.post_process(outputs, target_sizes)

        # Filter.
        bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
        scores_scaled = v_probs[keep]

        return {
            'boxes': np.array(bboxes_scaled), # Shape of N * 4 (N = number of bounding boxes, 4 = coordinates)
            'labels': np.array([1 for _ in scores_scaled]), # Shape of N
            'scores': np.array(scores_scaled), # Shape of N
        }
