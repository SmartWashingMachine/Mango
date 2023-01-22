from gandy.onnx_models.base_onnx_model import BaseONNXModel

class RCnnONNX(BaseONNXModel):
    def __init__(self, onnx_path, use_cuda):
        """
        This model uses a finetuned Resnet50 Faster-RCNN to detect bounding boxes.

        This model is used in the rcnn_image_detection app, to detect text boxes in images.
        """
        super().__init__(use_cuda=use_cuda)
        self.load_session(onnx_path)

    def forward(self, x):
        input_name = self.ort_sess.get_inputs()[0].name

        ort_inp = { input_name: x }

        outputs = self.ort_sess.run(None, ort_inp)
        return {
            'boxes': outputs[0], # Shape of N * 4 (N = number of bounding boxes, 4 = coordinates)
            'labels': outputs[1], # Shape of N
            'scores': outputs[2], # Shape of N
        }
