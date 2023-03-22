from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
import logging

logger = logging.getLogger('Gandy')


# If creating a model from this, make sure to manually call load_dataloader() (if needed), and load_session.
class BaseONNXModel():
    def __init__(self, use_cuda):
        self.use_cuda = use_cuda

    def forward(self, x):
        pass

    def load_dataloader(self):
        """
        The dataloader is used to preprocess the given raw input (inp) into the processed input (x).
        """
        pass

    def create_session(self, onnx_path):
        if self.use_cuda is None:
            raise RuntimeError('use_cuda must be True or False.')

        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        # TODO right now the beam search models are unoptimized, leading to a lot of warning messages. Will fix later. - 27 years ago
        # options.log_severity_level = 0
        options.log_severity_level = 3

        # Note that CUDA errors are not properly logged right now :/
        # Note that it's not actually CUDA - changed it to DirectML since installing cudNN is going to be a MAJOR PAIN IN THE *** for most end users.
        if self.use_cuda:
            logger.info('CUDA enabled. Will try to use DirectML if allowed.')
            provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            logger.info('CUDA disabled. Will only use CPU.')
            provider = ['CPUExecutionProvider']

        self.ort_sess = InferenceSession(onnx_path, options, provider)
        # ? self.ort_sess.disable_fallback()

        return self.ort_sess

    def load_session(self, onnx_path):
        self.ort_sess = self.create_session(onnx_path)

    def preprocess(self, inp):
        """
        Process the input into a form of data that can be passed to .forward().
        """
        return inp

    def postprocess(self, y_hat):
        return y_hat

    def begin_forward(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def full_pipe(self, inp, *forward_args, **forward_kwargs):
        """
        Given a raw input, fully process it and return the proper output.
        """

        x = self.preprocess(inp)
        return self.postprocess(self.begin_forward(x, *forward_args, **forward_kwargs))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
