import logging
from onnxruntime import (
    InferenceSession,
)
import numpy as np
from gandy.utils.knn_utils.modeling_outputs import (
    BaseModelOutput,
)
from gandy.onnx_models.auto_regressive.ar_utils import session_has_cuda, DEVICE_ID, DEVICE_TO_USE, get_from_buffer

logger = logging.getLogger('Gandy')

try:
    import torch
except:
    logger.debug('Pytorch not installed - GPU support disabled.')

class OnnxArVisionEncoder():
    def __init__(self, encoder_sess: InferenceSession, config):
        self.encoder = encoder_sess
        self.use_cuda = session_has_cuda(encoder_sess)

        self.main_input_name = 'pixel_values'

        self.config = config

    def prepare_inputs_binding(self, io_binding, pixel_values):
        pixel_values = pixel_values.contiguous()
        io_binding.bind_input(
            'pixel_values',
            pixel_values.device.type,
            DEVICE_ID,
            np.float32,
            tuple(pixel_values.shape),
            pixel_values.data_ptr(),
        )

    def prepare_outputs_binding(self, io_binding, pixel_values):
        bsz = pixel_values.size(0)

        output_shape = (bsz, 197, 768) # TODO: Not hardcode lol
        output_buffer = torch.empty(np.prod(output_shape), dtype=torch.float32, device=DEVICE_TO_USE).contiguous()
        io_binding.bind_output(
            'hidden_states',
            output_buffer.device.type,
            DEVICE_ID,
            np.float32,
            output_shape,
            output_buffer.data_ptr(),
        )

        output_shapes = {
            'encoder_hidden_state': output_shape,
        }
        output_buffers = {
            'encoder_hidden_state': output_buffer,
        }

        return output_shapes, output_buffers

    def prepare_io_binding(self, pixel_values):
        io_binding = self.encoder.io_binding()

        self.prepare_inputs_binding(io_binding, pixel_values)

        output_shapes, output_buffers = self.prepare_outputs_binding(io_binding, pixel_values)

        return io_binding, output_shapes, output_buffers

    def forward(
        self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if self.use_cuda:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(pixel_values)

            io_binding.synchronize_inputs()
            self.encoder.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            encoder_hidden_state = get_from_buffer('encoder_hidden_state', output_buffers, output_shapes)
        else:
            outputs = (
                self.encoder.run(
                    None,
                    {
                        'pixel_values': pixel_values,
                    },
                )
            )

            encoder_hidden_state = outputs[0]

        return BaseModelOutput(encoder_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)