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

class OnnxArEncoder():
    def __init__(self, encoder_sess: InferenceSession, config):
        self.encoder = encoder_sess
        self.use_cuda = session_has_cuda(encoder_sess)

        self.main_input_name = 'input_ids'

        self.config = config

    def prepare_inputs_binding(self, io_binding, input_ids, attention_mask):
        input_ids = input_ids.contiguous()
        io_binding.bind_input(
            'input_ids',
            input_ids.device.type,
            DEVICE_ID,
            np.int64,
            tuple(input_ids.shape),
            input_ids.data_ptr(),
        )

        attention_mask = attention_mask.contiguous()
        io_binding.bind_input(
            'attention_mask',
            attention_mask.device.type,
            DEVICE_ID,
            np.int64,
            tuple(attention_mask.shape),
            attention_mask.data_ptr(),
        )

    def prepare_outputs_binding(self, io_binding, input_ids, attention_mask):
        bsz = input_ids.size(0)
        seq_length = input_ids.size(1)
        hidden_size = self.config.d_model

        output_shape = (bsz, seq_length, hidden_size)
        output_buffer = torch.empty(np.prod(output_shape), dtype=torch.float32, device=DEVICE_TO_USE).contiguous()
        io_binding.bind_output(
            'hidden_states',
            output_buffer.device.type,
            DEVICE_ID,
            np.float32,
            output_shape,
            output_buffer.data_ptr(),
        )

        src_positions_shape = (seq_length, hidden_size)
        src_positions_buffer = torch.empty(np.prod(src_positions_shape), dtype=torch.float32, device=DEVICE_TO_USE).contiguous()
        io_binding.bind_output(
            'src_positions',
            src_positions_buffer.device.type,
            DEVICE_ID,
            np.float32,
            src_positions_shape,
            src_positions_buffer.data_ptr(),
        )

        output_shapes = {
            'encoder_hidden_state': output_shape,
            'src_positions': src_positions_shape,
        }
        output_buffers = {
            'encoder_hidden_state': output_buffer,
            'src_positions': src_positions_buffer,
        }

        return output_shapes, output_buffers

    def prepare_io_binding(self, input_ids, attention_mask):
        io_binding = self.encoder.io_binding()

        self.prepare_inputs_binding(io_binding, input_ids, attention_mask)

        output_shapes, output_buffers = self.prepare_outputs_binding(io_binding, input_ids, attention_mask)

        return io_binding, output_shapes, output_buffers

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if self.use_cuda:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_ids, attention_mask)

            io_binding.synchronize_inputs()
            self.encoder.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            encoder_hidden_state = get_from_buffer('encoder_hidden_state', output_buffers, output_shapes)
            src_positions = get_from_buffer('src_positions', output_buffers, output_shapes)
        else:
            encoder_hidden_state, src_positions = (
                self.encoder.run(
                    None,
                    {
                        "input_ids": input_ids.astype(np.int64),
                        "attention_mask": attention_mask.astype(np.int64),
                    },
                )
            )

        return BaseModelOutput(encoder_hidden_state, src_positions=src_positions)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
