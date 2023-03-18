from onnxruntime import (
    InferenceSession,
)
import numpy as np
import functools
import operator
from gandy.onnx_models.auto_regressive.ar_utils import session_has_cuda, DEVICE_ID, get_from_buffer
from gandy.onnx_models.auto_regressive.auto_regressive_decoder_init import OnnxArDecoderInit

class OnnxArDecoder(OnnxArDecoderInit):
    def __init__(self, decoder_sess: InferenceSession, config):
        self.decoder = decoder_sess
        self.use_cuda = session_has_cuda(decoder_sess)

        session_inputs = {output_key.name: idx for idx, output_key in enumerate(self.decoder.get_inputs())}
        self.session_inputs_names = list(session_inputs.keys())
        self.key_value_input_names = [k for k in self.session_inputs_names if 'key' in k or 'val' in k]

        session_outputs = {output_key.name: idx for idx, output_key in enumerate(self.decoder.get_outputs())}
        self.session_output_names = list(session_outputs.keys())
        self.key_value_output_names = [k for k in self.session_output_names if 'key' in k or 'val' in k]

        self.config = config

    def prepare_inputs(self, io_binding, input_ids, attention_mask, encoder_hidden_states, src_positions, past_key_values):
        super().prepare_inputs(io_binding, input_ids, attention_mask, encoder_hidden_states, src_positions)

        # bind past key values
        if past_key_values is not None:
            for input_name, past_key_value in zip(self.key_value_input_names, past_key_values):
                past_key_value = past_key_value.contiguous()
                io_binding.bind_input(
                    input_name,
                    past_key_value.device.type,
                    DEVICE_ID,
                    np.float32,
                    list(past_key_value.size()),
                    past_key_value.data_ptr(),
                )

    def prepare_io_binding(self, input_ids, attention_mask, encoder_hidden_states, src_positions, past_key_values):
        io_binding = self.decoder.io_binding()

        self.prepare_inputs(io_binding, input_ids, attention_mask, encoder_hidden_states=None, src_positions=None, past_key_values=past_key_values)

        output_shapes, output_buffers = self.prepare_outputs(io_binding, input_ids, attention_mask, encoder_hidden_states, src_positions, past_key_values)

        return io_binding, output_shapes, output_buffers

    def forward(self, input_ids, attention_mask, encoder_hidden_states, past_key_values, src_positions):
        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

        if self.use_cuda:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_ids, attention_mask, encoder_hidden_states, src_positions, flat_past_key_values)

            io_binding.synchronize_inputs()
            self.decoder.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()

            logits = get_from_buffer('logits', output_buffers, output_shapes)

            out_past_key_values = tuple()
            for name in self.session_output_names:
                if 'key' in name or 'val' in name:
                    out_past_key_values += (get_from_buffer(name, output_buffers, output_shapes), )

            out_past_key_values = tuple(
                out_past_key_values[i : i + 4] for i in range(0, len(out_past_key_values), 4)
            )

            list_hidden_states = [get_from_buffer('decoder_hidden_state', output_buffers, output_shapes)]
            list_cross_attentions = [get_from_buffer('cross_attentions', output_buffers, output_shapes)]
            
            return logits, out_past_key_values, list_hidden_states, list_cross_attentions
        else:
            input_ids = input_ids.astype(np.int64)

            input_names = [x.name for x in self.decoder.get_inputs()]
            inputs = [
                input_ids,
            ]

            if attention_mask is not None:
                attention_mask = attention_mask.astype(np.int64)
                inputs = inputs + [attention_mask]

            inputs = inputs + [
                tensor for tensor in flat_past_key_values
            ]

            decoder_inputs = dict(zip(input_names, inputs))
            decoder_outputs = self.decoder.run(None, decoder_inputs)

        hidden_states = [decoder_outputs[1]]
        cross_attentions = [decoder_outputs[2]]
        pkvs = []
        for x in decoder_outputs[3:]:
            if x.shape[2] == 512: # TODO: Filler code may not be needed anymore
                continue
            pkvs.append(x)
    
        list_pkv = tuple(x for x in pkvs)
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        list_hidden_states = tuple(x for x in hidden_states)

        list_cross_attentions = tuple(x for x in cross_attentions)

        return decoder_outputs[0], out_past_key_values, list_hidden_states, list_cross_attentions

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
