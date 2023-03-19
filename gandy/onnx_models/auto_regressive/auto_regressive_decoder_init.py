import logging
from onnxruntime import (
    InferenceSession,
)
import numpy as np
from gandy.onnx_models.auto_regressive.ar_utils import session_has_cuda, DEVICE_ID, DEVICE_TO_USE, get_from_buffer

logger = logging.getLogger('Gandy')

try:
    import torch
except:
    pass

class OnnxArDecoderInit():
    def __init__(self, decoder_sess: InferenceSession, config):
        self.decoder = decoder_sess
        self.use_cuda = session_has_cuda(decoder_sess)

        session_outputs = {output_key.name: idx for idx, output_key in enumerate(self.decoder.get_outputs())}
        self.session_output_names = list(session_outputs.keys())
        self.key_value_output_names = [k for k in self.session_output_names if 'key' in k or 'val' in k]

        self.config = config

    def prepare_pkv_buffer(self, batch_size: int, sequence_length = None, encoder_sequence_length = None, past_sequence_length = None, is_self_attn = False):
        num_heads = self.config.decoder_attention_heads if hasattr(self.config, 'decoder_attention_heads') else self.config.decoder.num_attention_heads
        hidden_size = self.config.d_model if hasattr(self.config, 'd_model') else self.config.decoder.hidden_size
        embed_size_per_head = hidden_size // num_heads
        if is_self_attn:
            if past_sequence_length is not None:
                sequence_length += past_sequence_length
            output_shape = (batch_size, num_heads, sequence_length, embed_size_per_head)
        else:
            output_shape = (batch_size, num_heads, encoder_sequence_length, embed_size_per_head)

        output_buffer = torch.empty(np.prod(output_shape), dtype=torch.float32, device=DEVICE_TO_USE).contiguous()

        return output_shape, output_buffer

    def bind_pkv(self, io_binding, input_ids, encoder_hidden_states, past_key_values, output_shapes, output_buffers):
        # Binds PKV for outputs.

        num_pkv = 4  # number of self-attention and cross-attention per decoder layer
        for pkv_names_per_layer in [
            self.key_value_output_names[i : i + num_pkv] for i in range(0, len(self.key_value_output_names), num_pkv)
        ]:
            # bind a self attention and a cross-attention each time(2)
            for i in range(2):
                # bind self-attention past key values(2)
                self_name = pkv_names_per_layer[i]
                self_pkv_shape, self_pkv_buffer = self.prepare_pkv_buffer(
                    batch_size=input_ids.size(0),
                    sequence_length=input_ids.size(1),
                    past_sequence_length=past_key_values[0].size(2)
                    if past_key_values
                    else None,  # sequence length of self-attention key for layer.0
                    is_self_attn=True,
                    encoder_sequence_length=None, # doesnt matter for self attn.
                )

                io_binding.bind_output(
                    self_name,
                    self_pkv_buffer.device.type,
                    DEVICE_ID,
                    np.float32,
                    self_pkv_shape,
                    self_pkv_buffer.data_ptr(),
                )
                # set -1 for sequence_length as it could be larger than the real sequence_length for creating buffer
                self_pkv_shape = self_pkv_shape[:2] + (-1,) + self_pkv_shape[3:]
                output_shapes[self_name] = self_pkv_shape
                output_buffers[self_name] = self_pkv_buffer

                # bind cross-attention past key values(2)
                cross_name = pkv_names_per_layer[i + 2]
                cross_pkv_shape, cross_pkv_buffer = self.prepare_pkv_buffer(
                    batch_size=input_ids.size(0),
                    encoder_sequence_length=encoder_hidden_states.size(1),
                )

                io_binding.bind_output(
                    cross_name,
                    cross_pkv_buffer.device.type,
                    DEVICE_ID,
                    np.float32,
                    cross_pkv_shape,
                    cross_pkv_buffer.data_ptr(),
                )
                # set -1 for sequence_length as it could be larger than the real sequence_length for creating buffer
                cross_pkv_shape = cross_pkv_shape[:2] + (-1,) + cross_pkv_shape[3:]
                output_shapes[cross_name] = cross_pkv_shape
                output_buffers[cross_name] = cross_pkv_buffer

    def prepare_inputs(self, io_binding, input_ids, attention_mask, encoder_hidden_states = None, src_positions = None):
        input_ids = input_ids.contiguous()
        io_binding.bind_input(
            'input_ids',
            input_ids.device.type,
            DEVICE_ID,
            np.int64,
            tuple(input_ids.shape),
            input_ids.data_ptr(),
        )

        if attention_mask is not None:
            attention_mask = attention_mask.contiguous()
            io_binding.bind_input(
                'encoder_attention_mask',
                attention_mask.device.type,
                DEVICE_ID,
                np.int64,
                tuple(attention_mask.shape),
                attention_mask.data_ptr(),
            )

        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.contiguous()
            io_binding.bind_input(
                'encoder_hidden_states',
                encoder_hidden_states.device.type,
                DEVICE_ID,
                np.float32,
                tuple(encoder_hidden_states.shape),
                encoder_hidden_states.data_ptr(),
            )

        if src_positions is not None:
            src_positions = src_positions.contiguous()
            io_binding.bind_input(
                'src_positions',
                src_positions.device.type,
                DEVICE_ID,
                np.float32,
                tuple(src_positions.shape),
                src_positions.data_ptr(),
            )

    def prepare_outputs(self, io_binding, input_ids, attention_mask, encoder_hidden_states, src_positions, past_key_values, has_cross_attentions = True):
        # Bind all outputs.
        bsz = input_ids.size(0)
        seq_length = input_ids.size(1)
        vocab_size = self.config.decoder_vocab_size if hasattr(self.config, 'decoder_vocab_size') else self.config.vocab_size
        num_heads = self.config.decoder_attention_heads if hasattr(self.config, 'decoder_attention_heads') else self.config.decoder.num_attention_heads

        logits_shape = (bsz, seq_length, vocab_size)
        logits_buffer = torch.empty(np.prod(logits_shape), dtype=torch.float32, device=DEVICE_TO_USE).contiguous()
        io_binding.bind_output(
            'logits',
            logits_buffer.device.type,
            DEVICE_ID,
            np.float32,
            logits_shape,
            logits_buffer.data_ptr(),
        )

        output_shapes = {'logits': logits_shape}
        output_buffers = {'logits': logits_buffer}

        if has_cross_attentions:
            hid_size = self.config.d_model

            # Hidden state for KNN
            decoder_hidden_state_shape = (bsz, seq_length, hid_size)
            decoder_hidden_state_buffer = torch.empty(np.prod(decoder_hidden_state_shape), dtype=torch.float32, device=DEVICE_TO_USE).contiguous()
            io_binding.bind_output(
                'decoder_hidden_states',
                decoder_hidden_state_buffer.device.type,
                DEVICE_ID,
                np.float32,
                decoder_hidden_state_shape,
                decoder_hidden_state_buffer.data_ptr(),
            )

            output_buffers['decoder_hidden_state'] = decoder_hidden_state_buffer
            output_shapes['decoder_hidden_state'] = decoder_hidden_state_shape

        if has_cross_attentions:
            # Cross attention for visualization.
            src_seq_length = encoder_hidden_states.size(1)
            cross_attentions_shape = (bsz, num_heads, seq_length, src_seq_length)
            cross_attentions_buffer = torch.empty(np.prod(cross_attentions_shape), dtype=torch.float32, device=DEVICE_TO_USE).contiguous()
            io_binding.bind_output(
                'cross_attentions',
                cross_attentions_buffer.device.type,
                DEVICE_ID,
                np.float32,
                cross_attentions_shape,
                cross_attentions_buffer.data_ptr(),
            )
        else:
            cross_attentions_buffer = None
            cross_attentions_shape = None

        output_buffers['cross_attentions'] = cross_attentions_buffer
        output_shapes['cross_attentions'] = cross_attentions_shape

        # PKVs. past_key_values must be flat. Adds to output_buffers/shapes inplace.
        self.bind_pkv(io_binding, input_ids, encoder_hidden_states, past_key_values, output_shapes, output_buffers)

        return output_shapes, output_buffers

    def prepare_io_binding(self, input_ids, attention_mask, encoder_hidden_states, src_positions, past_key_values, has_cross_attentions = True):
        io_binding = self.decoder.io_binding()

        self.prepare_inputs(io_binding, input_ids, attention_mask, encoder_hidden_states, src_positions)

        output_shapes, output_buffers = self.prepare_outputs(io_binding, input_ids, attention_mask, encoder_hidden_states, src_positions, past_key_values, has_cross_attentions=has_cross_attentions)

        return io_binding, output_shapes, output_buffers

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states, src_positions, has_cross_attentions = True):
        if self.use_cuda:
            io_binding, output_shapes, output_buffers = self.prepare_io_binding(input_ids, encoder_attention_mask, encoder_hidden_states, src_positions, past_key_values=None, has_cross_attentions=has_cross_attentions)

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
            input_feed = {
                "input_ids": input_ids.astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states,
            }

            if src_positions is not None:
                input_feed['src_positions'] = src_positions
            if encoder_attention_mask is not None:
                input_feed['encoder_attention_mask'] = encoder_attention_mask.astype(np.int64)

            decoder_outputs = self.decoder.run(
                None,
                input_feed
            )

        hidden_states = [decoder_outputs[1]]

        if has_cross_attentions:
            cross_attentions = [decoder_outputs[2]]
            pkvs = []
            for x in decoder_outputs[3:]:
                if x.shape[2] == 512:
                    continue
                pkvs.append(x)
        else:
            # TODO: has_cross_attentions should really be clarified as it also ignores hidden state outputs - it's just for the OCR models.
            cross_attentions = []
            pkvs = []
            for x in decoder_outputs[1:]:
                if x.shape[2] == 512:
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