from gandy.onnx_models.marian import BaseMarianONNX
from gandy.utils.mt_big_tokenizer import MtBigTokenizer
import numpy as np
from gandy.utils.knn_utils.modeling_outputs import (
    Seq2SeqLMOutput,
)
import copy
from gandy.onnx_models.auto_regressive.auto_regressive_decoder import OnnxArDecoder
from gandy.onnx_models.auto_regressive.auto_regressive_decoder_init import OnnxArDecoderInit
from gandy.onnx_models.auto_regressive.auto_regressive_encoder import OnnxArEncoder
from transformers.generation_utils import GenerationMixin
from gandy.utils.knn_utils.generation_mixin import GenerationMixinNumpy

class BigEncoder(OnnxArEncoder):

    def prepare_io_binding(self, input_ids, attention_mask):
        io_binding = self.encoder.io_binding()
        self.prepare_inputs_binding(io_binding, input_ids, attention_mask)

        # Unlike normal model, Big does not use P-transformer modifications.
        output_shapes, output_buffers = self.prepare_outputs_binding(io_binding, input_ids, attention_mask, has_src_positions=False)

        return io_binding, output_shapes, output_buffers

    def forward(self, input_ids, attention_mask, inputs_embeds=None, head_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        attention_mask = None

        return super().forward(input_ids, attention_mask, inputs_embeds, head_mask, output_attentions, output_hidden_states, return_dict)

class BigDecoderInit(OnnxArDecoderInit):

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):
        return super().forward(input_ids, encoder_attention_mask=None, encoder_hidden_states=encoder_hidden_states, src_positions=None)

class BigDecoder(OnnxArDecoder):

    def forward(self, input_ids, attention_mask, encoder_hidden_states, past_key_values):
        return super().forward(input_ids, attention_mask=None, encoder_hidden_states=encoder_hidden_states, past_key_values=past_key_values, src_positions=None)

class BaseMtBigONNX(BaseMarianONNX):
    def get_target_tokenizer(self):
        return self.en_tokenizer

    def get_decoder_input_ids(self, tgt_context_memory):
        # We use :-1 to slice off the last token, which is the EOS token. -2 means slicing off SEP as well.
        decoder_input_ids = self.en_tokenizer(tgt_context_memory, return_tensors='np', max_length=507).input_ids[:, :-1]
        return decoder_input_ids

    def load_session(self, enc_path, dec_path, dec_init_path):
        self.encoder = BigEncoder(self.create_session(enc_path), self.config)
        self.decoder = BigDecoder(self.create_session(dec_path), self.config)
        self.decoder_init = BigDecoderInit(self.create_session(dec_init_path), self.config)

    def load_dataloader(self, tokenizer_path):
        self.tokenizer = MtBigTokenizer.from_pretrained(tokenizer_path + '_ja', truncation_side='left', padding_side='right')
        self.en_tokenizer = MtBigTokenizer.from_pretrained(tokenizer_path + '_en', truncation_side='left', padding_side='right')

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        *args,
        **kwargs
    ):
        # This is called for every token.

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask,
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is None:
            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states,
            )
            logits, past_key_values, decoder_hidden_states, cross_attentions = init_onnx_outputs
        else:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values, decoder_hidden_states, cross_attentions = onnx_outputs

        outputs = Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values, decoder_hidden_states=decoder_hidden_states, cross_attentions=cross_attentions)

        if self.process_outputs_cb is not None:
            # Make sure the CB modifies in-place!
            # outputs = self.process_outputs_cb(outputs)
            pass

        return outputs

    def adjust_logits_during_generation(self, logits, cur_len):
        return logits

    # Copied from MarianMTModel.
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

class MtBigONNXNumpy(BaseMtBigONNX, GenerationMixinNumpy):
    pass

class MtBigONNXTorch(BaseMtBigONNX, GenerationMixin):
    pass