from gandy.onnx_models.marian import MarianONNX
from gandy.utils.mt_big_tokenizer import MtBigTokenizer
import numpy as np
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    ExecutionMode,
    RunOptions,
    OrtValue,
)
import functools
import operator
from gandy.utils.knn_utils.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput,
)
import copy

class OnnxMarianEncoder():
    def __init__(self, encoder_sess: InferenceSession):
        self.encoder = encoder_sess

        self.main_input_name = 'input_ids'

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
        encoder_hidden_state = (
            self.encoder.run(
                None,
                {
                    "input_ids": input_ids.astype(np.int64),
                    #"attention_mask": attention_mask.astype(np.int64),
                },
            )
        )

        return BaseModelOutput(encoder_hidden_state[0])

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class OnnxMarianDecoderInit():
    def __init__(self, decoder_sess: InferenceSession):
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):
        decoder_outputs = self.decoder.run(
            None,
            {
                "input_ids": input_ids.astype(np.int64),
                #"encoder_attention_mask": encoder_attention_mask.astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states,
            },
        )

        hidden_states = [decoder_outputs[1]]
        cross_attentions = [decoder_outputs[2]]
        pkvs = []
        for x in decoder_outputs[3:]:
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

class OnnxMarianDecoder():
    def __init__(self, decoder_sess: InferenceSession):
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_hidden_states, past_key_values):
        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

        input_names = [x.name for x in self.decoder.get_inputs()]
        inputs = [
            input_ids,
            #attention_mask,
        ] + [
            copy.deepcopy(tensor) for tensor in flat_past_key_values
        ]

        decoder_inputs = dict(zip(input_names, inputs))

        decoder_outputs = self.decoder.run(None, decoder_inputs)

        hidden_states = [decoder_outputs[1]]
        cross_attentions = [decoder_outputs[2]]
        pkvs = []
        for x in decoder_outputs[3:]:
            if x.shape[2] == 1024:
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

class MtBigONNX(MarianONNX):
    def get_target_tokenizer(self):
        return self.en_tokenizer

    def get_decoder_input_ids(self, tgt_context_memory):
        # We use :-1 to slice off the last token, which is the EOS token. -2 means slicing off SEP as well.
        decoder_input_ids = self.en_tokenizer(tgt_context_memory, return_tensors='np', max_length=507).input_ids[:, :-1]
        return decoder_input_ids

    def load_session(self, enc_path, dec_path, dec_init_path):
        self.encoder = OnnxMarianEncoder(self.create_session(enc_path))
        self.decoder = OnnxMarianDecoder(self.create_session(dec_path))
        self.decoder_init = OnnxMarianDecoderInit(self.create_session(dec_init_path))

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
                decoder_input_ids.astype(np.int64),
                attention_mask.astype(np.int64),
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
