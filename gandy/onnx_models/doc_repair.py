from gandy.onnx_models.base_onnx_model import BaseONNXModel
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    ExecutionMode,
    RunOptions
)
from transformers import (
    AutoTokenizer,
    AutoConfig
)
from gandy.utils.knn_utils.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput,
)
from gandy.utils.knn_utils.generation_mixin import GenerationMixinNumpy

import functools
import operator
from datetime import datetime
import numpy as np
import logging

logger = logging.getLogger('Gandy')

class OnnxDocRepairEncoder():
    def __init__(self, encoder_sess):
        super().__init__()
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
                    "attention_mask": attention_mask.astype(np.int64),
                },
            )[0]
        )

        return BaseModelOutput(encoder_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class OnnxDocRepairDecoder():
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_hidden_states, past_key_values):

        decoder_inputs = {
            "input_ids": input_ids.astype(np.int64),
            "encoder_attention_mask": attention_mask.astype(np.int64),
            #"encoder_hidden_states": encoder_hidden_states.cpu().numpy(),
        }

        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])
        
        input_names = [x.name for x in self.decoder.get_inputs()]
        inputs = [
            input_ids.astype(np.int64),
            attention_mask.astype(np.int64),
        ] + [
            tensor for tensor in flat_past_key_values
        ]

        decoder_inputs = dict(zip(input_names, inputs))
        decoder_outputs = self.decoder.run(None, decoder_inputs)
 
        list_pkv = tuple(x for x in decoder_outputs[1:])
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return decoder_outputs[0], out_past_key_values

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class OnnxDocRepairDecoderInit():
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states):

        decoder_outputs = self.decoder.run(
            None,
            {
                "input_ids": input_ids.astype(np.int64),
                "encoder_attention_mask": encoder_attention_mask.astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states,
            },
        )

        list_pkv = tuple(x for x in decoder_outputs[1:])
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return decoder_outputs[0], out_past_key_values

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def shift_tokens_right(input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids

class DocRepairONNX(BaseONNXModel, GenerationMixinNumpy):
    def __init__(self, onnx_path_enc, onnx_path_dec, onnx_path_dec_init, dataloader_path, use_cuda):
        super().__init__(use_cuda=use_cuda)

        self.config = AutoConfig.from_pretrained(dataloader_path)
        self.load_dataloader(dataloader_path)

        self.load_session(onnx_path_enc, onnx_path_dec, onnx_path_dec_init)

        self.main_input_name = 'input_ids'

    def load_session(self, enc_path, dec_path, dec_init_path):
        self.encoder = OnnxDocRepairEncoder(self.create_session(enc_path))
        self.decoder = OnnxDocRepairDecoder(self.create_session(dec_path))
        self.decoder_init = OnnxDocRepairDecoderInit(self.create_session(dec_init_path))

    # Copied from ModelingRoberta.
    def roberta_prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            # Old: attention_mask = input_ids.new_ones(input_shape)
            attention_mask = np.ones_like(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

    # From EncoderDecoder.
    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        decoder_inputs = self.roberta_prepare_inputs_for_generation(input_ids, past=past)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": decoder_inputs["past_key_values"],
            "use_cache": use_cache,
        }
        return input_dict

    def prepare_decoder_input_ids_from_labels(self, labels: np.ndarray):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # Old: reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
            reordered_past += (tuple(np.take(past_state, beam_idx, axis=0) for past_state in layer_past),)
        return reordered_past

    @property
    def device(self):
        return "cpu"

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return None

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

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            # (when using generate, we already get encoder_outputs generated
            #  by _prepare_encoder_decoder_kwargs_for_generation)
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is None:
            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )
            logits, past_key_values = init_onnx_outputs
        else:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values = onnx_outputs

        outputs = Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

        return outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_dataloader(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, truncation_side='left', padding_side='right')

    def load_session(self, enc_path, dec_path, dec_init_path):
        options = SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        provider = ['CPUExecutionProvider']
        def _load_sess(onnx_path):
            return InferenceSession(onnx_path, options, provider)

        self.encoder = OnnxDocRepairEncoder(_load_sess(enc_path))
        self.decoder = OnnxDocRepairDecoder(_load_sess(dec_path))
        self.decoder_init = OnnxDocRepairDecoderInit(_load_sess(dec_init_path))

    def preprocess(self, inp):
        inp_text = inp['text']

        x_dict = self.tokenizer([inp_text], return_tensors='np', max_length=512, truncation=True)

        x_dict['text'] = inp_text
        return x_dict

    def begin_forward(self, x_dict):
        start = datetime.now()
        outp = self.generate(
            input_ids=x_dict['input_ids'],
            attention_mask=x_dict['attention_mask'],
            num_beams=5,
            # no_repeat_ngram_size=12,
            do_sample=False,
            max_length=512,
        )
        end = datetime.now()
        logger.debug(f'Repairing item time elapsed: {(end - start).total_seconds()}s')
        logger.debug(f'For text: {x_dict["text"]}')
        return outp

    def postprocess(self, outp):
        # Old: decoded = self.tokenizer.decode(outp.squeeze(), skip_special_tokens=False)
        decoded = self.tokenizer.decode(outp[0, ...], skip_special_tokens=False)

        return decoded
