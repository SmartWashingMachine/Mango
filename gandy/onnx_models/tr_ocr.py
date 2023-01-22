from gandy.onnx_models.base_onnx_model import BaseONNXModel

from transformers import (
    AutoTokenizer,
    AutoConfig,
)
from transformers import ViTFeatureExtractor, AutoTokenizer
import functools
import operator

from gandy.utils.knn_utils.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput,
)
from gandy.utils.knn_utils.generation_mixin import GenerationMixinNumpy

from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)

import numpy as np

class OnnxVisionEncoder():
    def __init__(self, encoder_sess):
        super().__init__()
        self.encoder = encoder_sess

        self.main_input_name = 'pixel_values'

    def forward(
        self,
        pixel_values,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        encoder_hidden_state = (
            self.encoder.run(
                None,
                {
                    "pixel_values": pixel_values,
                },
            )[0]
        )

        return BaseModelOutput(encoder_hidden_state)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class OnnxVisionDecoder():
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_hidden_states, past_key_values, output_attentions=None, output_hidden_states=None, return_dict=None, encoder_attention_mask=None, **kwargs):
        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])
        
        input_names = [x.name for x in self.decoder.get_inputs()]
        inputs = [
            input_ids.astype(np.int64),
            encoder_attention_mask.astype(np.int64),
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

class OnnxVisionDecoderInit():
    def __init__(self, decoder_sess):
        super().__init__()
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_hidden_states, attention_mask, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        decoder_outputs = self.decoder.run(
            None,
            {
                'input_ids': input_ids.astype(np.int64),
                'encoder_attention_mask': attention_mask.astype(np.int64),
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

class OnnxVisionProj():
    def __init__(self, proj_sess):
        super().__init__()

        self.proj_sess = proj_sess

    def forward(
        self,
        encoder_hidden_states,
    ):
        inp_name = self.proj_sess.get_inputs()[0].name
        encoder_hidden_states = (
            self.proj_sess.run(
                None,
                {
                    inp_name: encoder_hidden_states,
                },
            )[0]
        )

        return encoder_hidden_states

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

class OnnxVision(BaseONNXModel, GenerationMixinNumpy):
    def __init__(self, onnx_path_enc, onnx_path_dec, onnx_path_dec_init, proj, tokenizer_path, feature_extractor_path, config_path, use_cuda):
        super().__init__(use_cuda=use_cuda)

        self.load_session(onnx_path_enc, onnx_path_dec, onnx_path_dec_init, proj)
        self.load_dataloader(tokenizer_path, feature_extractor_path)

        self.config = AutoConfig.from_pretrained(config_path)

        self.main_input_name = 'pixel_values'

    def load_session(self, enc_path, dec_path, dec_init_path, proj):
        self.encoder = OnnxVisionEncoder(self.create_session(enc_path))
        self.decoder = OnnxVisionDecoder(self.create_session(dec_path))
        self.decoder_init = OnnxVisionDecoderInit(self.create_session(dec_init_path))

        if proj is not None:
            self.enc_to_dec_proj = OnnxVisionProj(self.create_session(proj))
        else:
            self.enc_to_dec_proj = None

    def load_dataloader(self, tokenizer_path, feature_extractor_path):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def preprocess(self, inp):
        pixel_values = self.feature_extractor(inp, return_tensors='np').pixel_values

        return pixel_values

    def begin_forward(self, pixel_values):
        outp = self.generate(
            pixel_values,
            no_repeat_ngram_size=12, # TODO: Change at some point to a better value.
            length_penalty=1.0,
        )
        return outp

    def postprocess(self, outp):
        # Old: decoded = self.tokenizer.decode(outp.squeeze(), skip_special_tokens=True)
        decoded = self.tokenizer.decode(outp[0, ...], skip_special_tokens=True)

        decoded = decoded.replace(' ', '').strip()
        return decoded

    ## All utils below are for generation via huggingface ##

    def prepare_decoder_input_ids_from_labels(self, labels: np.ndarray):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        def _prepare_inputs_for_generation(input_ids, past=None, attention_mask=None, **model_kwargs):
            input_shape = input_ids.shape
            # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
            if attention_mask is None:
                # Old: attention_mask = input_ids.new_ones(input_shape)
                attention_mask = np.ones(input_shape)

            # cut decoder_input_ids if past is used
            if past is not None:
                input_ids = input_ids[:, -1:]

            return {"input_ids": input_ids, "attention_mask": attention_mask, "past_key_values": past}

        decoder_inputs = _prepare_inputs_for_generation(input_ids, past=past)
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

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            """
            Old:

            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
            """
            reordered_past += (
                tuple(np.take(past_state, beam_idx, axis=0) for past_state in layer_past[:2]) + layer_past[2:],
            )

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
        pixel_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")
            # Convert encoder inputs in embeddings if needed
            # (when using generate, we already get encoder_outputs generated
            #  by _prepare_encoder_decoder_kwargs_for_generation)
            encoder_outputs = self.encoder(
                pixel_values,
            )

        encoder_hidden_states = encoder_outputs[0]

        # optionally project encoder_hidden_states
        if (
            self.enc_to_dec_proj is not None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        if past_key_values is None:
            init_decoder_outputs = self.decoder_init(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=decoder_attention_mask,
            )

            logits, past_key_values = init_decoder_outputs
        else:
            encoder_attention_mask = np.ones((encoder_hidden_states.shape[0], encoder_hidden_states.shape[1]), dtype=np.int64)

            logits, past_key_values = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
            )

        outputs = Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

        return outputs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)        
