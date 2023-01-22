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
from gandy.utils.enhanced_ngram_logits import monkey_patch_model
import logging

logger = logging.getLogger('Gandy')

"""

Unlike the typical Marian model, it uses beam search logic borrowed from Huggingface but adapted to use numpy arrays instead of pytorch.

"""

class OnnxMarianEncoder():
    def __init__(self, encoder_sess):
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

class OnnxMarianDecoder():
    def __init__(self, decoder_sess):
        self.decoder = decoder_sess

    def forward(self, input_ids, attention_mask, encoder_hidden_states, past_key_values, src_positions):
        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])
        
        input_names = [x.name for x in self.decoder.get_inputs()]
        inputs = [
            input_ids,
            attention_mask,
        ] + [
            tensor for tensor in flat_past_key_values
        ]

        decoder_inputs = dict(zip(input_names, inputs))

        #for (k, v) in decoder_inputs.items():
            #print(f'{k} - {v.shape}')

        decoder_outputs = self.decoder.run(None, decoder_inputs)
 
        hidden_states, pkvs = [], []
        for x in decoder_outputs[1:]:
            if x.shape[2] == 512:
                hidden_states.append(x)
            else:
                pkvs.append(x)
    
        list_pkv = tuple(x for x in pkvs)
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        list_hidden_states = tuple(x for x in hidden_states)

        return decoder_outputs[0], out_past_key_values, list_hidden_states

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class OnnxMarianDecoderInit():
    def __init__(self, decoder_sess):
        self.decoder = decoder_sess

    def forward(self, input_ids, encoder_attention_mask, encoder_hidden_states, src_positions):
        decoder_outputs = self.decoder.run(
            None,
            {
                "input_ids": input_ids.astype(np.int64),
                "encoder_attention_mask": encoder_attention_mask.astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states,
                "src_positions": src_positions,
            },
        )

        hidden_states, pkvs = [], []
        for x in decoder_outputs[1:]:
            # TODO: Optimize
            if x.shape[2] == 512:
                hidden_states.append(x)
            else:
                pkvs.append(x)
    
        list_pkv = tuple(x for x in pkvs)
        out_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        list_hidden_states = tuple(x for x in hidden_states)

        return decoder_outputs[0], out_past_key_values, list_hidden_states

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MarianKNNONNX(BaseONNXModel, GenerationMixinNumpy):
    def __init__(self, onnx_path_enc, onnx_path_dec, onnx_path_dec_init, dataloader_path, process_outputs_cb = None, use_cuda = None, max_length_a = 0):
        """
        This model uses a Marian Transformer Encoder-Decoder architecture with a tokenizer from Huggingface.

        This model is used in the seq2seq_translation app, to translate source language text to target language text.

        NOTE: A lot of the code here (with adjustments) is from a notebook somebody else created for converting MBart models to ONNX, which is itself following fastT5, a library to convert T5 models to ONNX.
        Source: https://github.com/Ki6an/fastT5/issues/7
        """
        super().__init__(use_cuda=use_cuda)
        self.main_input_name = 'input_ids'

        self.load_session(onnx_path_enc, onnx_path_dec, onnx_path_dec_init)
        self.load_dataloader(dataloader_path)

        self.config = AutoConfig.from_pretrained(dataloader_path)

        self.process_outputs_cb = process_outputs_cb

        self.max_length_a = max_length_a

        monkey_patch_model(self)

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
        # cut decoder_input_ids if past is used
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

    # Copied from MarianMTModel.
    def adjust_logits_during_generation(self, logits, cur_len):
        logits[:, self.config.pad_token_id] = float("-inf")  # never predict pad token.
        return logits

    # Copied from MarianMTModel.
    def shift_tokens_right(self, input_ids: np.ndarray, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        # Old: shifted_input_ids = input_ids.new_zeros(input_ids.shape)

        shifted_input_ids = np.zeros_like(input_ids)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    # Copied from MarianMTModel.
    def prepare_decoder_input_ids_from_labels(self, labels: np.ndarray):
        return self.shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    # Copied from MarianMTModel.
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

    # Some of the following methods are used for GenerationMixin.
    @property
    def device(self):
        return "cpu"

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_output_embeddings(self):
        return None

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: np.ndarray, model_kwargs, model_input_name = None
    ):
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"] = encoder(**encoder_kwargs)

        return model_kwargs

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
        src_positions = encoder_outputs[-1]

        if past_key_values is None:
            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states, src_positions,
            )
            logits, past_key_values, decoder_hidden_states = init_onnx_outputs
        else:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]

            onnx_outputs = self.decoder(
                decoder_input_ids.astype(np.int64),
                attention_mask.astype(np.int64),
                encoder_hidden_states,
                past_key_values,
                src_positions,
            )

            logits, past_key_values, decoder_hidden_states = onnx_outputs

        outputs = Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values, decoder_hidden_states=decoder_hidden_states)

        if self.process_outputs_cb is not None:
            # Make sure the CB modifies in-place!
            outputs = self.process_outputs_cb(outputs)

        return outputs

    def load_dataloader(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, truncation_side='left', padding_side='right')

    def load_session(self, enc_path, dec_path, dec_init_path):
        self.encoder = OnnxMarianEncoder(self.create_session(enc_path))
        self.decoder = OnnxMarianDecoder(self.create_session(dec_path))
        self.decoder_init = OnnxMarianDecoderInit(self.create_session(dec_init_path))

    def preprocess(self, inp):
        inp_text = inp['text']

        x_dict = self.tokenizer([inp_text], return_tensors='np', max_length=512, truncation=True)

        x_dict['text'] = inp_text
        return x_dict

    def begin_forward(self, x_dict, force_words = None):
        did_fail = False
        if force_words:
            force_word_ids = []

            for item in force_words:
                if did_fail:
                    break

                if isinstance(item, list):
                    for subitem in item:
                        if not isinstance(subitem, str):
                            logger.warning('Invalid value given for force_words in translation. Ignoring values.')
                            did_fail = True
                            break
                elif not isinstance(item, str):
                    logger.warning('Invalid value given for force_words in translation. Ignoring values.')
                    did_fail = True

                input_ids = self.tokenizer(item, add_prefix_space=True, add_special_tokens=False).input_ids
                force_word_ids.append(input_ids)
        if did_fail or not force_words:
            force_word_ids = None

        start = datetime.now()
        outp = self.generate(
            input_ids=x_dict['input_ids'],
            attention_mask=x_dict['attention_mask'],
            num_beams=5,
            max_length=512 if self.max_length_a == 0 else (x_dict['input_ids'].shape[1] * self.max_length_a),
            # Awful awful. Just no. Why? Why? Don't enable this.
            do_sample=False,
            # Helps with computation time with beam search.
            early_stopping=True,
            # Modifies the higher order structure. Helps prevent repeating sentences. May not actually be needed anymore, since the model appears to be decently tuned.
            no_repeat_ngram_size=7,
            # Following the "Diverse Beam Search" paper, we set the number of groups to the number of beams.
            # This will attempt to encourage each beam to have diverse outcomes, by accounting for similarity of the beam groups at each step.
            # num_beam_groups=5 if force_word_ids is None else 1,
            # Following the "CRTL" paper, we set the repetition penalty to 1.2.
            # This will attempt to further discourage repetition of previously used tokens.
            repetition_penalty=1.2,
            force_words_ids=force_word_ids,
            # No patience which is kind of a bummer, even if it is likely detrimental. plz huggingface I don't want to scour the codebase anymore
        )
        end = datetime.now()
        logger.debug(f'Translating item time elapsed: {(end - start).total_seconds()}s')
        logger.debug(f'For text: {x_dict["text"]}')

        return outp

    def postprocess(self, outp):
        decoded = self.tokenizer.decode(outp[0, ...], skip_special_tokens=False)

        return decoded
