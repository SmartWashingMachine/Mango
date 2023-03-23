from gandy.onnx_models.base_onnx_model import BaseONNXModel
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
    ExecutionMode,
    RunOptions,
)
from transformers import (
    AutoTokenizer,
    AutoConfig
)
from gandy.utils.knn_utils.modeling_outputs import (
    Seq2SeqLMOutput,
)

from gandy.utils.knn_utils.generation_mixin import GenerationMixinNumpy
from datetime import datetime
import numpy as np
import logging
from gandy.utils.clean_text import clean_text

from gandy.onnx_models.auto_regressive.auto_regressive_decoder import OnnxArDecoder
from gandy.onnx_models.auto_regressive.auto_regressive_decoder_init import OnnxArDecoderInit
from gandy.onnx_models.auto_regressive.auto_regressive_encoder import OnnxArEncoder

logger = logging.getLogger('Gandy')

try:
    import torch
    from transformers.generation_utils import GenerationMixin
except:
    from gandy.utils.stub_generation_mixin import GenerationMixin

class BaseMarianONNX(BaseONNXModel):
    def __init__(self, onnx_path_enc, onnx_path_dec, onnx_path_dec_init, dataloader_path, process_outputs_cb = None, use_cuda = None, max_length_a = 0):
        """
        This model uses a Marian Transformer Encoder-Decoder architecture with a tokenizer from Huggingface.

        This model is used in the seq2seq_translation app, to translate source language text to target language text.

        NOTE: A lot of the code here (with adjustments) is from a notebook somebody else created for converting MBart models to ONNX, which is itself following fastT5, a library to convert T5 models to ONNX.
        Source: https://github.com/Ki6an/fastT5/issues/7
        """
        super().__init__(use_cuda=use_cuda)
        self.main_input_name = 'input_ids'

        self.config = AutoConfig.from_pretrained(dataloader_path)

        self.load_session(onnx_path_enc, onnx_path_dec, onnx_path_dec_init)
        self.load_dataloader(dataloader_path)

        self.process_outputs_cb = process_outputs_cb

        self.max_length_a = max_length_a

        # Not needed? monkey_patch_model(self)

        self.use_cuda = use_cuda
        # GenerationMixin ONLY works if use_cuda == True.
        # GenerationMixinNumpy ONLY works if use_cuda == False.

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

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            if not isinstance(layer_past[0][0], np.ndarray): # TODO - Is this right?
                reordered_past += (
                    tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
                )
            else:
                reordered_past += (
                    tuple(np.take(past_state, beam_idx, axis=0) for past_state in layer_past[:2]) + layer_past[2:],
                )

        return reordered_past

    # Some of the following methods are used for GenerationMixin.
    @property
    def device(self):
        return 'cuda:0' if self.use_cuda else 'cpu'

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

    def mc(self, tensor):
        if tensor is not None and self.use_cuda:
            return tensor.contiguous()
        else:
            return tensor

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

        # NOTE: So for some reason we NEED these contiguous checks for Iobinding (if using CUDA).
        # Even more bizarre is that the contiguous functions called above right before creating the inputs (see prepare_inputs()) are not enough. What in tarnation?
        decoder_input_ids = self.mc(decoder_input_ids)
        input_ids = self.mc(input_ids)
        attention_mask = self.mc(attention_mask)
        decoder_attention_mask = self.mc(decoder_attention_mask)
        head_mask = self.mc(head_mask)
        decoder_head_mask = self.mc(decoder_head_mask)
        encoder_outputs = self.mc(encoder_hidden_states)
        src_positions = self.mc(src_positions)
        inputs_embeds = self.mc(inputs_embeds)
        decoder_inputs_embeds = self.mc(decoder_inputs_embeds)
        if past_key_values is not None:
            for p in past_key_values:
                if p is None:
                    continue
                for q in p:
                    q = self.mc(q)

        if past_key_values is None:
            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states, src_positions,
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
                src_positions,
            )

            logits, past_key_values, decoder_hidden_states, cross_attentions = onnx_outputs

        outputs = Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values, decoder_hidden_states=decoder_hidden_states, cross_attentions=cross_attentions)

        if self.process_outputs_cb is not None:
            # Make sure the CB modifies in-place!
            outputs = self.process_outputs_cb(outputs)

        return outputs

    def load_dataloader(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, truncation_side='left', padding_side='right')

    def create_session(self, onnx_path):
        if self.use_cuda is None:
            raise RuntimeError('use_cuda must be True or False.')

        options = SessionOptions()

        options.intra_op_num_threads = 2
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

        #options.log_severity_level = 0

        if self.use_cuda:
            logger.info('CUDA enabled. Will try to use CUDA if allowed.')
            provider = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            logger.info('CUDA disabled. Will only use CPU.')
            provider = ['CPUExecutionProvider']

        ort_sess = InferenceSession(onnx_path, options, provider)

        return ort_sess

    def load_session(self, enc_path, dec_path, dec_init_path):
        self.encoder = OnnxArEncoder(self.create_session(enc_path), self.config)
        self.decoder = OnnxArDecoder(self.create_session(dec_path), self.config)
        self.decoder_init = OnnxArDecoderInit(self.create_session(dec_init_path), self.config)

    def preprocess(self, inp):
        inp_text = clean_text(inp)

        x_dict = self.tokenizer([inp_text], return_tensors='pt' if self.use_cuda else 'np', max_length=512, truncation=True)

        x_dict['text'] = inp_text
        return x_dict

    def get_decoder_input_ids(self, tgt_context_memory):
        with self.tokenizer.as_target_tokenizer():
            # We use :-1 to slice off the last token, which is the EOS token. -2 means slicing off SEP as well.
            decoder_input_ids = self.tokenizer(tgt_context_memory, return_tensors='pt' if self.use_cuda else 'np', max_length=507).input_ids[:, :-1]
        return decoder_input_ids

    def get_target_tokenizer(self):
        return self.tokenizer

    def begin_forward(self, x_dict, force_words = None, tgt_context_memory = None, output_attentions = False):
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

        if tgt_context_memory is not None:
            decoder_input_ids = self.get_decoder_input_ids(tgt_context_memory)
        else:
            decoder_input_ids = None

        extra_kwargs = {}

        if decoder_input_ids is not None and decoder_input_ids.shape[1] > 0:
            # Model performs less accurately when prepending or appending SOS token with predefined decoder input IDs.
            # batch_size = 1
            # decoder_start_token_id = self.config.decoder_start_token_id
            # initial_input_ids = np.ones((batch_size, 1), dtype=np.int64) * decoder_start_token_id
            # Prepending: extra_kwargs['decoder_input_ids'] = np.concatenate((initial_input_ids, decoder_input_ids), axis=1)
            # Appending: extra_kwargs['decoder_input_ids'] = np.concatenate((decoder_input_ids, initial_input_ids), axis=1)

            extra_kwargs['decoder_input_ids'] = decoder_input_ids

            # Model has poor coverage without this hack. Dang exposure bias...
            hacky_min_length = decoder_input_ids.shape[1] + 4
            if hacky_min_length < 512:
                extra_kwargs['min_length'] = hacky_min_length

        if x_dict['input_ids'].shape[1] > 512:
            logger.debug(f'Warning: Input is longer than supported. May affect performance. Length: {x_dict["input_ids"].shape[1]}')

        start = datetime.now()

        if self.use_cuda:
            x_dict['input_ids'] = x_dict['input_ids'].to('cuda:0')
            x_dict['attention_mask'] = x_dict['attention_mask'].to('cuda:0')

        outp = self.generate(
            input_ids=x_dict['input_ids'],
            attention_mask=x_dict['attention_mask'],
            num_beams=5,
            max_length=512 if self.max_length_a == 0 else (x_dict['input_ids'].shape[1] * self.max_length_a),
            # Awful awful. Just no. Why? Why? Don't enable this.
            do_sample=False,
            # Helps with computation time with beam search.
            early_stopping=True,
            #early_stopping=False,
            #length_penalty=1.0,
            # Modifies the higher order structure. Helps prevent repeating sentences. May not actually be needed anymore, since the model appears to be decently tuned.
            no_repeat_ngram_size=6,
            # Following the "Diverse Beam Search" paper, we set the number of groups to the number of beams.
            # This will attempt to encourage each beam to have diverse outcomes, by accounting for similarity of the beam groups at each step.
            num_beam_groups=5 if force_word_ids is None else 1,
            # Following the "CRTL" paper, we set the repetition penalty to 1.2.
            # This will attempt to further discourage repetition of previously used tokens.
            repetition_penalty=1.2,
            force_words_ids=force_word_ids,
            # No patience which is kind of a bummer, even if it is likely detrimental. plz huggingface I don't want to scour the codebase anymore
            return_dict_in_generate=output_attentions,
            output_attentions=output_attentions,
            output_scores=output_attentions, # Needed to get beam indices.
            **extra_kwargs,
        )
        end = datetime.now()
        logger.debug(f'Translating item time elapsed: {(end - start).total_seconds()}s')
        logger.debug(f'For text: {x_dict["text"]}')

        if tgt_context_memory is not None:
            logger.debug(f'With context memory: {tgt_context_memory}')

        if output_attentions:
            return outp['sequences'], self.map_attention(outp['cross_attentions'], outp['beam_indices']), x_dict['input_ids']
        else:
            return outp, None, None

    def postprocess(self, outp_data):
        seq, attentions, source_tokens_np = outp_data
        decoded = self.get_target_tokenizer().decode(seq[0, ...], skip_special_tokens=False)
        logger.debug(f'Translated: {decoded}')

        if attentions is not None:
            # Reference [0] since batch size == 1
            source_tokens = source_tokens_np[0, ...].tolist()
            target_tokens = seq[0, ...].tolist()

            source_tokens = self.tokenizer.convert_ids_to_tokens(source_tokens)
            target_tokens = self.get_target_tokenizer().convert_ids_to_tokens(target_tokens)
        else:
            source_tokens = None
            target_tokens = None

        return decoded, attentions, source_tokens, target_tokens

    def map_attention(self, cross_attentions, per_beam_idx):
        # Map the attentions returned by huggingface.

        # Ignore this - docs lie. cross_attentions is given per beam, so we need to get the best beam idx per token.
        # beam_idx = 0 # According to docs, it's just batch size no beams... so since batch is 1 we just pick the 1st.

        # Example of beam indices: [[ 0  0  0  0  0  0  1  4 -1]] (only 1 parent element since we're returning 1 sequence - the best one!
        # Index == generated target token, Value == beam index.

        # First element is always pad token, just add -1. I think... maybe?
        # Pick [0] in per_beam_idx since there's only 1 returned sequence.
        # beam_idx_per_target = [-1] + per_beam_idx[0].tolist() # Length == # of generated target tokens.
        beam_idx_per_target = per_beam_idx[0].tolist() # Length == # of generated target tokens.

        tgt_to_src_pairs = []
        for tgt_idx, tgt_info in enumerate(cross_attentions):
            last_layer_tgt_info = tgt_info[-1] # [beam, num_heads, 1, src_len]

            beam_idx_to_use = beam_idx_per_target[min(tgt_idx, len(beam_idx_per_target) - 1)]

            last_layer_tgt_info = last_layer_tgt_info[beam_idx_to_use, :, :, :] # [num_heads, 1, src_len]

            if self.use_cuda:
                tgt_to_src_info = torch.mean(last_layer_tgt_info, dim=0)
            else:
                tgt_to_src_info = np.mean(last_layer_tgt_info, axis=0) # [1, src_len]

            tgt_to_src_pairs.append(tgt_to_src_info)

        if self.use_cuda:
            tgt_to_src_pairs = torch.concat(tgt_to_src_pairs, dim=0)
        else:
            tgt_to_src_pairs = np.concatenate(tgt_to_src_pairs, axis=0) # [tgt_len, src_len]

        return tgt_to_src_pairs.tolist()

class MarianONNXNumpy(BaseMarianONNX, GenerationMixinNumpy):
    pass

class MarianONNXTorch(BaseMarianONNX, GenerationMixin):
    pass
