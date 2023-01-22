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
from datetime import datetime
import numpy as np
from gandy.utils.enhanced_ngram_logits import monkey_patch_model
import logging

logger = logging.getLogger('Gandy')

class MarianONNX(BaseONNXModel):
    def __init__(self, model_path, dataloader_path, process_outputs_cb = None, use_cuda = None, max_length_a = 0):
        """
        This model uses a Marian Transformer Encoder-Decoder architecture with a tokenizer from Huggingface.

        This model is used in the seq2seq_translation app, to translate source language text to target language text.

        """
        super().__init__(use_cuda=use_cuda)
        self.main_input_name = 'input_ids'

        self.load_session(model_path)
        self.load_dataloader(dataloader_path)

        self.config = AutoConfig.from_pretrained(dataloader_path)

        self.process_outputs_cb = process_outputs_cb # NOTE: Unused.

        self.max_length_a = max_length_a

        # May not be needed anymore. TODO
        monkey_patch_model(self)

    def create_session(self, onnx_path):
        # TODO: Fix up.

        if self.use_cuda is None:
            raise RuntimeError('use_cuda must be True or False.')

        options = SessionOptions()
        options.intra_op_num_threads = 2
        options.execution_mode = ExecutionMode.ORT_SEQUENTIAL

        if self.use_cuda:
            # Actually DirectML but go on.
            options.enable_mem_pattern = False

            # Using extended optimizations gives weird errors. I don't speak ONNXenese.
            #options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
        else:
            # Get weird errors with DirectML using this.
            #options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
            pass

        options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL

        # TODO right now the beam search models are unoptimized, leading to a lot of warning messages. Will fix later. - 27 years ago
        # For debugging: options.log_severity_level = 0

        # Note that CUDA errors are not properly logged right now :/
        # Note that it's not actually CUDA - changed it to DirectML since installing cudNN is going to be a MAJOR PAIN IN THE *** for most end users.

        try:
            if self.use_cuda:
                logger.info('CUDA enabled. Will try to use DirectML if allowed.')
                provider = ['DmlExecutionProvider', 'CPUExecutionProvider']
            else:
                logger.info('CUDA disabled. Will only use CPU.')
                provider = ['CPUExecutionProvider']

            self.ort_sess = InferenceSession(onnx_path, options, provider)
            # ? self.ort_sess.disable_fallback()
        except Exception as e:
            print('An error?:')
            logger.exception(e)

        return self.ort_sess

    def load_dataloader(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, truncation_side='left', padding_side='right')

    def load_session(self, model_path):
        self.ort_sess = self.create_session(model_path)

    def preprocess(self, inp):
        inp_text = inp['text']

        x_dict = self.tokenizer([inp_text], return_tensors='np', max_length=512, truncation=True)

        x_dict['text'] = inp_text
        return x_dict

    def begin_forward(self, x_dict, force_words = None):
        if force_words:
            logger.info('force_words is not supported for MT with baked in beam search... yet. Ignoring.')

        start = datetime.now()
        outp = self.ort_sess.run(
            None,
            {
                "input_ids": x_dict['input_ids'].astype(np.int64),
                "attention_mask": x_dict['attention_mask'].astype(np.int64),
                "num_beams": np.array(5, dtype=np.int64),
                "max_length": np.array(512 if self.max_length_a == 0 else (x_dict['input_ids'].shape[1] * self.max_length_a), dtype=np.int64),
                "decoder_start_token_id": np.array(self.config.decoder_start_token_id, dtype=np.int64),
            },
        )
        end = datetime.now()
        logger.debug(f'Translating item time elapsed: {(end - start).total_seconds()}s')
        logger.debug(f'For text: {x_dict["text"]}')

        return outp

    def postprocess(self, outp):
        # Old: decoded = self.tokenizer.decode(outp.squeeze(), skip_special_tokens=False)

        decoded = self.tokenizer.decode(outp[0][0, ...], skip_special_tokens=False)

        return decoded

"""
The neighbor may still work. Seems its just a warning. If not try with opset==11 non quantized.
The mem may just be an incompatibility.

Anyways for now;
gather more phantom data.
copy over apps after building and see if it works, after a quick debug round.
IF the neighbor works then raise a github issue
Also make opset==11 before copying just in case.

For programming? Try and figure out a finer sentence level agreement loss, only using the current non mask. Extend from focused mix.
"""