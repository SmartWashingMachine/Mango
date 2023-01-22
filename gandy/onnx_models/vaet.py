from gandy.onnx_models.base_onnx_model import BaseONNXModel
import numpy as np
from scipy.special import log_softmax
from onnxruntime import (
    GraphOptimizationLevel,
    InferenceSession,
    SessionOptions,
)
from transformers import AutoTokenizer

class VaetONNX(BaseONNXModel):
    def __init__(self, encoder_backbone_path, embeddings_path, decoder_with_lm_head_path, tokenizer_path, use_cuda):
        """
        This model uses a finetuned VAE transformer model to paraphrase text.

        This model is used in the vaet_paraphrase app.
        """
        super().__init__(use_cuda=use_cuda)

        self.ort_encoder_backbone_sess = self.load_session(encoder_backbone_path)
        self.ort_embeddings_sess = self.load_session(embeddings_path)
        self.ort_decoder_with_lm_head_sess = self.load_session(decoder_with_lm_head_path)

        self.load_dataloader(tokenizer_path)

    def load_session(self, onnx_path):
        return self.create_session(onnx_path)

    def load_dataloader(self, tokenizer_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def greedy_search(self, x):
        encoder_output = self.encoder_backbone_forward(x)[0]

        bs = 1
        sos_idx = 101
        eos_idx = 102
        output_len = 512
        hid_size = 384

        z = np.random.randn(bs, output_len, hid_size) # Latent variable.

        output = np.zeros((bs, output_len), dtype=np.uint8)
        output[:, 0] = sos_idx

        for t in range(1, output_len):
            tgt_emb = self.embeddings_forward(output[:, :t])

            tgt_emb = tgt_emb + z[:, :t, :]

            decoder_output = self.decoder_with_lm_head_forward(
                encoder_output,
                z,
                tgt_emb,
            ) # [batch, inputlen, vocab]

            #decoder_output = self.lm_head(decoder_output) # [batch, inputlen, vocab]
            decoder_output[:, :, sos_idx] = float('-inf')
            probs = log_softmax(probs, axis=2) # logits to probs

            tokens = np.argmax(probs, axis=2) # [batch, inputlen]

            output[:, t] = tokens[:, -1:]

            if tokens[:, -1] == eos_idx:
                break

        return output

    def concatenate_text(self, a, b):
        if b is None:
            return f'{a} <SEP> <NON>'
        return f'{a} <SEP> {b}'

    def preprocess(self, inp):
        # inp should be a string.

        tokenized = self.tokenizer(
            self.concatenate_text(inp, None),
            max_length=512, truncation=True, return_tensors='np',
        )['input_ids']

        tokenized = tokenized[0, :] # Remove batch dim so that we have shape [seqlength]

        return tokenized

    def encoder_backbone_forward(self, x):
        input_name = self.ort_encoder_backbone_sess.get_inputs()[0].name
        ort_inp = { input_name: x.cpu().numpy() }
        outputs = self.ort_encoder_backbone_sess.run(None, ort_inp)

        return outputs[0]

    def embeddings_forward(self, x):
        input_name = self.ort_embeddings_sess.get_inputs()[0].name
        ort_inp = { input_name: x.cpu().numpy() }
        outputs = self.ort_embeddings_sess.run(None, ort_inp)

        return outputs[0]

    def decoder_with_lm_head_forward(self, x):
        input_name = self.ort_decoder_with_lm_head_sess.get_inputs()[0].name
        ort_inp = { input_name: x.cpu().numpy() }
        outputs = self.ort_decoder_with_lm_head_sess.run(None, ort_inp)

        return outputs[0]

    def forward(self, x):
        return self.greedy_search(x)

    def postprocess(self, pred_tokenized):
        return self.tokenizer.batch_decode(pred_tokenized, skip_special_tokens=True)[0]
