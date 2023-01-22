from gandy.spell_correction.base_spell_correction import BaseSpellCorrection
from gandy.onnx_models.vaet import VaetONNX

class VaetParaphraseApp(BaseSpellCorrection):
    def __init__(self):
        """
        This app uses a VAE model to paraphrase the text.
        """
        super().__init__()

    def load_model(self):
        self.model = VaetONNX(
            encoder_backbone_path='models/vaet/encoder_backbone.onnx',
            embeddings_path='models/vaet/embeddings.onnx',
            decoder_with_lm_head_path='models/vaet/decoder_with_lm_head.onnx',
            tokenizer_path='models/vaet/tokenizer',
            use_cuda=self.use_cuda,
        )

        return super().load_model()

    def process(self, translation_input, translation_output):
        new_output = []
        for sentence in translation_output:
            new_output.append(self.model.full_pipe(sentence))

        return new_output
