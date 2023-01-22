from gandy.translation.seq2seq_translation import Seq2SeqTranslationApp
import logging

logger = logging.getLogger('Gandy')

class Seq2SeqLiteTranslationApp(Seq2SeqTranslationApp):
    """
    This app is used for KO/ZH translation models. Unlike Seq2Seq, this Seq2SeqLite app can only have one contextual sentence, and a <NON> token otherwise.
    """
    def map_input(self, input_text: str):
        split = input_text.split('<SEP>')

        if len(split) == 1 or self.max_context == 0:
            input_text = f'<NON> <SEP> {split[-1].strip()}'
        else:
            input_text = f'{split[-2].strip()} <SEP> {split[-1].strip()}'

        return {
            'text': input_text,
        }

    def strip_non(self, text: str):
        return text.replace('<NON>', '').strip()

    def process(self, i_frames=None, text=None, force_words=None):
        final_input, output = super().process(i_frames, text, force_words)

        final_output = []
        for o in output:
            final_output.append(self.strip_non(o))

        return final_input, final_output
