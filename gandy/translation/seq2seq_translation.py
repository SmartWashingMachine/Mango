from gandy.onnx_models.marian import MarianONNX
from gandy.translation.base_translation import BaseTranslation
import logging
from gandy.utils.get_sep_regex import get_last_sentence
from gandy.utils.frame_input import FrameInput, p_transformer_join

logger = logging.getLogger('Gandy')

class Seq2SeqTranslationApp(BaseTranslation):
    def __init__(self, model_sub_path = '/'):
        self.num_beams = 5

        # '/' = j. '/zh/' = c. '/ko/' = k.
        self.model_sub_path = model_sub_path

        self.max_context = 4
        self.max_length_a = 0

        super().__init__()

    def set_max_context(self, max_c):
        self.max_context = max_c
        logger.debug(f'Set context amount to: {max_c}')

    def set_max_length_a(self, max_a):
        self.max_length_a = max_a
        logger.debug(f'Set max length A to: {max_a}')

    def load_model(self):
        s = self.model_sub_path

        logger.info('Loading translation model...')

        self.translation_model = MarianONNX(
            f'models/marian{s}encoder_q.onnx',
            f'models/marian{s}decoder_q.onnx',
            f'models/marian{s}decoder_init_q.onnx',
            f'models/marian{s}tokenizer_mt',
            use_cuda=self.use_cuda,
            max_length_a=self.max_length_a,
        )

        logger.info('Done loading translation model!')

        return super().load_model()

    def strip_padding(self, prediction):
        # Strips padding tokens.
        # We also strip any "U+2047" characters as those appear to be unknown tokens.
        # Encode and then decode to remove the weird unknown token character.
        return prediction.replace('<pad>', '').encode('ascii', 'ignore').decode('utf-8').strip()

    def process(self, i_frame: FrameInput = None, text: str = None, force_words = None, tgt_context_memory = None, output_attentions = False):
        """
        force_words, if given, should be a list containing strings and/or sublists of strings.

        Each string in force_words will be treated as a required word, and the translation output will contain that word.
        Each sublist in force_words will be treated as a disjunctive constraint, where only one of the strings (words) in the sublist is required to be in the translation output.
        """
        if i_frame is None and text is None:
            raise RuntimeError('Either i_frame or text must be given.')
        if i_frame is not None and text is not None:
            raise RuntimeError('Either i_frame or text must be given, but not both.')

        if self.max_context == 0:
            # No point in caching context if we can't use it!
            tgt_context_memory = None

        output = []
        source_tokens = None
        target_tokens = None
        attentions = None

        if i_frame is not None:
            # Only used for tgt_context_memory, if set to -1 and translating images.
            # It's a list like outputs, but whereas each string in outputs contains contextual sentences, each string here is only the current sentence.
            only_last_outputs = []

            if tgt_context_memory is not None and tgt_context_memory != '-1':
                logger.debug(f'tgt_context_memory was provided as an argument while translating iframes that are likely from images. Ignoring tgt_context_memory since it is not == -1. Value: {tgt_context_memory}')
                tgt_context_memory = None

            input_sentences = i_frame.get_untranslated_sentences(self.max_context)

            for inp in input_sentences:
                logger.debug('Translating a section of text...')

                if tgt_context_memory is not None and len(only_last_outputs) > 0:
                    tgt_context_memory_to_use = p_transformer_join(only_last_outputs)
                else:
                    tgt_context_memory_to_use = None

                predictions, attentions, source_tokens, target_tokens = self.translation_model.full_pipe(
                    inp, force_words=force_words, tgt_context_memory=tgt_context_memory_to_use, output_attentions=output_attentions
                )
                predictions = self.strip_padding(predictions)

                output.append(predictions)

                if tgt_context_memory is not None:
                    only_last_outputs.append(get_last_sentence(predictions))

                logger.debug('Done translating a section of text!')
        else:
            logger.debug('Translating given a string of text...')

            if tgt_context_memory is not None:
                tgt_context_memory_to_use = p_transformer_join(tgt_context_memory)
            else:
                tgt_context_memory_to_use = None

            predictions, attentions, source_tokens, target_tokens = self.translation_model.full_pipe(
                text, force_words=force_words, tgt_context_memory=tgt_context_memory_to_use,
                output_attentions=output_attentions,
            )
            predictions = self.strip_padding(predictions)

            output.append(predictions)

            logger.debug('Done translating a string of text!')

        return output, attentions, source_tokens, target_tokens
