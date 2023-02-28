from gandy.onnx_models.marian import MarianONNX
from gandy.translation.base_translation import BaseTranslation
from typing import List
import logging
from gandy.utils.clean_text import clean_text
from gandy.utils.get_sep_regex import get_last_sentence

logger = logging.getLogger('Gandy')

class Seq2SeqTranslationApp(BaseTranslation):
    def __init__(self, concat_mode = 'frame', model_sub_path = '/'):
        self.concat_mode = concat_mode
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

    def convert_text(self, input_text):
        split = input_text.split(' <SEP> ')

        if len(split) <= self.max_context or self.max_context == -1:
            input_text = split
        else:
            input_text = split[-(self.max_context):]

        # Now we're using P-transformer. As a hacky fix, replace SEPs with SEP1, SEP2, etc...
        input_text = self.p_transformer_join(input_text)
        # If we were not using P-transformer: input_text = ' <SEP> '.join(input_text).strip()
        return input_text

    def map_input(self, input_text):
        input_text = self.convert_text(input_text)

        return {
            'text': clean_text(input_text),
        }

    def process(self, i_frames = None, text = None, force_words = None, tgt_context_memory = None, output_attentions = False):
        """
        force_words, if given, should be a list containing strings and/or sublists of strings.

        Each string in force_words will be treated as a required word, and the translation output will contain that word.
        Each sublist in force_words will be treated as a disjunctive constraint, where only one of the strings (words) in the sublist is required to be in the translation output.
        """
        if i_frames is None and text is None:
            raise RuntimeError('Either i_frames or text must be given.')
        if i_frames is not None and text is not None:
            raise RuntimeError('Either i_frames or text must be given, but not both.')

        if self.max_context == 0:
            # No point in caching context if we can't use it!
            tgt_context_memory = None

        output = []

        final_input = []

        # Only used for tgt_context_memory, if set to -1 and translating images.
        # It's a list like outputs, but whereas each string in outputs contains contextual sentences, each string here is only the current sentence.
        only_last_outputs = []

        if i_frames is not None:
            if tgt_context_memory is not None and tgt_context_memory != '-1':
                logger.debug(f'tgt_context_memory was provided as an argument while translating iframes that are likely from images. Ignoring tgt_context_memory since it is not == -1. Value: {tgt_context_memory}')
                tgt_context_memory = None

            for i_frame in i_frames:
                if self.concat_mode == '2plus2':
                    inp = i_frame.get_2plus2_input(with_separator=True)
                elif self.concat_mode == 'noconcat':
                    inp = i_frame.get_single_input()
                else:
                    inp = i_frame.get_full_input(with_separator=True)

                single_inp = i_frame.get_single_input() # list
                final_input.extend(single_inp)

                logger.debug('Translating a section of text...')

                if isinstance(inp, list):
                    for i in inp:
                        if tgt_context_memory is not None and len(only_last_outputs) > 0:
                            tgt_context_memory_to_use = self.map_input(' <SEP> '.join(only_last_outputs + [' ']))['text']
                        else:
                            tgt_context_memory_to_use = None

                        predictions, attentions, source_tokens, target_tokens = self.translation_model.full_pipe(self.map_input(i), force_words=force_words, tgt_context_memory=tgt_context_memory_to_use, output_attentions=output_attentions)
                        predictions = self.strip_padding(predictions)

                        output.append(predictions)

                        if tgt_context_memory is not None:
                            only_last_outputs.append(get_last_sentence(predictions))
                else:
                    if tgt_context_memory is not None and len(only_last_outputs) > 0:
                        tgt_context_memory_to_use = self.map_input(' <SEP> '.join(only_last_outputs + [' ']))['text']
                    else:
                        tgt_context_memory_to_use = None

                    predictions, attentions, source_tokens, target_tokens = self.translation_model.full_pipe(self.map_input(inp), force_words=force_words, tgt_context_memory=tgt_context_memory_to_use, output_attentions=output_attentions)
                    predictions = self.strip_padding(predictions)

                    output.append(predictions)

                    if tgt_context_memory is not None:
                        only_last_outputs.append(get_last_sentence(predictions))

                logger.debug('Done translating a section of text!')
        else:
            logger.debug('Translating given a string of text...')
            final_input.append(text)

            if tgt_context_memory is not None:
                tgt_context_memory_to_use = self.map_input(tgt_context_memory)['text']
            else:
                tgt_context_memory_to_use = None

            predictions, attentions, source_tokens, target_tokens = self.translation_model.full_pipe(
                self.map_input(text), force_words=force_words, tgt_context_memory=tgt_context_memory_to_use,
                output_attentions=output_attentions,
            )

            predictions = self.strip_padding(predictions)

            logger.debug('Done translating a text item!')
            output.append(predictions)

        return final_input, output, attentions, source_tokens, target_tokens
