from gandy.onnx_models.marian import MarianONNX
from gandy.translation.base_translation import BaseTranslation
from typing import List
import logging
from gandy.utils.clean_text import clean_text

logger = logging.getLogger('Gandy')

class Seq2SeqTranslationApp(BaseTranslation):
    def __init__(self, concat_mode = 'frame', model_sub_path = '/'):
        self.concat_mode = concat_mode
        self.num_beams = 5

        # '/' = j. '/zh/' = c. '/ko/' = k.
        self.model_sub_path = model_sub_path

        self.max_context = -1
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
            f'models/marian{s}optimized_marian.onnx',
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

    def p_transformer_join(self, input_texts: List[str]):
        new_input_text = ''
        for i, t in enumerate(input_texts):
            new_input_text += t

            # If not the last sentence, then append the appropriate SEP token.
            if i < (len(input_texts) - 1):
                sep_token = f' <SEP{i+1}> '
                new_input_text += sep_token
            else:
                new_input_text += ' '

        return new_input_text.strip()

    def map_input(self, input_text):
        split = input_text.split(' <SEP> ')

        if len(split) <= self.max_context or self.max_context == -1:
            input_text = split
        else:
            input_text = split[-(self.max_context):]

        # Now we're using P-transformer. As a hacky fix, replace SEPs with SEP1, SEP2, etc...
        input_text = self.p_transformer_join(input_text)
        # If we were not using P-transformer: input_text = ' <SEP> '.join(input_text).strip()

        return {
            'text': clean_text(input_text),
        }

    def process(self, i_frames = None, text = None, force_words = None, tgt_context_memory = None):
        """
        force_words, if given, should be a list containing strings and/or sublists of strings.

        Each string in force_words will be treated as a required word, and the translation output will contain that word.
        Each sublist in force_words will be treated as a disjunctive constraint, where only one of the strings (words) in the sublist is required to be in the translation output.
        """
        if i_frames is None and text is None:
            raise RuntimeError('Either i_frames or text must be given.')
        if i_frames is not None and text is not None:
            raise RuntimeError('Either i_frames or text must be given, but not both.')

        output = []

        final_input = []

        if i_frames is not None:
            if tgt_context_memory is not None:
                logger.debug('tgt_context_memory was provided as an argument while translating iframes that are likely from images. Ignoring tgt_context_memory.')

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
                        predictions = self.translation_model.full_pipe(self.map_input(i), force_words=force_words)

                        output.append([predictions])
                else:
                    predictions = self.translation_model.full_pipe(self.map_input(inp), force_words=force_words)
                    output.append([predictions])

                logger.debug('Done translating a section of text!')
        else:
            logger.debug('Translating given a string of text...')
            final_input.append(text)

            predictions = self.translation_model.full_pipe(self.map_input(text), force_words=force_words, tgt_context_memory=tgt_context_memory)

            logger.debug('Done translating a text item!')
            output.append([predictions])

        final_output = []

        logger.debug('Postprocessing the translated text...')

        # predictions is a list of lists, but each nested list only contains one element.
        for prediction in output:
            # We loop over the nested list anyways just to be safe.
            for pr in prediction:
                # output is a string
                pr = self.strip_padding(pr)

                final_output.append(pr)

        logger.debug('Done postprocessing text!')
        return final_input, final_output
