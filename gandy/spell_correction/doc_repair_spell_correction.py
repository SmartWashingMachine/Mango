from gandy.onnx_models.doc_repair import DocRepairONNX
from gandy.translation.base_translation import BaseTranslation
import logging

logger = logging.getLogger('Gandy')

class DocRepairApp(BaseTranslation):
    def __init__(self):
        super().__init__()

    def load_model(self):

        logger.info('Loading doc repair model...')
        self.translation_model = DocRepairONNX(
            f'models/doc_repair/encoder.onnx',
            f'models/doc_repair/decoder.onnx',
            f'models/doc_repair/decoder_init.onnx',
            f'models/doc_repair/tokenizer_dr',
            use_cuda=self.use_cuda,
        )
        logger.info('Done loading doc repair model!')

        return super().load_model()

    def strip_padding(self, prediction):
        # Strips padding tokens.
        # We also strip any "U+2047" characters as those appear to be unknown tokens of some sort.
        # Encode and then decode to remove the weird unknown token character.
        return prediction.replace('<pad>', '').encode('ascii', 'ignore').decode('utf-8').strip()

    def flatten_sep(self, prediction):
        # Only take the final sentence (the current one). This strips out contextual info we don't care about.
        split_preds = prediction.split('<SEP>')

        return split_preds[-1].replace('<s>', '').replace('</s>', '').strip()


    def map_input(self, input_text):
        # If using a ptransformer model, the input texts will probably contain <SEP1> <SEP2> <SEP3> tokens.
        # But DocRepair only supports <SEP> tokens, therefore we replace them as needed.
        input_text = input_text.replace('<SEP1>', '<SEP>').replace('<SEP2>', '<SEP>').replace('<SEP3>', '<SEP>')
        return {
            'text': input_text,
        }

    def process(self, translation_input, texts):
        output = []

        for inp in texts:
            logger.debug('Editing a section of text...')

            mapped_inp = self.map_input(inp)
            if mapped_inp['text'].count('<SEP>') == 3:
                predictions = self.translation_model.full_pipe(mapped_inp)
            else:
                logger.debug('Ignoring repair of text because there is not enough contextual info.')
                predictions = mapped_inp['text']
            output.append(predictions)

            logger.debug('Done editing a section of text!')

        final_output = []

        logger.debug('Postprocessing the edited text...')
        # predictions is a list of lists, but each nested list only contains one element.
        for pr in output:
            # Converts from a lists of lists to a list of strings.
            pr = self.strip_padding(pr)
            # Keeps it as a list of strings, but may extend the list.
            pr = self.flatten_sep(pr)

            final_output.append(pr)

        logger.debug('Done postprocessing text!')

        return final_output
