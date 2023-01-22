from nlprule import Tokenizer, Rules

from gandy.spell_correction.base_spell_correction import BaseSpellCorrection
import logging

logger = logging.getLogger('Gandy')

class RuleBasedSpellCorrectionApp(BaseSpellCorrection):
    def __init__(self):
        """
        This app uses simple rule based spell correction to help make text more readable after translation.
        """
        spell_tokenizer = Tokenizer.load("en")
        self.model = Rules.load("en", spell_tokenizer)

    def process(self, translation_input, translation_output):

        logger.debug('Correcting all text...')
        new_output = []
        for sentence in translation_output:
            new_output.append(self.model.correct(sentence))
        logger.debug('Done correcting all text!')

        return new_output
