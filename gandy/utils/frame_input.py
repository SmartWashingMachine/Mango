from typing import List, Dict
from gandy.utils.replace_terms import replace_terms
from gandy.utils.speech_bubble import SpeechBubble

class FrameInput():
    """
    A class containing all of the untranslated text bounding regions in an image ("frame") as well as their untranslated and translated text.
    """

    def __init__(self, speech_bboxes: List[SpeechBubble]):
        # A list of ndarrays consisting of speech bboxes, found via the speech bubble detection model.
        self.speech_bboxes = speech_bboxes
        # A string list of words in the speech bboxes, found via the OCR model. Each item corresponds to all the text in a speech bubble.
        self.untranslated_speech_text: List[str] = []

        # A string list of translated sentences from the post-editing model postprocessed outputs.
        self.translated_sentences = []

    def replace_terms_source_side(self, terms: List[Dict]):
        """
        Apply user terms to the untranslated texts. Called right before using the translation model.
        """
        self.untranslated_speech_text = replace_terms(self.untranslated_speech_text, terms, on_side='source')

    def replace_terms_target_side(self, terms: List[Dict]):
        """
        Apply user terms to the translated texts. Called right after using the post-editing model.
        """
        self.translated_sentences = replace_terms(self.translated_sentences, terms, on_side='target')

    def add_untranslated_speech_text(self, s):
        self.untranslated_speech_text.append(s)

    def add_translated_sentence(self, s):
        self.translated_sentences.append(s)

    def _p_transformer_join(self, input_texts: List[str]):
        new_input_text = ''
        for i, t in enumerate(input_texts):
            new_input_text += t

            # If not the last sentence, then append the appropriate SEP token.
            if i < (len(input_texts) - 1):
                sep_token = f' <SEP{i+1}> '
                new_input_text += sep_token
            else:
                new_input_text += ' '

        return new_input_text

    def get_untranslated_sentences(self, texts, max_context):
        """
        Retrieve the proper text inputs for the translation models with prior sentences used as context.
        """

        if len(texts) <= max_context or max_context == -1:
            input_text = texts
        else:
            input_text = texts[-(max_context):]

        # Concat sentences into a string in the form "A <SEP1> B <SEP2> C <SEP3> ... <SEPN> N"
        speech = self._p_transformer_join(input_text)

        return speech.strip()

    @classmethod
    def from_speech_bubbles(cls, speech_bubbles: List[SpeechBubble]):
        """
        Creates a FrameInput from the postprocessed output of a text detection app.
        """
        return cls(speech_bubbles=speech_bubbles)


def unite_i_frames(i_frames: List[FrameInput], context_input: List[str]):
    """
    Merge i_frame TEXT into one. No frame_bbox or speech_bboxes are merged.
    """
    speech_text = ''

    for i_f in i_frames:
        for st in i_f.untranslated_speech_text:
            speech_text += st

    united_frame = FrameInput(frame_bbox=None, speech_bboxes=None)

    if context_input is not None and len(context_input) > 0:
        split_context = ' <SEP> '.join(context_input).strip()
        full_input = f'{split_context} <SEP> {speech_text}'
    else:
        full_input = speech_text

    united_frame.add_untranslated_speech_text(full_input)
    return united_frame
