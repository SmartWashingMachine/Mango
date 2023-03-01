
from gandy.full_pipelines.base_app import BaseApp
from gandy.utils.frame_input import FrameInput

class BaseTranslation(BaseApp):
    def __init__(self):
        super().__init__()

    def process(self, i_frame: FrameInput = None, text: str = None, force_words = None, tgt_context_memory = None, output_attentions = False):
        """
        I frame is modified in-place.
        """
        pass
