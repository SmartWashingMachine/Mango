
from gandy.full_pipelines.base_app import BaseApp


class BaseImageRedraw(BaseApp):
    def __init__(self):
        super().__init__()

    def uppercase_text(self, text):
        return text.upper()

    def process(self, image, i_frames, texts, debug = False, font_size=6, adaptative_font_size = True):
        pass