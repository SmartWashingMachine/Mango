from gandy.full_pipelines.base_app import BaseApp
from gandy.utils.frame_input import FrameInput
from PIL import Image

class BaseImageRedraw(BaseApp):
    def __init__(self):
        super().__init__()

    def uppercase_text(self, text: str):
        return text.upper()

    def process(self, image: Image.Image, i_frame: FrameInput):
        pass
