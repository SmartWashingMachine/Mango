
from gandy.full_pipelines.base_app import BaseApp
from gandy.utils.speech_bubble import SpeechBubble
from typing import List
from PIL import Image

class BaseImageDetection(BaseApp):
    def __init__(self):
        super().__init__()

    def process(self, image: Image) -> List[SpeechBubble]:
        pass

    def begin_process(self, *args, **kwargs) -> List[SpeechBubble]:
        return super().begin_process(*args, **kwargs)
