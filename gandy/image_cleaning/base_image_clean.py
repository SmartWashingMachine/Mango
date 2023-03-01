
from gandy.full_pipelines.base_app import BaseApp
from PIL.Image import Image
from gandy.utils.frame_input import FrameInput

class BaseImageClean(BaseApp):
    def __init__(self):
        super().__init__()

    def process(self, image: Image, i_frame: FrameInput):
        pass

    def begin_process(self, *args, **kwargs) -> Image:
        return super().begin_process(*args, **kwargs)
