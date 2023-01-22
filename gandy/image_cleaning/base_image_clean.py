
from gandy.full_pipelines.base_app import BaseApp


class BaseImageClean(BaseApp):
    def __init__(self):
        super().__init__()

    def process(self, image, i_frames):
        pass
