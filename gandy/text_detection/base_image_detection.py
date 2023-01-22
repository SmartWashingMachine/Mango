
from gandy.full_pipelines.base_app import BaseApp


class BaseImageDetection(BaseApp):
    def __init__(self):
        super().__init__()

    def process(self, image):
        pass