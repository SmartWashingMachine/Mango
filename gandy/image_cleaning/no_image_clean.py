from PIL import ImageDraw

from gandy.image_cleaning.base_image_clean import BaseImageClean

class NoImageCleanApp(BaseImageClean):
    def __init__(self):
        super().__init__()

    def process(self, image, i_frames):
        return image
