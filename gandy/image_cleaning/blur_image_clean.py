from PIL import ImageDraw, Image, ImageFilter
import numpy as np
from math import floor
from gandy.utils.frame_input import FrameInput

from gandy.image_cleaning.base_image_clean import BaseImageClean

class BlurImageCleanApp(BaseImageClean):
    def __init__(self):
        super().__init__()

    def clean_image(self, image: Image.Image, i_frame: FrameInput):
        input_image = image.copy()
        blurred_image = input_image.filter(ImageFilter.GaussianBlur(5))

        # Size gives width, height.
        mask = np.zeros((input_image.size[1], input_image.size[0]), dtype=np.uint8)

        for s in i_frame.speech_bboxes:
            x1, y1, x2, y2 = s
            x1 = floor(x1)
            y1 = floor(y1)
            x2 = floor(x2)
            y2 = floor(y2)

            mask[y1 : y2, x1 : x2] = 255

        mask = Image.fromarray(mask, mode='L')

        input_image.paste(blurred_image, mask)
        return input_image

    def process(self, image, i_frames):
        return self.clean_image(image, i_frames)
