from PIL import Image, ImageFilter
import numpy as np
from math import floor
import cv2

from gandy.utils.frame_input import FrameInput
from gandy.utils.speech_bubble import SpeechBubble
from gandy.image_cleaning.tnet_image_clean import TNetImageClean

class TextFillCleanApp(TNetImageClean):
    def __init__(self):
        super().__init__()

    def process(self, image: Image.Image, i_frame: FrameInput):
        easy_mask, added_to_easy_mask = super().process(image, i_frame, return_masks_only=True)

        #easy_mask = np.repeat(easy_mask[None, :, :], 3, axis=0) # 3 for channels.
        easy_mask = Image.fromarray(easy_mask[:, :, 0].astype(np.uint8), mode='L')
        image.paste((255, 255, 255), mask=easy_mask)

        return image
