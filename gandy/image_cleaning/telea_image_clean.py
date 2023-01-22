from PIL import ImageDraw, Image
import cv2
import numpy as np
from gandy.image_cleaning.base_image_clean import BaseImageClean

class TeleaImageCleanApp(BaseImageClean):
    def __init__(self):
        super().__init__()

    def clean_image(self, image, i_frames):
        all_speech_bboxes = []
        for f in i_frames:
            all_speech_bboxes.extend(f.speech_bboxes)

        input_image = image.copy()
        input_width, input_height = input_image.size

        # mode=L refers to grayscale.
        mask_image = Image.new(mode='L', size=(input_width, input_height)) # The image will have a black background by default.
        mask_draw = ImageDraw.Draw(mask_image)
        for s in all_speech_bboxes:
            mask_draw.rectangle(s, outline=255, fill=255, width=1)

        input_image = np.asarray(input_image)
        mask_image = np.asarray(mask_image)

        inpainted_image = cv2.inpaint(input_image, mask_image, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        inpainted_image = Image.fromarray(inpainted_image)
        return inpainted_image

    def process(self, image, i_frames):
        return self.clean_image(image, i_frames)
