from PIL import Image, ImageFilter
import numpy as np
from math import floor
import cv2

from gandy.utils.frame_input import FrameInput
from gandy.utils.speech_bubble import SpeechBubble
from gandy.image_cleaning.tnet_image_clean import TNetImageClean

class BlurMaskImageCleanApp(TNetImageClean):
    def __init__(self):
        super().__init__()

    def blur_bbox(self, mask: np.ndarray, bbox: SpeechBubble):
        x1, y1, x2, y2 = bbox
        x1 = floor(x1)
        y1 = floor(y1)
        x2 = floor(x2)
        y2 = floor(y2)

        mask[y1 : y2, x1 : x2] = 255
        return mask

    def detect_mask(self, cropped_image):
        processed, confidence_scores = self.tnet_model.full_pipe(cropped_image)

        # TODO: Need to calibrate model better.
        req_threshold = 0.4 # If over this threshold, then we believe the cropped image is for a speech bubble and not a random background with text overlayed. Use to .42
        add_to_mask = np.mean(confidence_scores) >= req_threshold

        return add_to_mask, processed

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

    def process(self, image: Image.Image, i_frame: FrameInput):
        easy_mask, added_to_easy_mask = super().process(image, i_frame, return_masks_only=True)

        inpainted_image = cv2.inpaint(np.array(image), easy_mask, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
        inpainted_image = Image.fromarray(inpainted_image)

        hard_mask = np.zeros((inpainted_image.size[1], inpainted_image.size[0]), dtype=np.uint8)

        any_added_to_hard_mask = False

        for idx, bbox in enumerate(i_frame.speech_bboxes):
            if added_to_easy_mask[idx]:
                continue

            x1, y1, x2, y2 = bbox
            x1 = floor(x1)
            y1 = floor(y1)
            x2 = floor(x2)
            y2 = floor(y2)

            hard_mask[y1 : y2, x1 : x2] = 255

            any_added_to_hard_mask = True

        if any_added_to_hard_mask:
            blurred_image = inpainted_image.filter(ImageFilter.GaussianBlur(5))
            hard_mask = Image.fromarray(hard_mask, mode='L')
            inpainted_image.paste(blurred_image, hard_mask)

        return inpainted_image
