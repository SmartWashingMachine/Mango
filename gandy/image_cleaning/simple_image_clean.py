from PIL import ImageDraw, Image
from gandy.utils.frame_input import FrameInput
from gandy.image_cleaning.base_image_clean import BaseImageClean

class SimpleImageCleanApp(BaseImageClean):
    def __init__(self):
        super().__init__()

    # After we get our translated output, we want to clean the image so that we can place text in later on.
    # Fills the bounding box areas with white.

    def clean_image(self, image: Image.Image, i_frame: FrameInput):
        input_image = image.copy()
        input_draw = ImageDraw.Draw(input_image)

        mask_color = (255, 255, 255)
        for s in i_frame.speech_bboxes:
            input_draw.rounded_rectangle(s, outline=mask_color, fill=mask_color, width=1, radius=30)

        return input_image

    def process(self, image, i_frames):
        return self.clean_image(image, i_frames)
