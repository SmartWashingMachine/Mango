import logging
logger = logging.getLogger('Gandy')

try:
    from wand.image import Image
    from wand.drawing import Drawing
    from wand.font import Font
    from wand.color import Color
except Exception as e:
    logger.info('Error loading wand library for BlurRedraw (likely due to missing ImageMagick installation):')
    logger.info(e)

import numpy as np
from math import floor
from PIL import Image as PILImage

from gandy.image_redrawing.base_image_redraw import BaseImageRedraw

class WandBlurRedrawApp(BaseImageRedraw):
    def __init__(self):
        super().__init__()

    def process(self, image, i_frames, texts, debug = False, font_size=6):
        old_font_size = font_size # We restore the font size after each speech bubble, in case we use adaptative font sizing.
        new_image = image.copy()

        i = 0

        arr_img = np.array(new_image)
        with Drawing() as context:
            candidate_texts = []

            with Image.from_array(arr_img) as canvas:
                for frame in i_frames:
                    s_bboxes = frame.speech_bboxes

                    for j, s_bbox in enumerate(s_bboxes):
                        font_size = old_font_size
                        s_bb = s_bbox

                        if i >= len(texts):
                            print('WARNING: repaint_image has more speech bubbles than texts. Some speech bubbles were left untouched.')
                            break

                        text = texts[i]
                        if isinstance(text, list):
                            print('WARNING: texts should be a list of strings. Detected list of lists:')
                            print('Taking the first element out of the list and assuming it is a string.')
                            text = text[0]
                            if not text:
                                print('No item found in list. Skipping.')
                                continue

                        text = self.uppercase_text(text)

                        left = floor(s_bb[0])
                        top = floor(s_bb[1])
                        width = floor(floor(s_bb[2] - s_bb[0]) * 1.1)
                        height = floor(s_bb[3] - s_bb[1])

                        # Add all of the blurred boxes first.
                        with canvas.region(x=left, y=top, width=width, height=height) as box_canvas:
                            box_canvas.blur(sigma=6)
                            canvas.composite(box_canvas, left=left, top=top)

                        candidate_texts.append((left, top, width, height, text))

                        # Consider two frames, each containing one speech bubble.
                        # If we simply do (i = j), then i would still be 0 for the second frame and thus get the wrong speech bubble.
                        # So to fix that we add 1.
                        i += 1

                    # Then draw all of the texts.
                    for (left, top, width, height, text) in candidate_texts:
                        font = Font('resources/font.otf', stroke_width=2, stroke_color=Color('white'), color=Color('black'))
                        canvas.caption(text, font=font, gravity='center', left=left, top=top, width=width, height=height)
                        # Redraw for outside stroke.
                        font = Font('resources/font.otf', stroke_width=2, color=Color('black'))
                        canvas.caption(text, font=font, gravity='center', left=left, top=top, width=width, height=height)

                context(canvas)

                new_image = PILImage.fromarray(np.array(canvas)).convert('RGB')

        return new_image
