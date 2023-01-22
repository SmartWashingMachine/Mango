import logging
logger = logging.getLogger('Gandy')

try:
    from wand.image import Image
    from wand.drawing import Drawing
    from wand.font import Font
    from wand.color import Color
except Exception as e:
    logger.info('Error loading wand library for CaptionRedraw (likely due to missing ImageMagick installation):')
    logger.info(e)

import numpy as np
from math import floor
from PIL import Image as PILImage

from gandy.image_redrawing.base_image_redraw import BaseImageRedraw

class WandCaptionRedrawApp(BaseImageRedraw):
    def __init__(self):
        super().__init__()

    def process(self, image, i_frames, texts, debug = False, font_size=6, adaptative_font_size = True):
        old_font_size = font_size # We restore the font size after each speech bubble, in case we use adaptative font sizing.
        new_image = image.copy()

        i = 0

        with Drawing() as context:
            context.fill_color = Color('red')
            context.stroke_color = Color('yellow')
            context.stroke_width = 2

            with Image.from_array(np.array(new_image)) as canvas:
                font = Font('resources/font.otf')

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
                        width = floor(s_bb[2] - s_bb[0])
                        height = floor(s_bb[3] - s_bb[1])

                        #context.fill_color = Color('black')
                        #context.stroke_color = Color('red')
                        #context.stroke_width = 2
                        #context.rectangle(left=left, top=top, width=width, height=height)

                        # Source: https://github.com/emcconville/wand/blob/master/wand/image.py
                        canvas.caption(text, font=font, gravity='center', left=left, top=top, width=width, height=height)

                        # Consider two frames, each containing one speech bubble.
                        # If we simply do (i = j), then i would still be 0 for the second frame and thus get the wrong speech bubble.
                        # So to fix that we add 1.
                        i += 1

                context(canvas)

                new_image = PILImage.fromarray(np.array(canvas)).convert('RGB')

        return new_image
