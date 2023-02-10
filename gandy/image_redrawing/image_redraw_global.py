from PIL import ImageFont, ImageDraw
import textwrap
from math import floor
from string import ascii_letters

from gandy.image_redrawing.base_image_redraw import BaseImageRedraw

MAX_FONT_SIZE = 30
MIN_FONT_SIZE = 11

"""

Uses two font sizes for the entire image, and does not break characters in words.

Small text regions (e.g: SFX) have their own font sizes.
One font size is determined for normal-sized text regions.

"""
class ImageRedrawGlobalApp(BaseImageRedraw):
    def __init__(self):
        super().__init__()

    def wrap_text(self, text, max_chars_per_line):
        wrapped_text = textwrap.fill(text=text, width=max_chars_per_line, break_long_words=False)

        return wrapped_text

    def words_did_break(self, text, max_chars_per_line):
        # TODO: Optimize

        # If text is just one word, then it's always False.
        try:
            t = ' '.split(text)
        except ValueError:
            return False

        a = textwrap.wrap(text, width=max_chars_per_line, break_long_words=False)
        b = textwrap.wrap(text, width=max_chars_per_line, break_long_words=True)
        #['hello wo', 'rld its me']
        #['hello world', 'its me']

        if len(a) != len(b):
            return True 

        for (aa, bb) in zip(a, b):
            if aa != bb:
                return True

        return False

    def compute_font_metrics(self, s_bb, text, base_font_size):
        width = floor(s_bb[2] - s_bb[0])
        height = floor(s_bb[3] - s_bb[1])

        # Since text bboxes tend to be tight, let's increase the available space a bit.
        ADDITIONAL_FACTOR = 0.1
        width += (width * ADDITIONAL_FACTOR)
        height += (height * ADDITIONAL_FACTOR)

        best_size = None # tuple of width and height.
        last_one_fits = False
        max_char_count = 99

        best_font_size = base_font_size
        
        while (best_size is None or (best_size[0] > width or best_size[1] > height)):
            candidate_font_size = best_font_size

            font = ImageFont.truetype('resources/font.otf', candidate_font_size, encoding='unic')

            avg_char_width = sum(font.getsize(char)[0] for char in ascii_letters) / len(ascii_letters)
            candidate_max_char_count = max(1, int(width / avg_char_width)) # Max true chars before it overflows the width.
            wrapped_text = self.wrap_text(text, candidate_max_char_count)

            candidate_size = font.getsize_multiline(wrapped_text)

            if candidate_size[0] < width and candidate_size[1] < height and not self.words_did_break(text, candidate_max_char_count):
                # Stop at the first biggest size that fits.
                best_font_size = candidate_font_size
                last_one_fits = True
                best_size = candidate_size
                max_char_count = candidate_max_char_count

                break

            if candidate_font_size <= MIN_FONT_SIZE:
                # Could not fit at all :(
                best_font_size = candidate_font_size
                last_one_fits = False
                best_size = candidate_size
                max_char_count = candidate_max_char_count

                break
            else:
                best_font_size -= 1

        best_font_size = max(MIN_FONT_SIZE, best_font_size)

        return last_one_fits, best_font_size, max_char_count, best_size

    def find_best_global_font_sizes(self, frame, texts, i):
        best_font_size = MAX_FONT_SIZE
        text_details = []

        s_bboxes = frame.speech_bboxes

        _other_i = i

        areas = []
        for s_bb in s_bboxes:
            width = floor(s_bb[2] - s_bb[0])
            height = floor(s_bb[3] - s_bb[1])

            area = width * height
            areas.append(area)

            _other_i += 1

        mean_area = sum(areas) / len(areas)

        for j, s_bbox in enumerate(s_bboxes):
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
            area = width * height

            # MEAN_FACTOR = 0.7
            #MEAN_FACTOR = 0.55
            MEAN_FACTOR = 0.65
            if area <= (mean_area * MEAN_FACTOR):
                best_font_size_to_use = best_font_size
                text_will_fit, best_font_size_to_use, max_chars_per_line, (best_width, best_height) = self.compute_font_metrics(s_bb, text, best_font_size_to_use)
            else:
                text_will_fit, best_font_size, max_chars_per_line, (best_width, best_height) = self.compute_font_metrics(s_bb, text, best_font_size)
                best_font_size_to_use = best_font_size

            wrapped_text = self.wrap_text(text, max_chars_per_line)
            start_top = (height - best_height) // 2

            offset_left = (width - best_width) // 2

            text_details.append([wrapped_text, left + offset_left, top + start_top, best_font_size_to_use])

            i += 1

        # For each bucket:
        # Return i (integer) and a list of list of wrapped texts with their left and top positions and best font size.
        return i, text_details

    def process(self, image, i_frames, texts, debug = False, font_size=6, adaptative_font_size = True):
        new_image = image.copy()

        draw = ImageDraw.Draw(new_image)

        i = 0
        for frame in i_frames:
            i, text_details = self.find_best_global_font_sizes(frame, texts, i)

            for td in text_details:
                # td = A list containing [wrappedtextstring, leftinteger, topinteger, fontsizeinteger]
                best_font_size = td[3]
                font = ImageFont.truetype('resources/font.otf', best_font_size, encoding='unic')

                draw.multiline_text((td[1], td[2]), td[0], '#000', font, align='center', stroke_fill='white', stroke_width=max(2, best_font_size // 7))

        return new_image
