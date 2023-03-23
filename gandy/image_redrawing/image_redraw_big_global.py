from PIL import ImageFont, ImageDraw, Image
from gandy.utils.frame_input import FrameInput

from gandy.image_redrawing.image_redraw_global import ImageRedrawGlobalApp

class ImageRedrawBigGlobalApp(ImageRedrawGlobalApp):
    def box_overlaps(self, box, other_boxes):
        """
        Returns True if box overlaps with any of the other boxes, and False otherwise.

        Assume other_boxes is a list of list of coords, and box is a list of coords.
        """
        a_x1, a_y1, a_x2, a_y2 = box

        def _is_overlapping_1d(a_1, b_1, a_2, b_2):
            # a_2 must be greater than a_1. b_2 must be greater than b_1. a_1 and a_2 must be for the same box. b_1 and b_2 must be for another same box.
            return a_2 >= b_1 and b_2 >= a_1

        for other in other_boxes:
            # (x1, y1, x2, y2)
            b_x1, b_y1, b_x2, b_y2 = other

            # From: https://stackoverflow.com/questions/20925818/algorithm-to-check-if-two-boxes-overlap
            is_overlapping = _is_overlapping_1d(a_x1, b_x1, a_x2, b_x2) and _is_overlapping_1d(a_y1, b_y1, a_y2, b_y2)
            if is_overlapping:
                return True

        return False

    def expand_boxes(self, i_frame: FrameInput, image: Image.Image):
        # TODO: Super inefficient.

        inc_factor = 1.4 # +40% width

        for i, s_bb in enumerate(i_frame.speech_bboxes):
            try_factor = inc_factor

            old_x1, old_y1, old_x2, old_y2 = s_bb

            others = [i_frame.speech_bboxes[j] for j in range(len(i_frame.speech_bboxes)) if j != i]

            #while try_factor >= 1.0 and any(self.can_overlap(s_bb, o) for o in others):
            do_try = True
            while try_factor >= 1.0 and do_try:
                shift_factor = (old_x2 - old_x1) * (try_factor - 1.)
                s_bb[0] = old_x1 - shift_factor
                s_bb[2] = old_x2 + shift_factor

                try_factor -= 0.1
                does_overfill_image = s_bb[0] < 0 or s_bb[2] >= image.width
                do_try = self.box_overlaps(s_bb, others) or does_overfill_image

    def process(self, image: Image.Image, i_frame: FrameInput):
        new_image = image.copy()

        draw = ImageDraw.Draw(new_image)

        # Expand hack, since object detection model tends to fit the text tightly rather than the bubble.
        self.expand_boxes(i_frame, image)

        i = 0
        i, text_details = self.find_best_global_font_sizes(i_frame, i_frame.translated_sentences, i)

        for td in text_details:
            # td = A list containing [wrappedtextstring, leftinteger, topinteger, fontsizeinteger]
            best_font_size = td[3]
            font = ImageFont.truetype('resources/fonts/font.otf', best_font_size, encoding='unic')

            draw.multiline_text((td[1], td[2]), td[0], '#000', font, align='center', stroke_fill='white', stroke_width=max(2, best_font_size // 7))

        return new_image
