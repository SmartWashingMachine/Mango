import logging
logger = logging.getLogger('Gandy')

try:
    from wand.image import Image
    from wand.drawing import Drawing
    from wand.font import Font
    from wand.color import Color
except Exception as e:
    logger.info('Error loading wand library for NeighborRedraw (likely due to missing ImageMagick installation):')
    logger.info(e)
import numpy as np
from math import floor
from PIL import Image as PILImage

from gandy.image_redrawing.base_image_redraw import BaseImageRedraw

class NeighborRedrawApp(BaseImageRedraw):
    def __init__(self):
        super().__init__()

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

    def get_anchor_points(self, picked_direction_val, top, left, width, height, horz_margin, vert_margin, neighbor_width):
        if picked_direction_val == 0:
            # Bottom left
            neighbor_top = top + height + vert_margin
            neighbor_left = left - horz_margin
        elif picked_direction_val == 1:
            # Left
            neighbor_top = top
            neighbor_left = left - horz_margin

            # All left/top directions need to be further shifted to accomodate for the neighbor box width/height.
            neighbor_left -= neighbor_width
        elif picked_direction_val == 2:
            # Top left
            neighbor_top = top - vert_margin
            neighbor_left = left - horz_margin
        elif picked_direction_val == 3:
            # Top
            neighbor_top = top - vert_margin
            neighbor_left = left

            # All left/top directions need to be further shifted to accomodate for the neighbor box width/height.
            # neighbor_height is assumed to be equal to height.
            neighbor_top -= height
        elif picked_direction_val == 4:
            # Top right
            neighbor_top = top - vert_margin
            neighbor_left = left + width + horz_margin
        elif picked_direction_val == 5:
            # Right
            neighbor_top = top
            neighbor_left = left + width + horz_margin
        elif picked_direction_val == 6:
            # Bottom right
            neighbor_top = top + height + vert_margin
            neighbor_left = left + width + horz_margin
        elif picked_direction_val == 7:
            # Bottom
            neighbor_top = top + height + vert_margin
            neighbor_left = left + width + horz_margin

        return neighbor_left, neighbor_top

    def get_text_gravity(self, picked_direction_val):
        """
        Returns the intended text gravity given the picked direction. (East for left direction, Southeast for topleft direction, etc...)

        Refer to get_anchor_points to see which val corresponds to which direction.
        """
        if picked_direction_val == 0:
            # Bottom left
            return 'north_east'
        elif picked_direction_val == 1:
            # Left
            return 'east'
        elif picked_direction_val == 2:
            return 'south_east'
        elif picked_direction_val == 3:
            return 'south'
        elif picked_direction_val == 4:
            return 'south_west'
        elif picked_direction_val == 5:
            # Right
            return 'west'
        elif picked_direction_val == 6:
            return 'north_west'
        elif picked_direction_val == 7:
            return 'north'

    def does_overflow(self, box, image):
        """
        Returns True if the box overflows outside of the image.
        """
        height, width, _ = image.shape

        a_x1, a_y1, a_x2, a_y2 = box

        return a_x1 < 0 or a_y1 < 0 or a_x2 >= width or a_y2 >= height

    def get_image_mean_bg(self, image, left, top, width, height):
        """
        Get the average background color of the portion of the image from the given coords.
        """
        image_slice = image[top : top + height, left : left + width]
        return np.mean(image_slice, axis=(0, 1))

    def determine_font_styles(self, bg_color):
        # From: https://stackoverflow.com/questions/1855884/determine-font-color-based-on-background-color
        # We assume bg_color is an array of (r, g, b), where r, g, and b are floats.
        r = bg_color[0]
        g = bg_color[1]
        b = bg_color[2]

        r_comp = ((r / 255.0) ** 2.2) * 0.2126
        g_comp = ((g / 255.0) ** 2.2) * 0.7152
        b_comp = ((b / 255.0) ** 2.2) * 0.0722

        y = r_comp + g_comp + b_comp

        l = y ** 0.7

        font_weight = 'bold' if (l > 0.33 and l < 0.66) else 'normal'
        font_color = Color('white') if y < 0.4 else Color('black')
        font_stroke = Color('black') if y < 0.4 else Color('white')

        return font_weight, font_color, font_stroke

    def process(self, image, i_frames, texts, font_size=6):
        """
        Attempts to draw translated text NEAR ("neighboring") the original text.

        For each box and translated text, pick 1 out of 8 available directions (left, top left, bottom left, up, bottom, bottom, right, etc...)
        and attempt to create a text box (transparent background) on that direction, with the horizontal and vertical margin being fixed initially.

        Note that out of the 8 directions, the pick is not entirely fair - the left and right directions have a greater probability of being picked. Especially the left.

        The text box will be 50% wider than the original box, but will retain the same height.
        If this new text box overlaps with any other text boxes (real or neighboring), then attempt to recreate the box on a different direction.

        If the box could not be created on any direction, then increase the horizontal and vertical margin by half and try again. (This can happen four times)

        If it still failed, then do not create a neighboring box for that box and text. Instead, the text will be prepended to the text in the next neighboring box.
        """
        old_font_size = font_size # We restore the font size after each speech bubble, in case we use adaptative font sizing.
        new_image = image.copy()

        initial_directions = [i for i in range(8)] # 0 = bottom left, 1 = left, 2 = top left, 3 = up, 4 = top right, etc...
        initial_direction_probs = [0.1, 0.3, 0.1, 0.05, 0.1, 0.2, 0.1, 0.05]

        initial_horz_margin = 7
        initial_vert_margin = 7

        new_boxes_list = []
        for f in i_frames:
            for s_bbox in f.speech_bboxes:
                new_boxes_list.append(s_bbox)

        buffered_text = ''

        k = 0

        with Drawing() as context:
            with Image.from_array(np.array(new_image)) as canvas:
                for frame in i_frames:
                    s_bboxes = frame.speech_bboxes

                    for j, s_bbox in enumerate(s_bboxes):
                        font_size = old_font_size
                        s_bb = s_bbox

                        text = texts[k]
                        if k >= len(texts):
                            print('WARNING: repaint_image has more speech bubbles than texts. Some speech bubbles were left untouched.')
                            break

                        k += 1

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

                        # If there is any buffered text (text from the previous box), then increase this neighbor's width even further to help compensate for it.
                        if len(buffered_text) > 0:
                            neighbor_width = int(width * 2)
                        else:
                            neighbor_width = int(width * 1.5)

                        did_finally_succeed = False

                        horz_margin = initial_horz_margin
                        vert_margin = initial_vert_margin
                        # Only a few tries are given.
                        for _ in range(4):
                            if did_finally_succeed:
                                break

                            directions = initial_directions.copy()
                            direction_probs = initial_direction_probs.copy()
    
                            # Attempt to go through every direction, until finding a valid match.
                            while len(directions) > 0:
                                picked_direction_val = np.random.choice(directions, p=direction_probs)
                                picked_direction_idx = directions.index(picked_direction_val)
 
                                # Remove picked direction from available list.
                                directions.pop(picked_direction_idx)
                                direction_probs.pop(picked_direction_idx)

                                # Reweight probs equally.
                                missing_prob = 1.0 - sum(direction_probs)
                                direction_probs = [v + (missing_prob / len(direction_probs)) for v in direction_probs]

                                # Set anchor points depending on the picked direction.
                                neighbor_left, neighbor_top = self.get_anchor_points(
                                    picked_direction_val,
                                    top,
                                    left,
                                    width,
                                    height,
                                    horz_margin,
                                    vert_margin,
                                    neighbor_width,
                                )

                                # x1. y1. x2. y2
                                neighbor_box = (
                                    neighbor_left,
                                    neighbor_top,
                                    neighbor_left + neighbor_width,
                                    neighbor_top + height, # Neighbor has same width as the original box.
                                )

                                # PIL images are not subscriptable. Must convert to NP array before passing it to get_image_mean_bg and does_overflow.
                                np_img = np.array(new_image)
    
                                does_overlap = self.box_overlaps(neighbor_box, new_boxes_list)
                                does_overflow = self.does_overflow(neighbor_box, np_img)
                                if not does_overlap and not does_overflow:
                                    # Success!
                                    new_boxes_list.append(neighbor_box)

                                    text_to_draw = ' '.join([buffered_text, text]).strip()

                                    bg_color = self.get_image_mean_bg(np_img, neighbor_left, neighbor_top, neighbor_width, height)

                                    font_weight, font_color, font_stroke = self.determine_font_styles(bg_color)
                                    font_grav = self.get_text_gravity(picked_direction_val)

                                    font = Font(
                                        'resources/font-bold.otf' if font_weight == 'bold' else 'resources/font.otf',
                                        color=font_color,
                                        stroke_width=2,
                                        stroke_color=font_stroke,
                                    )

                                    canvas.caption(text_to_draw, font=font, gravity=font_grav, left=neighbor_left, top=neighbor_top, width=neighbor_width, height=height)

                                    # Redraw to get outside stroke.
                                    font = Font(
                                        'resources/font-bold.otf' if font_weight == 'bold' else 'resources/font.otf',
                                        color=font_color,
                                        stroke_width=2,
                                    )

                                    canvas.caption(text_to_draw, font=font, gravity=font_grav, left=neighbor_left, top=neighbor_top, width=neighbor_width, height=height)

                                    did_finally_succeed = True
                                    break

                            # If it did not finally succeed in drawing text, then the margins are reduced.
                            horz_margin = int(horz_margin + (initial_horz_margin * 2))
                            vert_margin = int(vert_margin + (initial_vert_margin * 2))

                        if not did_finally_succeed:
                            # If it couldn't draw the text on any direction even with reduced margins, then simply prepend the text to the next text.
                            buffered_text += text

                            
                            context.fill_color = Color('black')
                            context.stroke_color = Color('red')
                            context.stroke_width = 2
                            #context.rectangle(left=left, top=top, width=width, height=height)
                            
                        else:
                            # Remove any buffered text if it successfully drew the box.
                            buffered_text = ''

                            
                            context.fill_color = Color('green')
                            context.stroke_color = Color('blue')
                            context.stroke_width = 2
                            #context.rectangle(left=left, top=top, width=width, height=height)

                context(canvas)

                new_image = PILImage.fromarray(np.array(canvas)).convert('RGB')

        return new_image
