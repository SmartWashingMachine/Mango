import cv2
import numpy as np

from gandy.full_pipelines.base_app import BaseApp
import logging

logger = logging.getLogger('Gandy')

class BaseTextRecognition(BaseApp):
    def __init__(self, merge_split_lines = True, preload = False):
        """
        By default, the process() method will split the given image into multiple images (one per text line), and feed each text line into process_one_line().

        If we want to process the entire image, we can override the process() method.
        """
        self.merge_split_lines = merge_split_lines
        super().__init__(preload)

    # Pads the image and then attempts to filter out the background to a white color.
    def image_pad_and_filter_bg(self, image_ndarray, padding = 12, pad_value = 255):
        height, _width, depth = image_ndarray.shape

        # Yes, the types must be the same (float32) for the hstack function to work properly.
        pad = np.full((height, round(padding / 2), depth), pad_value).reshape(height, -1, 3).astype(np.float32)

        if padding % 2 != 0:
            raise RuntimeError('padding must be an even number.')

        padded = np.hstack([
            pad,
            image_ndarray,
            pad,
        ]) # depth should be 3.
        # Grayscale the image,
        padded = cv2.cvtColor(padded, cv2.COLOR_RGB2GRAY).astype(np.uint8)

        _ret, thresh = cv2.threshold(padded, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB) # Maybe not needed?
        return thresh

    # TO-DO. This function is highly unoptimized.
    def slice_image_text(self, cropped_image):
        # cropped_image should be a PIL image cropped to fit the contents of a speech bubble and then converted to an np array.
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB) # <-- WRONG? PROBABLY.

        edges = cv2.Canny(image=cropped_image, threshold1=35, threshold2=150)
        # If not converted back to RGB, we get very... strange colors.
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)

        cropped_height, width, _depth = edges.shape

        pixel_cols_activated = []
        for col_i in range(width):
            # Gets all edges in the Ith column.
            pixels = edges[:, col_i]

            # Check if any pixel in the edge column was activated.
            empty_pixel_value = 0
            activated_pixels = np.any(np.greater(pixels, empty_pixel_value))

            pixel_cols_activated.append(activated_pixels)

        items = []

        def _append_pixel_col(c, as_new = False):
            if as_new or len(items) == 0:
                items.append(c)
            else:
                items[-1] = np.hstack((items[-1], c))

        for col_i in range(width):
            # For every activated pixel column, if a neighboring column was also activated, then this column is added to the current item.
            # If no neighboring column was activated, then this column is added to a new item.

            # Gets all pixels in the Ith column.
            # This cropped_image slice will return [H, W], but we must reshape it to have [H, W, channels] where channels is 3.
            pixels = cropped_image[:, col_i].reshape((cropped_height, -1, 3))

            # Found this IF and the next to be buggy.
            if col_i > 0:
                prev_neighbor = pixel_cols_activated[col_i - 1]
                if prev_neighbor:
                    _append_pixel_col(pixels)
                    continue
            else:
                    _append_pixel_col(pixels)
                    continue

            # Check if next pixel column was
            if col_i < width-1:
                next_neighbor = pixel_cols_activated[col_i + 1]
                if next_neighbor:
                    _append_pixel_col(pixels)
                    continue
            else:
                    _append_pixel_col(pixels)
                    continue

            cur_activated = pixel_cols_activated[col_i]
            if cur_activated:
                _append_pixel_col(pixels)
                continue

            _append_pixel_col(pixels, as_new=True)


        # Every item that is less than half the width of the largest item is removed.
        # This helps filter out noisy random elements.
        largest_width = 0
        for idx, i in enumerate(items):
            _height, _width, _depth = i.shape
            if largest_width < _width:
                largest_width = _width

        new_items = []
        for i in items:
            _height, width, _depth = i.shape
            if width >= (largest_width / 2):
                j = np.float32(i)
                conv = cv2.cvtColor(j, cv2.COLOR_BGR2RGB)
                thresh = self.image_pad_and_filter_bg(conv, padding=10, pad_value=255)

                new_items.append(np.uint8(thresh))

        # Returns a group of images, each image containing one vertical text line.
        return cropped_image, edges, new_items

    def process_one_line(self, text_line_image):
        pass

    def process(self, image, i_frames):
        for i, i_frame in enumerate(i_frames):
            for j, bbox in enumerate(i_frame.speech_bboxes):
                cropped_image = image.crop(bbox)

                cropped_image = np.array(cropped_image)

                _, __, lines = self.slice_image_text(cropped_image)
                lines.reverse()

                # If merge_split_lines is true (which it should almost always be for PyTesseract), then:
                # Each vertical line will be scanned indivdually, and then combined into one speech bubble text.
                translated_lines = []

                for k, l in enumerate(lines):
                    str_words = self.process_one_line(l)

                    # For debugging.
                    logger.debug(f'(Frame {i}, Split {j+k}) == "{str_words}"')

                    if self.merge_split_lines:
                        translated_lines.append(str_words)
                    else:
                        i_frame.add_untranslated_speech_text(str_words)

                if self.merge_split_lines:
                    joined_lines = ' '.join(translated_lines)
                    i_frame.add_untranslated_speech_text(joined_lines)

        return i_frames
