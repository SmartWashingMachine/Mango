import pytesseract

from gandy.text_recognition.base_text_recognition import BaseTextRecognition

class TesseractTextRecognitionApp(BaseTextRecognition):
    def __init__(self):
        """
        This app uses PyTesseract OCR to identify text from cropped images given by the text detection app.
        """
        super().__init__()

    def process_one_line(self, text_line_image):
        # psm 5 assumes that we have a uniform block of vertical text, which should hopefully be true.
        str_words = pytesseract.image_to_string(text_line_image, lang='jpn_ver5', config='--psm 5')
        # We remove all page separators as without it our strings tend to end in \n\x0c
        str_words = str_words.replace('\n', '').replace('\x0c', '')
        return str_words
