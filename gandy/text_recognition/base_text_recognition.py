import cv2
import numpy as np
from PIL.Image import Image
from gandy.utils.frame_input import FrameInput
from gandy.full_pipelines.base_app import BaseApp
import logging

logger = logging.getLogger('Gandy')

class BaseTextRecognition(BaseApp):
    def __init__(self, merge_split_lines = True, preload = False):
        self.merge_split_lines = merge_split_lines
        super().__init__(preload)

    def process(self, image: Image, i_frame: FrameInput):
        """
        I frame is modified in-place.
        """
        pass
