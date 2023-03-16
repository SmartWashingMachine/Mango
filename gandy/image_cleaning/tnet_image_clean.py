from PIL import Image
import numpy as np
from gandy.onnx_models.ttnet import TTNetONNX
from gandy.image_cleaning.base_image_clean import BaseImageClean
from gandy.utils.frame_input import FrameInput
from math import floor
import cv2
import albumentations as A

class TNetImageClean(BaseImageClean):
    def __init__(self):
        super().__init__()

        self.transform = self.get_image_transform()

    def load_model(self):
        # TSeg detects segmentation binary masks for the text.
        self.tnet_model = TTNetONNX('models/ttnet/ttnet.onnx', use_cuda=self.use_cuda)

        return super().load_model()

    def get_image_transform(self):
        transforms = [A.ToGray(always_apply=True)]

        return A.Compose(transforms)

    def detect_mask(self, cropped_image):
        processed, confidence_scores = self.tnet_model.full_pipe(cropped_image)

        add_to_mask = True
        return add_to_mask, processed

    def validate_mask(self, detected_mask, cropped_image):
        return True

    def process(self, image: Image.Image, i_frame: FrameInput, return_masks_only = False):
        full_mask_image = image.copy()
        full_mask_image = np.array(full_mask_image)
        full_mask_image[:, :, :] = 0 # Fill background.
        full_mask_image = full_mask_image[:, :, :1] # Only get 1 channel.

        added_to_mask = []

        for bbox in i_frame.speech_bboxes:
            x1, y1, x2, y2 = bbox
            x1 = floor(x1)
            y1 = floor(y1)
            x2 = floor(x2)
            y2 = floor(y2)

            cropped_image = image.crop([x1, y1, x2, y2])

            cropped_image = np.array(cropped_image)
            cropped_image = self.transform(image=cropped_image)['image']

            add_to_mask, detected_mask = self.detect_mask(cropped_image) # Expected to be H * W * 1 (where 1 = channel)

            if add_to_mask:
                detected_mask = detected_mask * 255

                # Add that text mask. Should be in range [0, 255]
                # NOTE: Sometimes the mask is larger than the image due to bounding box rounding errors. Todo fix. Currently bandaid fix.
                if x2 > full_mask_image.shape[1]:
                    detected_mask = detected_mask[:, 0 : full_mask_image.shape[1] - x1, :]
                if y2 > full_mask_image.shape[0]:
                    detected_mask = detected_mask[0 : full_mask_image.shape[0] - y1, :, :]

                if self.validate_mask(detected_mask, cropped_image):
                    full_mask_image[y1 : y2, x1 : x2] = detected_mask
                else:
                    add_to_mask = False

                #f = Image.fromarray(detected_mask[:, :, 0], mode='L')
                #f.save(f'./maskseg_{i}.png')

            added_to_mask.append(add_to_mask)

        # For debugging: f = Image.fromarray(full_mask_image[:, :, 0], mode='L')
        # For debugging: f.save('./mask.png')

        if not return_masks_only:
            # Clean!
            inpainted_image = cv2.inpaint(np.array(image), full_mask_image, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
            inpainted_image = Image.fromarray(inpainted_image)

            # For debugging: inpainted_image.save('./inpainted.png')

            return inpainted_image
        else:
            # return_masks_only should only be used for child classes (e.g: BlurMaskImageClean).
            # In this case, it returns the mask itself and a list for each speech bubble - True if it was used for this mask and False otherwise.
            return full_mask_image, added_to_mask
