from PIL import Image
import numpy as np
from gandy.onnx_models.ttnet import TTNetONNX
from gandy.image_cleaning.base_image_clean import BaseImageClean
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
        return self.tnet_model.full_pipe(cropped_image)

    def process(self, image, i_frames):
        full_mask_image = image.copy()
        full_mask_image = np.array(full_mask_image)
        full_mask_image[:, :, :] = 0 # Fill background.
        full_mask_image = full_mask_image[:, :, :1] # Only get 1 channel.

        i = 0
        for i_frame in i_frames:
            for bbox in i_frame.speech_bboxes:
                x1, y1, x2, y2 = bbox
                x1 = floor(x1)
                y1 = floor(y1)
                x2 = floor(x2)
                y2 = floor(y2)

                cropped_image = image.crop([x1, y1, x2, y2])

                cropped_image = np.array(cropped_image)
                cropped_image = self.transform(image=cropped_image)['image']

                f = Image.fromarray(cropped_image)
                f.save(f'./maskog_{i}.png')

                detected_mask = self.detect_mask(cropped_image) # Expected to be H * W * 1 (where 1 = channel)
                detected_mask = detected_mask * 255

                # Add that text mask. Should be in range [0, 255]

                # NOTE: Sometimes the mask is larger than the image due to bounding box rounding errors. Todo fix. Currently bandaid fix.
                if x2 > full_mask_image.shape[1]:
                    detected_mask = detected_mask[:, 0 : full_mask_image.shape[1] - x1, :]
                if y2 > full_mask_image.shape[0]:
                    detected_mask = detected_mask[0 : full_mask_image.shape[0] - y1, :, :]

                #full_mask_image[y1 : y2, x1 : x2] = detected_mask

                full_mask_image[y1 : y2, x1 : x2] = detected_mask

                f = Image.fromarray(detected_mask[:, :, 0], mode='L')
                f.save(f'./maskseg_{i}.png')
                i += 1

        # For debugging: f = Image.fromarray(full_mask_image[:, :, 0], mode='L')
        # For debugging: f.save('./mask.png')

        # Clean!
        inpainted_image = cv2.inpaint(np.array(image), full_mask_image, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
        inpainted_image = Image.fromarray(inpainted_image)

        # For debugging: inpainted_image.save('./inpainted.png')

        return inpainted_image