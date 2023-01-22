from gandy.onnx_models.base_onnx_model import BaseONNXModel
import numpy as np
from skimage.feature import canny
import cv2
import albumentations as A

# From: https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
def pad_to(x, stride = 32):
    h, w = x.shape[-2:]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w

    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    pads = (lw, uw, lh, uh)

    # assumes channel, height, width.
    out = np.pad(x, ((0, 0), (lh, uh), (lw, uw)))

    return out, pads

def unpad(x, pad):
    # assumes height, width.

    if pad[2]+pad[3] > 0:
        x = x[pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,pad[0]:-pad[1]]
    return x

class EdgeConnectONNX(BaseONNXModel):
    def __init__(self, onnx_path, use_cuda):
        """
        This model uses EdgeConnect (further finetuned for in-domain images) to inpaint images.
        """
        super().__init__(use_cuda=use_cuda)
        self.load_session(onnx_path)

        # EdgeConnect is slower the larger the input, and may have poor accuracy. Resizing will definitely impact accuracy to some degree, but can make it faster.
        # Note that 256 works. 256 seems to give moderate quality.

        # At some point should probably retrain the model using a nearest-neighbor upsampling decoder rather than conv transpose...
        self.tfm = A.Compose([
            # A.LongestMaxSize(max_size=512, interpolation=cv2.INTER_NEAREST),
            A.LongestMaxSize(max_size=256, interpolation=cv2.INTER_AREA),
        ])

    def forward(self, inp_data):
        color_image, mask, edges, gray_image, pads, original_height, original_width, og_color_image, og_mask = inp_data

        # Unsqueeze for batch=1.
        ort_inputs = {
            'color_images': color_image[None, :, :, :],
            'masks': mask[None, :, :, :],
            'edges': edges[None, :, :, :],
            'gray_images': gray_image[None, :, :, :],
        }
        ort_outs = self.ort_sess.run(None, ort_inputs)

        outp_data = ort_outs[0], pads, original_height, original_width, og_color_image, og_mask

        return outp_data

    def channels_last_to_channels_first(self, n):
        # expects unbatched input.
        return np.transpose(n, (2, 0, 1))

    def channels_first_to_channels_last(self, n):
        # expects unbatched input.
        return np.transpose(n, (1, 2, 0))

    def preprocess(self, inp):
        color_image, mask = inp

        og_color_image, og_mask = np.copy(color_image), np.copy(mask)

        original_height, original_width = color_image.shape[0], color_image.shape[1] # channels is currently last at this point.

        transformed = self.tfm(image=color_image, mask=mask)
        color_image = transformed['image']
        mask = transformed['mask']

        # inputs should be float ndarrays.
        color_image = color_image.astype(np.float32)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        mask = mask.astype(np.float32)

        # Normalize to get values from 0 to 1.
        gray_image /= 255.
        color_image /= 255.
        mask /= 255. # And yes, EdgeConnect does NOT do well with masks not equal to 0 or 1. In the future, it may be worth training it with 0.5 values.

        edges = canny(gray_image, sigma=4.0).astype(np.float32) # use grayscale for edge detection.
        edges = np.array(edges, dtype=np.float32)

        # Convert from HWC to CHW format.
        color_image = self.channels_last_to_channels_first(color_image)
        mask = self.channels_last_to_channels_first(mask)

        # edges & gray_image have no channel dim. Add them here. No need ot use channels_last_to_channels_first since we can just add it at the start.
        edges = edges[None, :, :]
        gray_image = gray_image[None, :, :]

        # EdgeConnect requires image sizes to be divisible by 2. This util will pad around the image to ensure it can be processed.
        # Later on, the padding will be removed.
        # pads can be reassigned - they're all equivalent here.
        color_image, pads = pad_to(color_image, stride=8)
        mask, pads = pad_to(mask, stride=8)
        edges, pads = pad_to(edges, stride=8)
        gray_image, pads = pad_to(gray_image, stride=8)

        # The .forward method will unsqueeze inputs for batch=1.

        inp_data = (color_image, mask, edges, gray_image, pads, original_height, original_width, og_color_image, og_mask)
        return inp_data

    def postprocess(self, outp_data):
        outp, pads, original_height, original_width, og_color_image, og_mask = outp_data

        outp = outp.squeeze() # Remove batch from output.

        # Remove the padding applied earlier.
        outp = unpad(outp, pads)

        outp *= 255.
        outp = outp.astype(np.uint8)

        outp = self.channels_first_to_channels_last(outp)

        # Then resize to the original size.
        outp = cv2.resize(outp, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        # Mask has to be the original size.
        hole_mask = (og_mask > 0)

        merged_image = (og_color_image * (~hole_mask).astype(np.uint8)) + (outp * hole_mask.astype(np.uint8))
        return merged_image
