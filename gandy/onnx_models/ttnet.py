import albumentations as A
import numpy as np
import cv2

from gandy.onnx_models.base_onnx_model import BaseONNXModel

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

def binary_filter_outs(o):
    mask_o = o >= 0.15

    o[mask_o] = 1
    o[~mask_o] = 0

class TTNetONNX(BaseONNXModel):
    def __init__(self, onnx_path, use_cuda):
        """
        This model uses a ResNet18 UNET architecture to detect masks for text pixels in an image.
        """
        super().__init__(use_cuda=use_cuda)
        self.load_session(onnx_path)

        self.tfm = A.Compose([
            A.LongestMaxSize(max_size=256, interpolation=cv2.INTER_NEAREST, always_apply=True),
            A.ToGray(always_apply=True),
        ])

    def forward(self, inp_data):
        x, pads, original_height, original_width = inp_data

        input_name = self.ort_sess.get_inputs()[0].name
        ort_inputs = { input_name: x }
        ort_outs = self.ort_sess.run(None, ort_inputs)

        outp_data = ort_outs[0], pads, original_height, original_width

        return outp_data

    def preprocess(self, inp):
        # inp should be an ndarray.

        # NOTE: Maybe not - While the model was trained on images with the longest max size being 128, thankfully it seems to generalize well for different resolutions.
        original_height, original_width = inp.shape[0], inp.shape[1]

        inp = self.tfm(image=inp)['image']
        # Convert from HWC to CHW format.
        inp = inp.transpose(2, 0, 1)

        # TNet requires image sizes to be divisible by 32. This util will pad around the image to ensure it can be processed.
        # Later on, the padding will be removed.
        inp, pads = pad_to(inp, stride=32)

        # Model expects batch * c * h * w. Unsqueeze to get batch of 1.
        inp = inp[None, :, :, :]

        inp = inp.astype(np.float32)
        inp /= 255. # Normalize to get values from 0 to 1.

        inp_data = (inp, pads, original_height, original_width)
        return inp_data

    def postprocess(self, outp_data, to_255 = False):
        outp, pads, original_height, original_width = outp_data

        outp = outp.squeeze() # Remove batch from output.

        # TNet returns a probability of each pixel being activated. This util checks if the probability passes a fixed threshold and sets it to 1, and 0 otherwise.
        binary_filter_outs(outp)

        # Remove the padding applied earlier.
        outp = unpad(outp, pads)

        if to_255:
            # 1s become 255, and 0s stay 0s.
            # This must be disabled for FFC Inpainting.
            outp *= 255.

        outp = outp.astype(np.uint8)

        # May lose some precision here.
        outp = cv2.resize(outp, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

        ## Later on, the mask will be multiplied with an image of C * H * W.
        ## But our mask has a shape of H * W. In order to make it broadcastable, we repeat it along a new channel dim.
        #outp = np.repeat(outp[None, :, :], 3, axis=0)

        # Unsqueeze to get H * W * C, where C is 1.
        outp = outp[:, :, None]

        return outp
