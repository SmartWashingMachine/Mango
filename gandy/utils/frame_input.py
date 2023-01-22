import numpy as np
from typing import List, Dict
from gandy.utils.replace_terms import replace_terms

### These utils outside FrameInput are from Pytorch, but modified for numpy arrays.
### "Why go through all this work just to use Numpy??? Numpy sucks!" Compiled with PyInstaller, PyTorch can take up to 1.4 GB of space (wowzers!), so Numpy is ideal.
### If I'm going to make the user download GBs of data, it's going to be for datastores for the nearest neighbor model - not one library...

def _upcast(t: np.ndarray):
    # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
    if isinstance(t, np.floating):
        return t if t.dtype in (np.float32, np.float64) else t.astype(np.float32)
    else:
        return t if t.dtype in (np.int32, np.int64) else t.astype(np.int64)

def box_area(boxes: np.ndarray):
    """
    Computes the area of a set of bounding boxes, which are specified by their
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        Tensor[N]: the area for each box
    """
    boxes = _upcast(boxes)
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def _box_inter_union(boxes1: np.ndarray, boxes2: np.ndarray):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # Old: lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    # Old: rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    # Old: wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    wh = _upcast(rb - lt).clip(min=0)  # [N,M,2]

    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray):
    """
    Return intersection-over-union (Jaccard index) between two sets of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4]): first set of boxes
        boxes2 (Tensor[M, 4]): second set of boxes

    Returns:
        Tensor[N, M]: the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou

class FrameInput():
    def __init__(self, frame_bbox, speech_bboxes):
        self.frame_bbox = frame_bbox
        # A list of tensors consisting of speech bboxes, found via the speech bubble detection model.
        self.speech_bboxes = speech_bboxes
        # A string list of words in the speech bboxes, found via the OCR model. Each item corresponds to all the text in a speech bubble.
        self.untranslated_speech_text = []
        # A string list of words found via the frame feature detector.
        # The translator will append these words to each speech bbox.
        # UNUSED TODO
        self.additional_context = []

    def replace_terms(self, terms: List[Dict]):
        self.untranslated_speech_text = replace_terms(self.untranslated_speech_text, terms, on_side='source')

    def add_context(self, c, as_token=True):
        if as_token:
            self.additional_context.append(f'<{c}>')
        else:
            self.additional_context.append(c)

    def add_untranslated_speech_text(self, s):
        self.untranslated_speech_text.append(s)

    def add_speech_bbox(self, bbox):
        self.speech_bboxes.append(bbox)

    def get_sentence(self, texts, with_sep, no_context = False):
        sep_token = ' <SEP> ' if with_sep else ' '

        context = ''.join(self.additional_context)
        speech = sep_token.join(texts)

        if no_context:
            return speech.strip()
        return f'{context} {speech}'.strip()

    # this function takes all the texts in the frame into account.
    def get_full_input(self, no_context = True, with_separator=True):
        # Returns a single string containing all the context tokens and the speech text sentences.
        return self.get_sentence(self.untranslated_speech_text, with_separator, no_context)

    def get_single_input(self):
        return self.untranslated_speech_text

    def get_2plus2_input(self, no_context = True, with_separator=True):
        # Returns a list of strings with the same length as untranslated_speech_text.
        # Each string contains a speech text sentence along with the previous original speech text sentences.
        # The first sentence will be unchanged.
        inputs = []
        for i, t in enumerate(self.untranslated_speech_text):
            if i == 0:
                inputs.append(t)
                continue

            prev_texts = [self.untranslated_speech_text[i - 1]]
            if (i - 2) >= 0:
                prev_texts.append(self.untranslated_speech_text[i - 2])
            if (i - 3) >= 0:
                prev_texts.append(self.untranslated_speech_text[i - 3])

            prev_texts = prev_texts[::-1] # Reverse the context such that the newest context comes later.
            prev_texts.append(t)

            speech = self.get_sentence(prev_texts, with_separator, no_context)
            inputs.append(speech)

        return inputs

    # Unused for now. This function does not take the future texts in the frame into account.
    def get_incremental_input(self, no_context = True, with_separator=True):
        # Returns a list of strings with the same length as untranslated_speech_text.
        # Each string contains a speech text sentence along with the context tokens.
        inputs = []
        for i, t in enumerate(self.untranslated_speech_text):
            prev_texts = self.untranslated_speech_text[:i] if i > 0 else []
            prev_texts.append(t)

            speech = self.get_sentence(prev_texts, with_separator, no_context)
            inputs.append(speech)

        return inputs

# Return a list of frame inputs, where each item has a frame and it's speech bubbles.
def link_speechbubble_to_frame(speech_bubbles, frames):
    def _create_frame_inputs(frames):
        # Return a list where the Ith frame corresponds to the Ith FrameInput.

        #n_frames = frames.shape(0)
        return [FrameInput(f, speech_bboxes=[]) for f in frames]

    # speech_bubbles and frames are bounding box tensors.
    # Each speech bubble will be linked to the frame with the highest IoU value, as a high IoU value means that the speech bubble box is mostly inside the frame box.
    i_frame_for_speech_bubbles = []

    # Returns [N, M] tensor where N is the number of bubbles, and M is the number of frames.
    iou_values = box_iou(speech_bubbles, frames)

    v_len = len(iou_values)
    for i in range(v_len):
        # This list has an IoU value for the speech bubble for every frame.
        bubble_iou_values = iou_values[i, :]

        # This returns the INDEX of the frame with the highest IoU value.
        best_frame_index = np.argmax(bubble_iou_values)
        i_frame_for_speech_bubbles.append(best_frame_index)

    frame_inputs = _create_frame_inputs(frames)
    for i, f in enumerate(frame_inputs):
        # Returns a list of integers, where each integer is the index for a speech bubble for this frame.
        speech_indices = [idx for (idx, f_idx) in enumerate(i_frame_for_speech_bubbles) if f_idx == i]
        for si in speech_indices:
            bubble = speech_bubbles[si]

            # Convert bubble from tensor to list, as we need to perform other operations on it.
            bubble = bubble.tolist()

            f.add_speech_bbox(bubble)

    return frame_inputs

def unite_i_frames(i_frames: List[FrameInput], context_input: List[str]):
    """
    Merge i_frame TEXT into one. No frame_bbox or speech_bboxes are merged.
    """
    speech_text = ''

    for i_f in i_frames:
        for st in i_f.untranslated_speech_text:
            speech_text += st

    united_frame = FrameInput(frame_bbox=None, speech_bboxes=None)

    if context_input is not None and len(context_input) > 0:
        split_context = ' <SEP> '.join(context_input).strip()
        full_input = f'{split_context} <SEP> {speech_text}'
    else:
        full_input = speech_text

    united_frame.add_untranslated_speech_text(full_input)
    return united_frame
