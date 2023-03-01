from gandy.image_redrawing.base_image_redraw import BaseImageRedraw
from gandy.utils.frame_input import FrameInput

class AMGConvertApp(BaseImageRedraw):
    def __init__(self):
        super().__init__()

    def process(self, image, i_frame: FrameInput):

        amg_data = {
            'image': image, # Converted to base64 in task1_route.
            'annotations': [],
        }

        s_bboxes = i_frame.speech_bboxes
        texts = i_frame.translated_sentences

        for i, s_bb in enumerate(s_bboxes):
            text = texts[i]

            x1 = s_bb[0]
            y1 = s_bb[1]
            x2 = s_bb[2]
            y2 = s_bb[3]

            amg_data['annotations'].append({
                'text': text,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            })

        return amg_data
