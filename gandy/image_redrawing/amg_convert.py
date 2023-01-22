from gandy.image_redrawing.base_image_redraw import BaseImageRedraw

class AMGConvertApp(BaseImageRedraw):
    def __init__(self):
        super().__init__()

    def process(self, image, i_frames, texts, debug = False, font_size=6, adaptative_font_size = True):

        amg_data = {
            'image': image, # Converted to base64 in task1_route.
            'annotations': [],
        }

        i = 0
        for frame in i_frames:
            s_bboxes = frame.speech_bboxes

            for s_bb in s_bboxes:
                if i >= len(texts):
                    print('WARNING: repaint_image has more speech bubbles than texts. Some speech bubbles were left untouched.')
                    break

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

                i += 1

        return amg_data
