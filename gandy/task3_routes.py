from flask import request
from PIL import Image
import logging

from gandy.app import app, translate_pipeline, socketio

logger = logging.getLogger('Gandy')

# Task3 - translate images into text (from the OCR box).
# Context here is stored on the SERVER rather than the client.
# Why? Because we may be using textract, and this is the best way to keep a state of previous contexts if it comes to that.

class ContextState():
    def __init__(self):
        self.prev_source_text_list = []

    def update_list(self, text):
        self.prev_source_text_list.append(text.split('<SEP>')[-1].strip())
        if len(self.prev_source_text_list) > 3:
            self.prev_source_text_list = self.prev_source_text_list[1:]

context_state = ContextState()

def translate_task3_background_job(images, force_words, box_id = None, with_text_detect = True):
    try:
        socketio.emit('begin_translating_task3', {}, include_self=True)

        opened_images = []

        for img_file in images:
            opened_img = Image.open(img_file)
            opened_img.load()
            opened_images.append(opened_img)

        for img in opened_images:
            logger.debug(f'Task3 is using server-side-context of: {context_state.prev_source_text_list}')
            logger.debug(f'Task3 text detect mode: {"ON" if with_text_detect else "OFF"}')

            socketio.emit('progress_task3', 0.05, include_self=True)
            new_texts, source_texts = translate_pipeline.process_task3(
                img, translation_force_words=force_words, socketio=socketio, with_text_detect=with_text_detect, context_input=context_state.prev_source_text_list,
            )

            last_source = source_texts[-1].split('<SEP>')[-1].strip()
            context_state.update_list(last_source)
            logger.debug(f'Task3 is updating server-side-context with item: {last_source}')

            socketio.emit('item_task3', { 'text': new_texts, 'boxId': box_id, 'sourceText': [last_source], }, include_self=True)

        socketio.emit('done_translating_task3', {}, include_self=True)
    except Exception:
        logger.exception('An error happened while translating task3 as a background job:')

        socketio.emit('done_translating_task3', {}, include_self=True)

@app.route('/processtask3', methods=['POST'])
def process_task3_route():
    data = request.form.to_dict(flat=False)
    box_id = data['boxId'] if 'boxId' in data else None
    force_words = data['required_words'] if 'required_words' in data else None
    text_detect = data['textDetect'] if 'textDetect' in data else 'off'

    # It's an array? Huh? TODO
    with_text_detect = text_detect[0] == 'on'

    images = request.files.getlist('file')

    socketio.start_background_task(translate_task3_background_job, images, force_words, box_id, with_text_detect)

    return { 'processing': True }, 202
