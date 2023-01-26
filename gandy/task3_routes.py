from flask import request
from PIL import Image
import logging
from typing import List

from gandy.app import app, translate_pipeline, socketio

logger = logging.getLogger('Gandy')

# Task3 - translate images into text (from the OCR box).
# Context here is stored on the SERVER rather than the client.
# Why? Because we may be using textract, and this is the best way to keep a state of previous contexts if it comes to that.

class ContextState():
    def __init__(self):
        self.prev_source_text_list = []
        self.prev_target_text_list = []

    def update_list(self, text_list: List[str], text: str):
        text_list.append(text.split('<SEP>')[-1].strip())
        if len(text_list) > 3:
            text_list = text_list[1:]

    def update_source_list(self, text: str):
        self.update_list(self.prev_source_text_list, text)

    def update_target_list(self, text: str):
        self.update_list(self.prev_target_text_list, text)

context_state = ContextState()

def translate_task3_background_job(images, force_words, box_id = None, with_text_detect = True, tgt_context_memory = None):
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

            # If tgt_context_memory is -1, we assume that means that the user wants to use the prior contextual outputs as memory.
            if tgt_context_memory == '-1' and len(context_state.prev_target_text_list) > 0:
                sep_after = context_state.prev_target_text_list
                tgt_context_memory = ' <SEP> '.join(sep_after).strip() + ' <SEP> '
            elif tgt_context_memory == '-1':
                tgt_context_memory = None # Nothing in memory YET.

            new_texts, source_texts = translate_pipeline.process_task3(
                img, translation_force_words=force_words, socketio=socketio, with_text_detect=with_text_detect, context_input=context_state.prev_source_text_list,
                tgt_context_memory=tgt_context_memory,
            )

            last_source = source_texts[-1].split('<SEP>')[-1].strip()
            context_state.update_source_list(last_source)
            logger.debug(f'Task3 is updating server-side-context with item: {last_source}')

            context_state.update_target_list(new_texts[-1])

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

    # Proper values = None (no memory) | -1
    tgt_context_memory = data['tgt_context_memory'][0] if 'tgt_context_memory' in data else None

    images = request.files.getlist('file')

    socketio.start_background_task(translate_task3_background_job, images, force_words, box_id, with_text_detect, tgt_context_memory)

    return { 'processing': True }, 202
