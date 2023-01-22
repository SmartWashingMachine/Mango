from flask import request
import logging

from gandy.app import app, translate_pipeline, socketio
from gandy.task3_routes import context_state

logger = logging.getLogger('Gandy')

# Task2 - translate text into text (from the OCR box or the text field input).

def translate_task2_background_job(text, force_words, box_id = None):
    output = {
        'text': '',
        'sourceText': text,
        'boxId': box_id,
    }

    if box_id is not None:
        logger.debug('Using box context.')
        # Hacky for now. For OCR boxes. TODO
        li = context_state.prev_source_text_list + [text]
        text = ' <SEP> '.join(li).strip()

        context_state.update_list(text)

        socketio.emit('begin_translating_task2', {}, include_self=True)
        socketio.sleep()

    try:
        socketio.emit('progress_task2', 0.05, include_self=True)

        new_text = translate_pipeline.process_task2(text, translation_force_words=force_words, socketio=socketio)
        output['text'] = new_text

        socketio.emit('done_translating_task2', output, include_self=True)
    except Exception:
        logger.exception('An error happened while translating task2 as a background job:')

        socketio.emit('done_translating_task2', {}, include_self=True)

@app.route('/processtask2', methods=['POST'])
def process_task2_route():
    data = request.json
    text = data['text']
    box_id = data['boxId'] if 'boxId' in data else None
    force_words = data['required_words'] if 'required_words' in data else None

    socketio.start_background_task(translate_task2_background_job, text, force_words, box_id)

    return { 'processing': True }, 202
