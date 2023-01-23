from flask import request
import logging

from gandy.app import app, translate_pipeline, socketio
from gandy.task3_routes import context_state

logger = logging.getLogger('Gandy')

# Task2 - translate text into text (from the OCR box or the text field input or e-books).

def translate_task2_background_job(text, force_words, box_id = None, tgt_context_memory = None):
    output = {
        'text': '',
        'sourceText': text,
        'boxId': box_id,
    }

    if box_id is not None:
        # This is only used for clipboard copying on the frontend with the OCR box (normal OCR box method is task3). Context is stored on the server FOR NOW.

        logger.debug('Using box context.')
        # Hacky for now. For OCR boxes. TODO
        li = context_state.prev_source_text_list + [text]
        text = ' <SEP> '.join(li).strip()

        context_state.update_source_list(text)

        socketio.emit('begin_translating_task2', {}, include_self=True)
        socketio.sleep()

        # If tgt_context_memory is -1, we assume that means that the user wants to use the prior contextual outputs as memory.
        if tgt_context_memory == '-1':
            sep_after = context_state.prev_target_text_list + [' <SEP>']
            tgt_context_memory = ' <SEP> '.join(sep_after).strip()
    try:
        socketio.emit('progress_task2', 0.05, include_self=True)

        # tgt_context_memory, if provided, should be the string consisting of the target-side translations of the contextual sentences.
        # e.g: if our input text is like "Asource <SEP1> Bsource <SEP2> Csource", tgt_context_memory should be "Atarget <SEP1> Btarget <SEP2>" (Ctarget can't be provided as that is the target we wish to predict.)
        # This can speed up decoding for long lists of text since it doesn't have to translate the contextual sentences again and again, but can affect model accuracy.
        new_text = translate_pipeline.process_task2(text, translation_force_words=force_words, socketio=socketio, tgt_context_memory=tgt_context_memory)
        output['text'] = new_text

        if box_id is not None:
            context_state.update_target_list(new_text)

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

    # Proper values = None (no memory) | String (not allowed for clipboard copying) | -1 (allowed for clipboard copying)
    tgt_context_memory = data['tgt_context_memory'] if 'tgt_context_memory' in data else None

    socketio.start_background_task(translate_task2_background_job, text, force_words, box_id, tgt_context_memory)

    return { 'processing': True }, 202
