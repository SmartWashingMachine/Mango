from flask import request
import logging

from gandy.app import app, translate_pipeline, socketio

logger = logging.getLogger('Gandy')

def process_paraphrase_background_job(text):
    try:
        paraphrased = translate_pipeline.paraphrase_app.begin_process(text)
        socketio.emit('done_paraphrasing', { 'text': paraphrased, }, include_self=True)
    except Exception as e:
        logging.exception('An error happened while paraphrasing as a background job:')

        socketio.emit('done_paraphrasing', {}, include_self=True)

@app.route('/paraphrase', methods=['POST'])
def process_paraphrase_route():
    """
    Paraphrase some text.
    """
    data = request.json
    text = data['text']

    socketio.start_background_task(process_paraphrase_background_job, text)

    return { 'processing': True }, 202