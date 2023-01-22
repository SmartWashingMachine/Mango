from flask import request
from io import BytesIO
import base64
import logging

from gandy.tricks.translate_epub import translate_epub
from gandy.app import app, translate_pipeline, socketio

logger = logging.getLogger('Gandy')

def process_task2_book_background_job(file):
    try:
        translate_epub(file, translate_pipeline, checkpoint_every_pages=1, socketio=socketio)

        socketio.emit('done_translating_epub', {})
    except Exception:
        logging.exception('An error happened while translating a book as a background job:')

        socketio.emit('done_translating_epub', {})

@app.route('/processbookb64', methods=['POST'])
def process_book_route():
    """
    Process an EPUB file.
    """
    if 'file' not in request.files or request.files['file'].filename == '':
        logger.debug('No book was sent.')

        return {}, 404

    file = request.files['file']

    socketio.start_background_task(process_task2_book_background_job, file)

    return { 'processing': True }, 202
