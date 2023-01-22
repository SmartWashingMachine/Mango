from flask import request
from io import BytesIO
import base64

from gandy.app import app, translate_pipeline, socketio
from gandy.task1_routes import translate_task1_background_job
from gandy.tricks.translate_web import open_notepad_with_texts, translate_web

# There is a personal mobile app I made called Randy for testing purposes - these routes are to help with that.

def encode_image(new_image):
    buffer = BytesIO()
    new_image.save(buffer, format="JPEG")
    new_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return new_image_base64

@app.route('/processb64', methods=['POST'])
def process_b64_route():
    data = request.json
    images_base64 = data['images']

    socketio.start_background_task(translate_task1_background_job, images_base64, None, True)
    return { 'processing': True }, 202

@app.route('/processzipb64', methods=['POST'])
def process_zip_route():
    """
    Process a zip file containing images.
    """
    data = request.json

    zip_file_base64 = data['file']

    socketio.start_background_task(translate_task1_background_job, zip_file_base64, None, False, True)

    return { 'processing': True }, 202

@app.route('/lowmode', methods=['POST'])
def set_low_mode_route():
    translate_pipeline.low_mode()
    return '', 200

@app.route('/mediummode', methods=['POST'])
def set_high_mode_route():
    translate_pipeline.medium_mode()
    return '', 200

@app.route('/cyclemode', methods=['POST'])
def set_cycle_mode_route():
    print('Cycling mode.')

    new_mode = translate_pipeline.cycle_mode()

    output = {
        'new_mode': new_mode,
    }

    return output, 200

@app.route('/processweb', methods=['POST'])
def process_web_route():
    data = request.json

    web_link = data['weblink']

    translated_texts = translate_web(web_link, translate_pipeline)
    open_notepad_with_texts(translated_texts)

    return { 'processing': True }, 202
