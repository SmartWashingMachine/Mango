from flask import request
from PIL import Image
from io import BytesIO
import base64
import os
import zipfile
import logging

from gandy.app import app, translate_pipeline, socketio

logger = logging.getLogger('Gandy')

# Task1 - translate images into images.

def encode_image(new_image):
    buffer = BytesIO()
    new_image.save(buffer, format="JPEG")
    new_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return new_image_base64

def translate_task1_background_job(images, force_words, is_b64 = False, is_zip = False):
    try:
        image_streams = []
        image_names = []

        if is_zip:
            logging.info('Decoding zip.')

            zip_file = BytesIO(base64.b64decode(images))
            images = []

            z = zipfile.ZipFile(zip_file)
            names = z.namelist()

            for n in names:
                d = z.read(n)
                images.append(BytesIO(d))

        for idx, img_file in enumerate(images):
            if is_b64:
                logger.info('Decoding B64.')
                # is_b64 is only really used for Randy.
                img_file = BytesIO(base64.b64decode(img_file))

            img = Image.open(img_file)
            img.load()

            image_streams.append(img)

            try:
                if is_b64 or (img_file.filename == '' or img_file.filename is None):
                    image_names.append(f'{idx}')
                else:
                    image_names.append(img_file.filename)
            except:
                image_names.append(f'{idx}')

        for img, img_name in zip(image_streams, image_names):
            # The client really only uses progress for task1 anyways. The other progress_tasks aren't used... yet.
            socketio.emit('progress_task1', 0.05)
            socketio.sleep()

            new_image, is_amg = translate_pipeline.process_task1(img, translation_force_words=force_words, socketio=socketio)

            new_img_name = img_name

            if is_amg:
                new_image_base64 = encode_image(new_image['image'])
                annotations = new_image['annotations']

                img_name_no_ext = os.path.splitext(img_name)[0]

                new_img_name = f'{img_name_no_ext}.amg'
            else:
                new_image_base64 = encode_image(new_image)
                annotations = []

                img_name_no_ext = os.path.splitext(img_name)[0]

                new_img_name = f'{img_name_no_ext}.png'

            socketio.emit(
                'item_task1', 
                { 'image': new_image_base64, 'imageName': new_img_name, 'annotations': annotations, },
            )
            socketio.sleep()

        socketio.emit('done_translating_task1', {})
        socketio.sleep()
    except Exception:
        logger.exception('An error happened while translating task1 as a background job:')

        socketio.emit('done_translating_task1', {})
        socketio.sleep()

@app.route('/processtask1', methods=['POST'])
def process_task1_route():
    data = request.form.to_dict(flat=False)
    force_words = data['required_words'] if 'required_words' in data else None

    images = request.files.getlist('file')
    socketio.start_background_task(translate_task1_background_job, images, force_words)

    return { 'processing': True }, 202
