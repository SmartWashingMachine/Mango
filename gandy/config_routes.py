from flask import request
from gandy.app import app, translate_pipeline

@app.route('/changecleaning', methods=['POST'])
def change_cleaning_route():
    data = request.json
    new_mode = data['mode']

    new_mode = translate_pipeline.switch_cleaning_app(new_mode)

    return {}, 200

@app.route('/changeredrawing', methods=['POST'])
def change_redrawing_route():
    data = request.json
    new_mode = data['mode']

    new_mode = translate_pipeline.switch_redrawing_app(new_mode)

    return {}, 200

@app.route('/switchmodels', methods=['POST'])
def change_multiple_models_route():
    data = request.json

    translate_pipeline.translation_app.select_app(data['translationModelName'])
    translate_pipeline.text_recognition_app.select_app(data['textRecognitionModelName'])
    translate_pipeline.text_detection_app.select_app(data['textDetectionModelName'])
    translate_pipeline.spell_correction_app.select_app(data['spellCorrectionModelName'])

    translate_pipeline.switch_use_cuda(data['enableCuda'])

    if data['contextAmount'] == 'three':
        c_amount = 4
    elif data['contextAmount'] == 'two':
        # Context (2) + Current Sentence (1) == 3 total
        c_amount = 3
    elif data['contextAmount'] == 'one':
        c_amount = 2
    else:
        c_amount = 1

    translate_pipeline.translation_app.for_each_app('set_max_context', c_amount)

    max_length_a = float(data['maxLengthA'])
    if max_length_a > 0:
        translate_pipeline.translation_app.for_each_app('set_max_length_a', max_length_a)

    translate_pipeline.terms = data['terms']

    return {}, 200

@app.route('/changetextrecognition', methods=['POST'])
def change_tr_route():
    data = request.json
    new_mode = data['mode']

    new_mode = translate_pipeline.switch_text_recognition_app(new_mode)

    return {}, 200
