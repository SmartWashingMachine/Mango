from datetime import datetime
from gandy.spell_correction.base_spell_correction import BaseSpellCorrection
from gandy.utils.frame_input import link_speechbubble_to_frame, unite_i_frames
from gandy.text_detection.base_image_detection import BaseImageDetection
from gandy.utils.split_text import split_text
import numpy as np
import logging
import re
from gandy.utils.replace_terms import replace_terms
from gandy.utils.get_sep_regex import get_last_sentence

logger = logging.getLogger('Gandy')

def get_seconds(date1, date2):
    return (date2 - date1).total_seconds()

class DefaultFrameDetectionApp(BaseImageDetection):
    def __init__(self):
        super().__init__()

    def process(self, image):
        im_width, im_height = image.size
        frame_bboxes = np.array([[0, 0, im_width, im_height]])

        return frame_bboxes

class DefaultSpellCorrectionApp(BaseSpellCorrection):
    def __init__(self):
        super().__init__()

    def process(self, translation_input, translation_output):
        # Remove contextual info.
        return [get_last_sentence(o) for o in translation_output]

class BasePipeline():
    def __init__(
        self,
        frame_detection_app,
        text_detection_app,
        text_recognition_app,
        translation_app,
        spell_correction_app,
        image_cleaning_app,
        image_redrawing_app,
    ):
        if frame_detection_app is None:
            # If no frame detection app is given, then a default function is given which provides the bbox of the entire image.
            self.frame_detection_app = DefaultFrameDetectionApp()
        else:
            self.frame_detection_app = frame_detection_app

        if text_detection_app is None:
            raise RuntimeError('text_detection_app must be given.')
        else:
            self.text_detection_app = text_detection_app

        if text_recognition_app is None:
            raise RuntimeError('text_recognition_app must be given.')
        else:
            self.text_recognition_app = text_recognition_app

        if translation_app is None:
            raise RuntimeError('translation_app must be given.')
        else:
            self.translation_app = translation_app

        if spell_correction_app is None:
            # Identity function.
            self.spell_correction_app = DefaultSpellCorrectionApp()
        else:
            self.spell_correction_app = spell_correction_app

        if image_cleaning_app is None:
            raise RuntimeError('image_cleaning_app must be given.')
        else:
            self.image_cleaning_app = image_cleaning_app

        if image_redrawing_app is None:
            raise RuntimeError('image_redrawing_app must be given.')
        else:
            self.image_redrawing_app = image_redrawing_app

        self.n_apps = 7
        self._apps_done = 0

        # This is a user defined list of terms to replace/filter.
        self.terms = []

    def pre_app(self, app_name):
        self.last_app_name = app_name
        self.start = datetime.now()

    def get_progress(self):
        min_value = 0.05

        value = max(round(self._apps_done / self.n_apps, 2), min_value)
        value = min(1, value)

        return value

    def post_app(self, socketio, task_name):
        end = datetime.now()
        logger.debug(f'{self.last_app_name} took {get_seconds(self.start, end)}s')

        self._apps_done += 1
        if socketio is not None:
            socketio.emit(f'progress_{task_name}', self.get_progress())
            socketio.sleep()

    def in_app(self, socketio, task_name):
        self._apps_done = 0
        if socketio is not None:
            socketio.emit(f'progress_{task_name}', self.get_progress())
            socketio.sleep()

    def process_simulate(self, image, translation_force_words = None, socketio=None):
        self.in_app(socketio, 'simulate')

        self.pre_app('DEBUG')
        socketio.sleep(1)
        self.post_app(socketio, 'simulate')

        self.pre_app('DEBUG')
        socketio.sleep(1)
        self.post_app(socketio, 'simulate')

        self.pre_app('DEBUG')
        socketio.sleep(3)
        self.post_app(socketio, 'simulate')

        self.pre_app('DEBUG')
        socketio.sleep(2)
        self.post_app(socketio, 'simulate')

        return image

    def process_task1(self, image, translation_force_words = None, socketio = None, tgt_context_memory = None):
        self.in_app(socketio, 'task1')

        self.pre_app('frame_detection')
        frame_bboxes = self.frame_detection_app.begin_process(image)
        self.post_app(socketio, 'task1')

        self.pre_app('text_detection')
        speech_bboxes = self.text_detection_app.begin_process(image)
        self.post_app(socketio, 'task1')

        self.pre_app('linking')
        # Creates a list of FrameInputs, where each item is an object containing a single frame and all of the speech bubbles inside of that frame.
        i_frames = link_speechbubble_to_frame(speech_bboxes, frame_bboxes)
        self.post_app(socketio, 'task1')

        self.pre_app('text_recognition')
        # NOTE: This class will modify the i_frames inplace.
        i_frames = self.text_recognition_app.begin_process(image.convert('RGB'), i_frames)
        self.post_app(socketio, 'task1')

        # Modify user terms on the source side.
        for i_f in i_frames:
            i_f.replace_terms(self.terms)

        # Each translation_output will contain the contextual sentences too in the output strings.
        self.pre_app('translation')

        translation_input, translation_output, attentions, source_tokens, target_tokens = self.translation_app.begin_process(i_frames=i_frames, force_words=translation_force_words, tgt_context_memory=tgt_context_memory)
        self.post_app(socketio, 'task1')

        # But the spelling correction apps will take care of removing any contextual sentences from the final output.
        self.pre_app('spell_correction')
        translation_output = self.spell_correction_app.begin_process(translation_input, translation_output)
        self.post_app(socketio, 'task1')

        # Modify user terms on the target side.
        translation_output = replace_terms(translation_output, self.terms, on_side='target')

        self.pre_app('image_cleaning')
        rgb_image = image.convert('RGB')
        rgb_image = self.image_cleaning_app.begin_process(rgb_image, i_frames)
        self.post_app(socketio, 'task1')

        self.pre_app('image_redrawing')
        rgb_image = self.image_redrawing_app.begin_process(rgb_image, i_frames, translation_output)
        self.post_app(socketio, 'task1')

        is_amg = isinstance(rgb_image, dict) # AMG convert app returns a dict rather than an image directly.

        return rgb_image, is_amg

    def process_task2(self, text, translation_force_words = None, tgt_context_memory = None, socketio = None, output_attentions = False):
        self.in_app(socketio, 'task2')

        self.pre_app('translation')
        # There's a good chance that the user is actually inputting a huge document for this task, even though the model can only have up to 512 tokens as input...
        # In order to handle this, if large, we split the text into sentences, translate them via rotating window, and then concatenate them into one single output.
        # Currently, there are no spell correction systems here that take advantage of the source text, so translation_input does not matter.
        if len(text) > 1024:
            logger.debug('Task2 found text to be long - splitting.')

            text = split_text(text) # Now text is a list of strings, and not just a string.
            translation_output = []

            # Modify user terms on the source side.
            text = replace_terms(text, self.terms, on_side='source')

            for text_str in text:
                translation_input, translation_output_one = self.translation_app.begin_process(i_frames=None, text=text_str, force_words=translation_force_words)
                translation_output.extend(translation_output_one)

            # Then combine each output into one string.
            translation_output = [''.join(translation_output)]

            # Don't really have a good solution for reusing context memory with extremely long text yet, so just set it to None if it is not already.
            tgt_context_memory = None
        else:
            # Modify user terms on the source side.
            # Here, text is a str but replace_terms takes in a list.
            text = replace_terms([text], self.terms, on_side='source')[0]

            translation_input, translation_output, attentions, source_tokens, target_tokens = self.translation_app.begin_process(
                i_frames=None, text=text, force_words=translation_force_words, tgt_context_memory=tgt_context_memory,
                output_attentions=output_attentions
            )
        self.post_app(socketio, 'task2')

        self.pre_app('spell_correction')
        translation_output = self.spell_correction_app.begin_process(translation_input, translation_output)
        self.post_app(socketio, 'task2')

        # Modify user terms on the target side.
        translation_output = replace_terms(translation_output, self.terms, on_side='target')

        return translation_output, attentions, source_tokens, target_tokens

    def process_task3(self, image, translation_force_words = None, socketio = None, with_text_detect = False, context_input = None, tgt_context_memory = None):
        self.in_app(socketio, 'task3')

        self.pre_app('frame_detection')
        frame_bboxes = self.frame_detection_app.begin_process(image)
        self.post_app(socketio, 'task3')

        self.pre_app('text_detection')
        if with_text_detect:
            speech_bboxes = self.text_detection_app.begin_process(image)
        else:
            speech_bboxes = self.frame_detection_app.begin_process(image)
        self.post_app(socketio, 'task3')

        self.pre_app('linking')
        # Creates a list of FrameInputs, where each item is an object containing a single frame and all of the speech bubbles inside of that frame.
        i_frames = link_speechbubble_to_frame(speech_bboxes, frame_bboxes)
        self.post_app(socketio, 'task3')

        self.pre_app('text_recognition')
        # NOTE: This class will modify the i_frames inplace.
        i_frames = self.text_recognition_app.begin_process(image, i_frames)
        self.post_app(socketio, 'task3')

        # Task3 assumes that there is only one detected text item to translate, but sometimes there are multiple text items to translate. (Such as the case with DETR-VN).
        # How do we handle this? Simple! After text recognition, combine all of the recognized text (no sep except for contexts) into one unit and translate it all together.
        i_frames = [unite_i_frames(i_frames, context_input)]

        # Modify user terms on the source side.
        for i_f in i_frames:
            i_f.replace_terms(self.terms)

        self.pre_app('translation')
        translation_input, translation_output, attentions, source_tokens, target_tokens = self.translation_app.begin_process(i_frames=i_frames, force_words=translation_force_words, tgt_context_memory=tgt_context_memory)
        self.post_app(socketio, 'task3')

        self.pre_app('spell_correction')
        translation_output = self.spell_correction_app.begin_process(translation_input=translation_input, translation_output=translation_output)
        self.post_app(socketio, 'task3')

        # Modify user terms on the target side.
        translation_output = replace_terms(translation_output, self.terms, on_side='target')

        source_texts = []
        for i_f in i_frames:
            source_texts.extend(i_f.untranslated_speech_text)

        return translation_output, source_texts

class SwitchApp():
    def __init__(self, apps, app_names, default_idx = 0):
        """
        Some of our apps (the individual pipes) in a pipeline will want to be able to be switched around at runtime.

        This class expects an app name for each given app, and provides a util to easily switch between the two as needed.
        """

        if len(app_names) != len(apps):
            raise ValueError('app_names must be of same length as apps.')
        if len(apps) == 0 or len(app_names) == 0:
            raise ValueError('apps and app_names must have at least one item.')

        self.sel_idx = default_idx
        self.apps = apps
        self.app_names = app_names

    def select_app(self, app_name):
        """
        Select the app with the given name. All further process calls on this app will redirect to the newly selected app.
        """
        try:
            idx = self.app_names.index(app_name)
            self.sel_idx = idx
        except IndexError:
            logger.warning(f'No app with name {app_name} found. Ignoring the call to select new app.')

    def process(self, *args, **kwargs):
        return self.apps[self.sel_idx].begin_process(*args, **kwargs)

    def begin_process(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def get_sel_app(self):
        return self.apps[self.sel_idx]

    def for_each_app(self, func_name: str, value):
        """
        Call a method on each app. Only really used for translation app's set_max_context in the config route.
        """

        for a in self.apps:
            a_func = getattr(a, func_name)
            a_func(value)

    def set_each_app(self, var_name: str, value):
        """
        Sets a variable on each app. Only really used for setting use_cuda.
        """
        for a in self.apps:
            setattr(a, var_name, value)
