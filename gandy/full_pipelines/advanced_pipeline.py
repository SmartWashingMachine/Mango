
from gandy.full_pipelines.base_pipeline import BasePipeline, DefaultSpellCorrectionApp, SwitchApp
from gandy.image_cleaning.no_image_clean import NoImageCleanApp
from gandy.image_cleaning.simple_image_clean import SimpleImageCleanApp
from gandy.image_cleaning.telea_image_clean import TeleaImageCleanApp
from gandy.image_cleaning.tnet_image_clean import TNetImageClean
from gandy.image_cleaning.tnet_edge_image_clean import TNetEdgeImageClean
from gandy.image_cleaning.blur_image_clean import BlurImageCleanApp
from gandy.image_cleaning.blur_and_mask_image_clean import BlurMaskImageCleanApp
from gandy.image_redrawing.amg_convert import AMGConvertApp
from gandy.image_redrawing.image_redraw_v2 import ImageRedrawV2App
from gandy.image_redrawing.neighbor_redraw import NeighborRedrawApp
from gandy.image_redrawing.image_redraw_global import ImageRedrawGlobalApp
from gandy.text_detection.detrg_image_detection import DETRGBigImageDetectionApp, DETRVnImageDetectionApp
from gandy.text_detection.rcnn_image_detection import ResNetImageDetectionApp
from gandy.text_recognition.tr_recognition import TrOCRTextRecognitionApp
from gandy.translation.seq2seq_translation import Seq2SeqTranslationApp, Seq2SeqBigTranslationApp
from gandy.translation.kretrieval_translation import KRetrievalTranslationApp
from gandy.translation.graves_translation import GravesTranslationApp
from gandy.spell_correction.doc_repair_spell_correction import DocRepairApp

class AdvancedPipeline(BasePipeline):
    def __init__(self):
        super().__init__(
            text_detection_app=SwitchApp(
                apps=[
                    DETRGBigImageDetectionApp(),
                    ResNetImageDetectionApp(),
                    DETRVnImageDetectionApp(),
                ],
                app_names=[
                    'detrg',
                    'resnet',
                    'detr_vn',
                ],
            ),
            text_recognition_app=SwitchApp(
                apps=[
                    TrOCRTextRecognitionApp(has_proj=False), # TODO: Make others not use proj.
                    TrOCRTextRecognitionApp(model_sub_path='_ko/'),
                    TrOCRTextRecognitionApp(model_sub_path='_zh/'),
                ],
                app_names=[
                    'trocr',
                    'k_trocr',
                    'zh_trocr',
                ],
            ),
            translation_app=SwitchApp(
                apps=[
                    Seq2SeqTranslationApp(),
                    KRetrievalTranslationApp(),
                    GravesTranslationApp(),
                    Seq2SeqTranslationApp(model_sub_path='_zh/'),
                    Seq2SeqTranslationApp(model_sub_path='_ko/'),
                    Seq2SeqBigTranslationApp(model_sub_path='_jbig/'),
                ],
                app_names=[
                    'j_base',
                    'j_kaug',
                    'j_kaug_gaug',
                    'zh_base',
                    'ko_base',
                    'j_big',
                ],
            ),
            spell_correction_app=SwitchApp(
                apps=[
                    DefaultSpellCorrectionApp(),
                    DocRepairApp(),
                ],
                app_names=[
                    'default',
                    'docrepair',
                ],
            ),
            image_cleaning_app=SwitchApp(
                apps=[
                    NoImageCleanApp(),
                    SimpleImageCleanApp(),
                    TeleaImageCleanApp(),
                    TNetImageClean(),
                    TNetEdgeImageClean(),
                    BlurImageCleanApp(),
                    BlurMaskImageCleanApp(),
                ],
                app_names=[
                    'none',
                    'simple',
                    'telea',
                    'smart_telea',
                    'edge_connect',
                    'blur',
                    'blur_mask',
                ],
                # default_idx=1,
            ),
            image_redrawing_app=SwitchApp(
                apps=[
                    AMGConvertApp(),
                    NeighborRedrawApp(),
                    ImageRedrawV2App(),
                    ImageRedrawGlobalApp(),
                ],
                app_names=[
                    'amg_convert',
                    'neighbor',
                    'simple',
                    'global',
                ],
                # default_idx=-1
            ),
        )

        # For Mobile App (Randy)
        self.cur_mode = 'low'

    """ For Mobile App (Randy) """
    def low_mode(self):
        self.text_recognition_app.select_app('tesseract')
        self.translation_app.concat_mode = '2plus2'

        self.cur_mode = 'low'
        return self.cur_mode

    def medium_mode(self):
        self.text_recognition_app.select_app('trocr')
        self.translation_app.concat_mode = '2plus2'

        self.cur_mode = 'medium'
        return self.cur_mode

    def cycle_mode(self):
        if self.cur_mode == 'low':
            return self.medium_mode()
        elif self.cur_mode == 'medium':
            return self.low_mode()

    """ For Desktop App (Mango) """
    def switch_cleaning_app(self, app_name):
        self.image_cleaning_app.select_app(app_name)

    def switch_redrawing_app(self, app_name):
        self.image_redrawing_app.select_app(app_name)

    def switch_translation_app(self, app_name):
        self.translation_app.select_app(app_name)

    def switch_text_recognition_app(self, app_name):
        self.text_recognition_app.select_app(app_name)

    def switch_use_cuda(self, value):
        self.text_detection_app.set_each_app('use_cuda', value)
        self.text_recognition_app.set_each_app('use_cuda', value)
        self.translation_app.set_each_app('use_cuda', value)
        self.image_cleaning_app.set_each_app('use_cuda', value)
        self.image_redrawing_app.set_each_app('use_cuda', value)
        self.spell_correction_app.set_each_app('use_cuda', value)
