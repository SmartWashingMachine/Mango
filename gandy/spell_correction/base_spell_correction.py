
from gandy.full_pipelines.base_app import BaseApp
from typing import List

class BaseSpellCorrection(BaseApp):
    def __init__(self):
        super().__init__()

    def process(self, translation_input: List[str], texts: List[str]):
        pass
