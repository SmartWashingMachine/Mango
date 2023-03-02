from typing import List

class ContextState():
    def __init__(self):
        self.prev_source_text_list = []
        self.prev_target_text_list = []

    def update_list(self, text_list: List[str], text: str, max_context: int):
        if max_context is None:
            raise RuntimeError('max_context must be given.')

        text_list = text_list + [text]

        # max_context from translation app is the count for contextual sentences + current sentence. (so 1 context and 1 current would be 2).
        # Whereas ContextState is only concerned with the context sentence count, so we subtract max_context by 1.
        if len(text_list) > (max_context - 1):
            text_list = text_list[1:]

        return text_list

    def update_source_list(self, text: str, max_context: int):
        self.prev_source_text_list = self.update_list(self.prev_source_text_list, text, max_context)

    def update_target_list(self, text: str, max_context: int):
        self.prev_target_text_list = self.update_list(self.prev_target_text_list, text, max_context)