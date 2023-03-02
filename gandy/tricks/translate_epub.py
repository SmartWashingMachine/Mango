from ebooklib import epub
from bs4 import BeautifulSoup
from gandy.utils.clean_text import clean_text
import logging
import regex as re
import os
import uuid
from flask_socketio import SocketIO
from gandy.utils.frame_input import p_transformer_join
from gandy.full_pipelines.base_pipeline import BasePipeline
from gandy.utils.context_state import ContextState

logger = logging.getLogger('Gandy')

# See: https://docs.python.org/3/library/os.path.html#os.path.expanduser
save_folder_path = os.path.expanduser('~/Documents/Mango/books')

def make_folder(folder_path: str):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)

class DataCleaner():
    def __init__(self):
        pass

    @classmethod
    def replace_many(self, sentences):
        output_sentences = []

        for sentence in sentences:
            new_sentence = clean_text(sentence)
            output_sentences.append(new_sentence)

        return output_sentences

    @classmethod
    def strip_html(self, sentences):
        def _rem_newl(l):
            return BeautifulSoup(l, features="html.parser").get_text()

        return [_rem_newl(l) for l in sentences]

def emit_progress(socketio: SocketIO, j, sentences_len: int):
    data = {
        'progressFrac': max(j / max(sentences_len, 1), 0.01),
        'sentsTotal': sentences_len,
        'sentsDone': j,
    }
    socketio.emit('progress_epub', data, include_self=True)

sent_regex = re.compile(r'([.?!])([a-zA-Z0-9_])')

def translate_one_sentence(t: str, tgt_context_memory, app_pipeline: BasePipeline, context_state: ContextState):
    t_o: str = DataCleaner.replace_many(DataCleaner.strip_html([t]))[0].strip()

    # Create input.
    t = p_transformer_join(context_state.prev_source_text_list + [t_o])
    # Add current sentence to contextual inputs for future sentences.
    context_state.update_source_list(t_o, app_pipeline.translation_app.get_sel_app().max_context)

    # If tgt_context_memory is -1, we assume that means that the user wants to use the prior contextual outputs as memory.
    if tgt_context_memory == '-1' and len(context_state.prev_target_text_list) > 0:
        tgt_context_memory_to_use = p_transformer_join(context_state.prev_target_text_list + [' '])
    else:
        tgt_context_memory_to_use = None # Nothing in memory YET, or it's simply disabled.

    translated_text, _attentions, _source_tokens, _target_tokens = app_pipeline.process_task2(
        t, translation_force_words=None, socketio=None, tgt_context_memory=tgt_context_memory_to_use,
        output_attentions=False,
    )
    translated_text = translated_text[0] # Batch of 1. Only get the sentence string itself.

    # Because EPUBs are weird, sometimes the output text will have multiple sentences with poor spacing. Quick hack.
    translated_text = re.sub(sent_regex, r'\1 \2', translated_text)

    # Add current sentence to contextual outputs for future sentences.
    context_state.update_target_list(translated_text, app_pipeline.translation_app.get_sel_app().max_context)

    return translated_text

def translate_epub(file_path: str, app_pipeline: BasePipeline, checkpoint_every_pages = 0, socketio: SocketIO = None, tgt_context_memory = None):
    e_book = epub.read_epub(file_path)

    write_book_id = str(uuid.uuid4())
    book_folder_path = f'{save_folder_path}/{write_book_id}'
    book_checkpoint_folder_path = f'{save_folder_path}/{write_book_id}/checkpoints'
    make_folder(book_folder_path)
    make_folder(book_checkpoint_folder_path)

    context_state = ContextState()

    # Progress is based on # of sentences - not pages.
    # pages_len = len(list(e_book.get_items()))
    sentences_len = 0

    i = 0 # Every page, save a checkpoint.
    j = 2 # Every few translations (and 1 initially), ping the progress.

    min_per_doc = 3

    # First get sentences_len for progress updates.
    for doc in e_book.get_items():
        try:
            content = doc.content.decode("utf-8")

            soup = BeautifulSoup(content, features="html.parser")
            replacement_count = 0

            # Find text in <p> elements. If too little was found, try in <div> elements instead.
            p_results = soup.find_all('p')
            for p in p_results:
                p_text = p.get_text() # Strip HTML
                p_text = p_text.strip()

                if len(p_text) > 0:
                    replacement_count += 1

            if replacement_count < min_per_doc:
                replacement_count = 0

                # If little to no text was extracted, we will assume that the EPUB stores the text in <div> elements instead of <p>.
                div_results = soup.find_all('div')

                for d in div_results:
                    d_text = d.get_text() # Strip HTML
                    d_text = d_text.strip()

                    if len(d_text) > 0:
                        replacement_count += 1

            sentences_len += replacement_count
        except UnicodeError:
            pass

    # Now we actually translate the pages.
    for doc in e_book.get_items():
        # doc.content returns a bytes object. If we simply stringify it, we get... an ugly mess.
        # Instead, we want to utf-8 decode it. This gives us a nice mostly-probably-okay string representation.
        try:
            content = doc.content.decode("utf-8")

            soup = BeautifulSoup(content, features="html.parser")

            # DEV
            # with open(f"output{i}.html", "w", encoding='utf-8') as file:
                # file.write(str(soup))

            replacement_count = 0

            # Find text in <p> elements. If too little was found, try in <div> elements instead.

            p_results = soup.find_all('p')
            for p in p_results:
                p_text = p.get_text() # Strip HTML
                p_text = p_text.strip()

                if len(p_text) > 0:
                    # We assume all the text in a HTML paragraph element constitutes a sentence.
                    new_text = translate_one_sentence(p_text, tgt_context_memory, app_pipeline, context_state)
                    p.string = new_text
                    replacement_count += 1

                    j += 1

                    if j % 3 == 0:
                        # Need calls every now and then so the websocket doesn't lose "interest".

                        emit_progress(socketio, j, sentences_len)
                        socketio.sleep()

            if replacement_count < min_per_doc:
                # If little to no text was extracted, we will assume that the EPUB stores the text in <div> elements instead of <p>.
                div_results = soup.find_all('div')

                for d in div_results:
                    d_text = d.get_text() # Strip HTML
                    d_text = d_text.strip()

                    if len(d_text) > 0:
                        new_text = translate_one_sentence(d_text, tgt_context_memory, app_pipeline, context_state)
                        d.string = new_text
                        replacement_count += 1

                        j += 1

                        if j % 3 == 0:
                            # Need calls every now and then so the websocket doesn't lose "interest".

                            emit_progress(socketio, j, sentences_len)
                            socketio.sleep()

            if replacement_count > 0:
                # If some text was translated, then use the modified page as the new page.
                # Content must be re-encoded to utf-8.
                doc.content = str(soup).encode('utf-8')
        except UnicodeDecodeError:
            # Some pages have no text. They can be ignored and left as is.
            pass

        i += 1

        if checkpoint_every_pages != 0 and i % checkpoint_every_pages == 0:
            write_path = f'{book_checkpoint_folder_path}/checkpoint_{i}.epub'
            logger.debug(f'Saving book checkpoint in: {write_path}')
            epub.write_epub(write_path, e_book)

        emit_progress(socketio, j, sentences_len)
        socketio.sleep()

    epub.write_epub(f'{book_folder_path}/new_book.epub', e_book)
