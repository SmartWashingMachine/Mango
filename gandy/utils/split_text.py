import re

end_chars = [
    '!',
    '?',
    '.',
    '。',
    '？',
    '！',
    '．',
]

# NOTE: Currently unused.
after_chars = [
    '"',
    "'",
    '”',
    '“',
    '」',
    '「',
    '【',
    '】',
]

# See: https://stackoverflow.com/questions/4998629/split-string-with-multiple-delimiters-in-python
end_regex = '|'.join('(?<={})'.format(re.escape(delim)) for delim in end_chars)

def split_text(text: str):
    """
    Split a long block of text into sentences.
    """
    sentences = re.split(end_regex, text)

    if sentences is not None and isinstance(sentences, list):
        output = []
        # Each "sentence" should have prior sentences as context i.e: a rolling window.
        # NOTE: Currently this assumes that the input has no <SEP> tokens already in it.
        
        for i in range(len(sentences)):
            te = f'{sentences[i]}'

            if (i - 1) >= 0:
                te = f'{sentences[i - 1]} <SEP> {te}'
            if (i - 2) >= 0:
                te = f'{sentences[i - 2]} <SEP> {te}'
            if (i - 3) >= 0:
                te = f'{sentences[i - 3]} <SEP> {te}'

            if len(te.strip()) > 0:
                output.append(te)

        return output
    else:
        return [text]