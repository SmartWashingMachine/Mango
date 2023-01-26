import re

sep_splitter = re.compile(r'<SEP>|<SEP1>|<SEP2>|<SEP3>')

def get_last_sentence(s: str):
    return re.split(sep_splitter, s)[-1].strip()
