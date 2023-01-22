from typing import List, Dict

def replace_many(s: str, terms: List[Dict], on_side: str):
    for t in terms:
        if t['onSide'] == on_side:
            s = s.replace(t['original'], t['replacement'])
    return s

# on_side == "source" || "target"
def replace_terms(sentences: List[str], terms: List[Dict], on_side: str):
    new_texts = [replace_many(s, terms, on_side) for s in sentences]
    return new_texts
