# Most of the code here is from huggingface.

import logging
import os
from shutil import copyfile
from typing import List, Optional

import sentencepiece as spm

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.models.xlnet.tokenization_xlnet import SPIECE_UNDERLINE

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "sentencepiece.bpe.model"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "camembert-base": "https://s3.amazonaws.com/models.huggingface.co/bert/camembert-base-sentencepiece.bpe.model",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "camembert-base": None,
}

class MtBigTokenizer(PreTrainedTokenizer):
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        additional_special_tokens=["<s>NOTUSED", "</s>NOTUSED"],
        **kwargs
    ):
        if 'max_len' in kwargs:
            print('Popping max_len off tokenizer kwargs.')
            kwargs.pop('max_len')
        super().__init__(
            max_len=512,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            vocab_file=vocab_file + '_dict.txt',
            additional_special_tokens=additional_special_tokens,
            **kwargs,
        )
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(str(vocab_file))
        self.vocab_file = vocab_file
        self.dict_file = vocab_file + '_dict.txt'

        print('Tokenizer SPM:')
        print(str(vocab_file))
        print('Tokenizer Dict:')
        print(self.dict_file)

        print('Vocab file:')
        print(vocab_file)

        # HACK: These tokens were added by fairseq but don't seem to be actually used when duplicated in the actual
        # sentencepiece vocabulary (this is the case for <s> and </s>
        self.fairseq_tokens_to_ids = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        self.fairseq_offset = len(self.fairseq_tokens_to_ids)
        self.piece2id = {}
        self.id2piece = []
        id = 0
        with open(self.dict_file, 'r', encoding='utf-8') as dictfile:
            for line in dictfile:
                piece = line.split()[0]
                self.piece2id[piece] = id
                self.id2piece.append(piece)
                id += 1
        self.fairseq_tokens_to_ids["<mask>"] = len(self.piece2id) + len(self.fairseq_tokens_to_ids)
        self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

    @property
    def vocab_size(self):
        return len(self.fairseq_tokens_to_ids) + len(self.piece2id)

    def _tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        """ Converts a token (str) in an id using the vocab. """
        if token in self.fairseq_tokens_to_ids:
            return self.fairseq_tokens_to_ids[token]
        elif token not in self.piece2id:
            # Convert sentence piece unk token to fairseq unk token index
            return self.unk_token_id
        return self.fairseq_offset + self.piece2id[token]

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        if index in self.fairseq_ids_to_tokens:
            return self.fairseq_ids_to_tokens[index]
        return self.id2piece[index - self.fairseq_offset]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        try:
            import sentencepiece as spm
        except ImportError:
            logger.warning(
                "You need to install SentencePiece to use AlbertTokenizer: https://github.com/google/sentencepiece"
                "pip install sentencepiece"
            )
            raise
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(self.vocab_file)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (strings for sub-words) in a single string."""
        out_string = "".join(tokens).replace(SPIECE_UNDERLINE, " ").strip()
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix = None):
        """
        Save the sentencepiece vocabulary (copy original file) and special tokens file to a directory.
        Args:
            save_directory (:obj:`str`):
                The directory in which to save the vocabulary.
        Returns:
            :obj:`Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error("Vocabulary path ({}) should be a directory".format(save_directory))
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"])
        out_dict_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + '_dict.txt')

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        if os.path.abspath(self.dict_file) != os.path.abspath(out_dict_file):
            copyfile(self.dict_file, out_dict_file)

        return (out_vocab_file,)

    # From Marian.
    def num_special_tokens_to_add(self, *args, **kwargs):
        """Just EOS"""
        return 1

    def _special_token_mask(self, seq):
        all_special_ids = set(self.all_special_ids)  # call it once instead of inside list comp
        all_special_ids.remove(self.unk_token_id)  # <unk> is only sometimes special
        return [1 if x in all_special_ids else 0 for x in seq]

    def get_special_tokens_mask(
        self, token_ids_0: List, token_ids_1: Optional[List] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """Get list where entries are [1] if a token is [eos] or [pad] else 0."""
        if already_has_special_tokens:
            return self._special_token_mask(token_ids_0)
        elif token_ids_1 is None:
            return self._special_token_mask(token_ids_0) + [1]
        else:
            return self._special_token_mask(token_ids_0 + token_ids_1) + [1]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None) -> List[int]:
        """Build model inputs from a sequence by appending eos_token_id."""
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        # We don't expect to process pairs, but leave the pair logic for API consistency
        return token_ids_0 + token_ids_1 + [self.eos_token_id]
