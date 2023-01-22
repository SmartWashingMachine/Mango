from gandy.utils.knn_utils.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    LogitsProcessor,
    _calc_banned_ngram_tokens,
)
import numpy as np

# This is a MODIFIED variant that does NOT ban some tokens, e.g: <SEP>.
# This was important for older model variants, but not anymore.
class CustomNoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).
    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def replace_input_ids(self, input_ids: np.ndarray):
        # Let me explain WHY we allow these tokens to be repeated:
        # 60716 = <SEP> token. Very important, as otherwise the MT model may stop translating after a few contextual sentences.
        # 12 = Apostrophe. Very important, as otherwise the model may fail to complete words like "shouldn't". (instead giving "shouldn")
        # 14 = "\u2581\". Pretty important, as otherwise the model may generate empty sequences for non contextual sentences. Needs further investigation.
        # 15 = "?" No empirical evidence, but I think some sentences will fail to complete properly without it.
        # 17 = "s". Pretty important, as otherwise the model can't complete words like "she's".
        # 52 = "t". Very important, as otherwise the model may fail to complete words like "shouldn't". (Instead giving "shouldn'")
        # 195 = "d" No empircal evidence. Follows assumptions of token 17.
        # 214 = "ll" No empirical evidence. Follows assumptions of token 17.
        # 111 = "m" No empirical evidence. Follows assumptions of token 17.
        # 853 = "I" No empirical evidence. Follows assumptions of token 17.
        # 2 = ".". No empirical evidence. Follows assumptions of token 15.
        # 3 = ",". No empirical evidence. Follows assumptions of token 15.

        # This list is surely not complete, if any other poor sucker can help me complete this list, please do so!
        good_token_ids = [60716, 12, 14, 15, 17, 52, 195, 214, 111, 411, 853, 2, 3]

        mask = None
        for gi in good_token_ids:
            new_mask = input_ids == gi

            if mask is None:
                mask = new_mask
            else:
                mask = (mask | new_mask)

        # fill good input ids with some dumb token "\u2581\u611f".
        
        # Old: new_inp = input_ids.masked_fill(mask, 1988)
        # See: https://github.com/google/jax/discussions/9363
        new_inp = np.where(mask, 1988, input_ids)
        return new_inp

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]

        # MODIFICATION START
        # input_ids = self.replace_input_ids(input_ids)
        # MODIFICATION END

        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores

# This is a different modified variant that only disallows ngrams from the current sentence (any tokens after the last <SEP>).
# Experimental.
class SepNoRepeatNGramLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces no repetition of n-grams. See
    [Fairseq](https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345).
    Args:
        ngram_size (`int`):
            All ngrams of size `ngram_size` can only occur once.
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def replace_input_ids(self, input_ids: np.ndarray):
        sep_exact_mask = input_ids == 60716 # Positions of <SEP> tokens only.
        # Old: sep_mask = torch.cumsum(sep_exact_mask, dim=-1) # ? Positions of <SEP> and tokens after.
        sep_mask = np.cumsum(sep_exact_mask, axis=-1)

        # Old: max, _ = sep_mask.max(dim=-1, keepdim=True)
        max = np.max(sep_mask, dim=-1, keepdims=True)
        cur_mask = sep_mask == max # Positions of the current <SEP> and it's sentence.

        full_mask = cur_mask & (~sep_exact_mask) # Position of tokens after the current <SEP> only.

        # fill good input ids with some dumb token "\u2581\u611f".
        # Old: new_inp = input_ids.masked_fill(~full_mask, 1988)
        new_inp = np.where(~full_mask, 1988, input_ids)

        """ TEST CODE SNIPPET:
            import torch

            input_ids = torch.tensor([
            [1, 3, 5, 1, 4],
            [2, 1, 14, 0, 3],
            ])


            print(new_inp)
            sep_exact_mask = input_ids == 1 # Positions of <SEP> tokens only.
            sep_mask = torch.cumsum(sep_exact_mask, dim=-1) # ? Positions of <SEP> and tokens after.
            max, _ = sep_mask.max(dim=-1, keepdim=True) # Positions of the current <SEP> and it's sentence.
            print(max.shape)
            print(sep_mask.shape)
            cur_mask = sep_mask == max
            full_mask = cur_mask & (~sep_exact_mask) # Position of tokens after the current <SEP> only.

            new_inp = input_ids.masked_fill(~full_mask, 1988)
        """

        return new_inp

    def __call__(self, input_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        num_batch_hypotheses = scores.shape[0]
        cur_len = input_ids.shape[-1]

        # MODIFICATION START
        input_ids = self.replace_input_ids(input_ids)
        # MODIFICATION END

        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

        return scores

def _get_logits_processor(
    self,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    encoder_no_repeat_ngram_size: int,
    input_ids_seq_length: int,
    encoder_input_ids: np.ndarray,
    bad_words_ids,
    min_length: int,
    max_length: int,
    eos_token_id: int,
    forced_bos_token_id: int,
    forced_eos_token_id: int,
    prefix_allowed_tokens_fn,
    num_beams: int,
    num_beam_groups: int,
    diversity_penalty: float,
    remove_invalid_values: bool,
    exponential_decay_length_penalty,
    logits_processor,
    renormalize_logits,
    suppress_tokens = None,
    begin_suppress_tokens = None,
    forced_decoder_ids = None,
):
    """
    This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsProcessor`]
    instances used to modify the scores of the language model head.
    """
    processors = LogitsProcessorList()

    # init warp parameters
    repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
    )
    encoder_no_repeat_ngram_size = (
        encoder_no_repeat_ngram_size
        if encoder_no_repeat_ngram_size is not None
        else self.config.encoder_no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    diversity_penalty = diversity_penalty if diversity_penalty is not None else self.config.diversity_penalty
    forced_bos_token_id = (
        forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
    )
    forced_eos_token_id = (
        forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
    )
    remove_invalid_values = (
        remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
    )
    exponential_decay_length_penalty = (
        exponential_decay_length_penalty
        if exponential_decay_length_penalty is not None
        else self.config.exponential_decay_length_penalty
    )

    # MODIFIED. For legacy support.
    if not hasattr(self.config, 'suppress_tokens'):
        suppress_tokens = None
    else:
        suppress_tokens = suppress_tokens if suppress_tokens is not None else self.config.suppress_tokens

    # MODIFIED. For legacy support.
    if not hasattr(self.config, 'begin_suppress_tokens'):
        begin_suppress_tokens = None
    else:
        begin_suppress_tokens = (
            begin_suppress_tokens if begin_suppress_tokens is not None else self.config.begin_suppress_tokens
        )

    if forced_decoder_ids is None and hasattr(self.config, "forced_decoder_ids"):
        forced_decoder_ids = self.config.forced_decoder_ids
    # instantiate processors list

    # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
    # all samplers can be found in `generation_utils_samplers.py`
    if diversity_penalty is not None and diversity_penalty > 0.0:
        processors.append(
            HammingDiversityLogitsProcessor(
                diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
            )
        )
    if repetition_penalty is not None and repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
        processors.append(CustomNoRepeatNGramLogitsProcessor(no_repeat_ngram_size)) # MODIFIED LINE.
        # Needs work ->> processors.append(SepNoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
    if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
        if self.config.is_encoder_decoder:
            processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
        else:
            raise ValueError(
                "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
            )
    if bad_words_ids is not None:
        processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
    if min_length is not None and eos_token_id is not None and min_length > 0:
        processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
    if prefix_allowed_tokens_fn is not None:
        print('Prefix constrained not supported.')
    if forced_bos_token_id is not None:
        processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
    if forced_eos_token_id is not None:
        processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
    if remove_invalid_values is True:
        print('Inf nan not supporetd.')
    if exponential_decay_length_penalty is not None:
        print('Exp decay not supported.')

    processors = self._merge_criteria_processor_list(processors, logits_processor)
    # `LogitNormalization` should always be the last logit processor, when present
    if renormalize_logits is True:
        processors.append(LogitNormalization())
    return processors

def monkey_patch_model(model):
    def monk(*args, **kwargs):
        return _get_logits_processor(model, *args, **kwargs)

    model._get_logits_processor = monk
    return model
