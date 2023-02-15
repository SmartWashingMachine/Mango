from collections import deque
from scipy.special import softmax
import numpy as np

from gandy.translation.kretrieval_translation import MTRetrieval, process_outputs_cb_for_kretrieval
from gandy.translation.seq2seq_translation import Seq2SeqTranslationApp
from gandy.onnx_models.marian import MarianONNX
import logging

logger = logging.getLogger('Gandy')

def process_outputs_cb_for_graves(mt_retrieval, cache_retrieval, outputs):
    # The graves caching will be conditioned on the KRetrieval augmented distribution,
    outputs = process_outputs_cb_for_kretrieval(mt_retrieval, outputs)

    last_logits = outputs.logits[:, -1, :]
    last_hidden = outputs.decoder_hidden_states[-1][0][0]
    # Save current for future.
    cache_retrieval.prepare_translation_output(key=last_hidden)

    item_distances, item_values = cache_retrieval.retrieve_items_from_cache(last_hidden)

    if len(item_values) == 0:
        return outputs

    cache_representations = cache_retrieval.compute_cache_prob(item_distances, item_values, last_logits)
    fused_representation = cache_retrieval.fuse_probs(last_logits, cache_representations)

    # The CB must modify in-place.
    outputs.logits[:, -1, :] = fused_representation
    return outputs

class GravesCaching():
    def __init__(self):
        self.max_limit = 2000
        self.queue = deque(maxlen=self.max_limit)

        self._prepared_keys = []
        self._prepared_values = []

    def push_to_cache(self, k, v):
        self.queue.append((k, v))

    def reset_cache(self):
        self.queue = deque(maxlen=self.max_limit)

    def compute_cache_prob(self, dot_distances, values, mt_dist):
        """
        values is the list of values for the items in the cache. (the list of predicted token IDs)
        dot_distances is the dot product between the MT hidden state and the key for each item in the cache.

        mt_dist is the vocabulary distribution to predict a token from the MT system.
        """
        temperature = self.get_theta_value()

        new_distances = []

        for d in dot_distances:
            new_dist = d * temperature
            new_distances.append(new_dist)

        # Old: new_distances = torch.tensor(new_distances)
        new_distances = np.array(new_distances)

        # Old: normalized = torch.softmax(new_distances, dim=0)
        normalized = softmax(new_distances, axis=0)

        knn_dist = np.zeros_like(mt_dist)

        # Aggregate over multiple instances of the same target token:
        for i in range(len(values)):
            v = values[i] # This is the target token ID.
            normalized_score = normalized[i] # This is the normalized probability of the neighbor.

            knn_dist[v] += normalized_score # Neighbors with the same target token will have their scores combined.

        return knn_dist

    def get_theta_value(self):
        return 0.4

    def get_fuse_value(self):
        # Also known as lambda.
        return 0.5

    def retrieve_items_from_cache(self, decoder_final_hidden_states):
        """
        decoder_final_hidden_states is the hidden states outputted by the last decoder layer to predict a token.
        """

        distances = [] # Distances are computed with the dot product.
        values = []
        for (k, v) in self.queue:
            # Old: dot_dist = torch.dot(decoder_final_hidden_states, k)
            dot_dist = np.dot(decoder_final_hidden_states, k)

            distances.append(dot_dist)
            values.append(v)

        return distances, values

    def fuse_probs(self, mt_prob, cache_prob):
        lamb = self.get_fuse_value()
        return ((1 - lamb) * mt_prob) + (lamb * cache_prob)

    def prepare_translation_output(self, key = None, value = None):
        if key is not None:
            self._prepared_keys.append(key)

        if value is not None:
            self._prepared_values.append(value)

    def end_prepared(self):
        zipped = zip(self._prepared_keys, self._prepared_values)

        for (k, v) in zipped:
            self.push_to_cache(k, v)

        self.clear_prepared()

    def clear_prepared(self):
        self._prepared_keys = []
        self._prepared_values = []

class GravesTranslationApp(Seq2SeqTranslationApp):
    def __init__(self, concat_mode='frame'):
        """
        Augmented version of Seq2Seq translation. This app uses the caching method proposed by Graves to help the model use long range contextual info better.

        From personal experiments, it does not seem to work well, or at least the KRetrieval app does better, and this conflicts with it somehow? YMMV.
        
        There are two more things to note with this app:
            1. In theory, this should suffer from error propagation. (The translation errors will compound here)
            2. This can only help the model use the exact tokens stored in memory,
            e.g, boosting the probability of token "car" but not necessarily the token "vehicle".

        This app also uses the KRetrieval app.
        """
        super().__init__(concat_mode)

    def load_model(self):
        logger.info('Loading datastore - this may take a while...')
        self.mt_retrieval = MTRetrieval()
        logger.info('Done loading datastore!')

        self.graves_retrieval = GravesCaching()

        logger.info('Loading translation model...')
        # Only the J model supports KNN for now.
        s = '/'
        self.translation_model = MarianONNX(
            f'models/marian{s}encoder_q.onnx',
            f'models/marian{s}decoder_q.onnx',
            f'models/marian{s}decoder_init_q.onnx',
            f'models/marian{s}tokenizer_mt',
            process_outputs_cb=lambda x: process_outputs_cb_for_graves(self.mt_retrieval, self.graves_retrieval, x),
            use_cuda=self.use_cuda,
            max_length_a=self.max_length_a,
        )
        logger.info('Done loading translation model!')

        self.loaded = True
