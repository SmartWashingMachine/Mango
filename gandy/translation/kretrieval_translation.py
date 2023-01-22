from annoy import AnnoyIndex
import numpy as np
from joblib import load
from scipy.special import softmax
from random import sample
from gandy.translation.seq2seq_translation import Seq2SeqTranslationApp
from gandy.onnx_models.marian_knn import MarianKNNONNX
import logging

logger = logging.getLogger('Gandy')

def process_outputs_cb_for_kretrieval(mt_retrieval, outputs):
    # Called for every decoding timestep.

    # decoder_hidden_states is a tuple of length 7, the 0th being the embedding, and the last being the last layer output. We should always want to index with -1.
    # decoder_hidden_states[0] is a tensor of shape [beam, seq_length, 512(hidden)]

    # logits is a tensor of shape [beam, seq_length, 60719(vocab)]
    # last_logits is a tensor of shape [beam, 60719(vocab)]

    if mt_retrieval.datastore_targets is None:
        return outputs

    last_layer_output = outputs.decoder_hidden_states[-1]
    last_hidden_states = last_layer_output[:, -1, :] # [beam, 512]

    n_beams = last_hidden_states.shape[0]
    for beam_idx in range(n_beams):
        last_hidden = last_hidden_states[beam_idx, :] # [512]
        last_logits = outputs.logits[beam_idx, -1, :] # [60719]

        # Process to hopefully get a better output.
        neighbor_values, neighbor_distances = mt_retrieval.retrieve_neighbors_from_datastore(last_hidden)

        knn_representations = mt_retrieval.compute_knn_prob_true(neighbor_values, neighbor_distances, last_logits)
        fused_representation = mt_retrieval.fuse_probs(last_logits, knn_representations) #<- no different than logits...? whats this mean

        outputs.logits[beam_idx, -1, :] = fused_representation

    return outputs

class MTRetrieval():
    def __init__(self):
        # In order to make the datastore more compact, the keys (hidden states) for the datastore are reduced via PCA.
        # Don't think anybody wants to download a 30GB datastore lol.
        try:
            pca_path_to_save = 'models/knn/pca.joblib'
            self.pca = load(pca_path_to_save) # 128 components
        except Exception as e:
            logger.info('Error loading PCA for KNN:')
            logger.exception(e)

        self.initialize_datastore()

    def compute_knn_prob_true(self, neighbor_values, neighbor_distances, mt_dist):
        """
        neighbor_values is the list of values from the neighbors.
        neighbor_distances is the computed euclidean distance.

        mt_dist is the vocabulary distribution to predict a token from the MT system.
        """
        temperature = self.get_temperature_value()

        new_neighbor_distances = []

        for d in neighbor_distances:
            new_dist = -1 * d
            new_dist /= temperature
            new_neighbor_distances.append(new_dist)

        # Old: new_neighbor_distances = torch.tensor(new_neighbor_distances)
        new_neighbor_distances = np.array(new_neighbor_distances)

        # Old: normalized = torch.softmax(new_neighbor_distances, dim=0)
        normalized = softmax(new_neighbor_distances, axis=0)

        # Old: knn_dist = torch.zeros_like(mt_dist) # vocab_size
        knn_dist = np.zeros_like(mt_dist)

        # Aggregate over multiple instances of the same target token:
        for i in range(len(neighbor_values)):
            neighbor_value = neighbor_values[i] # This is the target token ID.
            normalized_score = normalized[i] # This is the normalized probability of the neighbor.
            knn_dist[neighbor_value] += normalized_score # Neighbors with the same target token will have their scores combined.

        return knn_dist

    def get_temperature_value(self):
        return 10

    def get_fusion_value(self):
        return 0.5

    def get_k_value(self):
        return 4

    def initialize_datastore(self):
        # Load index.
        self.datastore_hidden_index = AnnoyIndex(128, 'euclidean')

        try:
            # Though prefault=True does not work on Windows... I'm keeping it here for the future.
            self.datastore_hidden_index.load(f'models/knn/index_V2.ann', prefault=True) # Returns a bool, but modifies inplace.
            # Load values.
            self.datastore_targets = np.load('models/knn/targets_V2.npy', allow_pickle=True).item()

            logger.info('Datastore memory-mapped. Warming up...')
            indices = list(self.datastore_targets.keys())

            # See: https://github.com/spotify/annoy/issues/376
            every_n = 500
            if len(indices) > every_n:
                sample_n = len(indices) // every_n
                indices = sample(indices, k=sample_n)
                for iteration, idx in enumerate(indices):
                    self.datastore_hidden_index.get_item_vector(idx)

                    if iteration % (sample_n // 10) == 0:
                        logger.info(f'Warming up at: ({iteration+1} / {sample_n})')

            logger.info('Datastore loaded!')
        except Exception as e:
            logger.info('Error loading datastore:')
            logger.exception(e)

            self.datastore_hidden_index = None
            self.datastore_targets = None

    def retrieve_neighbors_from_datastore(self, decoder_final_hidden_states):
        """
        decoder_final_hidden_states is the hidden states outputted by the last decoder layer to predict a token.
        """
        # This returns a list of integers. I'm hoping that this is for the indices...
        reduced = self.reduce_dims(decoder_final_hidden_states)

        neighbor_indices, neighbor_distances = self.datastore_hidden_index.get_nns_by_vector(reduced, n=self.get_k_value(), include_distances=True)
        neighbor_values = [self.datastore_targets[n] for n in neighbor_indices]

        return neighbor_values, neighbor_distances

    def fuse_probs(self, mt_prob, knn_prob):
        fuse_param = self.get_fusion_value()
        return ((1 - fuse_param) * mt_prob) + (fuse_param * knn_prob)

    def reduce_dims(self, hidden):
        hidden = hidden[None, ...] # Scikit requires a single sample to be of shape (1, -1). This is equivalent to pytorch's .unsqueeze(dim=0)

        reduced = self.pca.transform(hidden)

        # Old: reduced = torch.from_numpy(reduced).squeeze(dim=0) # Then we can resqueeze it.
        reduced = reduced[0, ...]
        return reduced

class KRetrievalTranslationApp(Seq2SeqTranslationApp):
    def __init__(self, concat_mode='frame'):
        """
        Augmented version of Seq2Seq translation. This app uses KNN with a datastore to improve token probabilities for already seen tokens, supposedly improving accuracy.

        First, the Seq2Seq model generates decoder hidden states for a token.
        Then, this app will use approximate nearest neighbor search to find nearby tokens according to the Euclidean distance between the hidden states.
        Up to K tokens will be found, and then their values will be fused with the logits for decoding.

        The datastore is pre-constructed from some training data, and can not be added to on runtime. (The Graves caching app can sort of do that, if so desired.)
        """
        super().__init__(concat_mode)

    def load_model(self):
        logger.info('Loading datastore - this may take a while...')
        self.mt_retrieval = MTRetrieval()
        logger.info('Done loading datastore!')

        logger.info('Loading translation model...')
        # Only the J model supports KNN for now.
        s = '/'
        self.translation_model = MarianKNNONNX(
            f'models/marian{s}encoder.onnx',
            f'models/marian{s}decoder.onnx',
            f'models/marian{s}decoder_init.onnx',
            f'models/marian{s}tokenizer_mt',
            process_outputs_cb=lambda x: process_outputs_cb_for_kretrieval(self.mt_retrieval, x),
            use_cuda=self.use_cuda,
            max_length_a=self.max_length_a,
        )
        logger.info('Done loading translation model!')

        self.loaded = True

