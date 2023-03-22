import numpy as np
from scipy.special import softmax
from gandy.translation.seq2seq_translation import Seq2SeqTranslationApp
from gandy.onnx_models.marian import MarianONNXNumpy, MarianONNXTorch
import logging
import faiss

try:
    import torch
except:
    pass

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
        neighbor_values, neighbor_distances, did_succeed = mt_retrieval.retrieve_neighbors_from_datastore(last_hidden)
        if not did_succeed:
            continue

        knn_representations = mt_retrieval.compute_knn_prob_true(neighbor_values, neighbor_distances, last_logits)
        fused_representation = mt_retrieval.fuse_probs(last_logits, knn_representations)

        outputs.logits[beam_idx, -1, :] = fused_representation

    return outputs

class MTRetrieval():
    def __init__(self):
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

        if isinstance(mt_dist, np.ndarray):
            new_neighbor_distances = np.array(new_neighbor_distances)
            knn_dist = np.zeros_like(mt_dist) # vocab_size

            normalized = softmax(new_neighbor_distances, axis=0)
        else:
            #new_neighbor_distances = torch.tensor(new_neighbor_distances, device='cuda:0', dtype=torch.float32)
            knn_dist = torch.zeros_like(mt_dist, device='cuda:0', dtype=mt_dist.dtype) # vocab_size

            new_neighbor_distances = np.array(new_neighbor_distances)
            #new_neighbor_distances = new_neighbor_distances.cpu().numpy()
            normalized = softmax(new_neighbor_distances, axis=0)
            #normalized = torch.softmax(new_neighbor_distances, dim=0)

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
        try:
            # Load index. I tried to use the Annoy package, but I found it to be quite... Annoy-ing.
            self.datastore_hidden_index: faiss.IndexFlatL2 = faiss.read_index('models/knn/index.faiss', faiss.IO_FLAG_ONDISK_SAME_DIR)

            # Load values.
            self.datastore_targets = np.load('models/knn/targets.npy', allow_pickle=True).item()

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
        # start = datetime.now()

        neighbor_distances, neighbor_indices = self.datastore_hidden_index.search(
            decoder_final_hidden_states[None, ...] if isinstance(decoder_final_hidden_states, np.ndarray) else decoder_final_hidden_states.unsqueeze(dim=0),
            k=self.get_k_value()
        )
        neighbor_distances = neighbor_distances[0, ...] # Resqueeze.
        neighbor_indices = neighbor_indices[0, ...]

        # end = datetime.now()

        # logger.debug(f'KNN retrieval time elapsed: {(end - start).total_seconds()}s') # KNN takes almost no time. It's truly amazing...

        if -1 in neighbor_indices:
            # Failure case. Not enough neighbors so just don't bother.
            logger.info('Not enough neighbors - ignoring.')
            return None, None, False
            
        neighbor_values = [self.datastore_targets[n] for n in neighbor_indices]

        return neighbor_values, neighbor_distances, True

    def fuse_probs(self, mt_prob, knn_prob):
        fuse_param = self.get_fusion_value()
        return ((1 - fuse_param) * mt_prob) + (fuse_param * knn_prob)

class KRetrievalTranslationApp(Seq2SeqTranslationApp):
    def __init__(self):
        """
        Augmented version of Seq2Seq translation. This app uses KNN with a datastore to improve token probabilities for already seen tokens, supposedly improving accuracy.

        First, the Seq2Seq model generates decoder hidden states for a token.
        Then, this app will use approximate nearest neighbor search to find nearby tokens according to the Euclidean distance between the hidden states.
        Up to K tokens will be found, and then their values will be fused with the logits for decoding.

        The datastore is pre-constructed from some training data, and can not be added to on runtime. (The Graves caching app can sort of do that, if so desired.)
        """
        super().__init__()

    def load_model(self):
        logger.info('Loading datastore - this may take a while...')
        self.mt_retrieval = MTRetrieval()
        logger.info('Done loading datastore!')

        logger.info('Loading translation model...')
        # Only the J model supports KNN for now.
        s = '/'

        if self.use_cuda:
            model_cls = MarianONNXTorch
        else:
            model_cls = MarianONNXNumpy

        # KNN requires unquantized models to work properly.
        self.translation_model = model_cls(
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
