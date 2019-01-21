import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from keras.preprocessing.text import text_to_word_sequence


class SCDV:
    def __init__(self, embedding_matrix: np.ndarray, tokenizer, logger, num_clusters=50, gmm_path=None):
        self.embedding_matrix = embedding_matrix
        self.tokenizer = tokenizer
        self.num_clusters = num_clusters
        self.logger = logger
        self.gmm_path = gmm_path

        self.word_centroid_map = None
        self.word_centroid_prob_map = None
        self.word_idf_dict = None
        self.prob_wordvecs = None
        self.min_no = 0
        self.max_no = 0
        self.threshold = 0

    def fit_transform(self, texts, sequences):
        idx, idx_proba = self._cluster_gmm()

        self.logger.info('Form word centroid maps')
        self.word_centroid_map = dict(zip(list(range(self.embedding_matrix.shape[0])), idx))
        self.word_centroid_prob_map = dict(zip(list(range(self.embedding_matrix.shape[0])), idx_proba))

        self.logger.info('Fit TfidfVectorizer on train text data')
        vectorizer = TfidfVectorizer(dtype=np.float32, tokenizer=text_to_word_sequence)
        vectorizer.fit(texts)
        feature_names = vectorizer.get_feature_names()
        idf_list = vectorizer._tfidf.idf_

        self.logger.info('Form word idf dictionary')
        self.word_idf_dict = dict(zip([self.tokenizer.word_index[w] for w in feature_names], idf_list))
        self.word_idf_dict[0] = 0

        self.logger.info('Compute probability word vectors')
        self.prob_wordvecs = self._get_probability_word_vectors(self.word_centroid_map,
                                                                self.word_centroid_prob_map,
                                                                self.word_idf_dict)

        self.logger.info('Compute SCDV vector for each text')
        scdv_vectors = np.zeros([len(sequences), self.num_clusters * self.embedding_matrix.shape[1]], dtype=np.float32)
        for i, seq in enumerate(sequences):
            scdv_vectors[i] = self._compute_scdv_vector(seq, self.prob_wordvecs, self.word_centroid_map, train=True)

        self.threshold = 0.04 * (abs(self.min_no / len(sequences)) + abs(self.max_no / len(sequences))) / 2
        self.logger.info(f'Threshold: {self.threshold}')

        scdv_vectors[scdv_vectors < self.threshold] = 0

        return scdv_vectors

    def transform(self, sequences):
        self.logger.info('Compute SCDV vector for each text')
        scdv_vectors = np.zeros([len(sequences), self.num_clusters * self.embedding_matrix.shape[1]], dtype=np.float32)
        for i, seq in enumerate(sequences):
            scdv_vectors[i] = self._compute_scdv_vector(seq, self.prob_wordvecs, self.word_centroid_map, train=False)

        scdv_vectors[scdv_vectors < self.threshold] = 0

        return scdv_vectors

    def _cluster_gmm(self):
        if self.gmm_path is not None:
            if self.gmm_path.exists():
                mixture_model = joblib.load(str(self.gmm_path))
            else:
                mixture_model = GaussianMixture(n_components=self.num_clusters,
                                                covariance_type='tied',
                                                init_params='kmeans',
                                                max_iter=50)
                self.logger.info('Fit Gaussian Mixture model...')
                mixture_model.fit(self.embedding_matrix)
                joblib.dump(mixture_model, str(self.gmm_path))
        else:
            mixture_model = GaussianMixture(n_components=self.num_clusters,
                                            covariance_type='tied',
                                            init_params='kmeans',
                                            max_iter=50)
            self.logger.info('Fit Gaussian Mixture model...')
            mixture_model.fit(self.embedding_matrix)

        self.logger.info('Cluster word vectors')
        idx = mixture_model.predict(self.embedding_matrix)
        idx_proba = mixture_model.predict_proba(self.embedding_matrix)
        return idx, idx_proba

    def _get_probability_word_vectors(self, word_centroid_map, word_centroid_prob_map, word_idf_dict):
        prob_wordvecs = {}
        num_features = self.embedding_matrix.shape[1]
        for word_id in word_centroid_map:
            prob_wordvecs[word_id] = np.zeros(self.num_clusters * num_features, dtype="float32")
            for index in range(self.num_clusters):
                prob_wordvecs[word_id][index * num_features:(index + 1) * num_features] = \
                    self.embedding_matrix[word_id] * \
                    word_centroid_prob_map[word_id][index] * \
                    word_idf_dict[word_id]

        return prob_wordvecs

    def _compute_scdv_vector(self, sequence, prob_wordvecs, word_centroid_map, train=True):
        scdv_vector = np.zeros(self.num_clusters * self.embedding_matrix.shape[1], dtype="float32")

        for word_id in sequence:
            if word_id in word_centroid_map:
                scdv_vector += prob_wordvecs[word_id]

        norm = np.sqrt(np.einsum('...i,...i', scdv_vector, scdv_vector))
        if norm != 0:
            scdv_vector /= norm

        # To make feature vector sparse, make note of minimum and maximum values.
        if train:
            self.min_no += scdv_vector.min()
            self.max_no += scdv_vector.max()

        return scdv_vector
