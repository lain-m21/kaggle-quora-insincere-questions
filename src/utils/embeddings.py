import numpy as np
from numba import jit


def load_embeddings(word_index, max_words=200000, embed_type=0):
    if embed_type == 0:
        embedding_file = '../../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    elif embed_type == 1:
        embedding_file = '../../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    else:
        embedding_file = '../../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore')
                            if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = min(max_words, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)).astype(np.float32)
    for word, i in word_index.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


@jit('f8[:,:]()')
def load_embeddings_jit(word_index, max_words=200000, embed_type=0):
    if embed_type == 0:
        embedding_file = '../../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    elif embed_type == 1:
        embedding_file = '../../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    else:
        embedding_file = '../../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'