import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer

from .preprocess import clean_text, clean_numbers, replace_typical_misspell


def load_data(input_dir, logger):
    logger.info('Loading train and test data from {}'.format(input_dir))
    df_train = pd.read_csv(input_dir.joinpath('train.csv').open('r'))
    df_test = pd.read_csv(input_dir.joinpath('test.csv').open('r'))
    logger.info('Train shape: {}'.format(df_train.shape))
    logger.info('Test shape: {}'.format(df_test.shape))
    return df_train, df_test


def preprocess_text(df):
    df['question_text'] = df['question_text'].apply(
        lambda x: replace_typical_misspell(
            clean_numbers(
                clean_text(
                    x.lower()
                )
            )
        )
    )
    return df


def tokenize_text(df, logger, tokenizer=None):
    text = df['question_text']

    if tokenizer is None:
        logger.info('Fit tokenizer on train data')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(text))

    logger.info('Tokenize text data')
    sequence = tokenizer.texts_to_sequences(text)

    return sequence, tokenizer


def get_seq_length(sequence):
    return np.array([len(x) for x in sequence])


def load_embeddings(word_index, logger, embed_type=0):
    if embed_type == 0:
        embedding_file = '../../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
        logger.info('Loading Glove embeddings')
    elif embed_type == 1:
        embedding_file = '../../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
        logger.info('Loading fastText embeddings')
    else:
        embedding_file = '../../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
        logger.info('Loading Paragram embeddings')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore')
                            if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = len(word_index)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)).astype(np.float32)
    logger.info('Filling embedding matrix')
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    logger.info('Loading completed')

    return embedding_matrix


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')




