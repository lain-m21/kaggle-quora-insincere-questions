import re
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer

from .preprocess import clean_text, clean_numbers, replace_typical_misspell, puncts


def load_data(input_dir, logger):
    logger.info('Loading train and test data from {}'.format(input_dir))
    df_train = pd.read_csv(input_dir.joinpath('train.csv').open('r'))
    df_test = pd.read_csv(input_dir.joinpath('test.csv').open('r'))
    logger.info('Train shape: {}'.format(df_train.shape))
    logger.info('Test shape: {}'.format(df_test.shape))
    return df_train, df_test


def preprocess_text(df, return_df=True):
    processed_texts = df['question_text'].apply(
        lambda x: replace_typical_misspell(
            clean_numbers(
                clean_text(
                    x.lower()
                )
            )
        )
    )
    if return_df:
        df['question_text'] = processed_texts
        return df
    else:
        return processed_texts


def tokenize_text(df, logger, tokenizer=None):
    text = df['question_text']

    if tokenizer is None:
        logger.info('Fit tokenizer on train data')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(text))

    logger.info('Tokenize text data')
    sequence = tokenizer.texts_to_sequences(text)

    return sequence, tokenizer


def tokenize_texts(texts, logger, tokenizer=None):
    if tokenizer is None:
        logger.info('Fit tokenizer on train data')
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(list(texts))

    logger.info('Tokenize text data')
    sequences = tokenizer.texts_to_sequences(texts)

    return sequences, tokenizer


def extract_nlp_features(df):
    df['total_length'] = df['question_text'].apply(len)
    df['n_capitals'] = df['question_text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['n_words'] = df['question_text'].str.count('\S+')
    df['n_unique_words'] = df['question_text'].apply(lambda x: len(set(w for w in x.split())))
    df['n_puncts'] = df['question_text'].apply(lambda x: sum(1 for c in x if c in set(puncts)))
    df['n_?'] = df['question_text'].apply(lambda x: sum(1 for c in x if c == '?'))
    df['n_!'] = df['question_text'].apply(lambda x: sum(1 for c in x if c == '!'))
    df['n_you'] = df['question_text'].apply(lambda x: len(re.findall(r'you[^~a-z]', x.lower())))
    return df


def load_embeddings(embed_type, word_index):
    if embed_type == 0:
        embedding_file = '../../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    elif embed_type == 1:
        embedding_file = '../../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    else:
        embedding_file = '../../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(embedding_file, encoding="utf8", errors='ignore')
                            if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    nb_words = len(word_index) + 1
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size)).astype(np.float32)
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


def load_multiple_embeddings(word_index, embed_types=(0, 1, 2), debug=False):
    results = []
    for t in embed_types:
        if debug:
            results.append(np.random.rand(len(word_index) + 1, 300))
        else:
            results.append(load_embeddings(t, word_index))
    return results


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')




