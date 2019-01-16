import sys; sys.path.append('..')
import json
from pathlib import Path

from model.scdv import SCDV
from utils.prepare_data import load_data, preprocess_text, tokenize_text, load_embeddings
from utils.logger import Logger

INPUT_DIR = Path.cwd().joinpath('../../input')
DATA_DIR = Path.cwd().joinpath('../../data')
LOG_DIR = Path.cwd().joinpath('../../logs')
SLACK_URL = json.load(Path.cwd().joinpath('../config.json').open('r'))['slack']


def main(logger):
    df_train, df_test = load_data(INPUT_DIR, logger)
    logger.info('Preprocess text')
    df_train = preprocess_text(df_train)
    df_test = preprocess_text(df_test)
    seq_train, tokenizer = tokenize_text(df_train, logger)
    seq_test, _ = tokenize_text(df_test, logger, tokenizer=tokenizer)
    text_train = df_train['question_text'].values.tolist()
    embedding_matrix = load_embeddings(0, tokenizer.word_index)

    scdv = SCDV(embedding_matrix, tokenizer, logger, num_clusters=50)
    with logger.timer('SCDV computation on train data'):
        scdv_train = scdv.fit_transform(text_train, seq_train)

    logger.post(f'Computing SCDV for train data has been done: shape = {scdv_train.shape}')

    with logger.timer('SCDV computation on test data'):
        scdv_test = scdv.transform(seq_test)

    logger.post(f'Computing SCDV for test data has been done: shape = {scdv_test.shape}')


if __name__ == '__main__':
    script_name = Path(__file__).stem
    log_dir = LOG_DIR.joinpath(f'{script_name}')
    logger = Logger(script_name, log_dir=log_dir, webhook_url=SLACK_URL, overwrite=True)
    try:
        main(logger)
        logger.post('===== Script completed successfully! =====')
    except Exception as e:
        logger.exception(e)