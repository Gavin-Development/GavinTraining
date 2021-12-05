import os
import shutil
import sys
import pickle
import base64
import sqlite3
import logging
import typing
import re
import tqdm
import glob

import pandas as pd
import tensorflow_datasets as tfds

DEBUG = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format='%(process)d-%(levelname)s %(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger(__name__)


def preprocess_sentence(sentence):
    sentence = sentence.strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,'*\"])", r" \1 ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    sentence = re.sub(r"[^a-zA-z?.!,'*\"]+", " ", sentence)
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


def main(tokenizer_path: str, database_path: str, time_frames: typing.List[str]):
    total_tokenized_comments = 0
    for time_frame in time_frames:
        database_name = time_frame + ".db"
        tokenizer_name = os.path.basename(tokenizer_path)
        logger.info(f"{time_frame} Tokenizing {database_name} with {tokenizer_name}")
        if os.path.exists(tokenizer_path + ".subwords") and os.path.exists(database_path) and os.path.exists(
                os.path.join(database_path, database_name)):
            tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_path)
            if not os.path.exists('./cache/'):
                os.makedirs('./cache/')
            shutil.copy(os.path.join(database_path, database_name), './cache/')
            limit = 200_000
            connection = sqlite3.connect('./cache/' + database_name)
            last_unix = 0
            cur_length = limit
            count = 0
            offset = 0
            c = connection.cursor()
            c.execute("INSERT INTO tokenizers (tokenizer_id) VALUES (?)", (tokenizer_name,))
            connection.commit()
            tokenizer_row_id = c.lastrowid
            c.execute("SELECT MAX(id) FROM comment;")
            max_id = c.fetchone()[0]

            while cur_length == limit:
                logger.info(f"{time_frame} Tokenized & inserted {count * limit}/{max_id} comments.")
                try:
                    df = pd.read_sql(
                        "SELECT * FROM main.comment AS c WHERE c.id NOT IN (SELECT content_id FROM tokenized_comment WHERE tokenizer = {}) AND c.score > 0 AND unix > {} ORDER BY unix ASC LIMIT {} OFFSET {};".format(
                            tokenizer_row_id, last_unix, limit, (limit * offset)), connection)
                except Exception as e:
                    logger.error(f"{time_frame} Pandas read_sql error: {e}")
                else:
                    cur_length = len(df)
                    if cur_length > 0:
                        # Should look like [(tokenized_text, comment_id, tokenizer_id)]
                        last_unix = df.tail(1)['unix'].values[0]
                        tokenized_comments = []
                        logger.info(f"{time_frame} Starting tokenization of {cur_length}.")
                        for table_id, comment in tqdm.tqdm(zip(df['id'].values, df['content'].values),
                                                           desc=f"Tokenizing {time_frame}",
                                                           total=cur_length, unit='comments'):
                            tokenized_comments.append(
                                (str(base64.b64encode(pickle.dumps(tokenizer.encode(preprocess_sentence(comment))))),
                                 int(table_id), int(tokenizer_row_id)))
                            total_tokenized_comments += 1

                        c.executemany(
                            "INSERT INTO main.tokenized_comment (tokenized_content, content_id, tokenizer) VALUES (?,?,?)",
                            tokenized_comments)
                        count += 1
                        offset += 1
                        connection.commit()
            c.close()
            connection.close()
            shutil.copy('./cache/' + database_name, database_path)
            os.remove('./cache/' + database_name)
            logger.info(f"{time_frame} {count * limit} comments tokenized.")
            logger.info(f"{time_frame} {total_tokenized_comments} total tokenized comments.")
            logger.info(f"{time_frame} Finished.")

        else:
            if not os.path.exists(tokenizer_path + ".subwords"):
                logger.error(f"{time_frame} Tokenizer not found.")
            if not os.path.exists(os.path.join(database_path, database_name)):
                logger.error(f"{time_frame} Database not found.")


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python3 tokenize_comments.py <tokenizer_path> <database_path> <time_frame>")
        sys.exit(1)
    if not os.path.exists(sys.argv[1] + ".subwords"):
        print(f"Tokenizer {sys.argv[1]} not found.")
        sys.exit(1)
    if not os.path.exists(sys.argv[2]):
        print(f"Database path {sys.argv[2]} not found.")
        sys.exit(1)
    if sys.argv[3] in ["*", "all"]:
        databases = glob.glob(f"{sys.argv[2]}/*.db")
        databases = [os.path.basename(database)[:-3] for database in databases]
    else:
        databases = [f"{sys.argv[3]}-%.2d" % i for i in range(1, 13)]
    main(sys.argv[1], sys.argv[2], databases)
