import base64
import bz2
import io
import json
import logging
import os
import pickle
import queue
import shutil
import sqlite3
import sys
import typing
import zstandard

import tensorflow_datasets as tfds

DEBUG = True
SUPPORTED_ALGO = ["zst", "bz2"]
CLEANUP = 1_000_000
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO, format='%(process)d-%(levelname)s %(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger(__name__)
timeFrame = None
file_dir = None
compress_algo = None
dest_dir = None
tokenizer_path = None
tokenize = None
tokenizer = None
cache = {'subreddits': [], 'tokenizers': []}

if len(sys.argv) >= 6:
    timeFrame = sys.argv[1]
    file_dir = sys.argv[2]
    compress_algo = sys.argv[3].lower()
    dest_dir = sys.argv[4]
    tokenize = sys.argv[5].lower()
    if tokenize == "true":
        if len(sys.argv) >= 7:
            tokenizer_path = sys.argv[6]
        else:
            tokenizer_path = input("Enter Tokenizer path: ")
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_path)
    if compress_algo in SUPPORTED_ALGO:
        logger.debug(f"{timeFrame} Time frame provided is: {timeFrame} ")
        logger.debug(f"{timeFrame} File directory provided is: {file_dir} ")
        logger.debug(f"{timeFrame} Compression algorithm provided is: {compress_algo} ")
        logger.debug(f"{timeFrame} Destination file provided is: {dest_dir} ")
    else:
        logger.error(f"{timeFrame} Compression algorithm provided is not supported: {compress_algo}. Current supported: {SUPPORTED_ALGO}")
        sys.exit(1)
else:
    logger.error(f"{timeFrame} Incorrect Arguments (time_frame, file_dir, compress-algo [bz2, zst], dest_dir, tokenizer [true, false]). Quitting.")
    sys.exit(1)

last_utc = 0
if not os.path.exists('./cache/'):
    os.mkdir('./cache/')
connection = sqlite3.connect(f'./cache/{timeFrame}.db')
cursor = connection.cursor()


def cleanup_null():
    sql = "DELETE FROM comment WHERE parent_id IS NULL;"
    try:
        cursor.execute(sql)
    except Exception as e:
        logger.error(f"{timeFrame} Error running sql: {e}")
        logger.error(f"{timeFrame} SQL: {sql}")
        if DEBUG:
            raise e
        else:
            quit()
    connection.commit()


def format_data(data: str):
    data = data.replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")
    return data


def acceptable(data: str):
    if len(data) < 1:
        return False
    elif len(data) > 5000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True


def run_sql_insert_or_update(sql, data):
    try:
        cursor.execute(sql, data)
    except Exception as e:
        logger.error(f"{timeFrame} Error running sql: {e}")
        logger.error(f"{timeFrame} SQL: {sql}")
        logger.error(f"{timeFrame} Data: {data}")
        if DEBUG:
            raise e
        else:
            quit()
    connection.commit()
    return cursor.lastrowid


def create_tables():
    sql1 = """CREATE TABLE IF NOT EXISTS subreddits (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                name TEXT UNQIUE NOT NULL);"""
    sql2 = """CREATE TABLE IF NOT EXISTS comment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                comment_id TEXT,
                parent_id INTEGER,
                subreddit_id INTEGER,
                unix INTEGER,
                score INTEGER,
                FOREIGN KEY (parent_id) REFERENCES comment(id),
                FOREIGN KEY (subreddit_id) REFERENCES subreddits(id));"""
    sql3 = """CREATE TABLE IF NOT EXISTS tokenized_comment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tokenized_content TEXT NOT NULL,
                content_id INTEGER NOT NULL,
                tokenizer INTEGER NOT NULL,
                FOREIGN KEY (content_id) REFERENCES comment(id),
                FOREIGN KEY (tokenizer) REFERENCES tokenizers(id));"""
    sql4 = """
        CREATE TABLE IF NOT EXISTS tokenizers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tokenizer_id TEXT NOT NULL);
        """
    cursor.execute(sql1)
    cursor.execute(sql2)
    cursor.execute(sql3)
    cursor.execute(sql4)
    connection.commit()


def check_score(parent_id):
    sql = "SELECT score FROM comment WHERE comment_id == ?"
    try:
        cursor.execute(sql, (parent_id,))
    except Exception as e:
        logger.error(f"{timeFrame} Error running sql: {e}")
        logger.error(f"{timeFrame} SQL: {sql}")
        logger.error(f"{timeFrame} Data: {parent_id}")
        if DEBUG:
            raise e
        else:
            quit()
    result = cursor.fetchone()
    if result is not None:
        return result[0]
    else:
        return None


def check_subreddit(subreddit):
    if len(cache['subreddits']) == 0:
        sql = "SELECT id, name FROM subreddits;"
        try:
            cursor.execute(sql)
        except Exception as e:
            logger.error(f"{timeFrame} Error running sql: {e}")
            logger.error(f"{timeFrame} SQL: {sql}")
            logger.error(f"{timeFrame} Data: {subreddit}")
            if DEBUG:
                raise e
            else:
                quit()
        result = cursor.fetchall()
        cache['subreddits'] = result
        check_subreddit(subreddit)
    else:
        for row in cache['subreddits']:
            if row[1] == subreddit:
                return row[0]
        else:
            sql = "INSERT INTO subreddits (name) VALUES (?);"
            row_id = run_sql_insert_or_update(sql, (subreddit,))
            cache['subreddits'].append((row_id, subreddit))
            return row_id


def check_tokenizer(tokenizer_name):
    if len(cache['tokenizers']) == 0:
        sql = "SELECT id, tokenizer_id FROM tokenizers;"
        try:
            cursor.execute(sql)
        except Exception as e:
            logger.error(f"{timeFrame} Error running sql: {e}")
            logger.error(f"{timeFrame} SQL: {sql}")
            logger.error(f"{timeFrame} Data: {tokenizer_name}")
            if DEBUG:
                raise e
            else:
                quit()
        result = cursor.fetchall()
        cache['tokenizers'] = result
        check_tokenizer(tokenizer_name)
    else:
        for row in cache['tokenizers']:
            if row[1] == tokenizer_name:
                return row[0]
        else:
            sql = "INSERT INTO tokenizers (tokenizer_id) VALUES (?);"
            row_id = run_sql_insert_or_update(sql, (tokenizer_name,))
            cache['tokenizers'].append((row_id, tokenizer_name))
            return row_id


def check_parent(parent_id):
    sql = "SELECT id FROM comment WHERE comment_id = ?;"
    try:
        cursor.execute(sql, (parent_id,))
    except Exception as e:
        logger.error(f"{timeFrame} Error running sql: {e}")
        logger.error(f"{timeFrame} SQL: {sql}")
        logger.error(f"{timeFrame} Data: {parent_id}")
        if DEBUG:
            raise e
        else:
            quit()
    result = cursor.fetchone()
    if result is None:
        return None
    else:
        return result[0]


def sql_replace_comment(comment_id: str, comment: str, subreddit: str, time: int, score: int):
    subreddit = check_subreddit(subreddit)
    sql = "UPDATE comment SET content = ?, subreddit_id = ?, unix = ?, score = ? WHERE comment_id = ?;"
    run_sql_insert_or_update(sql, (comment, subreddit, time, score, comment_id))


def sql_insert_no_parent(comment_id: str, comment: str, subreddit: str, time: int, score: int):
    subreddit = check_subreddit(subreddit)
    sql = "INSERT INTO comment (content, comment_id, subreddit_id, unix, score) VALUES (?,?,?,?,?);"
    row_id = run_sql_insert_or_update(sql, (comment, comment_id, subreddit, time, score))
    return row_id


def sql_insert_tokenize(comment: str, content_id: int, tokenizer_name: str):
    tokenizer_name = check_tokenizer(tokenizer_name)
    sql = "INSERT INTO tokenized_comment (tokenized_content, content_id, tokenizer) VALUES (?,?,?);"
    run_sql_insert_or_update(sql, (comment, content_id, tokenizer_name))


def sql_insert_row(comment_id, parent_id, comment: str, subreddit: str, time: int, score: int):
    tokenizer_name = os.path.splitext(os.path.basename(tokenizer_path))[0]
    subreddit = check_subreddit(subreddit)
    parent_id = check_parent(parent_id)
    if parent_id is not None:
        sql = "INSERT INTO comment (content, comment_id, parent_id, subreddit_id, unix, score) VALUES (?,?,?,?,?,?)"
        row_id = run_sql_insert_or_update(sql, (comment, comment_id, parent_id, subreddit, time, score))
        if tokenize:
            content = tokenizer.encode(comment)
            content = pickle.dumps(content)
            content = base64.b64encode(content)
            sql_insert_tokenize(str(content), row_id, tokenizer_name)
        return "parent"
    else:
        row_id = sql_insert_no_parent(comment_id, comment, subreddit, time, score)
        if tokenize:
            content = tokenizer.encode(comment)
            content = pickle.dumps(content)
            content = base64.b64encode(content)
            sql_insert_tokenize(str(content), row_id, tokenizer_name)
        return "no_parent"


def main():
    create_tables()
    row_counter = 0
    paired_rows = 0
    file_path = f"RC_{timeFrame}.{compress_algo}"

    read_mode = "r" if compress_algo == "bz2" else "rb"
    if compress_algo == "bz2":
        text_stream = bz2.open(os.path.join(file_dir, file_path), read_mode)
    else:
        f = open(os.path.join(file_dir, file_path), read_mode)
        dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
        stream_reader = dctx.stream_reader(f, read_size=int(1.953e+6))
        text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
    logger.info(f"{timeFrame} Reading {file_path}")
    for row in text_stream:
        row_counter += 1
        row = json.loads(row)

        parent_id = row['parent_id']
        body = format_data(row['body'])
        created_utc = int(row['created_utc'])
        score = row['score']
        if compress_algo == "bz2":
            comment_id = row['name']
        else:
            comment_id = 't1_' + row['id']
        subreddit = row['subreddit']

        if score > 2:
            existing_score = check_score(comment_id)
            if existing_score:
                if existing_score > score:
                    if acceptable(body):
                        sql_replace_comment(comment_id, body, subreddit, created_utc, score)
        else:
            if acceptable(body):
                result = sql_insert_row(comment_id, parent_id, body, subreddit, created_utc, score)
                if result == "parent":
                    paired_rows += 1
        if row_counter % 100_000 == 0:
            logger.info(f"{timeFrame} Processed {row_counter} rows & {paired_rows} paired rows")
        else:
            if DEBUG:
                if row_counter % 1000 == 0:
                    logger.info(f"{timeFrame} Processed {row_counter} rows & {paired_rows} paired rows")

        if row_counter > 0:
            if row_counter % CLEANUP == 0:
                logger.info(f"{timeFrame} Cleaning up.")
                cleanup_null()
    logger.info(f"{timeFrame} Finishing...")
    logger.info(f"{timeFrame} Vacuum")
    cursor.execute(f"{timeFrame} VACUUM")
    connection.commit()
    logger.info(f"{timeFrame} Closing")
    cursor.close()
    connection.close()
    logger.info(f"{timeFrame} Total:")
    logger.info(f"--Processed {row_counter} rows")
    logger.info(f"--Paired {paired_rows} rows")
    logger.info(f"{timeFrame} Moving")
    shutil.move(f'./cache/{timeFrame}.db', os.path.join(dest_dir, f'{timeFrame}.db'))
    logger.info(f"{timeFrame} Finished.")


if __name__ == "__main__":
    main()
    logger.info("Done")
