import base64
import bz2
import enum
import io
import json
import logging
import os
import pickle
import shutil
import sqlite3
import sys
import zstandard


class InsertionType(enum.Enum):
    """
    Enum for the different types of data insertion.
    """
    INSERT_NO_PARENT = 0
    INSERT_PARENT = 1
    UPDATE_CHILD = 2


DEBUG = True
SUPPORTED_ALGO = ["zst", "bz2"]
DATABASE_COMMIT_RATE = 10_000

logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format='%(process)d-%(levelname)s %(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger(__name__)
timeFrame = None
file_dir = None
compress_algo = None
dest_dir = None
tokenizer_path = None
tokenize = None
tokenizer = None

# Cache['comments'] = {'comment_id' : 'id'} where id is the id of the comment in the database.

cache = {'subreddits': [], 'tokenizers': [], 'comments': {}}
sql_none_parents = []
sql_parents = []
sql_update = []
comment_ids = []


if len(sys.argv) >= 6:
    timeFrame = sys.argv[1]
    file_dir = sys.argv[2]
    compress_algo = sys.argv[3].lower()
    dest_dir = sys.argv[4]
    tokenize = sys.argv[5].lower()
    if tokenize == "true":
        tokenize = True
        import tensorflow_datasets as tfds

        if len(sys.argv) >= 7:
            tokenizer_path = sys.argv[6]
        else:
            tokenizer_path = input("Enter Tokenizer path: ")
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizer_path)
    else:
        tokenize = False

    if compress_algo in SUPPORTED_ALGO:
        logger.debug(f"{timeFrame} Time frame provided is: {timeFrame} ")
        logger.debug(f"{timeFrame} File directory provided is: {file_dir} ")
        logger.debug(f"{timeFrame} Compression algorithm provided is: {compress_algo} ")
        logger.debug(f"{timeFrame} Destination file provided is: {dest_dir} ")
    else:
        logger.error(
            f"{timeFrame} Compression algorithm provided is not supported: {compress_algo}. Current supported: {SUPPORTED_ALGO}")
        sys.exit(1)
else:
    logger.error(
        f"{timeFrame} Incorrect Arguments (time_frame, file_dir, compress-algo [bz2, zst], dest_dir, tokenizer [true, false]). Quitting.")
    sys.exit(1)

last_utc = 0
if not os.path.exists('./cache/'):
    os.mkdir('./cache/')
connection = sqlite3.connect(f'./cache/{timeFrame}.db')
cursor = connection.cursor()


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


def run_sql_insert_or_update(data, insertion_type: InsertionType = InsertionType.INSERT_NO_PARENT):
    if insertion_type == InsertionType.INSERT_NO_PARENT:
        sql_none_parents.append(data)
        comment_ids.append(data[1])  # Append comment ID to list, to stop having to iterate through a list 2x causing O(N^2) complexity
    elif insertion_type == InsertionType.INSERT_PARENT:
        sql_parents.append(data)
    elif insertion_type == InsertionType.UPDATE_CHILD:
        sql_update.append(data)
    if len(sql_none_parents) % DATABASE_COMMIT_RATE == 0 and len(sql_none_parents) > 0:
        logger.debug(f"{timeFrame} Committing {len(sql_none_parents)} to database")
        for none_data in sql_none_parents:
            cursor.execute("INSERT INTO comment (content, comment_id, subreddit_id, unix, score) VALUES (?,?,?,?,?);", none_data)
            rowid = cursor.lastrowid
            cache['comments'][none_data[1]] = rowid
        connection.commit()
        for with_data in sql_parents:
            with_data = list(with_data)
            parent_id = cache['comments'][with_data[2]]
            with_data[2] = parent_id
            cursor.execute("INSERT INTO comment (content, comment_id, parent_id, subreddit_id, unix, score) VALUES (?,?,?,?,?,?);", with_data)
            rowid = cursor.lastrowid
            cache['comments'][with_data[1]] = rowid
        connection.commit()
        for update_data in sql_update:
            update_data = list(update_data)
            comment_id = cache['comments'][update_data[4]]
            update_data[4] = comment_id
            cursor.execute("UPDATE comment SET content = ?, subreddit_id = ?, unix = ?, score = ? WHERE comment_id = ?;", update_data)
        sql_none_parents.clear()
        sql_parents.clear()
        sql_update.clear()
        connection.commit()


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


def check_subreddit(subreddit: str, override: bool = False) -> int:
    """
    Check the subreddits in current database to see if it exists.
    If it does, return the id.
    :param subreddit: str
        Name of the subreddit
    :param override: bool
        In the case of subreddits being empty after the first check, the override is set. """
    if subreddit is None:
        raise ValueError("Subreddit is None")
    if len(cache['subreddits']) == 0 and not override:
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
        if not result:
            check_subreddit(subreddit, True)
        check_subreddit(subreddit)
    else:
        for row in cache['subreddits']:
            if row[1] == subreddit:
                return row[0]
        else:
            sql = "INSERT INTO subreddits (name) VALUES (?);"
            cursor.execute(sql, (subreddit,))
            connection.commit()
            row_id = cursor.lastrowid
            cache['subreddits'].append((row_id, subreddit))
            return row_id


def check_tokenizer(tokenizer_name: str, override: bool = False):
    """
    Check the tokenizers in current database to see if it exists.
    If it does, return the id.
    :param tokenizer_name: str
        The tokenizer version name
    :param override:
        In the case of tokenizers being empty after the first check, the override is set.
    :return:
    """
    if len(cache['tokenizers']) == 0 and not override:
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
        if not result:
            check_tokenizer(tokenizer_name, True)
        check_tokenizer(tokenizer_name)
    else:
        for row in cache['tokenizers']:
            if row[1] == tokenizer_name:
                return row[0]
        else:
            sql = "INSERT INTO tokenizers (tokenizer_id) VALUES (?);"
            cursor.execute(sql, (tokenizer_name,))
            row_id = cursor.lastrowid
            cache['tokenizers'].append((row_id, tokenizer_name))
            return row_id


def check_parent(parent_id):
    # Rewritten to check cache first
    # return parent_id in (data[1] for data in sql_none_parents) or parent_id in cache['comments'].keys()
    if parent_id in cache['comments'].keys():
        return True
    else:
        if parent_id in comment_ids:
            return True


def sql_replace_comment(comment_id: str, comment: str, subreddit: str, time: int, score: int):
    subreddit = check_subreddit(subreddit)
    run_sql_insert_or_update((comment, subreddit, time, score, comment_id), insertion_type=InsertionType.UPDATE_CHILD)


def sql_insert_no_parent(comment_id: str, comment: str, subreddit: int, time: int, score: int):
    run_sql_insert_or_update((comment, comment_id, subreddit, time, score), insertion_type=InsertionType.INSERT_NO_PARENT)


def sql_insert_row(comment_id, parent_id, comment: str, subreddit: str, time: int, score: int):
    subreddit = check_subreddit(subreddit)
    has_parent = check_parent(parent_id)
    if has_parent:
        run_sql_insert_or_update((comment, comment_id, parent_id, subreddit, time, score), insertion_type=InsertionType.INSERT_PARENT)
        return "parent"
    else:
        sql_insert_no_parent(comment_id, comment, subreddit, time, score)
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

    logger.info(f"{timeFrame} Finishing...")
    logger.info(f"{timeFrame} Vacuum")
    cursor.execute(f"VACUUM")
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
