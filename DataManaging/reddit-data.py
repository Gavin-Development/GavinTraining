import asyncio
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
import threading
import typing
import zstandard

import tensorflow_datasets as tfds

from utils import ErrorCatcher, ArgumentError, StoppableThread, SQLInsertFlags

logging.basicConfig(level=logging.INFO, format='%(process)d-%(levelname)s %(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class CategoriseReddit:
    __metaclass__ = ErrorCatcher
    """
    Reddit Dataset only.
    Handles all data from the compressed form.
    """

    def __init__(self, data_file_path: typing.AnyStr, database_file_path: typing.AnyStr, process_year: bool = False,
                 year: typing.AnyStr = None, time_frame: typing.AnyStr = None,
                 threads: int = 4, tokenize_at_process_time: bool = False, tokenizer_path: typing.AnyStr = None,
                 write_freq: int = 100_000, debug=False):
        """
        :param data_file_path: typing.AnyStr
            The path in which the reddit files can be found. (Compressed form)
        :param process_year: bool
             Whether we're processing an entire year or not.
        :param year: typing.AnyStr
            Which year we're processing. Only applicable if process_year is True.
        :param time_frame: typing.AnyStr
            Which specific time_frame we're processing. Only applicable if process_year is False
        :param threads: int
            How many threads to use in processing. Only applicable if process_year is True
        :param tokenize_at_process_time: bool
            If this is true, we tokenize all samples & create the tokenized table at process time.
        :param write_freq: int
            Every x rows write to a statement.
        :param debug: bool
            Whether to print debug statements or not & to catch errors out of a thread.
        """
        if not process_year and time_frame is None:
            raise ArgumentError("If processing year is false, you have to specify the exact time frame.")
        elif process_year and year is None:
            raise ArgumentError("You must specify the year if you're in process year mode.")
        elif year is None and time_frame is None:
            raise ArgumentError("You must specify at least year or time_frame.")
        elif tokenize_at_process_time and tokenizer_path is None:
            raise ArgumentError("You must supply the path to the tokenizer file inorder to use this feature.")
        elif process_year and debug:
            raise ArgumentError("Cannot process years & be in debug mode.")
        self.data_file_path = data_file_path
        self.database_file_path = database_file_path
        self.process_year = process_year
        self.threads = threads
        self.tokenize = tokenize_at_process_time
        if self.tokenize:
            self.tokenizer_path = tokenizer_path
            self.tokenizer_name = os.path.basename(tokenizer_path)
        self.write_freq = write_freq
        self.logger = self.__metaclass__.logger
        self.bucket = queue.Queue()
        self.loop = asyncio.get_event_loop()
        self.lock = threading.Lock()

        self.debug = debug
        self.loop.set_debug(self.debug)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        # Interval at which to print progress on read statements.
        self.info_freq = self.write_freq // 16
        self.stopped = False

        if self.process_year:
            self.time_frames = [year + "-%.2d" % i for i in range(1, 13)]
            thread_generator = self.__chunks(self.time_frames, ((len(self.time_frames) // self.threads) + 1))
            self.time_frames = [next(thread_generator) for _ in range(self.threads)]
        else:
            self.time_frames = [time_frame]

        # Samples dict:
        # If tokenize_at_process_time is True then *_tokenized is empty.
        # {'year': {'data': [{'parent_id': typing.AnyStr, 'comment_id': typing.AnyStr, 'parent': typing.AnyStr,
        #                     'comment': typing.AnyStr, 'parent_tokenized': typing.AnyStr,
        #                     'comment_tokenized': typing.AnyStr, 'subreddit': typing.AnyStr,
        #                     'unix': int, 'score': int
        #                     }]
        #          }
        # }
        # Samples: typing.Dict Schematic for dict is above.
        self.__samples = {}

        # Internal attribute for recovery purposes.
        # {'year': int}
        self.at_row = {}
        # Keep list of all threads running.
        self.threads = []

    @staticmethod
    def get_subreddits_by_name(cur: sqlite3.Cursor) -> typing.Dict[str, int]:
        """Fetch the subreddits."""
        cur.execute("""SELECT * FROM main.subreddits;""")
        results = cur.fetchall()
        return {result[1]: result[0] for result in results}

    @staticmethod
    def get_tokenizers_by_name(cur: sqlite3.Cursor) -> typing.Dict[str, int]:
        """Fetch the tokenizers."""
        cur.execute("""SELECT * FROM main.tokenizers;""")
        results = cur.fetchall()
        return {result[1]: result[0] for result in results}

    def __get_subreddits(self, conn: sqlite3.Connection, cur: sqlite3.Cursor, subreddit: str) -> typing.Dict[str, int]:
        subreddits = self.get_subreddits_by_name(cur)
        if subreddit not in subreddits.keys():
            try:
                cur.execute("INSERT INTO main.subreddits (name) VALUES (?)", (subreddit,))
            except Exception as e:
                self.logger.error(f"Error inserting subreddit {subreddit} into database: {e}")
                if self.debug:
                    raise e
            conn.commit()
            return self.get_subreddits_by_name(cur)
        else:
            return subreddits

    def __get_tokenizers(self, conn: sqlite3.Connection, cur: sqlite3.Cursor, tokenizer_name: str) -> typing.Dict[str, int]:
        tokenizers = self.get_tokenizers_by_name(cur)
        if tokenizer_name not in tokenizers.keys():
            try:
                cur.execute("INSERT INTO main.tokenizers (tokenizer_id) VALUES (?)", (tokenizer_name,))
            except Exception as e:
                self.logger.error(f"Error inserting tokenizer {tokenizer_name} into database: {e}")
                if self.debug:
                    raise e
            conn.commit()
            return self.get_tokenizers_by_name(cur)
        else:
            return tokenizers

    def sql_insert(self, conn: sqlite3.Connection, cur: sqlite3.Cursor, data: typing.Dict,
                   time_frame: typing.AnyStr, flag: SQLInsertFlags = SQLInsertFlags.PARENT):
        def get_parent_id(pid):
            cur.execute("""SELECT * FROM main.comment WHERE id = ?;""", (pid,))
            return cur.fetchone()

        def get_comment_id(cid):
            cur.execute("SELECT id FROM comment WHERE comment_id = ?", (cid,))
            return cur.fetchone()

        data['subreddit'] = self.__get_subreddits(conn, cur, data['subreddit'])[str(data['subreddit'])]
        tokenizers = self.__get_tokenizers(conn, cur, data['tokenizer_name'])

        if flag == SQLInsertFlags.PARENT:
            result = get_comment_id(data['parent_id'])
            if result is not None:
                parent_id = int(result[0])
                sql = """INSERT INTO comment 
                (content, comment_id, parent_id, subreddit_id, unix, score) VALUES (?, ?, ?, ?, ?, ?);"""
                cur.execute(sql, [data['comment'], data['comment_id'],
                                  parent_id, data['subreddit'],
                                  data['unix'], data['score']])
                if self.tokenize:
                    sql = "SELECT id FROM comment WHERE comment_id=?;"
                    cur.execute(sql, [data['comment_id'], ])
                    result = cur.fetchone()
                    if result is not None:
                        comment_id = int(result[0])
                        sql = """INSERT INTO main.tokenized_comment (tokenized_content, content_id, tokenizer) VALUES (?, ?, ?);"""
                        cur.execute(sql, [data['tokenized_comment'], comment_id, tokenizers[self.tokenizer_name]])
                    else:
                        raise Exception("Unexpected null value.")
            else:
                raise Exception("Unexpected null value.")
        elif flag == SQLInsertFlags.NO_PARENT:
            del data['parent']
            cur.execute("""INSERT INTO comment 
            (content, comment_id, subreddit_id, unix, score) VALUES (?, ?, ?, ?, ?);""",
                        [data['comment'], data['comment_id'], data['subreddit'],
                         data['unix'], data['score']])

            if self.tokenize:
                sql = "SELECT id FROM comment WHERE comment_id = ?;"
                cur.execute(sql, [data['comment_id'], ])
                comment_id = cur.fetchone()
                if comment_id is not None:
                    comment_id = int(comment_id[0])
                    del data['tokenized_parent']
                    cur.execute("""
                    INSERT INTO tokenized_comment (
                    tokenized_content,
                    content_id,
                    tokenizer
                    )
                    VALUES
                    (?, ?, ?);
                    """, [data['tokenized_comment'], comment_id,
                          tokenizers[self.tokenizer_name]])
                else:
                    raise Exception("Unexpected null value.")
        elif flag == SQLInsertFlags.REPLACE_COMMENT:
            result = get_comment_id(data['parent_id'])
            if result is None:
                result = self.__get_from_cid(data['comment_id'], time_frame)
                if not result:
                    raise Exception("Unexpected null value.")
                else:
                    self.sql_insert(conn, cur, result, time_frame, flag=SQLInsertFlags.NO_PARENT)
                    result = get_comment_id(data['parent_id'])
            if result is not None:
                parent_id = int(result[0])
                result = get_parent_id(parent_id)
                if result is None:
                    result = self.__get_from_pid(data['parent_id'], time_frame)
                    if not result:
                        raise Exception("Unexpected null value.")
                    else:
                        self.sql_insert(conn, cur, result, time_frame, flag=SQLInsertFlags.PARENT)
                        result = get_parent_id(parent_id)
                if result is not None:
                    old_comment_id = result[0]
                    cur.execute("DELETE FROM main.tokenized_comment WHERE main.tokenized_comment.content_id = ?",
                                (old_comment_id,))
                    cur.execute("DELETE FROM comment WHERE main.comment.parent_id = ?", (parent_id,))
                    sql = """INSERT INTO comment (content, comment_id, parent_id,  subreddit_id, unix, score) VALUES (?, ?, ?, ?, ?, ?);"""
                    cur.execute(sql, [data['comment'], data['comment_id'], parent_id, data['subreddit'],
                                      data['unix'], data['score']])
                    if self.tokenize:
                        sql = "SELECT id FROM comment WHERE comment_id=?;"
                        cur.execute(sql, [data['comment_id'], ])
                        result = cur.fetchone()
                        if result is not None:
                            comment_id = int(result[0])
                            sql = """INSERT INTO tokenized_comment (tokenized_content, content_id, tokenizer) VALUES (?, ?, ?);"""
                            cur.execute(sql, [data['tokenized_comment'], comment_id, tokenizers[self.tokenizer_name]])
                        else:
                            raise Exception("Unexpected null value.")

                else:
                    raise Exception("Unexpected null value.")
        conn.commit()

    @staticmethod
    def __create_cache():
        if not os.path.exists('./cache/'):
            os.mkdir("./cache/")

    @staticmethod
    def __clear_cache():
        if os.path.exists("./cache/"):
            shutil.rmtree("./cache/")

    def __create_tables(self, time_frame):
        self.__create_cache()
        connection = sqlite3.connect(f'./cache/{time_frame}.db')
        cursor = connection.cursor()
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
        sql4 = """CREATE TABLE IF NOT EXISTS tokenizers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tokenizer_id TEXT NOT NULL);"""
        cursor.execute(sql1)
        cursor.execute(sql2)
        cursor.execute(sql3)
        cursor.execute(sql4)
        connection.commit()
        if self.tokenize:
            sql = """
                INSERT into tokenizers (tokenizer_id) VALUES (?);
            """
            cursor.execute(sql, (self.tokenizer_name,))
            connection.commit()
        return connection, cursor

    def __find_existing_score(self, pid: str, year: str) -> typing.Union[bool, int]:
        if year in self.__samples.keys():
            data_blocks = self.__samples[year]['data']
            for o_data in data_blocks:
                if o_data['parent_id'] == pid:
                    return o_data['score']
            return False
        else:
            return False

    def __get_from_pid(self, pid: str, year: str) -> typing.Union[bool, dict]:
        if year in self.__samples.keys():
            data_blocks = self.__samples[year]['data']
            for o_data in data_blocks:
                if o_data['parent_id'] == pid:
                    return o_data
            return False
        else:
            return False

    def __get_from_cid(self, cid: str, year: str) -> typing.Union[bool, dict]:
        if year in self.__samples.keys():
            data_blocks = self.__samples[year]['data']
            for o_data in data_blocks:
                if o_data['comment_id'] == cid:
                    return o_data
            return False
        else:
            return False

    async def __write(self):  # Prevent the workers from adding more data to samples as we do our work.
        self.logger.debug("Running write.")
        self.lock.acquire()
        if 'total' not in self.at_row.keys():
            self.at_row['total'] = 0
        for time_frame in self.at_row.keys():
            self.at_row['total'] += self.at_row[time_frame]
        if self.at_row['total'] != 0:
            ten_percent_up = (self.at_row['total'] + self.at_row['total'] * 0.1) % self.write_freq
            ten_percent_down = (self.at_row['total'] - self.at_row['total'] * 0.1) % self.write_freq
            if self.at_row['total'] % self.write_freq == 0 or (
                    ten_percent_down <= self.at_row['total'] % self.write_freq <= ten_percent_up):
                self.logger.info("Beginning Write.")
                for time_frame in self.__samples.keys():
                    data_list = self.__samples[time_frame]['data']
                    connection, cursor = self.__create_tables(time_frame)
                    for i, data_block in enumerate(data_list):
                        parent_id = data_block['parent_id']
                        score = data_block['score']
                        parent_body = data_block['parent']
                        if score >= 2:
                            existing_comment_score = self.__find_existing_score(parent_id, time_frame)
                            if existing_comment_score:
                                if score > existing_comment_score:
                                    try:
                                        self.sql_insert(connection, cursor, data_block,
                                                        flag=SQLInsertFlags.REPLACE_COMMENT, time_frame=time_frame)
                                    except Exception as e:
                                        self.logger.error(f"Error on SQL insert into database: {e}")
                                        if self.debug:
                                            raise e
                        else:
                            if parent_body:
                                try:
                                    self.sql_insert(connection, cursor, data_block, flag=SQLInsertFlags.PARENT,
                                                    time_frame=time_frame)
                                except Exception as e:
                                    self.logger.error(f"Error on SQL insert into database: {e}")
                                    if self.debug:
                                        raise e
                            else:
                                try:
                                    self.sql_insert(connection, cursor, data_block, flag=SQLInsertFlags.NO_PARENT,
                                                    time_frame=time_frame)
                                except Exception as e:
                                    self.logger.error(f"Error on SQL insert into database: {e}")
                                    if self.debug:
                                        raise e
                            if i % self.info_freq == 0 and i != 0:
                                self.logger.info(f"{time_frame} rows written: {i}/{self.at_row[time_frame]}")
                    self.logger.info(f"{time_frame} rows written: {self.at_row[time_frame]}/{self.at_row[time_frame]}")
                    connection.close()
                if self.at_row["total"] % self.write_freq == 0 and self.at_row["total"] != 0:
                    self.logger.info(f'{self.at_row["total"]} rows written.')
                # Clear the samples afterwards.
                for time_frame in self.time_frames:
                    self.__samples[time_frame]['data'] = []
        self.lock.release()

    def __worker_thread(self, time_frames):
        running = True
        find_parent_cache = {}

        def find_parent(pid, year, obj: CategoriseReddit) -> typing.Union[bool, typing.AnyStr]:
            if year in obj.__samples.keys():
                data_list = obj.__samples[year]['data']
                if pid not in find_parent_cache.keys():
                    for o_data in data_list:
                        if o_data['parent_id'] == pid:
                            find_parent_cache[pid] = o_data['parent']
                            return o_data['parent']
                else:
                    return find_parent_cache[pid]
                return False
            else:
                return False

        if self.tokenize:
            tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)
        else:
            tokenizer = None
        while running:
            if 'total' not in self.at_row.keys():
                self.at_row['total'] = 0
            elif 'mean' not in self.at_row.keys():
                self.at_row['mean'] = 0
            for time_frame in time_frames:
                self.at_row[time_frame] = 0
                if os.path.exists(os.path.join(self.data_file_path, f'RC_{time_frame}.zst')):
                    compression_mode = "zst"
                elif os.path.exists(os.path.join(self.data_file_path, f'RC_{time_frame}.bz2')):
                    compression_mode = "bz2"
                else:
                    raise FileNotFoundError(f"{time_frame} was not found in {self.data_file_path}")
                self.__samples[time_frame] = {'data': []}
                if compression_mode == "bz2":
                    text_stream = bz2.open(os.path.join(self.data_file_path, f"RC_{time_frame}.{compression_mode}"))
                else:
                    f = open(os.path.join(self.data_file_path, f"RC_{time_frame}.{compression_mode}"), "rb")
                    dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
                    stream_reader = dctx.stream_reader(f, read_size=int(1.953e+6))
                    text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                for i, row in enumerate(text_stream):
                    if i >= self.at_row[time_frame]:
                        row = json.loads(row)
                        parent_id = row['parent_id']
                        body = row['body']
                        created_utc = row['created_utc']
                        score = row['score']
                        if compression_mode == "zst":
                            comment_id = 't1_' + row['id']
                        else:
                            comment_id = row['name']
                        subreddit = row['subreddit']
                        parent_body = find_parent(parent_id, time_frame, self)
                        data = {
                            'parent_id': parent_id,
                            'comment_id': comment_id,
                            'parent': False if not parent_body else parent_body,
                            'comment': body,
                            'tokenized_parent': '' if not self.tokenize or not parent_body else base64.b64encode(
                                pickle.dumps(tokenizer.encode(parent_body))),
                            'tokenized_comment': '' if not self.tokenize else base64.b64encode(
                                pickle.dumps(tokenizer.encode(body))),
                            'subreddit': subreddit,
                            'unix': int(created_utc),
                            'score': score}
                        self.__samples[time_frame]['data'].append(data)
                        # Increment the read_rows data.
                        self.at_row[time_frame] += 1
                    if self.at_row[time_frame] % self.info_freq == 0:
                        self.logger.info(f"Time Frame: {time_frame}, {i} samples read.")

            running = False

    @staticmethod
    def __chunks(input_list, size_chunk):
        for i in range(0, len(input_list), size_chunk):
            yield input_list[i:i + size_chunk]

    def __clear_stopped(self):
        for i, thread in enumerate(self.threads):
            if thread.stopped:
                del self.threads[i]

    def stop(self):
        self.stopped = True
        self.logger.info("Stopping all threads.")
        self.loop.stop()
        self.__clear_stopped()
        for thread in self.threads:
            thread.stop()
        self.__clear_stopped()
        self.__clear_cache()

    def __setup_threads(self):
        for time_frame in self.time_frames:
            self.threads.append(StoppableThread(target=self.__worker_thread, args=(time_frame,), bucket=self.bucket))

    async def __manager(self):
        """Keep track of the threads & manage when to write data to the database."""
        self.logger.debug("Running Manager loop.")
        try:
            exc = self.bucket.get(block=False)
        except queue.Empty:
            self.logger.debug("Exception Queue Empty")
            pass
        else:
            exc_type, exc_obj, exc_trace = exc
            # Write code to deal with exceptions here.
            # For example could call self.stop() & raise the error.
            if self.debug:
                self.logger.error(f"Type: {exc_type}, Exception: {exc_obj}, Trace: {exc_trace}")
                self.stop()
                raise exc_obj
            else:
                self.logger.error(f"Type: {exc_type}, Exception: {exc_obj}, Trace: {exc_trace}")
                self.stop()
        if all([(thread.is_alive()) or (not thread.started) for thread in self.threads]):
            self.logger.debug("All threads are alive.")
            await self.__write()
            self.logger.debug("Finished write.")
        else:
            self.stopped = True
            self.stop()

        await asyncio.Future()

    def start(self):
        self.__clear_cache()
        self.logger.info("Starting all threads")
        if self.process_year:
            self.__setup_threads()
            for thread in self.threads:
                thread.start()
        else:
            self.threads.append(
                StoppableThread(target=self.__worker_thread, args=(self.time_frames,), bucket=self.bucket))

        self.threads = self.threads[::-1]
        for thread in self.threads:
            if not thread.started:
                thread.start()

        asyncio.run(self.__manager())


if __name__ == "__main__":
    reddit = CategoriseReddit(debug=True, time_frame='2020-01',
                              data_file_path="D:\\Datasets\\reddit_data\\temp_download\\",
                              database_file_path="D:\\Datasets\\reddit_data\\databases\\",
                              tokenize_at_process_time=True,
                              tokenizer_path="../Tokenizer-3", write_freq=10_000)
    reddit.start()
