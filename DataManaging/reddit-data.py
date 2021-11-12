import base64
import bz2
import io
import json
import logging
import os
import pickle
import queue
import sched
import shutil
import sqlite3
import time
import typing
import zstandard

import tensorflow_datasets as tfds

from datetime import datetime
from utils import ErrorCatcher, ArgumentError, StoppableThread, SQLInsertFlags

logging.basicConfig(level=logging.WARNING, format='%(process)d-%(levelname)s %(asctime)s - %(message)s',
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
                 write_freq: typing.Union[int, float] = 120, debug=False):
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
        :param write_freq: typing.Union[int, float]
            Time in seconds in which the data in memory should be written to database, default is
            every 120 seconds (2 minutes)
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
        self.tokenizer_path = tokenizer_path
        self.tokenizer_name = os.path.basename(tokenizer_path).split('.')[0]
        self.write_freq = write_freq
        self.logger = self.__metaclass__.logger
        self.bucket = queue.Queue()
        self.debug = debug

        if self.process_year:
            self.time_frames = [year + "-%.2d" % i for i in range(1, 13)]
            thread_generator = self.__chunks(self.time_frames, ((len(self.time_frames) // self.threads) + 1))
            self.time_frames = [next(thread_generator) for _ in range(self.threads)]
        else:
            self.time_frame = [time_frame]

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
        self.__at_row = {}
        # Internal attribute for whether the read threads should continue
        self.__read = True
        # Keep list of all threads running.
        self.threads = []
        # Management Thread
        self.management_thread = None
        self.schedule = sched.scheduler(time.time, time.sleep)

    @staticmethod
    def get_subreddits(cur: sqlite3.Cursor) -> typing.Dict[str, int]:
        """Fetch the subreddits."""
        cur.execute("""SELECT * FROM main.subreddits;""")
        results = cur.fetchall()
        return {result[1]: result[0] for result in results}

    @staticmethod
    def get_tokenizers(cur: sqlite3.Cursor) -> typing.Dict[str, int]:
        """Fetch the tokenizers."""
        cur.execute("""SELECT * FROM main.tokenizers;""")
        results = cur.fetchall()
        return {result[1]: result[0] for result in results}

    def sql_insert(self, conn: sqlite3.Connection, cur: sqlite3.Cursor, data: typing.Dict,
                   flag: SQLInsertFlags = SQLInsertFlags.PARENT):
        subreddits = self.get_subreddits(cur)
        tokenizers = self.get_tokenizers(cur)
        if data['subreddit'] not in subreddits.keys():
            try:
                cur.execute("INSERT INTO main.subreddits (name) VALUES (?)", (data['subreddit'],))
            except [sqlite3.Error, Exception] as e:
                self.logger.error(f"Error inserting subreddit {data['subreddit']} into database: {e}")
                if self.debug:
                    raise e
            conn.commit()
            subreddits = self.get_subreddits(cur)
        if self.tokenizer_name not in tokenizers.keys():
            try:
                cur.execute("INSERT INTO main.tokenizers (tokenizer_id) VALUES (?)", (self.tokenizer_name, ))
            except [sqlite3.Error, Exception] as e:
                self.logger.error(f"Error inserting subreddit {data['subreddit']} into database: {e}")
                if self.debug:
                    raise e
            conn.commit()
            tokenizers = self.get_tokenizers(cur)
        else:
            data['subreddit'] = subreddits[data['subreddit']]
        if not self.tokenize:
            del data['tokenized_parent'], data['tokenized_comment']
        elif self.tokenize:
            del data['parent'], data['comment']
            data['tokenizer_name'] = os.path.basename(self.tokenizer_path)
        if flag == SQLInsertFlags.PARENT:
            sql = """SELECT id FROM comment WHERE comment_id=%(parent_id)s;"""
            cur.execute(sql, {'parent_id': data['parent_id']})
            result = cur.fetchone()
            if result is not None:
                parent_id = int(result[0])
                sql = """INSERT INTO comment 
                (content, comment_id, parent_id, subreddit_id, unix, score) VALUES (
                %(comment)s, 
                %(comment_id)s, 
                %(parent_id)s,
                %(subreddit_id)s, 
                %(unix)s, 
                %(score)s);"""
                cur.execute(sql, {'comment': data['comment'], 'comment_id': data['comment_id'],
                                  'parent_id': parent_id, 'subreddit_id': data['subreddit'],
                                  'unix': data['unix'], 'score': data['score']})
                if self.tokenize:
                    sql = "SELECT id FROM comment WHERE comment_id=%(comment_id)s;"
                    cur.execute(sql, {'comment_id': data['comment_id']})
                    result = cur.fetchone()
                    if result is not None:
                        comment_id = int(result[0])
                        sql = """INSERT INTO main.tokenized_comment (
                        tokenized_content, 
                        content_id, 
                        tokenizer) 
                        VALUES 
                        (
                        %(tokenized_content)s, 
                        %(content_id)s, 
                        %(tokenizer)s);"""
                        cur.execute(sql, {'tokenized_comment': data['tokenized_comment'], 'content_id': comment_id, 'tokenizer_name': tokenizers[self.tokenizer_name]})
                    else:
                        raise Exception("Unexpected null value.")
            else:
                raise Exception("Unexpected null value.")
        elif flag == SQLInsertFlags.NO_PARENT:
            del data['parent']
            cur.execute("""INSERT INTO comment 
            (content, comment_id, subreddit_id, unix, score) VALUES (
            %(content)s, 
            %(comment_id)s, 
            %(subreddit_id)s, 
            %(unix)s, 
            %(score)s);""", {'content': data['comment'], 'comment_id': data['comment_id'],
                             'subreddit_id': data['subreddit'], 'unix': data['unix'],
                             'score': data['score']})

            if self.tokenize:
                sql = "SELECT id FROM comment WHERE comment_id=%(comment_id)s;"
                cur.execute(sql, {'comment_id': data['comment_id']})
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
                    (%(tokenized_content)s, %(content_id)s, %(tokenizer_name)s);
                    """, {'tokenized_content': data['tokenized_parent'], 'content_id': comment_id, 'tokenizer_name': tokenizers[self.tokenizer_name]})
                else:
                    raise Exception("Unexpected null value.")
        elif flag == SQLInsertFlags.REPLACE_COMMENT:
            sql = "SELECT id FROM comment WHERE comment_id = ?"
            cur.execute(sql, (data['parent_id'], ))
            result = cur.fetchone()
            if result is not None:
                parent_id = int(result[0])
                cur.execute("SELECT id FROM comment WHERE main.comment.parent_id = ?", (parent_id, ))
                result = cur.fetchone()
                if result is not None:
                    old_comment_id = result[0]
                    cur.execute("DELETE FROM main.tokenized_comment WHERE main.tokenized_comment.content_id = ?", (old_comment_id, ))
                    cur.execute("DELETE FROM comment WHERE main.comment.parent_id = ?", (parent_id, ))
                    sql = """INSERT INTO comment (
                    content, 
                    comment_id, 
                    parent_id, 
                    subreddit_id, 
                    unix, 
                    score) VALUES (
                    %(comment)s, 
                    %(comment_id)s, 
                    %(parent_id)s,
                    %(subreddit_id)s, 
                    %(unix)s,
                    %(score)s);"""
                    cur.execute(sql, {'comment': data['comment'], 'comment_id': data['comment_id'],
                                      'parent_id': parent_id, 'subreddit_id': data['subreddit'],
                                      'unix': data['unix'], 'score': data['score']})
                    if self.tokenize:
                        sql = "SELECT id FROM comment WHERE comment_id=%(comment_id)s;"
                        cur.execute(sql, {'comment_id': data['comment_id']})
                        result = cur.fetchone()
                        if result is not None:
                            comment_id = int(result[0])
                            sql = """INSERT INTO main.tokenized_comment (
                            tokenized_content, 
                            content_id, 
                            tokenizer) 
                            VALUES 
                            (
                            %(tokenized_content)s, 
                            %(content_id)s, 
                            %(tokenizer)s);"""
                            cur.execute(sql, {'tokenized_comment': data['tokenized_comment'], 'content_id': comment_id,
                                              'tokenizer_name': tokenizers[self.tokenizer_name]})
                        else:
                            raise Exception("Unexpected null value.")

                else:
                    raise Exception("Unexpected null value.")
            else:
                raise Exception("Unexpected null value.")
        conn.commit()

    @staticmethod
    def __create_cache():
        if not os.path.exists('./cache/'):
            os.mkdir("./cache/")

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

    def __write(self, sc):
        def find_existing_score(obj: CategoriseReddit, pid: str, year: str) -> typing.Union[bool, int]:
            if year in obj.__samples.keys():
                data_blocks = obj.__samples[year]['data']
                for o_data in data_blocks:
                    if o_data['parent_id'] == pid:
                        return o_data['score']
                return False
            else:
                return False

        self.__read = False  # Prevent the workers from adding more data to samples as we do our work.
        for time_frame in self.__samples.keys():
            data_list = self.__samples[time_frame]['data']
            connection, cursor = self.__create_tables(time_frame)
            for data_block in data_list:
                parent_id = data_block['parent_id']
                score = data_block['score']
                parent_body = data_block['parent']
                if score >= 2:
                    existing_comment_score = find_existing_score(self, parent_id, time_frame)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            try:
                                self.sql_insert(connection, cursor, data_block, flag=SQLInsertFlags.REPLACE_COMMENT)
                            except [sqlite3.Error, Exception] as e:
                                self.logger.error(f"Error on SQL insert into database: {e}")
                                if self.debug:
                                    raise e
                else:
                    if parent_body:
                        try:
                            self.sql_insert(connection, cursor, data_block, flag=SQLInsertFlags.PARENT)
                        except [sqlite3.Error, Exception] as e:
                            self.logger.error(f"Error on SQL insert into database: {e}")
                            if self.debug:
                                raise e
                    else:
                        try:
                            self.sql_insert(connection, cursor, data_block, flag=SQLInsertFlags.NO_PARENT)
                        except [sqlite3.Error, Exception] as e:
                            self.logger.error(f"Error on SQL insert into database: {e}")
                            if self.debug:
                                raise e

        sc.enter(self.write_freq, 1, self.__write, (sc,))

    def __worker_thread(self, time_frames):
        samples = {}
        running = True

        def find_parent(pid, year) -> typing.Union[bool, typing.AnyStr]:
            if year in samples.keys():
                data_list = samples[year]['data']
                for o_data in data_list:
                    if o_data['commend_id'] == pid:
                        return o_data['comment']
                return False
            else:
                return False

        def merge_samples(obj: CategoriseReddit):
            for key, value in samples.items():
                if key not in obj.__samples.keys():
                    obj.__samples[key] = value

        if self.tokenize:
            tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(self.tokenizer_path)
        else:
            tokenizer = None
        while running:
            if 'total' not in self.__at_row.keys():
                self.__at_row['total'] = 0
            elif 'mean' not in self.__at_row.keys():
                self.__at_row['mean'] = 0
            for time_frame in time_frames:
                self.__at_row[time_frame] = 0
                if os.path.exists(os.path.join(self.data_file_path, f'{time_frame}.zst')):
                    compression_mode = "zst"
                elif os.path.exists(os.path.join(self.data_file_path, f'{time_frame}.bz2')):
                    compression_mode = "bz2"
                else:
                    raise FileNotFoundError(f"{time_frame} was not found in {self.data_file_path}")
                samples[time_frame] = {'data': []}
                if self.__read:
                    if compression_mode == "bz2":
                        text_stream = bz2.open(os.path.join(self.data_file_path, f"{time_frame}.{compression_mode}"))
                    else:
                        f = open(os.path.join(self.data_file_path, f"{time_frame}.{compression_mode}"), "rb")
                        dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
                        stream_reader = dctx.stream_reader(f, read_size=int(1.953e+6))
                        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
                    for i, row in enumerate(text_stream):
                        if i >= self.__at_row[time_frame] and self.__read:
                            row = json.loads(row)
                            parent_id = row['parent_id']
                            body = row['body']
                            created_utc = row['created_utc']
                            score = row['score']
                            comment_id = row['name']
                            subreddit = row['subreddit']
                            parent_body = find_parent(parent_id, time_frame)
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
                                'score': score
                            }
                            samples[time_frame]['data'].append(data)
                            # Increment the read_rows data.
                            self.__at_row[time_frame] += 1
                        # Pause the reading process if __read is False.
                        elif not self.__read:
                            merge_samples(self)
                            while not self.__read:
                                continue

            running = False
            merge_samples(self)

    @staticmethod
    def __chunks(input_list, size_chunk):
        for i in range(0, len(input_list), size_chunk):
            yield input_list[i:i + size_chunk]

    def __clear_stopped(self):
        for i, thread in enumerate(self.threads):
            if thread.stopped:
                del self.threads[i]

    def stop(self):
        self.__clear_stopped()
        for thread in self.threads:
            thread.stop()
        self.management_thread.stop()
        self.management_thread = None
        self.__clear_stopped()

    def __setup_threads(self):
        for time_frame in self.time_frames:
            self.threads.append(StoppableThread(target=self.__worker_thread, args=(time_frame,), bucket=self.bucket))

    def __manager(self):
        """Keep track of the threads & manage when to write data to the database."""
        # Schedule the checker writer for every 30 seconds W/ priority 1.
        self.schedule.enter(self.write_freq, 1, self.__write, (self.schedule,))
        while True:
            try:
                exc = self.bucket.get(block=False)
            except queue.Empty:
                pass
            else:
                exc_type, exc_obj, exc_trace = exc
                # Write code to deal with exceptions here.
                # For example could call self.stop() & raise the error.
                pass
            if all([thread.isAlive() for thread in self.threads]):
                continue
            else:
                break

    def start(self):
        if not self.debug:
            if self.process_year:
                self.__setup_threads()
                for thread in self.threads:
                    thread.start()
            else:
                self.threads.append(
                    StoppableThread(target=self.__worker_thread, args=(self.time_frame,), bucket=self.bucket))
            self.management_thread = StoppableThread(target=self.__manager, bucket=self.bucket)
        else:
            self.management_thread = StoppableThread(target=self.__manager, bucket=self.bucket)
            self.management_thread.start()
            self.__worker_thread(self.time_frame)
