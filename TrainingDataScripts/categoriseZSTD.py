import io
import os
import sqlite3
import json
from datetime import datetime
import shutil
import sys
import zstandard

timeFrame = sys.argv[1]
last_utc = 0
sql_transaction = []

if not os.path.exists('./cache/'):
    os.mkdir('./cache/')
connection = sqlite3.connect(f'./cache/{timeFrame}.db')
c = connection.cursor()
start_row = 0
cleanup = 1000000


def create_table():
    c.execute(
        "CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")


def format_data(data):
    data = data.replace('\n', ' newlinechar ').replace('\r', ' newlinechar ').replace('"', "'")
    return data


def find_parent(pid):
    try:
        sql_to_execute = "SELECT comment FROM parent_reply WHERE comment_id = '{}' LIMIT 1".format(pid)
        c.execute(sql_to_execute)
        result = c.fetchone()
        if result is not None:
            return result[0]
        else:
            return False
    except Exception as e:
        print(str(e))
        return False


def find_existing_score(pid):
    try:
        sql_to_execute = "SELECT score FROM parent_reply WHERE parent_id = '{}' LIMIT 1".format(pid)
        c.execute(sql_to_execute)
        result = c.fetchone()
        if result is not None:
            return result[0]
        else:
            return False
    except Exception as e:
        print(str(e))
        return False


def acceptable(data):
    if len(data.split(' ')) > 50 or len(data) < 1:
        return False
    elif len(data) > 1000:
        return False
    elif data == '[deleted]':
        return False
    elif data == '[removed]':
        return False
    else:
        return True


def transaction_bldr(p_sql):
    global sql_transaction
    sql_transaction.append(p_sql)
    if len(sql_transaction) > 1000:
        c.execute('BEGIN TRANSACTION')
        for s in sql_transaction:
            try:
                c.execute(s)
            except:
                pass
        connection.commit()
        sql_transaction = []


# noinspection PyShadowingNames
def sql_insert_replace_comment(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """UPDATE parent_reply SET parent_id = ?, comment_id = ?, parent = ?, comment = ?, subreddit = ?, unix = ?, score = ? WHERE parent_id =?;""".format(
            parentid, commentid, parent, comment, subreddit, int(time), score, parentid)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


# noinspection PyShadowingNames
def sql_insert_has_parent(commentid, parentid, parent, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, parent, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}","{}",{},{});""".format(
            parentid, commentid, parent, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


# noinspection PyShadowingNames
def sql_insert_no_parent(commentid, parentid, comment, subreddit, time, score):
    try:
        sql = """INSERT INTO parent_reply (parent_id, comment_id, comment, subreddit, unix, score) VALUES ("{}","{}","{}","{}",{},{});""".format(
            parentid, commentid, comment, subreddit, int(time), score)
        transaction_bldr(sql)
    except Exception as e:
        print('s0 insertion', str(e))


if __name__ == "__main__":
    try:
        create_table()
        row_counter = 0
        paired_rows = 0

        with open(f"D:\\Datasets\\reddit_data\\{timeFrame[:4]}\\RC_{timeFrame}.zst", "rb") as f:
            dctx = zstandard.ZstdDecompressor()
            stream_reader = dctx.stream_reader(f)
            text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8')
            for row in text_stream:
                row_counter += 1
                # print(row)
                row = json.loads(row)
                parent_id = row['parent_id']
                body = format_data(row['body'])
                created_utc = row['created_utc']
                score = row['score']
                comment_id = 't1_' + row['id']
                subreddit = row['subreddit']
                parent_data = find_parent(parent_id)

                if score >= 2:
                    existing_comment_score = find_existing_score(parent_id)
                    if existing_comment_score:
                        if score > existing_comment_score:
                            if acceptable(body):
                                sql_insert_replace_comment(comment_id, parent_id, parent_data, body, subreddit,
                                                           created_utc,
                                                           score)
                else:
                    if acceptable(body):
                        if parent_data:
                            sql_insert_has_parent(comment_id, parent_id, parent_data, body, subreddit, created_utc,
                                                  score)
                            paired_rows += 1
                        else:
                            sql_insert_no_parent(comment_id, parent_id, body, subreddit, created_utc, score)
                if row_counter % 100000 == 0:
                    print(
                        'Total Rows Read: {}, Paired Rows: {}, Time: {}, File: {}'.format(row_counter, paired_rows,
                                                                                          str(datetime.now()),
                                                                                          str(timeFrame)))
                if row_counter > start_row:
                    if row_counter % cleanup == 0:
                        print("Cleaning up!")
                        sql = "DELETE FROM parent_reply WHERE parent IS NULL"
                        c.execute(sql)
                        connection.commit()
        print("Starting VACUUM")
        c.execute("VACUUM")
        connection.commit()
        connection.close()
        print("Moving File...")
        shutil.move(f'./cache/{timeFrame}.db', f'D:\\Datasets\\reddit_data\\databases\\{timeFrame}.db')
    except Exception as e:
        if e:
            print(f"Error: {e}")
        connection.close()
        os.remove(f'./cache/{timeFrame}.db')
        print("Quitting....")
