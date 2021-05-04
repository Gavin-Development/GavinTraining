"""
This script allows me to tokenize all the entries on a database into their own table within the database.
Cutting down on preprocessing time. Because they just have to fetch the data from output files. Rather then computing
the Tokens for each sentence.
"""
import glob
import os
import shutil
import sqlite3
import pickle
import base64

import tensorflow_datasets as tfds
import pandas as pd

if not os.path.exists('./temp/'):
    os.mkdir('./temp/')

databases = glob.glob("D:/Datasets/reddit_data/databases/*.db")
databases = [os.path.basename(database) for database in databases]
subword_file = input("Please enter the name of the subword file: ")
subword_file_path = f"../{subword_file}"
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(subword_file_path)
print(f"Databases to process: {len(databases)}\nVocab Size on the Tokenizer: {tokenizer.vocab_size}")
for database in databases:
    runs = 0
    try:
        print(f"Starting Work on: {database}")
        limit = 3_000_000
        shutil.copy('D:/Datasets/reddit_data/databases/{}'.format(database), './temp/{}'.format(database))
        connection = sqlite3.connect('./temp/{}'.format(database))
        sql = "CREATE TABLE IF NOT EXISTS tokenized_comment_data_new (id INTEGER PRIMARY KEY AUTOINCREMENT, parent_id TEXT, comment_id TEXT, parent_tokenized TEXT, comment_tokenized TEXT, subreddit TEXT, unix INT, score INT, tokenizer_name TEXT)"
        cursor = connection.cursor()
        cursor.execute(sql)
        connection.commit()
        last_unix = 0
        cur_length = limit

        while cur_length == limit:
            runs += 1
            try:
                df = pd.read_sql(
                    "SELECT * FROM parent_reply WHERE unix > {} and parent NOT NULL and score > 0 ORDER BY unix ASC LIMIT {}".format(
                        last_unix, limit), connection)
            except Exception as e:
                print(f"Timeframe: {database} Error: {e}")
            else:
                last_unix = df.tail(1)['unix'].values[0]
                cur_length = len(df)
                i = 0
                for parent, comment in list(zip(df['parent'].values, df['comment'].values)):
                    parent_id = df['parent_id'].values[i]
                    comment_id = df['comment_id'].values[i]
                    subreddit = df['subreddit'].values[i]
                    unix = df['unix'].values[i]
                    score = df['score'].values[i]
                    t_parent = tokenizer.encode(parent)
                    t_parent = pickle.dumps(t_parent)
                    t_parent = base64.b64encode(t_parent)
                    t_comment = tokenizer.encode(comment)
                    t_comment = pickle.dumps(t_comment)
                    t_comment = base64.b64encode(t_comment)
                    sql = """INSERT INTO tokenized_comment_data_new (parent_id, comment_id, parent_tokenized, comment_tokenized, subreddit, unix, score, tokenizer_name) VALUES ("{}", "{}", "{}", "{}", "{}", {}, {}, "{}");""".format(
                        parent_id + '_' + subword_file.split('.')[0].replace('-', '_'), comment_id, t_parent, t_comment, subreddit, unix, score,
                        subword_file.split('.')[0].replace('-', '_'))
                    cursor.execute(sql)
                    i += 1
        print(f"Starting Vacuum on: {database}")
        connection.commit()
        cursor.execute("VACUUM")
        connection.commit()
        print(f"Finished Vacuum on: {database}")
        connection.close()
        shutil.copy('./temp/{}'.format(database), 'D:/Datasets/reddit_data/databases/{}'.format(database))
        print(f"Finished Work on: {database}")
    except sqlite3.IntegrityError or sqlite3.OperationalError as e:
        print(f"Error: {e} Run: {runs}")
        try:
            connection.close()
        except Exception as e:
            print(f"Error: {e}")
        shutil.copy('./temp/{}'.format(database), 'D:/Datasets/reddit_data/databases/{}'.format(database))
