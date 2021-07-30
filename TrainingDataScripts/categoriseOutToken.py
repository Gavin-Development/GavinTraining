import os
import sqlite3
import glob
import shutil
import pandas as pd


TEMP_DIR = './token_temp/'
timeframes = glob.glob("D:/Datasets/reddit_data/databases/*.db")
timeframes = [os.path.basename(timeframe) for timeframe in timeframes]
timeframes = timeframes[::-1]


def sort_out(time_frame):
    tokenizer_name = input("Please enter the tokenizer_name: ")
    for t_frame in time_frame:
        try:
            print(f"Staring on: {t_frame}")
            try:
                os.mkdir(TEMP_DIR)
            except FileExistsError:
                pass

            try:
                os.remove(f"D:/Datasets/reddit_data/files/{tokenizer_name}.*")
            except FileNotFoundError:
                pass
            except OSError:
                pass
            tokenizer_name_formatted = tokenizer_name.split('.')[0].replace('-', '_')
            shutil.copy('D:/Datasets/reddit_data/databases/{}'.format(t_frame), '{}{}'.format(TEMP_DIR, t_frame))
            limit = 3_000_000
            connection = sqlite3.connect('{}{}'.format(TEMP_DIR, t_frame))
            last_unix = 0
            cur_length = limit
            count = 0
            write_inputs = []
            write_outputs = []

            while cur_length == limit:
                try:
                    df = pd.read_sql(
                        """SELECT * FROM  tokenized_comment_data WHERE unix > {} and parent_tokenized NOT NULL and score > 0 and tokenizer_name = "{}" ORDER BY unix ASC LIMIT {}""".format(
                            last_unix, tokenizer_name_formatted, limit), connection)
                except Exception as error:
                    print(f"Timeframe: {t_frame} Error: {error}")
                else:
                    last_unix = df.tail(1)['unix'].values[0]
                    cur_length = len(df)
                    for content in df['parent_tokenized'].values:
                        write_inputs.append(str(content))

                    for content in df['comment_tokenized'].values:
                        write_outputs.append(str(content))

                    count += 1
                    if count % 10 == 0:
                        print(f"{count * limit} rows down so far.")

                with open(f"D:\\Datasets\\reddit_data\\files\\{tokenizer_name}.from", "a", encoding='utf8') as f:
                    for sentence in write_inputs:
                        f.write(sentence + '\n')
                    f.close()
                write_inputs = []

                with open(f"D:\\Datasets\\reddit_data\\files\\{tokenizer_name}.to", "a", encoding='utf-8') as f:
                    for sentence in write_outputs:
                        f.write(sentence + '\n')
                    f.close()

                write_outputs = []
            connection.close()
            try:
                os.remove('{}{}'.format(TEMP_DIR, t_frame))
            except PermissionError:
                continue
            print(f"{t_frame} finished.")
        except Exception as err:
            print(f"Error {err} on {t_frame}")
            continue


if __name__ == "__main__":
    try:
        shutil.rmtree(TEMP_DIR)
    except FileNotFoundError or OSError:
        pass
    sort_out(timeframes)
    print("Clearing up.")
    try:
        shutil.rmtree(TEMP_DIR)
    except FileNotFoundError or OSError:
        pass
    finished = True
