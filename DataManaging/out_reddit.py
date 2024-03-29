import os
import shutil
import sqlite3
import logging
import typing
import sys
import glob
import tqdm

import pandas as pd

DEBUG = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format='%(process)d-%(levelname)s %(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

logger = logging.getLogger(__name__)


def main(database_path: str, time_frames: typing.List[str], file_output_path: str, file_name: str):
    no_lines = 0
    for time_frame in time_frames:
        logging.info(f"{time_frame} Outputting {time_frame}")
        if os.path.exists(database_path) and os.path.exists(
                os.path.join(database_path, time_frame)):
            if not os.path.exists('./cache1/'):
                os.mkdir('./cache1/')
            shutil.copy(os.path.join(database_path, time_frame), './cache1/')
            limit = 1_500_000
            offset = 0
            connection = sqlite3.connect('./cache1/' + time_frame)
            c = connection.cursor()
            cur_length = limit
            count = 0

            filename_input = os.path.join(file_output_path, file_name + ".from")
            filename_output = os.path.join(file_output_path, file_name + ".to")

            while cur_length == limit:
                logger.info(f"{time_frame} Wrote {no_lines} lines so far.")
                try:
                    df = pd.read_sql(
                        f"SELECT c.id, comment.id, c.content, comment.content FROM comment INNER JOIN comment as c ON comment.parent_id = c.id LIMIT {limit} OFFSET {limit * offset};",
                        connection)
                except Exception as e:
                    logger.error(f"{time_frame} Pandas read sql error: {e}")
                else:
                    cur_length = len(df)
                    sql_inputs = []
                    sql_outputs = []

                    for data in df.itertuples():
                        sql_inputs.append(data[3])
                        sql_outputs.append(data[4])

                    count += 1
                    offset += 1

                    with open(filename_input, 'a', encoding="utf-8") as f:
                        if len(sql_inputs) <= 0:
                            logger.warning(f"{time_frame} No inputs found for {cur_length} comments.")
                            f.close()
                            continue
                        else:
                            logger.info(f"{time_frame} Writing inputs to {filename_input}")
                            for sentence in tqdm.tqdm(sql_inputs, total=len(sql_inputs),
                                                      desc=f"{time_frame} Writing inputs to {filename_input}",
                                                      unit="samples"):
                                f.write(sentence + "\n")
                                no_lines += 1
                            f.close()

                    with open(filename_output, 'a', encoding="utf-8") as f:
                        if len(sql_outputs) <= 0:
                            logger.warning(f"{time_frame} No outputs found for {cur_length} comments.")
                            f.close()
                            continue
                        else:
                            logger.info(f"{time_frame} Writing outputs to {filename_output}")
                            for sentence in tqdm.tqdm(sql_outputs, total=len(sql_outputs),
                                                      desc=f"{time_frame} Writing inputs to {filename_input}",
                                                      unit="samples"):
                                f.write(sentence + "\n")
                                no_lines += 1
                            f.close()

            logger.info(f"{time_frame} Finished outputting {time_frame}")
            c.close()
            connection.close()
            os.remove('./cache1/' + time_frame)

        else:
            logger.error(f"{time_frame} Database not found")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 get_comments.py <database_path> <file_output_path> <file_name>")
        exit(1)
    glob_path = os.path.join(sys.argv[1], "*.db")
    timeframes = glob.glob(glob_path)
    timeframes = [os.path.basename(timeframe) for timeframe in timeframes]
    timeframes = timeframes[::-1]
    main(sys.argv[1], timeframes, sys.argv[2], sys.argv[3])
