"""
This is used to bulk run commands on my Databases w/ the reddit data
"""
import glob
import os
import sqlite3


timeframes = glob.glob("D:/Datasets/reddit_data/databases/*.db")
timeframes = [os.path.basename(timeframe) for timeframe in timeframes]
total_rows = 0
sql = """DROP TABLE tokenized_comment_data_new;"""
for timeframe in timeframes:
    connection = sqlite3.connect('D:/Datasets/reddit_data/databases/{}'.format(timeframe))
    try:
        cursor = connection.cursor()
        cursor.execute(sql)
        total_rows += cursor.rowcount
        connection.commit()
    except Exception as e:
        print(f"{timeframe} Error: {e}")
    else:
        print(f"{timeframe} SQL completed successfully")

print(f"Total Row counts: {abs(total_rows)}")
