"""
This is used to bulk run commands on my Databases w/ the reddit data
"""
import glob
import os
import sqlite3


timeframes = glob.glob("D:/Datasets/reddit_data/databases/*.db")
timeframes = [os.path.basename(timeframe) for timeframe in timeframes]
total_rows = 0
sql = """DELETE FROM parent_reply WHERE comment LIKE "%dogetipbot%" or comment LIKE "%shiny iv%" or comment like "%retard%" or main.parent_reply.comment LIKE "%nigga%";"""
for timeframe in timeframes:
    connection = sqlite3.connect('D:/Datasets/reddit_data/databases/{}'.format(timeframe))
    cursor = connection.cursor()
    cursor.execute(sql)
    total_rows += cursor.rowcount
    connection.commit()
    print(f"{timeframe} SQL completed successfully")

print(f"Total Row counts: {total_rows}")
