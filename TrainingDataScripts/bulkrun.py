"""
This is used to bulk run commands on my Databases w/ the reddit data
"""
import glob
import os
import sqlite3


timeframes = glob.glob("D:/Datasets/reddit_data/databases/*.db")
timeframes = [os.path.basename(timeframe) for timeframe in timeframes]
total_rows = 0
count_total = 0
is_counting = True
sql = """SELECT COUNT(*) FROM comment WHERE content LIKE "%nigga%" OR content LIKE "%nigger%";"""
for timeframe in timeframes:
    connection = sqlite3.connect('D:/Datasets/reddit_data/databases/{}'.format(timeframe))
    try:
        count = 0
        cursor = connection.cursor()
        cursor.execute(sql)
        if is_counting:
            try:
                count = cursor.fetchone()[0]
                count_total += int(count)
            except Exception as e:
                print(f"{timeframe} On Count Error: {e}")
        total_rows += cursor.rowcount
        connection.commit()
    except Exception as e:
        print(f"{timeframe} Error: {e}")
    else:
        print(f"{timeframe} SQL completed successfully: {count if is_counting else ''}")
    finally:
        connection.close()

print(f"Total Row counts: {abs(total_rows)}")
if is_counting:
    print(f"Total: {count_total}")
