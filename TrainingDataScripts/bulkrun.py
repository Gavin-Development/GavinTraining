"""
This is used to bulk run commands on my Databases w/ the reddit data
"""
import glob
import os
import sqlite3


timeframes = glob.glob("D:/Datasets/reddit/databases/*.db")
timeframes = [os.path.basename(timeframe) for timeframe in timeframes]
total_rows = 0
count_total = 0
is_counting = True
is_select = False
sql_main = """SELECT COUNT(c.id) FROM comment as c;"""
for timeframe in timeframes:
    connection = sqlite3.connect('D:/Datasets/reddit/databases/{}'.format(timeframe))
    cursor = connection.cursor()
    try:
        count = 0
        cursor.execute(sql_main)
        if is_counting:
            try:
                count = cursor.fetchone()[0]
                count_total += int(count)
            except Exception as e:
                print(f"{timeframe} On Count Error: {e}")
        if is_select:
            try:
                rows = cursor.fetchall()
                total_rows += len(rows)
                print(f"{timeframe} On Select: {rows}")
            except Exception as e:
                print(f"{timeframe} On Select Error: {e}")
        total_rows += cursor.rowcount
        connection.commit()
    except Exception as e:
        print(f"{timeframe} Error: {e}")
    else:
        print(f"{timeframe} SQL completed successfully: {count if is_counting else ''}")
        print(f"{timeframe} On Commit: {cursor.rowcount}")
    finally:
        connection.close()

print(f"Total Row counts: {abs(total_rows)}")
if is_counting:
    print(f"Total: {count_total}")
