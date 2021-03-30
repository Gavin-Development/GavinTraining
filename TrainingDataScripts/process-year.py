import os
from concurrent.futures import ThreadPoolExecutor

year = input("Enter the year: ")
processes = int(input("How many files to be processed at once?: "))

files = [year + "-%.2d" % i for i in range(1, 13)]


def run(filename):
    os.system(f"categorise.py {filename}")
    print(f"{filename} Finished")


with ThreadPoolExecutor(processes) as executor:
    for file in files:
        executor.submit(run, file)
