import os
from concurrent.futures import ProcessPoolExecutor


def get_file(url_dl, output_dir):
    os.system(f"wget {url_dl} -o {output_dir}")


if __name__ == "__main__":
    year = input("Enter year: ")
    threads = int(input("How many files would you like to download at once: "))
    files = ["RC_" + year + "-%.2d" % i for i in range(1, 13)]
    files = [file + ".zst" for file in files]
    url = "https://files.pushshift.io/reddit/comments/"
    out_dir = "D:/Datasets/zst/"

    for file in files:
        get_file(url+file, out_dir+file)
