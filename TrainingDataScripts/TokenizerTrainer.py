import os
import marshal
import nltk
import re

from collections import Iterable
from nltk.sentiment import SentimentIntensityAnalyzer
from statistics import mean
from multiprocessing import Pool


f_path_a = "D:\\Datasets\\reddit_data\\files\\train.to"
f_path_q = "D:\\Datasets\\reddit_data\\files\\train.from"


def preprocess_sentence(sentence):
    sentence = sentence.strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,'])", r"\1", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,']+", " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-z?.!,']+", " ", sentence)
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


def file_generator(f_path):
    with open(f_path, "r", encoding="utf-8") as f:
        for line in f:
            line = preprocess_sentence(line)
            yield line


def chunk(lst, count):  # Make a list into N number of lists
    size = len(lst) // count  # figure out the size of them all
    for i in range(0, count):
        s = slice(i * size, None if i == count - 1 else (
                                                                i + 1) * size)  # Using slice here because you can't store 2:3 as a variable
        yield lst[s]  # Yield the list


def get_results(data):
    sia = SentimentIntensityAnalyzer()
    results = []
    for sentence in data:
        results.append(sia.polarity_scores(sentence))
    return results


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for i in flatten(item):
                yield i
        else:
            yield item


def manager():
    data_gen_a = file_generator(f_path_a)
    data_gen_q = file_generator(f_path_q)
    cores = round(os.cpu_count() * 0.75)
    count = 0
    while True:
        try:
            print(f"Beginning run {count}")
            data = [next(data_gen_a) for _ in range(200_000)]
            data.extend([next(data_gen_q) for _ in range(200_000)])
            data_generator = chunk(data, cores)
            p = Pool(processes=cores)
            process_outputs = p.map(get_results, next(data_generator))
            file_name = f"D:\\Datasets\\reddit_data\\caches\\cache{count}.marshal"
            p.close()
            file = open(file_name, "wb")
            marshal.dump(list(flatten(process_outputs)), file)
            file.close()
            count += 1
            del process_outputs, data, data_generator
        except StopIteration:
            break


if __name__ == "__main__":
    manager()
    # calculate_results()
