import os
import io
import zstandard
import sys
import json
import tqdm
import pickle
import base64

import tensorflow_datasets as tfds
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

filepath = sys.argv[1]
f = open(filepath, 'rb')
dctx = zstandard.ZstdDecompressor(max_window_size=2147483648)
stream_reader = dctx.stream_reader(f, read_size=int(1.953e+6))
text_stream = io.TextIOWrapper(stream_reader, encoding="utf-8")
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file("../Tokenizer-3")
handle_to = open("C:/Users/jkshi/Datasets/Tokenizer-3-to", 'wb')
handle_from = open("C:/Users/jkshi/Datasets/Tokenizer-3-from", 'wb')
if __name__ == "__main__":
    for row in tqdm.tqdm(text_stream, unit=" rows", desc="Processing"):
        row = json.loads(row)
        text = row['text']
        sentences = list(filter(lambda x: len(x) > 1, list(map(lambda x: x.strip() + ".", text.split('.')))))

        with ProcessPoolExecutor(4) as executor:
            results = executor.map(tokenizer.encode, sentences)
            results = list(results)
            executor.shutdown()
        results = list(map(lambda x: base64.b64encode(pickle.dumps(x)), results))
        for to_text, from_text in combinations(results, 2):
            handle_to.write(to_text + b'\n')
            handle_from.write(from_text + b'\n')
