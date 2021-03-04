import re
from concurrent.futures import ProcessPoolExecutor


def preprocess_sentence(sentence):
    sentence = sentence.strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,'*\":@])", r" \1 ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    sentence = re.sub(r"[^a-zA-z?.!,'*:\"@]+", " ", sentence)
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


def preprocess_context(sentence):
    sentence = re.sub(r"[^a-zA-Z ]+", "", sentence)
    return sentence


def read_thread(path, reddit_set_max):
    lines = []
    with open(path, "r", encoding='utf-8') as f:
        for i in range(reddit_set_max//2):
            line = next(f)
            # line = preprocess_sentence(line)
            lines.append(line)
    return lines


# noinspection PyShadowingNames,PyPep8Naming
def load_data(reddit_set_max, path):
    train_from = open(f"{path}train.from", "r", encoding="utf-8")
    train_to = open(f"{path}train.to", "r", encoding="utf-8")
    with ProcessPoolExecutor(2) as executor:
        inputs_fn = executor.submit(read_thread, f"{path}train.from", reddit_set_max)
        outputs_fn = executor.submit(read_thread, f"{path}train.to", reddit_set_max)
    return inputs_fn.result(), outputs_fn.result()

