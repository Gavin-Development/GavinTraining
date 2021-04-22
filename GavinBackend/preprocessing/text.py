import re
import pickle
import base64
import tqdm

from GavinBackend.models import tf
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
        for i in range(reddit_set_max // 2):
            newline = " newlinechar "
            line = next(f)
            if newline in line:
                line = line.replace(newline, "\n")
            # line = preprocess_sentence(line)
            lines.append(line)
    return lines


def tokenized_read_thread(path, reddit_set_max, s_token, e_token, thread_id=0):
    lines = []
    pbar = tqdm.tqdm(total=reddit_set_max//2, desc=f"Thread: {thread_id}")
    with open(path, "r") as f:
        for i in range(reddit_set_max // 2):
            line = next(f).strip("'b'")
            line = line.strip("'\n'")
            line = line.strip("'")
            # line = preprocess_sentence(line)
            line = pickle.loads(base64.b64decode(line))
            line.insert(0, s_token[0])
            line.append(e_token[0])
            lines.append(line)
            pbar.update(1)
    return lines


# noinspection PyShadowingNames,PyPep8Naming
def load_data(reddit_set_max, path):
    with ProcessPoolExecutor(2) as executor:
        inputs_fn = executor.submit(read_thread, f"{path}train.from", reddit_set_max)
        outputs_fn = executor.submit(read_thread, f"{path}train.to", reddit_set_max)
    return inputs_fn.result(), outputs_fn.result()


def load_tokenized_data(reddit_set_max, path, tokenizer_name, max_len, s_token, e_token):
    with ProcessPoolExecutor(2) as executor:
        inputs_fn = executor.submit(tokenized_read_thread, f"{path}{tokenizer_name}.from", reddit_set_max, s_token, e_token, 0)
        outputs_fn = executor.submit(tokenized_read_thread, f"{path}{tokenizer_name}.to", reddit_set_max, s_token, e_token, 1)
        executor.shutdown()
    print("Beginning padding.")

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs_fn.result(), maxlen=max_len, padding='post')
    outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs_fn.result(), maxlen=max_len, padding='post')
    return inputs, outputs
