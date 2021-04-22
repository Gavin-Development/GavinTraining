import re
import pickle
import base64
import tqdm

from GavinBackend import tf, np
from concurrent.futures import ProcessPoolExecutor
from typing import AnyStr, List, Tuple


def preprocess_sentence(sentence: AnyStr) -> AnyStr:
    sentence = sentence.strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,'*\":@])", r" \1 ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    sentence = re.sub(r"[^a-zA-z?.!,'*:\"@]+", " ", sentence)
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


def preprocess_context(sentence: AnyStr) -> AnyStr:
    sentence = re.sub(r"[^a-zA-Z ]+", "", sentence)
    return sentence


def read_thread(path: AnyStr, reddit_set_max: int) -> List[AnyStr]:
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


def tokenized_read_thread(path: AnyStr, reddit_set_max: int, s_token: List[int], e_token: List[int], thread_id: int = 0):
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
def load_data(reddit_set_max: int, path: AnyStr) -> Tuple[List[AnyStr], List[AnyStr]]:
    with ProcessPoolExecutor(2) as executor:
        inputs_fn = executor.submit(read_thread, f"{path}train.from", reddit_set_max)
        outputs_fn = executor.submit(read_thread, f"{path}train.to", reddit_set_max)
    return inputs_fn.result(), outputs_fn.result()


def load_tokenized_data(max_samples: int, tokenizer_path: AnyStr, tokenizer_name: AnyStr, max_len: int, s_token: List[int], e_token: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    with ProcessPoolExecutor(2) as executor:
        inputs_fn = executor.submit(tokenized_read_thread, f"{tokenizer_path}{tokenizer_name}.from", max_samples, s_token, e_token, 0)
        outputs_fn = executor.submit(tokenized_read_thread, f"{tokenizer_path}{tokenizer_name}.to", max_samples, s_token, e_token, 1)
        executor.shutdown()
    print("Beginning padding.")

    inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs_fn.result(), maxlen=max_len, padding='post')
    outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs_fn.result(), maxlen=max_len, padding='post')
    return inputs, outputs
