import base64
import pickle
import typing
import sys
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import tqdm

path = Path(__file__).resolve().parent.parent
sys.path.append(os.path.join(str(path), 'CustomPackages'))


def tokenized_read_thread(path: typing.AnyStr, reddit_set_max: int, s_token: typing.List[int], e_token: typing.List[int], thread_id: int = 0):
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


def load_tokenized_data(max_samples: int, data_path: typing.AnyStr, tokenizer_name: typing.AnyStr,
                        s_token: typing.List[int], e_token: typing.List[int], max_len: int = None, legacy: bool = False) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """Load tokenized data from the data files:
    {data_path}{tokenizer_name}.from
    {data_path}{tokenizer_name}.to these will be configurable eventually."""
    if not legacy and max_len is None:
        raise Exception("Max Length can't be none when Legacy is false.")
    if legacy:
        with ProcessPoolExecutor(2) as executor:
            inputs_fn = executor.submit(tokenized_read_thread, f"{data_path}{tokenizer_name}.from", max_samples, s_token, e_token, 0)
            outputs_fn = executor.submit(tokenized_read_thread, f"{data_path}{tokenizer_name}.to", max_samples, s_token, e_token, 1)
            executor.shutdown()

        return inputs_fn.result(), outputs_fn.result()
    else:
        import LoadTrainData
        inputs = LoadTrainData.LoadTrainDataST(max_samples//2, f"{data_path}", f"{tokenizer_name}.from", s_token[0], e_token[0], max_len, 0)
        outputs = LoadTrainData.LoadTrainDataST(max_samples//2, f"{data_path}", f"{tokenizer_name}.to", s_token[0], e_token[0], max_len, 0)
        inputs = np.asarray(inputs)
        outputs = np.asarray(outputs)
        return inputs, outputs
