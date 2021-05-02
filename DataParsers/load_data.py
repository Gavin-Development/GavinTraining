import tqdm
import typing
import pickle
import base64

from concurrent.futures import ProcessPoolExecutor


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


def load_tokenized_data(max_samples: int, tokenizer_path: typing.AnyStr, tokenizer_name: typing.AnyStr,
                        s_token: typing.List[int], e_token: typing.List[int]) -> typing.Tuple[typing.List[str], typing.List[str]]:
    """Load tokenized data from the data files:
    {tokenizer_path}{tokenizer_name}.from
    {tokenizer_path}{tokenizer_name}.to these will be configurable eventually."""
    with ProcessPoolExecutor(2) as executor:
        inputs_fn = executor.submit(tokenized_read_thread, f"{tokenizer_path}{tokenizer_name}.from", max_samples, s_token, e_token, 0)
        outputs_fn = executor.submit(tokenized_read_thread, f"{tokenizer_path}{tokenizer_name}.to", max_samples, s_token, e_token, 1)
        executor.shutdown()

    return inputs_fn.result(), outputs_fn.result()
