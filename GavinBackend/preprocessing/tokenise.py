from GavinBackend import tf, tfds, np
from tqdm import tqdm
from GavinBackend.preprocessing.concurrent import chunk
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor
from typing import List, Tuple, AnyStr, Dict


def tokenize(s_token: List[int], e_token: List[int], u_tokenizer: tfds.deprecated.text.SubwordTextEncoder, train_data: List[Tuple[AnyStr, AnyStr]], thread_id: int) -> Dict[AnyStr: List[List[int]]]:
    print(f"Size: {len(train_data)*2} on {thread_id}")
    # Create empty arrays
    # Add the Start token at the start of all rows
    outputs = {'inputs': [], 'outputs': []}
    # Iterate over each sentence in both inputs and outputs
    p_bar = tqdm(total=len(train_data)*2, desc=f"{thread_id}")
    while len(train_data) > 0:
        sentences = train_data[0]
        outputs['inputs'].append(s_token + u_tokenizer.encode(sentences[0]) + e_token)
        p_bar.update()
        outputs['outputs'].append(s_token + u_tokenizer.encode(sentences[1]) + e_token)
        p_bar.update()
        train_data.pop(0)
    p_bar.close()
    return outputs


def tokenize_dataset(inputs: List[AnyStr], outputs: List[AnyStr], cores: int, max_len: int, s_token: List[int], e_token: List[int], tokenizer: tfds.deprecated.text.SubwordTextEncoder) -> Tuple[np.ndarray, np.ndarray]:
    training_data = list(zip(inputs, outputs))
    del inputs, outputs
    data_gen = chunk(training_data, cores)

    lists = [next(data_gen) for _ in range(cores)]
    futures = []

    with ProcessPoolExecutor(cores) as executor:
        for i in range(cores):
            futures.append(executor.submit(tokenize, s_token, e_token, tokenizer, lists[i], f"Thread_{i}"))
        wait(futures)
        executor.shutdown()
    del lists, data_gen, training_data

    inputs = []
    outputs = []
    for future in futures:
        data = future.result()
        inputs.extend(data['inputs'])
        outputs.extend(data['outputs'])
    del futures

    print("Beginning Padding.")
    with ThreadPoolExecutor() as executor:
        fut1 = executor.submit(tf.keras.preprocessing.sequence.pad_sequences, inputs, maxlen=max_len, padding='post')
        fut2 = executor.submit(tf.keras.preprocessing.sequence.pad_sequences, outputs, maxlen=max_len, padding='post')
        wait((fut1, fut2))

    return fut1.result(), fut2.result()
