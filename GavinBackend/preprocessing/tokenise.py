import gc
import numpy as np
from GavinBackend.preprocessing.concurrent import chunk
from concurrent.futures import ProcessPoolExecutor, wait, ThreadPoolExecutor, ALL_COMPLETED


def tokenize(e_token, u_tokenizer, train_data, thread_id):
    print(f"Size: {len(train_data)*2} on {thread_id}")
    # Create empty arrays
    # Add the Start token at the start of all rows
    outputs = {'inputs': [], 'outputs': []}
    # Iterate over each sentence in both inputs and outputs
    while len(train_data) > 0:
        sentences = train_data[0]
        outputs['inputs'].append(u_tokenizer.encode(sentences[0]) + e_token)
        outputs['outputs'].append(u_tokenizer.encode(sentences[1]) + e_token)
        train_data.pop(0)
    return list(zip(outputs['inputs'], outputs['outputs']))


def tokenize_and_filter(inputs, outputs, cores, max_len, s_token, e_token, tokenizer):
    # Do the same but tokenizer everything instead of check length
    training_data = list(zip(inputs, outputs))
    del inputs, outputs
    data_gen = chunk(training_data, cores)

    lists = [next(data_gen) for _ in range(cores)]
    futures = []
    with ProcessPoolExecutor(cores) as executor:
        for i in range(cores):
            futures.append(executor.submit(tokenize, e_token, tokenizer, lists[i], f"Thread_{i}"))
        wait(futures)
        inputs_array = np.zeros(shape=(len(training_data), max_len), dtype=np.float32)
        inputs_array[:, 0] = s_token[0]
        outputs_array = np.zeros(shape=(len(training_data), max_len), dtype=np.float32)
        outputs_array[:, 0] = s_token[0]
        wait(futures, return_when=ALL_COMPLETED)
        del lists, data_gen, training_data
        gc.collect()
        for future in futures:
            data = future.result()
            for _, sentences in enumerate(data):
                sentence1, sentence2 = sentences[0], sentences[1]
                if len(sentence1) <= max_len - 2 and len(sentence2) <= max_len - 2:
                    inputs_array[_, 1:len(sentence1) + 1] = sentence1

                    outputs_array[_, 1:len(sentence2) + 1] = sentence2
        del futures

    return inputs_array, outputs_array
