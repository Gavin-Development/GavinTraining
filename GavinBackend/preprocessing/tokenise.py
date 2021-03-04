import numpy as np
from GavinBackend.preprocessing.concurrent import chunk
from concurrent.futures import ProcessPoolExecutor


def tokenize(m_length, s_token, e_token, u_tokenizer, train_data, thread_id):
    # Init the shapes for the arrays
    shape_inputs = (len(train_data), m_length)
    shape_outputs = (len(train_data), m_length)
    print(f"Size: {len(train_data)} on {thread_id}")
    # Create empty arrays
    # Add the Start token at the start of all rows
    inputs_array = np.zeros(shape=shape_inputs, dtype=np.float32)
    inputs_array[:, 0] = s_token[0]
    outputs_array = np.zeros(shape=shape_outputs, dtype=np.float32)
    outputs_array[:, 0] = s_token[0]

    # Iterate over each sentence in both inputs and outputs
    for _, sentences in enumerate(train_data):

        # Encode the sentences
        tokenized_sentence1 = u_tokenizer.encode(sentences[0]) + e_token
        tokenized_sentence2 = u_tokenizer.encode(sentences[1]) + e_token

        if len(tokenized_sentence1) <= m_length - 2 and len(tokenized_sentence2) <= m_length - 2:
            inputs_array[_, 1:len(tokenized_sentence1) + 1] = tokenized_sentence1

            outputs_array[_, 1:len(tokenized_sentence2) + 1] = tokenized_sentence2
    return {'inputs': inputs_array, 'outputs': outputs_array}


def tokenize_and_filter(inputs, outputs, cores, max_len, s_token, e_token, tokenizer):
    assert cores % 2 == 0
    # Do the same but tokenizer everything instead of check length
    training_data = list(zip(inputs, outputs))
    del inputs, outputs
    data_gen = chunk(training_data, cores)

    lists = [next(data_gen) for _ in range(cores)]
    futures = []
    with ProcessPoolExecutor(cores) as executor:
        for i in range(cores):
            futures.append(executor.submit(tokenize, max_len, s_token, e_token, tokenizer, lists[i], f"Thread_{i}"))
    inputs_array = futures[0].result()['inputs']
    outputs_array = futures[0].result()['inputs']
    futures = futures[:-1]
    while i != cores-1:
        inputs_array = np.append(inputs_array, futures[0].result()['inputs'], axis=0)
        outputs_array = np.append(outputs_array, futures[0].result()['outputs'], axis=0)
        futures = futures[:-1]
        i += 1
    del futures

    return inputs_array, outputs_array
