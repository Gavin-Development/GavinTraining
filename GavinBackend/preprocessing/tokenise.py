import numpy as np
from GavinBackend.preprocessing.concurrent import chunk
from functools import partial
from multiprocessing import Pool


# Function the process will be mapped to
def check_length(m_length, train_data):
    # Get rid of any inputs/outputs that don't meet the max_length requirement (save the model training on large sentences)
    output_data = {'inputs': [], 'outputs': []}
    for _, sentences in enumerate(train_data):
        if len(sentences[0]) <= m_length - 2 and len(sentences[1]) <= m_length - 2:
            output_data['inputs'].append(sentences[0])
            output_data['outputs'].append(sentences[1])
    return output_data


def tokenize(m_length, s_token, e_token, u_tokenizer, train_data):
    # Init the shapes for the arrays
    shape_inputs = (len(train_data), m_length)
    shape_outputs = (len(train_data), m_length)
    # Create empty arrays
    # Add the Start token at the start of all rows
    inputs_array = np.zeros(shape=shape_inputs, dtype=np.float32)
    inputs_array[:, 0] = s_token[0]
    outputs_array = np.zeros(shape=shape_outputs, dtype=np.float32)
    outputs_array[:, 0] = s_token[0]

    # Iterate over each sentence in both inputs and outputs
    for _, sentences in enumerate(train_data):

        # Encode the sentences
        tokenized_sentence1 = u_tokenizer.encode(sentences[0])
        tokenized_sentence2 = u_tokenizer.encode(sentences[1])

        # This check length doesn't technically matter but its here as a fail safe.
        if len(tokenized_sentence1) <= m_length - 2 and len(tokenized_sentence2) <= m_length - 2:
            # Add the tokenized sentence into array.
            # This acts as padding for the
            inputs_array[_, 1:len(tokenized_sentence1) + 1] = tokenized_sentence1
            inputs_array[_, len(tokenized_sentence1) + 1] = e_token[0]

            outputs_array[_, 1:len(tokenized_sentence2) + 1] = tokenized_sentence2
            outputs_array[_, len(tokenized_sentence2) + 1] = e_token[0]
    return {'inputs': inputs_array, 'outputs': outputs_array}


def tokenize_and_filter(inputs, outputs, cores, max_len, s_token, e_token, tokenizer):
    training_data = list(zip(inputs, outputs))
    data_gen = chunk(training_data, cores)

    partial_iter = partial(check_length, max_len)

    process_pool = Pool(processes=cores)

    lists = [next(data_gen) for _ in range(cores)]

    _process_outputs = process_pool.map(partial_iter, lists)
    process_pool.close()

    inputs = []
    for i in range(cores):
        inputs.extend(_process_outputs[i]['inputs'])

    outputs = []
    for i in range(cores):
        outputs.extend(_process_outputs[i]['outputs'])

    # Do the same but tokenizer everything instead of check length
    training_data = list(zip(inputs, outputs))
    del inputs, outputs
    data_gen = chunk(training_data, cores)

    partial_iter = partial(tokenize, max_len, s_token, e_token, tokenizer)
    process_pool = Pool(processes=cores)
    lists = [next(data_gen) for _ in range(cores)]
    _process_outputs = process_pool.map(partial_iter, lists)
    process_pool.close()

    # inputs_array = np.concatenate((_process_outputs[i]['inputs'] for i in range(cores)))
    inputs_array = _process_outputs[0]['inputs']
    for i in range(cores-2):
        inputs_array = np.append(inputs_array, _process_outputs[i+1]['inputs'], axis=0)

    # outputs_array = np.concatenate((_process_outputs[i]['outputs'] for i in range(cores)))
    outputs_array = _process_outputs[0]['outputs']
    for i in range(cores-2):
        outputs_array = np.append(outputs_array, _process_outputs[i+1]['outputs'], axis=0)
    del lists, data_gen

    return inputs_array, outputs_array
