from .preprocessing.text import np
from .models import tf


def create_data_objects(questions, answers, buffer_size, batch_size):
    sizes = (len(questions), len(answers))
    questions_train = questions[0: int(sizes[0] * .80)]
    questions_val = questions[int(sizes[0] * 0.80):]
    answers_train = answers[0: int(sizes[1] * .80)]
    answers_val = answers[int(sizes[1] * .80):]

    dec_inputs_train = answers_train.copy()
    dec_inputs_train[:, -1] = 0
    dec_inputs_val = answers_val.copy()
    dec_inputs_val[:, -1] = 0
    
    outputs_train = answers_train.copy()
    outputs_train[:, 0] = 0
    outputs_train = np.roll(outputs_train.copy(), -1)  # Roll back values -1 to not leave an empty value.
    outputs_val = answers_val.copy()
    outputs_val[:, 0] = 0
    del answers_train, answers_val
    # decoder inputs use the previous target as input
    # remove s_token from targets
    # print("Beginning Dataset Shuffling, Batching and Prefetch.")
    dataset_t = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_train,  # Source
            'dec_inputs': dec_inputs_train  # Targets
        },
        {
            'outputs': outputs_train  # Outputs
        }))
    dataset_v = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_val,  # Source
            'dec_inputs': dec_inputs_val  # Targets
        },
        {
            'outputs': outputs_val  # Outputs
        }))

    dataset_t = dataset_t.cache()
    dataset_v = dataset_v.cache()
    dataset_t = dataset_t.shuffle(buffer_size)
    dataset_v = dataset_v.shuffle(buffer_size)
    dataset_t = dataset_t.batch(batch_size)
    dataset_v = dataset_v.batch(batch_size)
    dataset_t = dataset_t.prefetch(tf.data.experimental.AUTOTUNE)
    dataset_v = dataset_v.prefetch(tf.data.experimental.AUTOTUNE)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset_t.with_options(options)
    dataset_v.with_options(options)

    return dataset_t, dataset_v
