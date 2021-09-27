from .models import tf
from .preprocessing.text import np


class DatasetAPICreator:
    def __init__(self, questions: list, answers: list, buffer_size: int, batch_size: int, vocab_size: int):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.questions_train = questions
        self.answers_train = answers
        self.vocab_size = vocab_size

    def change_to_probabilities(self, first_part, second_part):
        outputs = second_part['outputs'].eval(session=tf.compat.v1.Session(), feed_dict={second_part['outputs'].name: np.zeros(shape=(self.batch_size, 52), dtype=np.int32)})
        new_outputs = np.zeros(shape=(self.batch_size, outputs.shape[1], self.vocab_size), dtype=np.int32)
        for sentence in range(0, self.batch_size - 1):
            for Index in range(0, new_outputs.shape[0] - 1):
                vocab_id = outputs[sentence][Index] - 1
                vocab_id = vocab_id if vocab_id != -1 else vocab_id+1
                if vocab_id == self.vocab_size - 1:
                    break
                new_outputs[sentence][Index][vocab_id] = 1
        second_part = {'outputs': new_outputs}
        return first_part, second_part

    @classmethod
    def create_data_objects(cls, questions: list, answers: list, buffer_size: int, batch_size: int, vocab_size: int):
        self = cls(questions, answers, buffer_size, batch_size, vocab_size)

        dec_inputs_train = self.answers_train.copy()
        dec_inputs_train[:, -1] = 0
        outputs_train = self.answers_train.copy()
        outputs_train[:, 0] = 0
        outputs_train = np.roll(outputs_train.copy(), -1)  # Roll back values -1 to not leave an empty value.
        del self.answers_train
        # decoder inputs use the previous target as input
        # remove s_token from targets
        # print("Beginning Dataset Shuffling, Batching and Prefetch.")
        dataset_all = tf.data.Dataset.from_tensor_slices((
            {
                'inputs': self.questions_train,  # Source
                'dec_inputs': dec_inputs_train  # Targets
            },
            {
                'outputs': outputs_train  # Outputs
            }))
        dataset_all = dataset_all.shuffle(self.buffer_size)
        dataset_all = dataset_all.batch(self.batch_size)
        # dataset_all = dataset_all.map(self.change_to_probabilities, num_parallel_calls=os.cpu_count())
        dataset_all = dataset_all.cache()
        dataset_all = dataset_all.prefetch(tf.data.experimental.AUTOTUNE)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

        dataset_t = dataset_all.take(int(len(self.questions_train)*.8))
        dataset_v = dataset_all.skip(int(len(self.questions_train)*.8))
        del dataset_all

        return dataset_t, dataset_v
