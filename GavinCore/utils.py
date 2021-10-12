import tensorflow as tf
from .preprocessing.text import np


def convert_to_probabilities(y_true, vocab_size) -> tf.Tensor:
    """When doing loss for Transformers we should be comparing probabilities between
    2 distributions as such y_true, must be converted to a Tensor of samples, max_len, vocab_size
    Where 1 sentence looks like, [[0,1,0], [1,0,0], [1,0,0]], where the vocab size here is 3."""
    shape = (y_true.shape[0], y_true.shape[1], vocab_size)
    new_y_true = np.zeros(shape=shape, dtype=np.int32)
    # y_true = tf.multiply(y_true, tf.cast(tf.not_equal(y_true, 0), y_true.dtype))
    for sentence in range(0, y_true.shape[0] - 1):
        for Index in range(0, y_true.shape[1] - 1):
            vocab_id = y_true[sentence][Index] - 1
            if vocab_id == vocab_size - 1:
                break
            new_y_true[sentence][Index][vocab_id] = 1

    del y_true
    return tf.convert_to_tensor(new_y_true)
