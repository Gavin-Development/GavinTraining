from .utils import tf


@tf.function(experimental_follow_type_hints=True)
def SparseCategoricalCrossentropy(y_true: tf.Tensor, y_pred: tf.Tensor, numeric_stabilizer: float = 5e-6):
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    y_true = tf.cast(tf.add(y_true, numeric_stabilizer), tf.float32)
    breakpoint()
    summation = tf.zeros(shape=(tf.shape(y_pred)[tf.rank(y_pred) - 1], tf.shape(y_true)[0]))
    y_pred = tf.transpose(y_pred, [0, 2, 1])
    for i in range(batch_size):
        i = tf.cast(i, tf.int64)
        y_hat_i = y_pred[i]
        y_i = y_true[i]
        log_1 = tf.cast(tf.math.log(y_hat_i), tf.float32)
        mul_1 = tf.multiply(y_i, log_1, name="mul_1")

        subtract_1 = 1 - y_i
        log_2 = tf.cast(tf.math.log(1 - y_hat_i), tf.float32)
        mul_2 = tf.cast(tf.multiply(subtract_1, log_2, name="mul_2"), tf.float32)
        breakpoint()
        summation = tf.add(summation, tf.add(mul_1, mul_2, name="summation_add_2"), name="summation_add_1")

    return -tf.multiply(tf.cast(1 / batch_size, tf.float32), summation)
