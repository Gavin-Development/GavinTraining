from .utils import tf


@tf.function(experimental_follow_type_hints=True)
def SparseCategoricalCrossentropy(y_true: tf.Tensor, y_pred: tf.Tensor, numeric_stabilizer: float = 5e-6):
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    y_true = tf.cast(tf.add(y_true, numeric_stabilizer), tf.float32)

    # y_pred = tf.reshape(y_pred, shape=(tf.shape(y_pred)[tf.rank(y_pred) - 1], -1))
    # y_true = tf.reshape(y_true, shape=(-1,))
    y_true = tf.reshape(y_true, shape=(tf.shape(y_true)[0], tf.shape(y_true)[1], 1))
    return - (tf.reduce_sum(
            tf.multiply(y_true,
                        tf.math.log(y_pred), name="mul_1")
            + tf.multiply((1 - y_true),
                          tf.math.log(1 - y_pred)),
            axis=1) / batch_size)
