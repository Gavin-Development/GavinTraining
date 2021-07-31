import typing
from .models import tf, TransformerIntegration, PerformerIntegration


class Precision(tf.keras.metrics.Precision):

    def __init__(self, max_len: int, from_logits: bool = False, **kwargs):
        super(Precision, self).__init__(**kwargs)
        self.max_len = max_len
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, shape=(-1, self.max_len))
        super(Precision, self).update_state(y_true, y_pred if not self.from_logits else tf.keras.activations.sigmoid(y_pred), sample_weight=sample_weight)

    def result(self):
        super(Precision, self).result()


class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, max_len: int, **kwargs):
        super(Perplexity, self).__init__(**kwargs)
        self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.max_len = max_len
        self.perplexity = self.add_weight(name='p', initializer="zeros")

    def result(self):
        return self.perplexity

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, shape=(-1, self.max_len))

        loss = self.cross_entropy(y_true, y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), y_pred.dtype)
        loss = tf.multiply(loss, mask)
        loss = tf.reduce_mean(loss)
        self.perplexity.assign_add(tf.exp(loss))
