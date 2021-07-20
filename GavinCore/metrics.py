import typing
from .models import tf, TransformerIntegration, PreformerIntegration


class Precision(tf.keras.metrics.Precision):

    def __init__(self, wrapper_model: typing.Union[TransformerIntegration, PreformerIntegration], **kwargs):
        super(Precision, self).__init__(**kwargs)
        self.wrapper_model = wrapper_model

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, shape=(-1, self.wrapper_model.max_len))
        super(Precision, self).update_state(y_true, y_pred, sample_weight=sample_weight)
