from tensorflow.python.keras.utils import tf_utils

from .models import tf
from typing import Dict


def iid_gaussian(m, d):
    """Generate random values that are I.I.D (independent identically distributed)"""
    return tf.random.normal(size=(m, d))


def orthogonal_gaussian(m, d):
    """Generate Orthogonal Gaussian distribution's. This is to improve upon MSE (mean squared error)
    inside a preformer."""
    def orthogonal_square():
        q, _ = tf.linalg.qr(iid_gaussian(d, d))
        return q.T

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = tf.experimental.numpy.vstack(blocks)
    matrix /= tf.sqrt(num_squares + remainder / d)

    return matrix


def phi(h, fs, random_feats, m):
    return lambda x: (
        h(x) / tf.sqrt(m) *
        tf.concat([f(tf.einsum("...d_model,md->...num_feature", x, random_feats)) for f in fs],
                  axis=-1)
    )


def attn_hat(query, key, value, phi_fun, normalize=True):
    l, d = query.shape
    normalizer = 1 / (d ** 0.25)
    q_prime = phi_fun(query * normalizer)
    k_prime = phi_fun(key * normalizer)
    d_inv = tf.linalg.diag(1 / (q_prime @ (k_prime.T @ tf.ones(l))))
    return d_inv @ (q_prime @ (k_prime.T @ value))


def positive_attention(query, key, value, random_feats, mask, normalize=True):
    """Instead of using ScaledDotProduction, this uses the above Gaussian elements to estimate the answer that
    the full ScaledDotProduction would give. """
    def h(x):
        return tf.exp(-tf.math.square(x).sum(axis=-1, keepdims=True) / 2)
    kernel = phi(h, [tf.exp], random_feats, mask)
    return attn_hat(query, key, value, kernel, normalize)


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(attention_weights, value)


def generate_random_features():
    pass


# noinspection PyMethodOverriding,PyMethodMayBeStatic
class PositionalEncoding(tf.keras.layers.Layer):
    """Positional Encoding

    Acts as input for the model, attention to where words appear in an input etc...

    Attributes:
        :arg position: int
            The position the word appears in
        :arg d_model: int
            This is for the attention math, acts as units for other layers in the model too.
    """

    def __init__(self, position: int, d_model: int):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model=d_model)

    def get_angles(self, position: int, i, d_model: int):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)

        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        cfg = super().get_config()
        return cfg


# noinspection PyMethodOverriding,PyShadowingNames
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, num_heads: int, name: str = "multi_head_attention"):
        """Multi Head Attention Layer

        ...
        Attributes:
            :arg d_model: int
                Embeddings Size.
            :arg num_heads: int
                The number of heads the layer should have
            :arg name: str
                The name of layer, for output with model.summary
        """
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size: int):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs: Dict):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs

    def get_config(self):
        cfg = super().get_config()
        return cfg


class MultiHeadPreformerAttention(MultiHeadAttention):
    """MultiHead attention using the performers specification,
    significantly improving memory and time complexity allowing for
    higher values of sequence length, whilst maintaining as good or
    some cases better accuracy compared to standard transformer.

    Attributes:
            :arg d_model: int
                Embeddings Size.
            :arg num_heads: int
                The number of heads the layer should have
            :arg num_features: int
                Number of features to be used in Gaussian Matrix.
            :arg name: str
                The name of layer, for output with model.summary
    """

    def __init__(self, d_model: int, num_heads: int, num_features: int, name: str):
        super().__init__(d_model, num_heads, name)
        self.random_feats = orthogonal_gaussian(num_features, self.d_model)

    def call(self, inputs: Dict):
        query, key, value, mask = inputs['query'], inputs['key'], inputs[
            'value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = positive_attention(query=query, key=key, value=value, mask=mask, random_feats=self.random_feats)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs


class GPUEnabledEmbedding(tf.keras.layers.Embedding):
    """Embedding Layers are forced to run on CPUs which seriously
    hurts training performance this fixes that issue."""
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name="embeddings",
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
        )
        self.built = True
