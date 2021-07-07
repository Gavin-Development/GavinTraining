from pprint import pprint

from tensorflow.python.keras.utils import tf_utils

from .models import tf
from typing import Dict


def iid_gaussian(m, d):
    """Generate random values that are I.I.D (independent identically distributed)"""
    return tf.random.normal(shape=(m, d))


def orthogonal_gaussian(m, d):
    """Generate Orthogonal Gaussian distribution's. This is to improve upon MSE (mean squared error)
    inside a preformer."""

    def orthogonal_square():
        q, _ = tf.linalg.qr(iid_gaussian(d, d))
        return tf.transpose(q)

    num_squares = int(m / d)
    blocks = [orthogonal_square() for _ in range(num_squares)]

    remainder = m - d * num_squares
    if remainder:
        blocks.append(orthogonal_square()[:remainder])

    matrix = tf.experimental.numpy.vstack(blocks)
    matrix /= tf.sqrt(num_squares + remainder / d)

    return matrix


def softmax_kernel_transformation(data,
                                  is_query,
                                  projection_matrix=None,
                                  numerical_stabilizer=0.000001):
    """Computes random features for the softmax kernel using FAVOR+ mechanism.

  Computes random features for the softmax kernel using FAVOR+ mechanism from
  https://arxiv.org/pdf/2009.14794.pdf.

  Args:
    data: input data tensor of the shape [B, L, H, D], where: B - batch
      dimension, L - attention dimensions, H - heads, D - features.
    is_query: indicates whether input data is a query oor key tensor.
    projection_matrix: random Gaussian matrix of shape [M, D], where M stands
      for the number of random features and each D x D sub-block has pairwise
      orthogonal rows.
    numerical_stabilizer: small positive constant for numerical stability.

  Returns:
    Corresponding kernel feature map.
  """
    data_normalizer = 1.0 / (
        tf.math.sqrt(tf.math.sqrt(tf.dtypes.cast(data.shape[-1], tf.float32))))
    data = data_normalizer * data
    ratio = 1.0 / tf.math.sqrt(
        tf.dtypes.cast(projection_matrix.shape[0], tf.float32))
    # noinspection SpellCheckingInspection
    data_dash = tf.einsum("blhd,md->blhm", data, projection_matrix, name="SoftmaxKernel")
    diag_data = tf.math.square(data)
    diag_data = tf.math.reduce_sum(
        diag_data, axis=tf.keras.backend.ndim(data) - 1)
    diag_data = diag_data / 2.0
    diag_data = tf.expand_dims(diag_data, axis=tf.keras.backend.ndim(data) - 1)
    last_dims_t = (len(data_dash.shape) - 1,)
    attention_dims_t = (len(data_dash.shape) - 3,)
    if is_query:
        data_dash = ratio * (
                tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
                    data_dash, axis=last_dims_t, keepdims=True)) + numerical_stabilizer)
    else:
        data_dash = ratio * (
                tf.math.exp(data_dash - diag_data - tf.math.reduce_max(
                    data_dash, axis=last_dims_t + attention_dims_t, keepdims=True)) +
                numerical_stabilizer)

    return data_dash


def attn_hat(query, key, value, phi_fun=None, normalize=True, random_feats=None):
    l = tf.shape(query)[0]

    query = tf.transpose(query, perm=[0, 2, 1, 3])
    key = tf.transpose(key, perm=[0, 2, 1, 3])

    if phi_fun is not None:
        q_prime = phi_fun(query)
        k_prime = phi_fun(key)
    else:
        q_prime = softmax_kernel_transformation(query, projection_matrix=random_feats, is_query=True)
        k_prime = softmax_kernel_transformation(key, projection_matrix=random_feats, is_query=False)

    value = tf.transpose(value, [0, 2, 1, 3])

    # noinspection SpellCheckingInspection
    av_attention = tf.einsum("lbhm,lbhd->bhmd", k_prime, value, name="AVAttention_PA")

    # noinspection SpellCheckingInspection
    av_attention = tf.einsum("lbhm,bhmd->lbhd", q_prime, av_attention, name="AVAttention_PB")
    # noinspection SpellCheckingInspection
    normalizer = tf.einsum("lbhm,l->bhm", k_prime, tf.ones(l), name="NormalizerPA")
    # noinspection SpellCheckingInspection
    normalizer = tf.einsum("lbhm,bhm->lbh", q_prime, normalizer, name="NormalizerPB")
    av_attention = tf.transpose(av_attention, [1, 0, 2, 3])
    normalizer = tf.transpose(normalizer, [1, 0, 2])
    normalizer = tf.expand_dims(normalizer, len(tf.shape(normalizer)))
    return av_attention / normalizer


def positive_attention(query, key, value, random_feats, normalize=True):
    """Instead of using ScaledDotProduction, this uses the above Gaussian elements to estimate the answer that
    the full ScaledDotProduction would give. """

    return attn_hat(query, key, value, normalize=normalize, random_feats=random_feats)


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

        self.depth = d_model // 2

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size: int):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
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
        self.num_features = num_features
        super().__init__(d_model, num_heads, name)
        self.random_feats = orthogonal_gaussian(self.num_features, self.depth)

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

        scaled_attention = positive_attention(query=query, key=key, value=value,
                                              random_feats=self.random_feats)

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
