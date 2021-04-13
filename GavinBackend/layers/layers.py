import tensorflow as tf
from GavinBackend.ThirdParty.TFutils import sort_key_val, batched_index_select, make_unit_length, chunked_sum, \
    process_inputs_chunk


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    return tf.matmul(attention_weights, value)


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

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model=d_model)

    def get_angles(self, position, i, d_model):
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
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
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

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
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


# ANYTHING BELOW THIS LINE APPLIES TO THIS LICENCE

# MIT License

# Copyright (c) 2020 Streack, Jayakrishna Sahit

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class TFLSHAttention(tf.keras.layers.Layer):
    def __init__(self,
                 dropout=0.,
                 bucket_size=64,
                 n_hashes=8,
                 causal=False,
                 allow_duplicate_attention=True,
                 attend_across_buckets=True,
                 rehash_each_round=True,
                 random_rotations_per_head=False):
        super(TFLSHAttention, self).__init__()
        if dropout >= 1.0:
            raise ValueError('Dropout rates must be lower than 1.')

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dropout_for_hash = tf.keras.layers.Dropout(dropout)

        assert rehash_each_round or allow_duplicate_attention, (
            'The setting {allow_duplicate_attention=False, rehash_each_round=False}'
            ' is not implemented.')

        self.causal = causal
        self.n_hashes = n_hashes
        self.bucket_size = bucket_size

        self._allow_duplicate_attention = allow_duplicate_attention
        self._attend_across_buckets = attend_across_buckets
        self._rehash_each_round = rehash_each_round
        self._random_rotations_per_head = random_rotations_per_head

    def hash_vectors(self, n_buckets, vectors_to_hash):
        batch_size = vectors_to_hash.shape[0]
        # See https://arxiv.org/pdf/1509.02897.pdf
        # We sample a different random rotation for each round of hashing to
        # decrease the probability of hash misses.
        assert n_buckets % 2 == 0

        rot_size = n_buckets

        rotations_shape = (
            batch_size if self._random_rotations_per_head else 1,
            vectors_to_hash.shape[-1],
            self.n_hashes if self._rehash_each_round else 1,
            rot_size // 2)

        random_rotations = tf.broadcast_to(tf.random.normal(rotations_shape), (
            batch_size, vectors_to_hash.shape[-1], self.n_hashes if self._rehash_each_round else 1, rot_size // 2))

        dropped_vecs = self.dropout_for_hash(vectors_to_hash)
        rotated_vecs = tf.einsum('btf,bfhi->bhti', dropped_vecs, random_rotations)

        if self._rehash_each_round:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            buckets = tf.math.argmax(rotated_vecs, axis=-1)
            # buckets is now (self.n_hashes, seq_len). Next we add offsets so that
            # bucket numbers from different hashing rounds don't overlap.
            offsets = tf.range(self.n_hashes)
            offsets = tf.reshape(offsets * n_buckets, (1, -1, 1))
            offsets = tf.cast(offsets, tf.int64)
            buckets = tf.reshape(buckets + offsets, (batch_size, -1,))
        else:
            rotated_vecs = tf.concat([rotated_vecs, -rotated_vecs], axis=-1)
            # In this configuration, we map each item to the top self.n_hashes buckets
            rotated_vecs = tf.squeeze(rotated_vecs, axis=0)
            bucket_range = tf.range(rotated_vecs.shape[-1])
            bucket_range = tf.reshape(bucket_range, (1, -1))
            bucket_range = tf.broadcast_to(bucket_range, rotated_vecs.shape)

            _, buckets = sort_key_val(rotated_vecs, bucket_range, axis=-1)
            buckets = buckets[:, -self.n_hashes:]

            h, *_ = buckets.shape
            buckets = tf.reshape(buckets.permute((*_, h)), (-1,))

        return buckets

    # noinspection PyMethodOverriding
    def call(self, qk, v):
        batch_size, seq_len, _ = qk.shape
        device = qk.device

        n_buckets = seq_len // self.bucket_size
        n_bins = n_buckets

        buckets = self.hash_vectors(n_buckets, qk)
        # We use the same vector as both a query and a key.
        assert int(buckets.shape[1]) == self.n_hashes * seq_len

        ticker = tf.expand_dims(tf.range(self.n_hashes * seq_len), axis=0)
        buckets_and_t = seq_len * buckets + tf.cast((ticker % seq_len), tf.int64)
        buckets_and_t = tf.stop_gradient(buckets_and_t)

        # Hash-based sort ("s" at the start of variable names means "sorted")
        sbuckets_and_t, sticker = sort_key_val(buckets_and_t, ticker, axis=-1)
        _, undo_sort = sort_key_val(sticker, ticker, axis=-1)
        del ticker

        sbuckets_and_t = tf.stop_gradient(sbuckets_and_t)
        sticker = tf.stop_gradient(sticker)
        undo_sort = tf.stop_gradient(undo_sort)

        st = (sticker % seq_len)
        sqk = batched_index_select(qk, st)
        sv = batched_index_select(v, st)

        # Split off a "bin" axis so that attention only occurs within chunks.
        bq_t = bkv_t = tf.reshape(st, (batch_size, self.n_hashes * n_bins, -1))
        bqk = tf.reshape(sqk, (batch_size, self.n_hashes * n_bins, -1, sqk.shape[-1]))
        bv = tf.reshape(sv, (batch_size, self.n_hashes * n_bins, -1, sv.shape[-1]))
        bq_buckets = bkv_buckets = tf.reshape(sbuckets_and_t // seq_len, (batch_size, self.n_hashes * n_bins, -1))

        # Hashing operates on unit-length vectors. Unnormalized query vectors are
        # fine because they effectively provide a learnable temperature for the
        # attention softmax, but normalizing keys is needed so that similarity for
        # the purposes of attention correctly corresponds to hash locality.
        bq = bqk
        bk = make_unit_length(bqk)

        # Allow each chunk to attend within itself, and also one chunk back. Chunk
        # boundaries might occur in the middle of a sequence of items from the
        # same bucket, so this increases the chances of attending to relevant items.
        def look_one_back(x):
            x_extra = tf.concat([x[:, -1:, ...], x[:, :-1, ...]], axis=1)
            return tf.concat([x, x_extra], axis=2)

        bk = look_one_back(bk)
        bv = look_one_back(bv)
        bkv_t = look_one_back(bkv_t)
        bkv_buckets = look_one_back(bkv_buckets)

        # Dot-product attention.
        dots = tf.einsum('bhie,bhje->bhij', bq, bk) * (bq.shape[-1] ** -0.5)

        # Causal masking
        if self.causal:
            mask = bq_t[:, :, :, None] < bkv_t[:, :, None, :]
            dots = tf.math.multiply(dots, tf.cast(mask, tf.float32)) + (1 - tf.cast(mask, tf.float32)) * float('-inf')
            del mask

        # Mask out attention to self except when no other targets are available.
        self_mask = bq_t[:, :, :, None] == bkv_t[:, :, None, :]
        dots = tf.math.multiply(dots, tf.cast(self_mask, tf.float32)) + (1 - tf.cast(self_mask, tf.float32)) * (- 1e5)
        del self_mask

        # Mask out attention to other hash buckets.
        if not self._attend_across_buckets:
            bucket_mask = bq_buckets[:, :, :, None] != bkv_buckets[:, :, None, :]
            dots = tf.math.multiply(dots, tf.cast(bucket_mask, tf.float32)) + (
                    1 - tf.cast(bucket_mask, tf.float32)) * float('-inf')
            del bucket_mask

        # Don't double-count query-key pairs across multiple rounds of hashing.
        # There are two possible strategies here. (1) The default is to count how
        # many times a query-key pair is repeated, and to lower its log-prob
        # correspondingly at each repetition. (2) When hard_k is set, the code
        # instead masks all but the first occurence of each query-key pair.
        if not self._allow_duplicate_attention:
            locs1 = undo_sort // bq_t.shape[-1]
            locs2 = (locs1 + 1) % (self.n_hashes * n_bins)
            if not self._attend_across_buckets:
                locs1 = buckets * (self.n_hashes * n_bins) + locs1
                locs2 = buckets * (self.n_hashes * n_bins) + locs2
            locs = tf.transpose(
                tf.concat([
                    tf.reshape(locs1, (batch_size, self.n_hashes, seq_len)),
                    tf.reshape(locs2, (batch_size, self.n_hashes, seq_len)),
                ], 1),
                perm=[0, 2, 1])

            slocs = batched_index_select(locs, st)
            b_locs = tf.reshape(slocs, (batch_size, self.n_hashes * n_bins, -1, 2 * self.n_hashes))

            b_locs1 = b_locs[:, :, :, None, :self.n_hashes]

            bq_locs = b_locs1.expand(b_locs.shape[:3] + (2, self.n_hashes))
            bq_locs = tf.reshape(bq_locs, b_locs.shape)
            bkv_locs = look_one_back(b_locs)

            dup_counts = (bq_locs[:, :, :, None, :] == bkv_locs[:, :, None, :, :])
            # for memory considerations, chunk summation of last dimension for counting duplicates
            dup_counts = chunked_sum(dup_counts, chunks=(self.n_hashes * batch_size))
            dup_counts = tf.stop_gradient(dup_counts)
            assert dup_counts.shape == dots.shape
            dots = dots - tf.math.log(dup_counts + 1e-9)
            del dup_counts

        # Softmax.
        dots_logsumexp = tf.math.reduce_logsumexp(dots, axis=-1, keepdims=True)
        dots = tf.exp(dots - dots_logsumexp)
        dots = self.dropout(dots)

        bo = tf.einsum('buij,buje->buie', dots, bv)
        so = tf.reshape(bo, (batch_size, -1, bo.shape[-1]))
        slogits = tf.reshape(dots_logsumexp, (batch_size, -1,))

        class UnsortLogits(tf.keras.layers.Layer):
            def __init__(self):
                super(UnsortLogits, self).__init__()

            # noinspection PyMethodOverriding
            def call(self, _so, _slogits):
                _so, _slogits = tf.stop_gradient(_so), tf.stop_gradient(_slogits)
                _o = batched_index_select(_so, undo_sort)
                _, _logits = sort_key_val(sticker, _slogits, axis=-1)
                return _o, _logits

        unsort_logits = UnsortLogits()
        o, logits = unsort_logits(so, slogits)

        if self.n_hashes == 1:
            out = o
        else:
            o = tf.reshape(o, (batch_size, self.n_hashes, seq_len, o.shape[-1]))
            logits = tf.reshape(logits, (batch_size, self.n_hashes, seq_len, 1))
            probs = tf.exp(logits - tf.math.reduce_logsumexp(logits, axis=1, keepdims=True))
            out = tf.reduce_sum(o * probs, axis=1)

        assert out.shape == v.shape
        return out, buckets


class TFLSHSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, heads=8, bucket_size=64, causal=False, attn_chunks=None,
                 random_rotations_per_head=False, attend_across_buckets=True, allow_duplicate_attention=True, **kwargs):
        super(TFLSHSelfAttention, self).__init__()
        assert d_model % heads == 0, 'dimensions must be divisible by number of heads'

        self.emb = d_model
        self.heads = heads
        self.attn_chunks = heads if attn_chunks is None else attn_chunks

        self.toqk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.tov = tf.keras.layers.Dense(d_model, use_bias=False)
        self.to_out = tf.keras.layers.Dense(d_model)

        self.bucket_size = bucket_size
        self.lsh_attn = TFLSHAttention(bucket_size=bucket_size, causal=causal,
                                       random_rotations_per_head=random_rotations_per_head,
                                       attend_across_buckets=attend_across_buckets,
                                       allow_duplicate_attention=allow_duplicate_attention, **kwargs)

    def call(self, inputs, **kwargs):
        # noinspection PyTupleAssignmentBalance
        b, t, e, h = *inputs.shape, self.heads
        assert t % self.bucket_size == 0, f'Sequence length needs to be divisible by target bucket size - {self.bucket_size}'

        qk = self.toqk(inputs)
        v = self.tov(inputs)

        def merge_heads(v):
            return tf.reshape(tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3]), (b * h, t, -1))

        def split_heads(v):
            return tf.transpose(tf.reshape(v, (b, t, h, -1)), perm=[0, 2, 1, 3])

        qk = merge_heads(qk)
        v = merge_heads(v)

        outputs = process_inputs_chunk(self.lsh_attn, qk, v, chunks=self.attn_chunks)
        attn_out = tf.concat([output for (output, _) in outputs], axis=0)

        out = tf.reshape(split_heads(attn_out), (b, t, e))

        return self.to_out(out)
