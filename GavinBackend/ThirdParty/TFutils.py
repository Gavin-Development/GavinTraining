# == BEGIN 3RD PARTY ==
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

import tensorflow as tf


def make_unit_length(x, epsilon=1e-6):
    norm = tf.norm(x, ord=2, axis=-1, keepdims=True)
    return tf.math.truediv(x, norm + epsilon)


def sort_key_val(t1, t2, axis=-1):
    values = tf.sort(t1, axis=axis)
    t2 = tf.broadcast_to(t2, t1.shape)
    return values, tf.gather(t2, tf.argsort(t1, axis=axis), axis=axis)


def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return tf.squeeze(tf.gather(values, indices[:, :, None], axis=1))


def process_inputs_chunk(fn, *args, chunks=1):
    chunked_inputs = list(map(lambda x: tf.split(x, chunks, axis=0), args))
    outputs = [fn(*input_pair) for input_pair in zip(*chunked_inputs)]
    return outputs


def chunked_sum(tensor, chunks=1):
    *orig_size, last_dim = tensor.shape
    tensor = tf.reshape(tensor, [-1, last_dim])
    summed_tensors = [c.sum(axis=-1) for c in tf.split(tensor, chunks, axis=0)]
    return tf.reshape(tf.concat(summed_tensors, axis=0), orig_size)


def cache_fn(f):
    cache = None

    def cached_fn(*args, **kwargs):
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache

    return cached_fn

# == END 3RD Party ==
