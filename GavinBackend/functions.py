from GavinBackend.models import tf
from GavinBackend.preprocessing.text import preprocess_sentence


def loss_function(y_true, y_pred, max_len=52):
    y_true = tf.reshape(y_true, shape=(-1, max_len - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


def evaluate(sentence, model, max_len, s_token, e_token, tokenizer):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(s_token + tokenizer.encode(sentence) + e_token, axis=0)

    output = tf.expand_dims(s_token, 0)

    for i in range(max_len):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq length dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, e_token[0]):
            break

        # concatenated the predicted_id to the output which is given the decoder
        # as its input
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0)


def accuracy(y_true, y_pred, max_len):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, max_len - 1))
    return tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)


def predict(sentence, model, max_len, s_token, e_token, tokenizer):
    prediction = evaluate(sentence, model, max_len, s_token, e_token, tokenizer)

    predicated_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    print("Input: {}".format(sentence))
    print("Output: {}".format(predicated_sentence))

    return predicated_sentence
