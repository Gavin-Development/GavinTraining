from GavinCore import tf, tfds
from GavinCore.preprocessing.text import preprocess_sentence


def accuracy(y_true, y_pred, max_len: int):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, max_len - 1))
    return tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)


def predict(sentence: str, model: tf.keras.models.Model, max_len: int, s_token: list, e_token: list, tokenizer: tfds.deprecated.text.SubwordTextEncoder):
    prediction = evaluate(sentence, model, max_len, s_token, e_token, tokenizer)

    predicated_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])

    print("Input: {}".format(sentence))
    print("Output: {}".format(predicated_sentence))

    return predicated_sentence
